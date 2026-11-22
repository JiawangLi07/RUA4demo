"""Use the URDF symmetry plane to plan a real left-arm motion in a right-hand chart."""

from types import SimpleNamespace

import numpy as np

OPTICAL_FLIP = np.diag([-1.0, 1.0, 1.0, 1.0])
LOCAL_FLIP = np.diag([1.0, -1.0, 1.0, 1.0])
ARM_SIGNS = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def reflected_frame(frame, mirror, joints):
    k = frame["K"].copy()
    k[0, 2] = frame["depth"].shape[1] - 1 - k[0, 2]
    k[0, 1] *= -1
    return dict(
        frame,
        rgb=frame["rgb"][:, ::-1].copy(),
        depth=frame["depth"][:, ::-1].copy(),
        pose=mirror @ frame["pose"] @ OPTICAL_FLIP,
        K=k,
        q=joints,
        station_pose=frame["pose"],
        image_mirrored=True,
        hand="left",
    )


def opposite(name):
    if name.startswith(("left_", "right_")):
        prefix, tail = name.split("_", 1)
        return ("right_" if prefix == "left" else "left_") + tail
    if name.endswith(("_R", "_L")):
        return name[:-1] + ("L" if name.endswith("_R") else "R")
    return name


def canonical_left(io):
    tree, names = io.tree, io.names
    zeros = dict.fromkeys(names, 0.0)
    mount = tree.pose("dual_arm_mount", zeros)
    mirror = mount @ np.diag([-1.0, 1.0, 1.0, 1.0]) @ np.linalg.inv(mount)
    indices = [names.index(opposite(n)) for n in names]
    signs = np.array(
        [ARM_SIGNS[int(n[5]) - 1] if n.startswith("Joint") else 1.0 for n in names]
    )

    def command():
        return io.commanded_joints()[indices] * signs

    def feedback():
        return getattr(io, "joint_feedback", io.commanded_joints)()[indices] * signs

    def fk(link, angles=None):
        return tree.pose(
            link, dict(zip(names, feedback() if angles is None else angles))
        )

    def camera(role):
        frame = reflected_frame(io.camera(role), mirror, feedback())
        frame["q_command"] = command()
        return frame

    def local_transform(name):
        return np.linalg.solve(
            tree.pose(name, zeros), mirror @ tree.pose(opposite(name), zeros)
        )

    def mirrored_vertices(name, points):
        transform = local_transform(name)
        return points @ transform[:3, :3].T + transform[:3, 3]

    # Use the actual left CAD, including asymmetries, in the canonical link frames.
    model = SimpleNamespace(**vars(io.model))
    model.hand_support = {
        name: mirrored_vertices(name, io.model.hand_support[opposite(name)])
        for name in io.model.hand_support
    }
    forearms = getattr(io.model, "forearm_support", {})
    model.forearm_support = {
        name: mirrored_vertices(name, forearms[opposite(name)]) for name in forearms
    }
    model.fixture_palm = {
        name: mirrored_vertices("right_hand_C_MC", v)
        for name, v in io.model.fixture_palm.items()
    }
    model.limits = dict(io.model.limits, right=io.model.limits["left"])
    from .scene_fixture import box_corners

    bounds = {}
    for physical, b in io.bounds.items():
        name = opposite(physical)
        corners = mirrored_vertices(name, box_corners(b))
        bounds[name] = np.array([corners.min(0), corners.max(0)])
    wrist_k = None if io.wrist_K is None else io.wrist_K.copy()
    if wrist_k is not None:
        wrist_k[0, 2] = 479 - wrist_k[0, 2]
    canonical = SimpleNamespace(**vars(io))
    canonical.arm = [names.index("Joint" + str(i) + "_R") for i in range(1, 8)]
    canonical.hand = [names.index(n) for n in io.model.hand_names["right"]]
    canonical.commanded_joints = command
    canonical.joint_feedback = feedback
    canonical.step = lambda arm, hand: io.step(ARM_SIGNS * arm, hand)
    canonical.camera, canonical.fk = camera, fk
    canonical.model, canonical.bounds = model, bounds
    canonical.palm_from_wrist = (
        None
        if io.palm_from_wrist is None
        else local_transform("right_hand_C_MC") @ io.palm_from_wrist @ OPTICAL_FLIP
    )
    canonical.wrist_K = wrist_k
    canonical.station_reflection = mirror
    return canonical


def reflected_bundle(bundle):
    def mesh(vertices, faces):
        return vertices * LOCAL_FLIP.diagonal()[:3], faces[:, ::-1].copy()

    v, f, c = bundle["visual"]
    return dict(
        visual=(*mesh(v, f), c),
        collisions=[mesh(v, f) for v, f in bundle["collisions"]],
    )
