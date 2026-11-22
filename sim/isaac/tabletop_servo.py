"""Check the open hand against the head depth and correct its pose before closing."""

import numpy as np
from scipy.spatial.transform import Rotation

from .tabletop_control import PALM, NoPath, shifted
from .tabletop_handle import reobserve


def depth_support(ctrl, frame):
    """Count distinct depth pixels agreeing with the exposed hand CAD above the desk."""
    rows = []
    q = frame["q"]
    k, camera, depth = frame["K"], frame["pose"], frame["depth"]
    for link in (
        PALM,
        "right_index_PP",
        "right_middle_PP",
        "right_ring_PP",
        "right_pinky_PP",
        "right_thumb_DP",
    ):
        vertices = ctrl.io.model.hand_support.get(link)
        if vertices is None:
            continue
        tf = ctrl.io.fk(link, q)
        world = vertices @ tf[:3, :3].T + tf[:3, 3]
        world = world[world[:, 2] > ctrl.table + 0.012]
        cam = (world - camera[:3, 3]) @ camera[:3, :3]
        cam = cam[cam[:, 2] > 0.04]
        if not len(cam):
            continue
        uv = np.rint(
            cam[:, :2] * [k[0, 0], k[1, 1]] / cam[:, 2, None] + k[:2, 2]
        ).astype(int)
        inside = (
            (uv[:, 0] >= 0)
            & (uv[:, 1] >= 0)
            & (uv[:, 0] < depth.shape[1])
            & (uv[:, 1] < depth.shape[0])
        )
        u, v = uv[inside].T
        matching = np.abs(depth[v, u] - cam[inside, 2]) < 0.007
        rows.extend((v[matching] * depth.shape[1] + u[matching]).tolist())
    return len(set(rows))


def seat_before_close(ctrl, surface, pose, grasp, opened):
    # Refresh the target while it is still visible beside the open hand.
    updated, registration = reobserve(ctrl, surface, pose, tag="preclose_target")
    shift = registration[:3, 3] - pose[:3, 3]
    goal = shifted(grasp, shift)
    actual = ctrl.io.joint_feedback()[ctrl.io.arm]
    try:
        target = ctrl.ik.solve(goal, actual)
    except NoPath:
        reject(ctrl, opened, "measured arm has no nearby wrist solution")
    if np.max(np.abs(target - actual)) > 1.0:
        reject(ctrl, opened, "large joint branch change during final approach")
    try:
        ctrl.command([actual, target], opened, 0.54)
    except NoPath:
        reject(ctrl, opened, "measured approach trajectory obstructed")
    bias = np.zeros_like(target)
    for iteration in range(4):
        measured = ctrl.io.joint_feedback()[ctrl.io.arm]
        palm = ctrl.ik.fk(measured)
        gap = float(np.linalg.norm(palm[:3, 3] - goal[:3, 3]))
        angle = float(Rotation.from_matrix(palm[:3, :3].T @ goal[:3, :3]).magnitude())
        ctrl.event(
            "palm_tracking",
            iteration=iteration,
            position_error_m=gap,
            rotation_error_deg=float(np.rad2deg(angle)),
            target_shift_m=shift.tolist(),
        )
        if gap <= 0.005 and angle <= np.deg2rad(5):
            frame = ctrl.snapshot(ctrl.camera_roles[0], "palm_at_grasp")
            pixels = depth_support(ctrl, frame)
            if pixels >= 8:
                ctrl.event(
                    "palm_verified",
                    depth_pixels=pixels,
                    target_refreshed=updated is not surface,
                    position_error_m=gap,
                    rotation_error_deg=float(np.rad2deg(angle)),
                    commanded_palm=goal.tolist(),
                )
                ctrl.seat_trim = bias.copy()
                return updated, registration, shift
            reject(ctrl, opened, f"{ctrl.camera_roles[0]} depth supports only {pixels} hand pixels")
        if gap > 0.055 or angle > np.deg2rad(22):
            reject(ctrl, opened, "arm failed to follow final approach")
        # Bounded steady-state compensation, always using physical joint readback.
        bias = np.clip(bias + 0.85 * (target - measured), -0.065, 0.065)
        corrected = np.clip(target + bias, *ctrl.ik.bounds)
        try:
            ctrl.command([ctrl.arm, corrected], opened, 0.22)
        except NoPath:
            reject(ctrl, opened, "bounded wrist correction obstructed")
    reject(ctrl, opened, "open hand outside closure tolerance")


def reject(ctrl, opened, reason):
    measured = ctrl.io.joint_feedback()[ctrl.io.arm]
    ctrl.arm = measured.copy()
    measured_palm = ctrl.ik.fk(measured)
    ctrl.event("palm_gate_rejected", reason=reason)
    # Restart from the measured position, withdrawing locally before a new row.
    try:
        ctrl.move(shifted(measured_palm, [-0.012, 0, 0.036]), opened, 0.36)
    except NoPath:
        ctrl.hold(0.08)
    raise NoPath(reason)


def move_native_sites(sites, old, new):
    transform = new @ np.linalg.inv(old)
    for site in sites:
        site["goal"] = transform[:3, :3] @ site["goal"] + transform[:3, 3]
        site["normal"] = transform[:3, :3] @ site["normal"]


def opposed_touch(ctrl, surface, pose, sites, palm, native_q):
    """Approach opposing native sites together and check visible target motion before squeezing."""
    from .tabletop_handle import approach_contacts

    start = ctrl.io.fk(PALM)
    limits = ctrl.io.model.limits["right"]
    knots = approach_contacts(
        ctrl.hand_contacts, start, palm, ctrl.hand, native_q, sites, limits, ctrl.table
    )
    mid = len(knots) // 2
    for _, q in knots[1 : mid + 1]:
        ctrl.command([ctrl.arm, ctrl.arm], q, 0.046)
    fresh, reg = reobserve(ctrl, surface, pose, tag="opposed_touch_midpoint")
    shift = reg[:3, 3] - pose[:3, 3]
    if np.linalg.norm(shift) > 0.018:
        raise NoPath("visible object displaced beyond local closure during opposition")
    if fresh is not surface and np.linalg.norm(shift) > 0.002:
        move_native_sites(sites, pose, reg)
        palm = shifted(palm, shift)
        ctrl.closing(palm, ctrl.hand, 0.18)
        knots = approach_contacts(
            ctrl.hand_contacts,
            ctrl.io.fk(PALM),
            palm,
            ctrl.hand,
            native_q,
            sites,
            limits,
            ctrl.table,
        )
        tail = knots[2::2]
        surface, pose = fresh, reg
    else:
        tail = knots[mid + 1 :]
        shift = np.zeros(3)
    for _, q in tail:
        ctrl.command([ctrl.arm, ctrl.arm], q, 0.046)
    ctrl.event(
        "opposed_native_touch",
        tracked_shift_m=shift.tolist(),
        links=[s["link"] for s in sites],
    )
    return surface, pose, shift


def follow_during_closure(ctrl, surface, pose, sites, palm, tight, q, others):
    """Follow a displaced long object while the native contacts can still be recovered."""
    from .tabletop_opposition import LowerEdge, long_horizontal_axis

    if long_horizontal_axis(surface.world) is None:
        ctrl.closing(palm, q, 0.65)
        return surface, pose, palm, tight

    correction = None

    def observe(fraction):
        nonlocal correction
        updated, reg = reobserve(
            ctrl, surface, pose, tag=f"closing_{int(fraction * 100)}"
        )
        displacement = float(np.linalg.norm(reg[:2, 3] - pose[:2, 3]))
        turn = float(
            np.rad2deg(Rotation.from_matrix(reg[:3, :3] @ pose[:3, :3].T).magnitude())
        )
        ctrl.event(
            "closing_displacement",
            fraction=fraction,
            registered=updated is not surface,
            move_m=displacement,
            rotation_deg=turn,
        )
        if (
            updated is surface
            or displacement < 0.005
            or (displacement < 0.008 and turn < 7)
        ):
            return False
        correction = updated, reg, displacement, turn
        return True

    ctrl.closing(palm, q, 0.65, observe=observe)
    if correction is None:
        return surface, pose, palm, tight
    updated, reg, displacement, turn = correction
    if displacement > 0.019 or turn > 24:
        raise NoPath("target moved outside the current opposed contacts during closure")
    transform = reg @ np.linalg.inv(pose)
    new_palm, new_tight = transform @ palm, transform @ tight
    if not ctrl.clear(new_palm, ctrl.hand, others) or not ctrl.clear(
        new_palm, q, others
    ):
        raise NoPath("shifted contact frame has no clear local continuation")
    guard = LowerEdge(updated, ctrl.table)
    if ctrl.lower_guard is not None:
        guard.used.update(ctrl.lower_guard.used)
    ctrl.lower_guard = guard
    ctrl.check_closure_path("lightning_grasp", ctrl.arm, new_palm, new_tight)
    ctrl.closing(new_palm, ctrl.hand, 0.24)
    ctrl.closing(new_palm, q, 0.37)
    move_native_sites(sites, pose, reg)
    ctrl.event("follow_during_closure", displacement_m=displacement, rotation_deg=turn)
    return updated, reg, new_palm, new_tight
