"""Refit each finger to the observed thick handle using wrist RGB-D and the hand kinematic model."""

from copy import copy
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull

from .handle_grasp import TOOL_GRASPS, handle_grasps, handle_sections
from .head_wrist_trial import FINGERS, PALM, Rejected
from .tabletop_vision import wrist_points


def volume_center(parts):
    """Uniform-density first moment of the registered convex mesh components."""
    centers, volumes = [], []
    for vertices in parts:
        origin = vertices.mean(0)
        faces = vertices[ConvexHull(vertices).simplices]
        volume = np.abs(
            np.einsum(
                "ij,ij->i",
                faces[:, 0] - origin,
                np.cross(faces[:, 1] - origin, faces[:, 2] - origin),
            )
        )
        centers.extend((faces.sum(1) + origin) / 4)
        volumes.extend(volume)
    return np.average(centers, weights=volumes, axis=0)


def plan(controller, surface, used, others, cards):
    from .tabletop_meshfit import gravity_residual, thumb_height

    surface = copy(surface)
    surface.world = getattr(surface, "handle_world", surface.world)
    context = SimpleNamespace(
        scene=controller.io.model,
        surface=surface,
        palm=lambda: controller.io.fk(PALM),
        tool_library=controller.handle_library,
        cfg={
            "pinch_postures": cards,
            "gesture_order": TOOL_GRASPS,
            "handle_tilts_deg": (0, 12, 24, -12, 36),
        },
        used_postures=set(used),
    )
    center = volume_center(surface.world_parts)
    envelope = (surface.world.min(0) + surface.world.max(0)) / 2
    pool = []
    for name in TOOL_GRASPS:
        if name in used or name not in cards:
            continue
        context.cfg["gesture_order"] = (name,)
        context.cfg["catalog_grasps"] = {name: {"mode_id": name}}
        try:
            candidates = handle_grasps(context, None)
        except Rejected:
            continue
        for g in candidates:
            if not controller.clear(g.pre, g.opened, others) or not controller.clear(
                g.close, g.closed, others
            ):
                continue
            sites = handle_sites(controller.hand_contacts, surface, g.close, g.closed)
            contacts = np.array([c["goal"] for c in sites])
            # The thumb balances the combined load of the opposing fingers.
            midpoint = (contacts[0] + contacts[1:].mean(0)) / 2
            axis = np.asarray(g.details["handle_axis_xy"])
            lever = max(abs((midpoint[:2] - c[:2]) @ axis) for c in (center, envelope))
            gaps = [
                np.linalg.norm(
                    controller.hand_contacts.point_jac(c, g.close, g.closed)[0]
                    - c["goal"]
                )
                for c in sites
            ]
            wrench = gravity_residual(sites, center)
            quality = lever + 0.035 * wrench + 0.45 * (gaps[0] + np.mean(gaps[1:]))
            dz = thumb_height(sites)
            quality += 2.0 * max(0.0, -0.002 - dz) + 0.20 * max(gaps[1:])
            pool.append((quality, lever, wrench, dz, name, g))
    for _, lever, wrench, dz, name, g in sorted(pool, key=lambda row: row[0]):
        for initial in (
            controller.arm,
            controller.mid_arm,
            controller.reference_arm,
        ):
            try:
                q = controller.ik.solve(g.pre, initial)
                controller.lift_ik(g.close, q)
            except ValueError:
                continue
            controller.hints[g.pre.round(5).tobytes()] = q
            seed = {
                "posture": name,
                "origin": "wrist_handle_cross_sections",
                "q_open": g.opened,
                "q_touch": g.closed,
                "q_tight": g.closed,
            }
            controller.event(
                "handle_plan",
                posture=name,
                pregrasp_palm=g.pre.tolist(),
                palm=g.close.tolist(),
                q_open=g.opened.tolist(),
                q_target=g.closed.tolist(),
                uniform_density_center_m=center.tolist(),
                thumb_height_m=round(dz, 5),
                axial_balance_error_m=round(float(lever), 5),
                wrench_residual=round(float(wrench), 5),
                **{
                    k: v
                    for k, v in g.details.items()
                    if k.startswith("handle_") or k in ("tilt_deg", "patch_world_m")
                },
            )
            return seed, {"pre": g.pre, "grasp": g.close, "tight": g.close}, 0
    return None


def reobserve(controller, surface, pose, tag="handle_seated"):
    samples = []
    live = {}
    for role in getattr(controller, "camera_roles", ("head", "wrist")):
        fr = controller.snapshot(role, tag)
        cloud = wrist_points(fr, controller.io, surface, controller.table, 0.022)
        if len(cloud) >= 30:
            live[role] = cloud
            samples.append(cloud[:: max(1, len(cloud) // 410)])
    if samples:
        try:
            new, reg, info = controller.models.fit(
                controller.name, np.vstack(samples), controller.table, surface
            )
            dx = np.linalg.norm(reg[:3, 3] - pose[:3, 3])
            if dx < 0.023:
                controller.event(
                    "handle_reobserved"
                    if tag == "handle_seated"
                    else "mesh_reobserved",
                    points=sum(map(len, samples)),
                    shift_m=round(float(dx), 5),
                    median_m=info["median_residual_m"],
                )
                from .tabletop_vision import add_live_samples

                add_live_samples(new, live.get("head", []), live.get("wrist", []))
                new.allow_hand_table_contact = getattr(surface, "allow_hand_table_contact", False)
                return new, reg
        except ValueError:
            pass
    return surface, pose


def seat(controller, surface, pose, closed):
    from .tabletop_meshfit import close_from_mesh

    controller.hold(0.10)
    surface, pose = reobserve(controller, surface, pose)
    contacts = handle_sites(
        controller.hand_contacts, surface, controller.ik.fk(controller.arm), closed
    )
    controller.event(
        "handle_surface_targets",
        targets={
            c["link"]: {
                "point_m": c["goal"].tolist(),
                "normal": c["normal"].tolist(),
                "local_width_m": float(c["section_width"]),
            }
            for c in contacts
        },
    )
    close_from_mesh(controller, contacts)
    updated, new_pose = reobserve(controller, surface, pose, tag="before_probe")
    movement = new_pose @ np.linalg.inv(pose)
    for site in contacts:
        site["goal"] = movement[:3, :3] @ site["goal"] + movement[:3, 3]
        site["normal"] = movement[:3, :3] @ site["normal"]
    if np.linalg.norm(new_pose[:3, 3] - pose[:3, 3]) >= 0.002:
        close_from_mesh(controller, contacts)
    return updated, new_pose, contacts


def handle_sites(hand, surface, palm, closed):
    """Assign individual pads to the two sides of locally fitted handle cross-sections."""
    from .tabletop_meshfit import unit

    origin, axis, normal, sections = handle_sections(
        getattr(surface, "handle_world", surface.world)
    )
    along, across = np.r_[axis, 0.0], np.r_[normal, 0.0]
    base = np.r_[origin, 0.0]
    sign = np.sign(palm[:2, 2] @ normal) or 1.0
    contacts = []
    for digit in FINGERS:
        outward = sign * (-1 if digit == "thumb" else 1)
        link = "right_" + digit + "_elastomer"
        frame, _ = hand.link(link, palm, closed)
        vertices = hand.io.model.hand_support[link]
        inner = np.argsort(outward * (vertices @ (frame[:3, :3].T @ across)))[:8]
        anchor = vertices[inner].mean(0)
        point = frame[:3, :3] @ anchor + frame[:3, 3]
        station = np.clip(
            along @ (point - base), sections[0, 0] + 0.004, sections[-1, 0] - 0.004
        )
        lateral, z, radius, half_height = [
            np.interp(station, sections[:, 0], sections[:, k]) for k in (1, 2, 3, 4)
        ]
        below = min(0.005, 0.24 * half_height)
        side = radius * np.sqrt(1 - (below / half_height) ** 2)
        goal = base + station * along + (lateral + outward * side) * across
        goal[2] = z - below
        normal_world = unit(
            outward * side / radius**2 * across
            - np.array([0, 0, below / half_height**2])
        )
        contacts.append(
            {
                "link": link,
                "anchor": anchor,
                "hand_normal": unit(anchor - vertices.mean(0)),
                "goal": goal,
                "normal": normal_world,
                "section_width": 2 * side,
                "weight": 1.35 if digit == "thumb" else 1.0,
            }
        )
    return contacts


def approach_contacts(
    hand, start_palm, end_palm, q_open, q_touch, sites, limits, table
):
    """Move all finger pads from their open CAD positions to the target sides together."""
    from scipy.spatial.transform import Rotation, Slerp

    rotor = Slerp([0, 1], Rotation.from_matrix([start_palm[:3, :3], end_palm[:3, :3]]))
    initial = np.array([hand.point_jac(c, start_palm, q_open)[0] for c in sites])
    goals = np.array([c["goal"] - 0.0015 * c["normal"] for c in sites])
    goals[:, 2] = np.maximum(goals[:, 2], table + 0.0015)
    q = q_open.copy()
    path = [(start_palm.copy(), q.copy())]
    for fraction in np.linspace(0, 1, 19)[1:]:
        f = fraction * fraction * (3 - 2 * fraction)
        palm = end_palm.copy()
        palm[:3, :3] = rotor(f).as_matrix()
        palm[:3, 3] = (1 - f) * start_palm[:3, 3] + f * end_palm[:3, 3]
        targets = (1 - f) * initial + f * goals
        prior = (1 - f) * q_open + f * q_touch
        for _ in range(3):
            j, err = [], []
            for c, goal in zip(sites, targets):
                pt, normal, jp, jn = hand.point_jac(c, palm, q)
                j.extend([jp[:, 6:], 0.003 * f * jn[:, 6:]])
                err.extend([goal - pt, 0.003 * f * (-c["normal"] - normal)])
            j = np.vstack(j)
            step = np.linalg.solve(
                j.T @ j + 0.000045 * np.eye(len(q)),
                j.T @ np.concatenate(err) + 0.000005 * (prior - q),
            )
            q = np.clip(q + np.clip(step, -0.07, 0.07), *limits.T)
        path.append((palm, q.copy()))
    return path


def surround_handle(controller, surface, palm, q_touch):
    hand = controller.hand_contacts
    sites = handle_sites(hand, surface, palm, q_touch)
    knots = approach_contacts(
        hand,
        controller.ik.fk(controller.arm),
        palm,
        controller.hand,
        q_touch,
        sites,
        controller.io.model.limits["right"],
        controller.table,
    )
    arms = [controller.arm.copy()]
    for pose, _ in knots[1:]:
        arms.append(controller.ik.solve(pose, arms[-1]))
    controller.event(
        "synchronous_handle_approach", q_path=[q.tolist() for _, q in knots]
    )
    for i, (_, q_next) in enumerate(knots[1:]):
        q = knots[i][1]
        ticks = max(2, round(0.055 / controller.io.dt))
        for f in np.arange(1, ticks + 1) / ticks:
            controller.io.step(
                (1 - f) * arms[i] + f * arms[i + 1], (1 - f) * q + f * q_next
            )
    controller.arm, controller.hand = arms[-1].copy(), knots[-1][1].copy()
