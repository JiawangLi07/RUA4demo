"""Refit each finger to the observed thick handle with wrist RGB-D and encoders."""

from copy import copy
from types import SimpleNamespace

import numpy as np

from .handle_grasp import TOOL_GRASPS, handle_grasps, handle_sections
from .head_wrist_trial import FINGERS, PALM, Rejected
from .tabletop_vision import wrist_points


def plan(controller, surface, used, others, cards):
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
            for initial in (
                controller.arm,
                controller.mid_arm,
                controller.reference_arm,
            ):
                try:
                    q = controller.ik.solve(g.pre, initial)
                    controller.ik.solve(g.close, q)
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
    for role in ("head", "wrist"):
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
                return new, reg
        except ValueError:
            pass
    return surface, pose


def seat(controller, surface, pose, closed):
    io = controller.io
    names = io.model.hand_names["right"]
    joints = {
        d: [i for i, n in enumerate(names) if n.startswith("right_" + d + "_")]
        for d in FINGERS
    }
    lo, hi = io.model.limits["right"].T
    # Preserve the fresh closure target while the table-supported handle settles.
    controller.hold(0.10)
    surface, pose = reobserve(controller, surface, pose)
    origin, axis, normal, sections = handle_sections(
        getattr(surface, "handle_world", surface.world)
    )
    along, across = np.r_[axis, 0.0], np.r_[normal, 0.0]
    base = np.r_[origin, 0.0]
    palm = io.fk(PALM)
    sign = np.sign(palm[:2, 2] @ normal) or 1.0
    controller.event("seat_handle_encoders")
    stable = 0
    for iteration in range(23):
        measured = io.q()
        state = dict(zip(io.names, measured))
        actual = measured[io.hand]
        hand = controller.hand.copy()
        target = hand.copy()
        gaps = {}
        deflection = {}
        for d, ids in joints.items():
            outward = sign * (-1 if d == "thumb" else 1)
            link = "right_" + d + "_elastomer"
            t = io.tree.pose(link, state)
            local = io.model.hand_support[link]
            inner = np.argsort(outward * (local @ (t[:3, :3].T @ across)))[:8]
            anchor = local[inner].mean(0)
            point = t[:3, :3] @ anchor + t[:3, 3]
            station = np.clip(
                along @ (point - base), sections[0, 0] + 0.004, sections[-1, 0] - 0.004
            )
            u, z, radius, half = [
                np.interp(station, sections[:, 0], sections[:, k]) for k in (1, 2, 3, 4)
            ]
            below = min(0.005, 0.24 * half)
            radius *= np.sqrt(1 - (below / half) ** 2)
            gaps[d] = float(outward * ((point - base) @ across - u) - radius)
            depth = min(0.010, 0.0025 + iteration * 0.00065)
            goal = base + station * along + (u + outward * (radius - depth)) * across
            goal[2] = z - below
            jac = []
            for j in ids:
                altered = dict(state)
                altered[names[j]] += 0.001
                ti = io.tree.pose(link, altered)
                jac.append((ti[:3, :3] @ anchor + ti[:3, 3] - point) / 0.001)
            jac = np.array(jac).T
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + 0.000055 * np.eye(3), goal - point
            )
            target[ids] += np.clip(actual[ids] + dq - hand[ids], -0.095, 0.095)
            flex = [j for j in ids if names[j].endswith(("_FE", "PIP", "DIP", "_IP"))]
            deflection[d] = float(np.max(np.abs(hand[flex] - actual[flex])))
        target = np.clip(target, lo, hi)
        for k in range(1, 17):
            io.step(controller.arm, hand + (target - hand) * k / 16.0)
        controller.hand = target
        loaded = [d for d in FINGERS if deflection[d] > 0.042 and abs(gaps[d]) < 0.010]
        stable = stable + 1 if "thumb" in loaded and len(loaded) >= 3 else 0
        if iteration >= 10 and stable >= 3:
            break
    controller.event(
        "handle_encoder_readback",
        pad_gap_m={d: round(g, 4) for d, g in gaps.items()},
        joint_deflection_rad={d: round(g, 4) for d, g in deflection.items()},
        candidate_digits=loaded,
    )
    return reobserve(controller, surface, pose)
