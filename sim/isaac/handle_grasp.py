"""Plan an opposed multi-finger grasp on the wide section of a horizontal handle."""

from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .direct_pregrasp import Grasp, rotation_error
from .head_wrist_geometry import hand_shapes
from .handle_contacts import handle_observations, loaded_digits, tool_supported
from .head_wrist_trial import FINGERS, Rejected

TOOL_GRASPS = ("5_Light_Tool", "3_Medium_Wrap", "2_Small_Diameter", "1_Large_Diameter")


def handle_sections(points):
    """Measure cross sections and keep the contiguous, thick handle away from the shaft."""
    origin = points[:, :2].mean(0)
    centered = points[:, :2] - origin
    _, axes = np.linalg.eigh(np.cov(centered.T))
    along = axes[:, -1]
    across = np.array([along[1], -along[0]])
    t = centered @ along
    lo, hi = np.quantile(t, (0.004, 0.996))
    sections = []
    for pos in np.linspace(lo, hi, 35)[1:-1]:
        strip = points[np.abs(t - pos) < max(0.004, (hi - lo) / 25)]
        if len(strip) >= 10:
            sides = np.quantile((strip[:, :2] - origin) @ across, (0.02, 0.98))
            levels = np.quantile(strip[:, 2], (0.02, 0.98))
            sections.append([pos, sides.mean(), levels.mean(), np.ptp(sides) / 2, np.ptp(levels) / 2])
    if len(sections) < 8:
        raise Rejected("insufficient cross sections on the observed handle")
    data = np.array(sections)
    wide = (data[:, 3] >= 0.65 * np.quantile(data[:, 3], 0.9)) & (data[:, 4] >= 0.007)
    runs = np.split(np.flatnonzero(wide), np.flatnonzero(np.diff(np.flatnonzero(wide)) > 1) + 1)
    ids = max(runs, key=len)
    if len(ids) < 3 or np.ptp(data[ids, 0]) < 0.045:
        raise Rejected("handle is too short for distributed finger contacts")
    return origin, along, across, data[ids]


def _hand_library(scene, card):
    names = scene.hand_names["right"]
    bounds = scene.limits["right"].T
    samples = []
    clouds = {d: [] for d in FINGERS}
    lower_support = {d: [] for d in FINGERS}
    for m, p in product(np.linspace(0.30, 1.32, 8), np.linspace(0.28, 1.56, 9)):
        q = np.array(card, dtype=float)
        for j, n in enumerate(names):
            if n == "right_thumb_CMC_FE":
                q[j] = 0.65 * m + 0.32 * q[j]
            elif n == "right_thumb_MCP_FE":
                q[j] = 0.70 * p - 0.30 + 0.23 * q[j]
            elif n == "right_thumb_IP":
                q[j] = 0.75 * p + 0.22 * q[j]
            elif n.endswith("MCP_FE"):
                q[j] = 0.88 * m + 0.12 * q[j]
            elif n.endswith("PIP"):
                q[j] = 0.86 * p + 0.14 * q[j]
            elif n.endswith("DIP"):
                q[j] = 0.70 * p + 0.18 * q[j]
        q = np.clip(q, *bounds)
        shapes = hand_shapes(scene, q)
        samples.append(q)
        for d in FINGERS:
            stem = "right_" + d + "_"
            clouds[d].append(shapes[stem + "elastomer"])
            lower_support[d].append(np.vstack([v for k, v in shapes.items() if k.startswith(stem)]))
    return (
        np.asarray(samples),
        {d: np.asarray(v) for d, v in clouds.items()},
        {d: np.asarray(v) for d, v in lower_support.items()},
    )


def _fit_hand(scene, surface, section, axis, normal, card, library, tilt):
    origin, data, center_s = section
    center_u, z = [np.interp(center_s, data[:, 0], data[:, i]) for i in (1, 2)]
    center = np.r_[origin + center_s * axis[:2] + center_u * normal[:2], z]
    base = np.stack(([0.0, 0.0, -1.0], axis, normal), axis=1)
    rot = base @ Rotation.from_euler("y", tilt, degrees=True).as_matrix()
    grid, pads, links = library
    indexes = {
        d: [i for i, n in enumerate(scene.hand_names["right"]) if n.startswith("right_" + d + "_")] for d in FINGERS
    }
    # One mixed posture can use independently fitted joints from each finger's library.
    support = {}
    for d in FINGERS:
        outward = -1 if d == "thumb" else 1
        projected = pads[d] @ (rot.T @ normal)
        edge_ids = np.argpartition(outward * projected, 5, axis=1)[:, :6]
        edge = np.take_along_axis(pads[d], edge_ids[:, :, None], axis=1).mean(axis=1) @ rot.T
        low = (links[d] @ rot[2]).min(1)
        support[d] = (edge @ normal, edge @ axis, edge[:, 2], low, outward)
    fits = []
    for cx, cz in product((0.051, 0.058, 0.065, 0.072, 0.079), (0.084, 0.092, 0.100, 0.108, 0.116)):
        wrist = center - rot @ [cx, 0.010, cz]
        q = np.asarray(card).copy()
        errors = []
        for d in FINGERS:
            lat, long, ht, floor, outward = support[d]
            station = long + (wrist[:2] - origin) @ axis[:2]
            lateral = lat + (wrist[:2] - origin) @ normal[:2]
            wall = np.interp(station, data[:, 0], data[:, 1])
            radius = np.interp(station, data[:, 0], data[:, 3])
            mid = np.interp(station, data[:, 0], data[:, 2])
            gap = outward * (lateral - wall) - radius
            err = (
                abs(gap + 0.0045)
                + 0.82 * abs(ht + wrist[2] - mid - 0.0005)
                + 6 * np.maximum(0, surface.table + 0.0025 - floor - wrist[2])
                + 1.5 * (np.maximum(data[0, 0] - station, 0) + np.maximum(station - data[-1, 0], 0))
            )
            cols = indexes[d]
            err += 0.0005 * np.linalg.norm(grid[:, cols] - np.asarray(card)[cols], axis=1)
            k = int(np.argmin(err))
            q[cols] = grid[k, cols]
            errors.append(err[k])
        value = 1.3 * errors[0] + np.mean(sorted(errors[1:])[:3]) + 0.25 * np.max(errors[1:])
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = wrist
        fits.append((value, pose, q, center))
    return sorted(fits, key=lambda r: r[0])[:4]


def handle_grasps(trial, coarse):
    cfg, scene, surface = trial.cfg, trial.scene, trial.surface
    origin, axis, normal, bands = handle_sections(surface.world)
    library = cfg["pinch_postures"]
    allowed = cfg.get("catalog_grasps", TOOL_GRASPS)
    order = cfg.get("gesture_order", allowed)
    names = [
        n for n in dict.fromkeys((*order, *allowed)) if n in allowed and n in library and n not in trial.used_postures
    ]
    if not names:
        raise Rejected("multi-finger handle postures exhausted")
    name = coarse.details["posture"] if coarse else names[0]
    card = np.asarray(library[name])
    if not hasattr(trial, "tool_library"):
        trial.tool_library = {}
    key = (name, tuple(card))
    if key not in trial.tool_library:
        trial.tool_library[key] = _hand_library(scene, card)
    pool = trial.tool_library[key]
    rows = []
    for sign in (1, -1):
        a, n = sign * np.r_[axis, 0], sign * np.r_[normal, 0]
        profile = bands.copy()
        if sign < 0:
            profile = profile[::-1].copy()
            profile[:, :2] *= -1
        for frac in (0.49, 0.58):
            station = profile[0, 0] + frac * np.ptp(profile[:, 0])
            for tilt in cfg.get("handle_tilts_deg", (12, 24, 0)):
                for score, pose, q, center in _fit_hand(
                    scene, surface, (origin, profile, station), a, n, card, pool, tilt
                ):
                    if coarse and (
                        rotation_error(pose, coarse.close) > 0.30
                        or np.linalg.norm(pose[:3, 3] - coarse.close[:3, 3]) > 0.035
                    ):
                        continue
                    cost = score + 0.025 * rotation_error(trial.palm() if coarse is None else coarse.close, pose)
                    if coarse:
                        cost += np.linalg.norm(pose[:3, 3] - coarse.close[:3, 3])
                    rows.append((cost, pose, q, center, tilt))
    candidates = []
    for cost, pose, q, center, tilt in sorted(rows, key=lambda r: r[0]):
        if any(
            rotation_error(pose, g.close) < 0.09 and np.linalg.norm(pose[:3, 3] - g.close[:3, 3]) < 0.009
            for g in candidates
        ):
            continue
        closed = hand_shapes(scene, q)
        if min((v @ pose[2, :3]).min() + pose[2, 3] for v in closed.values()) < surface.table + 0.001:
            continue
        opened = q.copy()
        for j, k in enumerate(scene.hand_names["right"]):
            if k.startswith("right_thumb_"):
                if k.endswith(("MCP_FE", "_IP")):
                    opened[j] -= 0.20
            elif k.endswith(("MCP_FE", "PIP", "DIP")):
                opened[j] -= 0.24
        opened = np.clip(opened, *scene.limits["right"].T)
        shape = hand_shapes(scene, opened)
        for clearance in (0.024, 0.033, 0.045):
            pre = pose.copy()
            pre[2, 3] += clearance
            if surface.clear(shape, pre, 0.002):
                break
        else:
            continue
        details = dict(
            posture=name,
            tool_handle=True,
            patch_world_m=center.tolist(),
            handle_axis_xy=axis.tolist(),
            handle_origin_xy=origin.tolist(),
            handle_bounds_m=bands[[0, -1], 0].tolist(),
            handle_sections_m=bands.tolist(),
            tilt_deg=tilt,
        )
        if cfg.get("catalog_grasps"):
            details["catalog_mode_id"] = cfg["catalog_grasps"][name]["mode_id"]
        candidates.append(Grasp(pre, pose, opened, q, q, details))
        if len(candidates) >= 18:
            break
    return candidates


def close_handle(trial, goal, closed):
    """Close all five digits. Require thumb plus two other loaded digits at every hold step."""
    scene = trial.scene
    start = trial.palm()
    joints, rots = trial.joint_path(start, goal)
    hand = scene.target[0, trial.hand_ids].cpu().numpy().copy()
    initial = hand.copy()
    names = scene.hand_names["right"]
    bounds = scene.limits["right"].T
    row = None
    stable = 0
    trial.event("multifinger_handle_closure")
    for i in range(1, round(2.3 / trial.dt) + 1):
        seconds = i * trial.dt
        f = min(1.0, seconds / 0.9)
        f = f * f * (3 - 2 * f)
        u = f * (len(joints) - 1)
        j = min(int(u), len(joints) - 2)
        arm = joints[j] + (u - j) * (joints[j + 1] - joints[j])
        pose = start.copy()
        pose[:3, :3] = rots(f).as_matrix()
        pose[:3, 3] = start[:3, 3] + f * (goal[:3, 3] - start[:3, 3])
        loads = {} if row is None else row["link_normal_N"]
        for d, ids in trial.digit_indices.items():
            force = sum(loads.get(f"right_{d}_{s}", 0.0) for s in ("PP", "MP", "DP", "elastomer"))
            if force >= 2.5:
                continue
            for k in ids:
                target = initial[k] + f * (closed[k] - initial[k])
                if seconds > 0.90 and names[k].endswith(("MCP_FE", "PIP", "DIP", "_IP")):
                    target += min(0.43, (seconds - 0.90) * 0.55)
                hand[k] += np.clip(target - hand[k], -1.1 * trial.dt, 1.1 * trial.dt)
        np.clip(hand, *bounds, out=hand)
        row = trial.command(arm, hand, pose, allow_contact=True)
        stable = stable + 1 if tool_supported(row["link_normal_N"]) else 0
        if f == 1 and stable >= round(0.16 / trial.dt):
            break
    hand, row = seat_handle(trial, arm, hand, row, pose)
    trial.event("handle_opposition_confirmed", digits=sorted(loaded_digits(row["link_normal_N"])))

    def verify_handle(held):
        grip = next(e for e in reversed(trial.events) if e["stage"] == "refined_grasp")
        anchor = min(trial.samples, key=lambda r: abs(r["time"] - grip["sim_time"]))
        # This physical acceptance check only adjudicates an attempt after the hold.
        support = handle_observations(held, grip, anchor["object_pose_wxyz"])
        if support["opposed_on_handle_ticks"] != len(held):
            raise Rejected("opposed multi-finger loads did not remain on the handle throughout the hold")
        trial.outcome.update(support)
        trial.outcome["minimum_loaded_digits"] = min(len(loaded_digits(r["link_normal_N"])) for r in held)
        trial.outcome["digit_contact_ticks"] = {
            d: sum(d in loaded_digits(r["link_normal_N"]) for r in held) for d in FINGERS
        }

    trial.lift_and_hold(
        trial.palm(),
        hand,
        lambda row: tool_supported(row["link_normal_N"]),
        maintain_preload=True,
        verify_hold=verify_handle,
    )


def seat_handle(trial, arm, hand, row, pose):
    """Seat the pads around the registered handle using measured loads and fingertip FK."""
    scene = trial.scene
    origin, axis, normal, bands = handle_sections(trial.surface.world)
    sign = np.sign(pose[:2, 2] @ normal)
    if not sign:
        sign = 1
    target = np.r_[origin, 0.0]
    across = np.r_[normal, 0.0]
    along = np.r_[axis, 0.0]
    index = trial.digit_indices
    seed = dict(zip(scene.robot.joint_names, trial.actual()))
    palm_inv = np.linalg.inv(scene.tree.pose("right_hand_C_MC", seed))
    names = scene.hand_names["right"]
    lo, hi = scene.limits["right"].T
    pads = {d: "right_" + d + "_elastomer" for d in FINGERS}
    trial.event("seat_handle_fingers")
    stable = 0
    offset = 0.0
    for iteration in range(26):
        palm = trial.palm()
        actual = trial.actual()[trial.hand_ids]
        seed.update(zip(names, actual))
        # Flank contacts on both sides track small horizontal shifts caused by closure.
        estimates = []
        for key, (point, pressure_normal) in row["link_contact_surfaces"].items():
            if key not in pads.values() and not key.endswith(("_MP", "_DP")):
                continue
            point = np.asarray(point)
            station = along @ (point - target)
            lat = across @ (point - target)
            if station < bands[0, 0] - 0.004 or station > bands[-1, 0] + 0.004:
                continue
            u, radius = [np.interp(station, bands[:, 0], bands[:, k]) for k in (1, 3)]
            nx = abs(np.dot(pressure_normal, across))
            if nx > 0.55:
                estimates.append(lat - u - np.sign(lat - u - offset) * nx * radius)
        if estimates:
            offset += np.clip(np.median(estimates) - offset, -0.0012, 0.0012)
            offset = float(np.clip(offset, -0.014, 0.014))
        next_hand = hand.copy()
        for d in FINGERS:
            ids = index[d]
            outward = sign * (-1 if d == "thumb" else 1)
            link = pads[d]
            chain = scene.tree.pose(link, seed)
            world_from_pad = palm @ palm_inv @ chain
            cloud = scene.hand_support[link]
            inner = np.argsort(outward * (cloud @ (world_from_pad[:3, :3].T @ across)))[:8]
            anchor = cloud[inner].mean(0)

            def pad_point(state):
                transform = palm @ palm_inv @ scene.tree.pose(link, state)
                return transform[:3, :3] @ anchor + transform[:3, 3]

            current = pad_point(seed)
            station = np.clip(along @ (current - target), bands[0, 0] + 0.004, bands[-1, 0] - 0.004)
            u, z, radius, half_height = [np.interp(station, bands[:, 0], bands[:, k]) for k in (1, 2, 3, 4)]
            # Aim under the handle midline, with gradually increasing motor preload.
            below = min(0.005, 0.24 * half_height)
            radius *= np.sqrt(1 - (below / half_height) ** 2)
            depth = min(trial.cfg.get("handle_preload_m", 0.010), 0.0025 + iteration * 0.00065)
            goal = target + station * along + (u + offset + outward * (radius - depth)) * across
            goal[2] = z - below
            jac = []
            for k in ids:
                changed = dict(seed)
                changed[names[k]] += 0.001
                jac.append((pad_point(changed) - current) / 0.001)
            jac = np.array(jac).T
            error = goal - current
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.000055 * np.eye(3), error)
            # A loaded motor can be ahead of the measured angle. Keep this deflection
            # until the distal segment reaches the side wall at the target height.
            desired = actual[ids] + dq
            next_hand[ids] += np.clip(desired - hand[ids], -0.095, 0.095)
        next_hand = np.clip(next_hand, lo, hi)
        for tick in range(1, 17):
            q = hand + (next_hand - hand) * (tick / 16)
            row = trial.command(arm, q, pose, allow_contact=True)
            loaded = row["link_normal_N"]
            thumb = sum(v for k, v in loaded.items() if k.startswith("right_thumb_") and not k.endswith("_MC"))
            others = sum(v for k, v in loaded.items() if any(k.startswith("right_" + d + "_") for d in FINGERS[1:]))
            stable = stable + 1 if tool_supported(loaded) and min(thumb, others) > 1.3 else 0
        hand = next_hand
        if iteration >= 10 and stable >= round(0.20 / trial.dt):
            break
    trial.event("handle_grip_load", links_N=row["link_normal_N"], sustained_steps=stable)
    if stable < round(0.12 / trial.dt):
        raise Rejected("insufficient sustained multi-finger force on the handle")
    return hand, row
