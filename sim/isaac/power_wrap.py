"""Seat a cylindrical wall against the palm, then flex several fingers around it."""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .direct_pregrasp import Grasp, rotation_error
from .head_wrist_geometry import hand_shapes
from .head_wrist_trial import FINGERS, PALM, Rejected


def envelope_contacts(links):
    links = {name for name, normal in links.items() if normal > 0}
    fingers = {d for d in FINGERS[1:] if any(k.startswith("right_" + d + "_") for k in links)}
    proximal = any(k.endswith(("_PP", "_MP")) and not k.startswith("right_thumb_") for k in links)
    thumb = any(k.startswith("right_thumb_") for k in links)
    return PALM in links and len(fingers) >= 2 and proximal and (thumb or len(fingers) >= 3)


def phalangeal_support(links):
    """A supported power grip can load multiple phalanges while its central palm is unloaded."""
    active = {name for name, force in links.items() if force > 0}
    digits = {d for d in FINGERS[1:] if any(k.startswith("right_" + d + "_") for k in active)}
    bearing = {d for d in digits if any("right_" + d + "_" + suffix in active for suffix in ("PP", "MP"))}
    metacarpal = any(k in active for k in ("right_hand_C_MC", "right_pinky_MC", "right_thumb_MC"))
    return (
        any(k.startswith("right_thumb_") for k in active)
        and len(digits) >= 3
        and (len(bearing) >= 2 or (len(bearing) >= 1 and metacarpal))
    )


def distances(xz, points):
    """Distance to a link's convex silhouette normal to the cylinder axis."""
    v = points[:, [0, 2]]
    v = v[ConvexHull(v).vertices]
    e = np.roll(v, -1, axis=0) - v
    diff = xz[:, None, :] - v
    f = np.clip(np.sum(diff * e, axis=2) / np.sum(e * e, axis=1), 0, 1)
    d = np.linalg.norm(diff - f[:, :, None] * e, axis=2).min(1)
    cross = e[:, 0] * diff[:, :, 1] - e[:, 1] * diff[:, :, 0]
    return np.where(np.all(cross >= 0, axis=1), -d, d)


def cylinder_frame(cant, roll):
    return (
        Rotation.from_euler("x", -cant, degrees=True) * Rotation.from_euler("z", -roll, degrees=True)
    ).as_matrix()


def hand_fits(scene, name, card, radius, cant, roll):
    bounds = scene.limits["right"].T
    # Express the hand in a frame whose Y axis follows the measured cylinder.
    x, z = np.meshgrid(np.arange(radius + 0.006, radius + 0.027, 0.0025), np.arange(0.057, 0.101, 0.003))
    xz = np.c_[x.ravel(), z.ravel()]
    axes = cylinder_frame(cant, roll)
    hand_names = scene.hand_names["right"]
    ids = {d: np.array([j for j, n in enumerate(hand_names) if n.startswith("right_" + d + "_")]) for d in FINGERS}
    closed = np.tile(np.array(card), (len(xz), 1))
    best = {d: np.full(len(xz), np.inf) for d in FINGERS}
    data = {}
    base_gaps = {}
    palm = None
    for mcp, pip in (
        (0.66, 0.67),
        (0.76, 0.88),
        (0.85, 1.02),
        (0.94, 1.16),
        (1.04, 1.28),
        (1.16, 1.40),
        (1.29, 1.54),
        (1.40, 1.62),
    ):
        q = np.array(card, dtype=float)
        for j, n in enumerate(hand_names):
            if n.startswith("right_thumb_"):
                q[j] *= mcp + 0.04
            elif n.endswith("MCP_FE"):
                q[j] = 0.76 * mcp + 0.24 * q[j]
            elif n.endswith("PIP"):
                q[j] = 0.77 * pip + 0.23 * q[j]
            elif n.endswith("DIP"):
                q[j] = 0.68 * pip + 0.24 * q[j]
        q = np.clip(q, *bounds)
        shapes = {k: v @ axes.T for k, v in hand_shapes(scene, q).items()}
        if palm is None:
            palm = distances(xz, shapes[PALM]) - radius
        for d in FINGERS:
            root = "right_" + d + "_"
            proximal = (
                np.minimum(
                    distances(xz, shapes[root + "PP"]),
                    distances(xz, shapes[root + ("DP" if d == "thumb" else "MP")]),
                )
                - radius
            )
            pad = shapes[root + "elastomer"]
            distal = distances(xz, pad) - radius
            # Require the distal pad to reach the far half of the wall. Its
            # centroid must straddle the shaft with the palmar face.
            far = pad[:, 0].mean() - xz[:, 0]
            tangent = pad[:, 2].mean() - xz[:, 1]
            if d == "thumb":
                cost = (
                    abs(distal + 0.003)
                    + 0.6 * np.maximum(0, -0.012 - proximal)
                    + 0.7 * np.maximum(0, tangent + 0.004)
                )
            else:
                cost = (
                    abs(distal + 0.004)
                    + 0.8 * abs(proximal + 0.004)
                    + 1.5 * np.maximum(0, 0.40 * radius - far)
                    + 0.6 * np.maximum(0, tangent - 0.82 * radius)
                )
            base_gap = None
            if d in ("thumb", "pinky"):
                base_gap = distances(xz, shapes[root + "MC"]) - radius
                cost += 0.9 * np.maximum(0, 0.001 - base_gap)
            take = cost < best[d]
            if base_gap is not None:
                if d not in base_gaps:
                    base_gaps[d] = base_gap.copy()
                base_gaps[d][take] = base_gap[take]
            closed[np.ix_(take, ids[d])] = q[ids[d]]
            if d not in data:
                data[d] = [proximal.copy(), distal.copy()]
            else:
                data[d][0][take] = proximal[take]
                data[d][1][take] = distal[take]
            best[d][take] = cost[take]
    cost = (
        3.5 * abs(palm + 0.005)
        + 0.75 * best["thumb"]
        + np.mean([best[d] for d in FINGERS[1:]], axis=0)
        + 0.30 * np.maximum(0, xz[:, 1] - 0.085)
        + 2.8 * np.maximum(0, 0.001 - base_gaps["pinky"])
        + 1.6 * np.maximum(0, 0.001 - base_gaps["thumb"])
    )
    i = int(np.argmin(cost))
    q = closed[i].copy()
    qo = np.clip(np.array(card) * 0.14, *bounds)
    return (
        float(cost[i]),
        xz[i],
        qo,
        q,
        dict(
            posture=name,
            power_wrap=True,
            palm_gap_m=float(palm[i]),
            thumb_gap_m=float(data["thumb"][1][i]),
            proximal_gap_m=float(min(data[d][0][i] for d in FINGERS[1:])),
            finger_gaps_m=[float(data[d][1][i]) for d in FINGERS[1:]],
            axis_cant_deg=cant,
            axis_roll_deg=roll,
            metacarpal_gaps_m={d: float(v[i]) for d, v in base_gaps.items()},
        ),
    )


def wrap_grasps(trial, coarse):
    surface = trial.surface
    world = surface.world
    bottom, top = np.quantile(world[:, 2], [0.01, 0.99])
    band = world[(world[:, 2] > bottom + 0.25 * (top - bottom)) & (world[:, 2] < bottom + 0.60 * (top - bottom))]
    a, b = np.quantile(band[:, :2], [0.008, 0.992], axis=0)
    center = (a + b) / 2
    radius = float(np.mean(b - a) / 2)
    library = trial.cfg["pinch_postures"]
    choices = trial.cfg.get("wrap_postures", ("3_Medium_Wrap", "1_Large_Diameter", "2_Small_Diameter", "30_Palmar"))
    if coarse is not None:
        choices = [coarse.details["posture"]]
    cache = getattr(trial, "wrap_fit_cache", {})
    trial.wrap_fit_cache = cache
    home = trial.palm()
    proposals = []
    for name in choices:
        if name in trial.used_postures or name not in library:
            continue
        cant = int(trial.cfg.get("wrap_cant_deg", 26))
        roll = int(trial.cfg.get("wrap_roll_deg", 23))
        key = (name, round(radius, 3), cant, roll)
        if key not in cache:
            cache[key] = hand_fits(trial.scene, name, library[name], key[1], cant, roll)
        cost, (cx, cz), opened, closed, meta = cache[key]
        opened_shapes = hand_shapes(trial.scene, opened)
        for sign in (1, -1):
            for yaw in (-15, 10, 35, -40, 60, -65, 85, 110, -90, 135):
                az = np.deg2rad(yaw)
                z_axis = np.array([np.cos(az), np.sin(az), 0])
                y_axis = np.array([0.0, 0.0, float(sign)])
                base = np.stack([np.cross(y_axis, z_axis), y_axis, z_axis], axis=1)
                rot = base @ cylinder_frame(cant, roll)
                for level in (0.38, 0.46, 0.54):
                    h = bottom + level * (top - bottom)
                    goal = np.eye(4)
                    goal[:3, :3] = rot
                    goal[:3, 3] = np.r_[center, h] - base @ np.array([cx, 0, cz])
                    if coarse is not None and (
                        rotation_error(goal, coarse.close) > 0.12
                        or np.linalg.norm(goal[:3, 3] - coarse.close[:3, 3]) > 0.032
                    ):
                        continue
                    extent_y = min(np.min(v @ rot[2]) for v in opened_shapes.values())
                    if goal[2, 3] + extent_y < surface.table + 0.002:
                        continue
                    # Retract mainly normal to the palm so the open fingers can straddle the cylinder.
                    for retreat in (0.020, 0.030, 0.042, 0.055):
                        pre = goal.copy()
                        pre[:3, 3] -= rot @ np.array([retreat, 0, 0.006])
                        if surface.clear(opened_shapes, pre, 0.0015):
                            break
                    else:
                        continue
                    info = dict(
                        meta,
                        local_center_xz_m=[float(cx), float(cz)],
                        radius_m=radius,
                        yaw_deg=yaw,
                        thumb_up=bool(sign > 0),
                        height_fraction=level,
                        palm_standoff_m=retreat,
                    )
                    grip = Grasp(pre, goal, opened, closed, closed, info)
                    score = (
                        cost
                        + 0.22 * np.linalg.norm(pre[:3, 3] - home[:3, 3])
                        + 0.019 * rotation_error(home, pre)
                        + 0.25 * retreat
                        + 0.025 * abs(level - 0.41)
                    )
                    if coarse is not None:
                        score = np.linalg.norm(goal[:3, 3] - coarse.close[:3, 3]) + 0.07 * rotation_error(
                            goal, coarse.close
                        )
                    proposals.append((score, grip))
    proposals.sort(key=lambda row: row[0])
    if coarse is not None:
        grips = [g for _, g in proposals]
    else:
        grips = []
        seen = set()
        for _, g in proposals:
            k = (g.details["posture"], g.details["yaw_deg"], g.details["thumb_up"])
            if k not in seen:
                seen.add(k)
                grips.append(g)
        grips = grips[:35]
    if not trial.cfg.get("fit_finite_surface"):
        return grips
    return snap_to_mesh(trial, grips, coarse)


def snap_to_mesh(trial, grips, coarse):
    """Refine the palmar gap on the observed finite wall, including the bottle shoulder."""
    surface = trial.surface
    faces = [np.unique(np.round(ConvexHull(part).equations, 7), axis=0) for part in surface.world_parts]
    cache = {}
    refined = []
    for rank, grip in enumerate(grips[:16]):
        key = grip.closed.tobytes()
        if key not in cache:
            shapes = hand_shapes(trial.scene, grip.closed)
            compact = {}
            for link in (
                PALM,
                "right_thumb_elastomer",
                "right_pinky_MC",
                "right_thumb_MC",
                "right_index_PP",
                "right_middle_MP",
                "right_ring_MP",
                "right_pinky_MP",
            ):
                v = shapes[link]
                ids = np.unique(
                    np.r_[np.argmax(v @ surface.directions.T, axis=0), np.arange(0, len(v), max(1, len(v) // 48))]
                )
                compact[link] = v[ids]
            cache[key] = (compact, hand_shapes(trial.scene, grip.opened))
        compact, opened = cache[key]
        points = {k: v @ grip.close[:3, :3].T + grip.close[:3, 3] for k, v in compact.items()}
        # Point-to-plane distances are affine in translation, so reuse each
        # link's projections over the pose search.
        planes = {k: [v @ eq[:, :3].T + eq[:, 3] for eq in faces] for k, v in points.items()}
        best = None
        for normal in (-0.003, 0, 0.004, 0.008, 0.012, 0.016):
            for across in (-0.006, 0, 0.006):
                for height in (0, -0.009):
                    shift = normal * grip.close[:3, 0] + across * grip.close[:3, 2] + np.array([0, 0, height])
                    move = [eq[:, :3] @ shift for eq in faces]
                    g = {
                        k: min((d + m).max(1).min() for d, m in zip(all_planes, move))
                        for k, all_planes in planes.items()
                    }
                    prox = np.sort(
                        [g[k] for k in ("right_index_PP", "right_middle_MP", "right_ring_MP", "right_pinky_MP")]
                    )
                    cost = (
                        4.0 * abs(g[PALM] + 0.0055)
                        + 1.0 * abs(g["right_thumb_elastomer"] + 0.004)
                        + 1.2 * np.maximum(0, -0.005 - min(g["right_thumb_MC"], g["right_pinky_MC"]))
                        + 0.4 * np.mean(abs(prox[1:3] + 0.005))
                        + 0.15 * np.linalg.norm(shift)
                    )
                    goal = grip.close.copy()
                    goal[:3, 3] += shift
                    if coarse is not None and np.linalg.norm(goal[:3, 3] - coarse.close[:3, 3]) > 0.030:
                        continue
                    if best is not None and cost >= best[0]:
                        continue
                    for retreat in (0, 0.006, 0.014, 0.024):
                        pre = grip.pre.copy()
                        pre[:3, 3] += shift - retreat * goal[:3, 0]
                        if surface.clear(opened, pre, 0.0015):
                            break
                    else:
                        continue
                    details = dict(
                        grip.details,
                        finite_mesh_gaps_m={k: float(v) for k, v in g.items()},
                        mesh_translation_m=shift.tolist(),
                    )
                    best = (cost, Grasp(pre, goal, grip.opened, grip.closed, grip.limit, details))
        if best is not None:
            cost, solution = best
            preference = (
                0.00075 * rank if coarse is None else np.linalg.norm(solution.close[:3, 3] - coarse.close[:3, 3])
            )
            refined.append((cost + preference, solution))
    return [g for _, g in sorted(refined, key=lambda item: item[0])]


def close_envelope(trial, goal, closed, limit):
    # A free-space wrist setpoint beyond a contacted palm displaces light objects.
    # Keep the loaded arm command and finish wrapping the fingers at that seat.
    scene = trial.scene
    trial.event("seat_palm_and_wrap", goal=goal.tolist())
    start = trial.palm()
    loaded_goal = goal.copy()
    loaded_goal[:3, 3] += 0.008 * goal[:3, 0]
    joints, rotations = trial.joint_path(start, loaded_goal)
    initial = scene.target[0, trial.hand_ids].cpu().numpy().copy()
    fanned = initial.copy()
    fanned[:5] = [-0.14, -0.31, -0.47, -0.30, 0.04]
    names = scene.hand_names["right"]
    arm = joints[0]
    hand = initial.copy()
    # Keep the D405 unobstructed at exposure, then spread the thumb in free space.
    for i in range(1, round(0.40 / trial.dt) + 1):
        f = i / round(0.40 / trial.dt)
        hand = initial + f * f * (3 - 2 * f) * (fanned - initial)
        trial.command(arm, hand, start, allow_contact=True)
    digits = {d: np.array([j for j, n in enumerate(names) if n.startswith("right_" + d + "_")]) for d in FINGERS}
    mcp = np.array([j for j, n in enumerate(names) if "thumb" not in n and n.endswith(("MCP_FE", "MCP_AA", "CMC"))])
    travel = 0.0
    palm_force = 0.0
    on_palm = 0
    seat_reported = False
    loads = {}
    latched = False

    def ease(value):
        t = np.clip(value, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    for i in range(1, round(2.6 / trial.dt) + 1):
        time = i * trial.dt
        if palm_force < 0.35:
            travel = min(1.0, travel + trial.dt / 1.0)
        f = ease(travel)
        u = f * (len(joints) - 1)
        j = min(int(u), len(joints) - 2)
        arm = joints[j] + (u - j) * (joints[j + 1] - joints[j])
        pose = np.eye(4)
        pose[:3, :3] = rotations(f).as_matrix()
        pose[:3, 3] = start[:3, 3] + f * (loaded_goal[:3, 3] - start[:3, 3])
        # Form hooks with PIP/DIP first, bring the thumb across the near flank,
        # then flex the four MCPs into an already opposed cylinder.
        target = fanned + ease((time - 0.17) / 0.85) * (closed - fanned)
        target[mcp] = fanned[mcp] + ease((time - 0.82) / 0.85) * (closed[mcp] - fanned[mcp])
        target[:5] = fanned[:5] + ease((time - 0.42) / 0.85) * (closed[:5] - fanned[:5])
        for digit, ids in digits.items():
            force = sum(v for k, v in loads.items() if k.startswith("right_" + digit + "_"))
            if force < 1.75:
                hand[ids] += np.clip(target[ids] - hand[ids], -1.95 * trial.dt, 1.95 * trial.dt)
            elif force > 5.0:
                hand[ids] += np.clip(fanned[ids] - hand[ids], -0.18 * trial.dt, 0.18 * trial.dt)
        row = trial.command(arm, hand, pose, allow_contact=True)
        loads = row["link_normal_N"]
        if envelope_contacts(loads) and hold_contact_commands(trial, arm, hand, pose):
            latched = True
            break
        palm_force = loads.get(PALM, 0.0)
        on_palm = on_palm + 1 if palm_force > 0.05 else 0
        if not seat_reported and on_palm >= 3:
            seat_reported = True
            trial.event(
                "palm_seated",
                fraction=f,
                links_N=loads,
                actual_palm=trial.palm().tolist(),
                contact_local_m=row.get("palm_contact_local_m"),
            )
    trial.event(
        "load_envelope",
        links_N=loads,
        requested_to_measured_m=float(np.linalg.norm(trial.palm()[:3, 3] - goal[:3, 3])),
    )
    tight = hand.copy()
    actual = trial.palm()
    if not latched:
        for _ in range(round(0.25 / trial.dt)):
            row = trial.command(arm, tight, actual, allow_contact=True)
    if trial.cfg.get("tactile_seating") and not envelope_contacts(row["link_normal_N"]):
        arm, tight = seat_with_tactile(trial, goal, arm, tight, row)
    window = trial.samples[-round(0.10 / trial.dt) :]
    trial.event(
        "envelope_contact_check",
        links_N=window[-1]["link_normal_N"],
        support_ticks=sum(envelope_contacts(r["link_normal_N"]) for r in window),
        required_ticks=len(window),
        object_pose_wxyz=window[-1]["object_pose_wxyz"],
    )
    central = all(envelope_contacts(r["link_normal_N"]) for r in window)
    probe = trial.cfg.get("probe_supported_wrap", False)
    if not central and not (probe and all(phalangeal_support(r["link_normal_N"]) for r in window)):
        raise Rejected("palm and distributed finger support did not persist before lifting")
    trial.event(
        "envelope_confirmed" if central else "phalangeal_support_confirmed", links_N=window[-1]["link_normal_N"]
    )

    def supported(row):
        links = row["link_normal_N"]
        # A supported phalangeal seat may start the 2 cm probe. Check actual palm
        # loading at the top of that probe before committing to the full lift.
        allow_finger_support = probe and not trial.cfg.get("require_palm_hold")
        return envelope_contacts(links) or (allow_finger_support and phalangeal_support(links))

    trial.lift_and_hold(trial.palm(), tight, supported, maintain_preload=True)
    held = trial.samples[-round(2 / trial.dt) :]
    trial.outcome["strict_envelope_ticks"] = sum(envelope_contacts(r["link_normal_N"]) for r in held)
    trial.outcome["central_palm_ticks"] = sum(PALM in r["link_normal_N"] for r in held)


def hold_contact_commands(trial, arm, hand, pose):
    # Continuing a geometric correction after opposition is first achieved
    # can rotate the cylinder off the palm. Hold the loaded motors and measure.
    for _ in range(round(0.12 / trial.dt)):
        row = trial.command(arm, hand, pose, allow_contact=True)
        if not envelope_contacts(row["link_normal_N"]):
            return False
    return True


def tactile_cylinder_center(surfaces, prior, radius):
    """Reconstruct the waist centre from side-wall pressure locations and normals."""
    estimates = []
    for point, normal in surfaces.values():
        point, normal = np.asarray(point), np.asarray(normal)
        horiz = np.linalg.norm(normal[:2])
        if horiz < 0.77 or abs(point[2] - prior[2]) > 0.075:
            continue
        inward = normal[:2] / horiz
        if inward @ (prior[:2] - point[:2]) < 0:
            inward = -inward
        centre = point[:2] + radius * inward
        if np.linalg.norm(centre - prior[:2]) < 0.048:
            estimates.append(centre)
    return np.median(estimates, axis=0) if estimates else prior[:2]


def seat_with_tactile(trial, goal, arm, hand, row):
    """Roll tip-only contacts onto flexed proximal fingers while the thumb opposes."""
    grip = trial.selected_grasp.details
    scene = trial.scene
    angle = grip.get("axis_cant_deg", 0)
    cx, cz = grip["local_center_xz_m"]
    offset = cylinder_frame(angle, grip.get("axis_roll_deg", 0)).T @ [cx, 0, cz]
    origin = goal[:3, 3] + goal[:3, :3] @ offset
    centre = origin.copy()
    initial = trial.palm()
    seat_reference = hand.copy()
    commanded = initial.copy()
    names = scene.hand_names["right"]
    ids = {d: [j for j, n in enumerate(names) if n.startswith("right_" + d + "_")] for d in FINGERS}
    trial.event("roll_into_palm", initial_center_world_m=origin.tolist())
    for _ in range(28):
        observed = tactile_cylinder_center(row["link_contact_surfaces"], centre, grip["radius_m"])
        centre[:2] += np.clip(observed - centre[:2], -0.005, 0.005)
        actual = trial.palm()
        model = actual[:3, 3] + actual[:3, :3] @ offset
        shift = centre[:2] - model[:2]
        # Use remaining palm travel only with other contacts. The manipulator
        # cannot apply an external wrench to the object or set its position.
        if row["link_normal_N"] and PALM not in row["link_normal_N"]:
            shift += 0.003 * actual[:2, 0]
        shift = np.clip(shift, -0.0035, 0.0035)
        dest = commanded.copy()
        dest[:2, 3] = np.clip(actual[:2, 3] + shift, initial[:2, 3] - 0.035, initial[:2, 3] + 0.035)
        if trial.cfg.get("roll_on_fingers"):
            # Rotate about contacts that already hold the far wall. This lets
            # the palmar surface approach without translating the entire cage.
            anchors = [
                point
                for k, (point, _) in row["link_contact_surfaces"].items()
                if any(k.startswith("right_" + d + "_") for d in FINGERS[1:])
                and k.endswith(("_elastomer", "_MP", "_PP"))
            ]
            if len(anchors) >= 2 and PALM not in row["link_normal_N"]:
                pivot = np.median(anchors, axis=0)
                local_pivot = actual[:3, :3].T @ (pivot - actual[:3, 3])
                options = []
                for axis in (1, 2):
                    for sign in (-1, 1):
                        twist = np.zeros(3)
                        twist[axis] = sign * 0.042
                        proposed = actual.copy()
                        proposed[:3, :3] = actual[:3, :3] @ Rotation.from_rotvec(twist).as_matrix()
                        if rotation_error(initial, proposed) > 0.34:
                            continue
                        proposed[:3, 3] = pivot - proposed[:3, :3] @ local_pivot
                        proposed[:3, 3] += 0.0008 * actual[:3, 0]
                        world = scene.hand_support[PALM] @ proposed[:3, :3].T + proposed[:3, 3]
                        radial = np.linalg.norm(world[:, :2] - centre[:2], axis=1)
                        cost = np.quantile(radial, 0.025) + 0.03 * np.linalg.norm(proposed[:3, 3] - initial[:3, 3])
                        options.append((cost, proposed))
                if options:
                    dest = min(options, key=lambda item: item[0])[1]
        joints, _ = trial.joint_path(actual, dest)
        for tick in range(1, 17):
            w = tick / 16
            qarm = arm + w * (joints[-1] - arm)
            loads = row["link_normal_N"]
            if trial.cfg.get("require_palm_hold"):
                trial.share_thumb_load(hand, seat_reference, loads)
            for d, indexes in ids.items():
                digit_load = sum(
                    loads.get("right_" + d + "_" + k, 0) for k in ("PP", "MP", "DP", "elastomer", "MC")
                )
                proximal = any(loads.get("right_" + d + "_" + k, 0) > 0.1 for k in ("PP", "MP"))
                for j in indexes:
                    joint = names[j]
                    if d == "thumb":
                        direction = trial.selected_grasp.closed[j] - hand[j]
                        if digit_load < 1.5:
                            hand[j] += np.clip(direction, -0.40 * trial.dt, 0.40 * trial.dt)
                    elif joint.endswith("MCP_FE") and not proximal and digit_load < 6.0:
                        hand[j] += 0.43 * trial.dt
                    elif joint.endswith(("PIP", "DIP")):
                        if trial.cfg.get("roll_on_fingers") and proximal and PALM not in loads and digit_load < 5.0:
                            hand[j] += 0.30 * trial.dt
                        elif digit_load > 1.5 and not proximal:
                            hand[j] -= 0.38 * trial.dt
                        elif digit_load < 0.8:
                            hand[j] += 0.31 * trial.dt
            hand = np.clip(hand, *scene.limits["right"].T)
            row = trial.command(qarm, hand, dest, allow_contact=True)
            if envelope_contacts(row["link_normal_N"]) and hold_contact_commands(trial, qarm, hand, dest):
                return qarm, hand
        arm = qarm
        commanded = dest
    return arm, hand
