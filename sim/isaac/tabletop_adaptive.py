"""Table-relative reachable grasps from measured sections and the Sharpa link geometry."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .head_wrist_trial import PALM
from .paths import RIGHT_JOINTS
from .power_wrap import cylinder_frame
from .tabletop_control import NoPath, shifted
from .tabletop_grips import patch_candidates
from .tabletop_lightning import central_palm_samples

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
FLEX = np.array([n.endswith(("_FE", "_IP", "_PIP", "_DIP")) for n in RIGHT_JOINTS])


def wants_palm(surface):
    lo, hi = np.quantile(surface.world, [0.005, 0.995], axis=0)
    span = hi - lo
    return (
        span[2] >= 0.090 and min(span[:2]) >= 0.042 and span[2] > 1.18 * max(span[:2])
    )


def nearest_arm(ctrl, pre):
    atlas = ctrl.bank.palm
    dot = np.einsum("nij,ij->n", atlas[:, :3, :3], pre[:3, :3])
    distance = np.linalg.norm(atlas[:, :3, 3] - pre[:3, 3], axis=1) + 0.08 * (3 - dot)
    ids = np.argsort(distance)[:4]
    return distance[ids[0]], ctrl.bank.q[ids]


def find_reachable(ctrl, proposals, surface, others, previous):
    ranked = []
    for score, card, poses in proposals:
        if card["posture"] in previous:
            continue
        dist, joints = nearest_arm(ctrl, poses["pre"])
        ranked.append((score + 0.22 * dist, card, poses, joints))
    for _, card, poses, joints in sorted(ranked, key=lambda row: row[0]):
        pre, grasp, tight = [poses[k] for k in ("pre", "grasp", "tight")]
        qo, qt, qc = [card[k] for k in ("q_open", "q_touch", "q_tight")]
        if not ctrl.clear(grasp, qt, others) or not ctrl.clear(tight, qc, others):
            continue
        open_shapes = ctrl.link_shapes(qo)
        retreats = [pre]
        if card["origin"] == "head_opposed_section":
            # A fully opened hand can still extend into a wide or low object at
            # the nominal start. Retain the fitted grasp and move its start clear.
            for dz in (0.030, 0.060):
                retreats.append(shifted(pre, [0, 0, dz]))
        pre = next(
            (
                p
                for p in retreats
                if ctrl.clear(p, qo, others) and surface.clear(open_shapes, p, 0.0005)
            ),
            None,
        )
        if pre is None:
            continue
        poses = dict(poses, pre=pre)
        atlas_id = card["fit"].get("arm_sample")
        arm_seeds = (joints[0], ctrl.arm, *joints[1:])
        if atlas_id is not None:
            arm_seeds = (ctrl.bank.q[atlas_id], *arm_seeds)
        for seed in arm_seeds:
            try:
                arm = ctrl.ik.solve(pre, seed)
                ctrl.check_closure_path(card["origin"], arm, grasp, tight)
            except NoPath:
                continue
            ctrl.hints[pre.round(5).tobytes()] = arm
            ctrl.event(
                "selected",
                posture=card["posture"],
                origin=card["origin"],
                palm_grasp=grasp.tolist(),
                rgbd_fit=card["fit"],
            )
            return card, poses
    raise NoPath("no reachable mesh section with clear pregrasp and lift")


@lru_cache(maxsize=1)
def cylinder_postures():
    source = Path(__file__).with_name("config") / "sharpa_cylinder_postures.json"
    calibration = json.loads(source.read_text())
    return np.array(calibration["radii_m"]), np.array(calibration["commands"])


def wall_fit(ctrl, radius, cant, roll, station):
    # The two radius knots are motor targets from the station's previously
    # measured large-diameter catalog grips (ps_tube and ps_bottle). Only these
    # dimensionless joint angles transfer. Surface, wrist pose and force-free
    # closing trajectory are solved again on the currently registered body.
    radii, commands = cylinder_postures()
    fraction = np.clip((radius - radii[0]) / (radii[1] - radii[0]), 0.0, 1.0)
    nominal = commands[0] * (1 - fraction) + commands[1] * fraction
    axes = cylinder_frame(cant, roll)
    patch = central_palm_samples(ctrl.io.model.hand_support[PALM]) @ axes.T
    cut = np.abs(patch[:, 2] - station) < radius
    if not np.any(cut):
        raise NoPath("the cylinder does not overlap the central palm in this chart")
    first_touch = patch[cut, 0] + np.sqrt(radius**2 - (patch[cut, 2] - station) ** 2)
    # Take up the compliance of the broad palm patch before the fingers trap
    # the wall. A translation after closure can move the whole object away.
    cx = float(np.quantile(first_touch, 0.92) - 0.003)
    axis_xz = np.array([cx, station])
    closed = nominal.copy()
    lower, upper = ctrl.io.model.limits["right"].T
    for digit in DIGITS:
        cols = [
            i
            for i, n in enumerate(RIGHT_JOINTS)
            if n.startswith("right_" + digit + "_")
        ]
        joints = [RIGHT_JOINTS[i] for i in cols]
        proximal = np.array([n.endswith(("MCP_FE", "CMC_FE")) for n in joints])
        distal = np.array([n.endswith(("_PIP", "_DIP", "_IP")) for n in joints])
        best = None
        pad = "right_" + digit + "_elastomer"
        phalanx = "right_" + digit + ("_PP" if digit == "thumb" else "_MP")
        for flexion in (-0.16, 0.0, 0.16, 0.30):
            for bend in (-0.16, 0.0, 0.16):
                q = nominal.copy()
                q[cols] = np.clip(
                    q[cols] + flexion * proximal + bend * distal,
                    lower[cols],
                    upper[cols],
                )
                gaps = []
                for link in (pad, phalanx):
                    frame, _ = ctrl.hand_contacts.link(link, np.eye(4), q)
                    vertices = (
                        ctrl.io.model.hand_support[link] @ frame[:3, :3].T
                        + frame[:3, 3]
                    )
                    xz = (vertices @ axes.T)[:, [0, 2]]
                    gaps.append(
                        np.quantile(np.linalg.norm(xz - axis_xz, axis=1), 0.045)
                        - radius
                    )
                cost = (
                    abs(gaps[0] + 0.003)
                    + 0.60 * abs(gaps[1] + 0.002)
                    + 0.008 * (flexion**2 + bend**2)
                )
                if best is None or cost < best[0]:
                    best = cost, q[cols]
        closed[cols] = best[1]
    opened = np.clip(0.15 * nominal, lower, upper)
    opened[:5] = np.clip([-0.14, -0.31, -0.47, -0.30, 0.04], lower[:5], upper[:5])
    tight = closed.copy()
    tight[FLEX] = np.minimum(upper[FLEX], tight[FLEX] + 0.055)
    return axes, axis_xz, opened, closed, tight


def wall_grasps(ctrl, surface):
    points = surface.world
    lo, hi = np.quantile(points[:, 2], [0.01, 0.99])
    band = points[
        (points[:, 2] > lo + 0.25 * (hi - lo)) & (points[:, 2] < lo + 0.63 * (hi - lo))
    ]
    if len(band) < 16:
        return
    xy_lo, xy_hi = np.quantile(band[:, :2], [0.01, 0.99], axis=0)
    center = (xy_lo + xy_hi) / 2
    radius = round(float(np.mean(xy_hi - xy_lo) * 0.5), 3)
    if radius < 0.004:
        return
    # A tapered wall changes width across the finger span. Fit its local section
    # at each candidate height, retaining the broad waist estimate at the centre.
    upper = points[np.abs(points[:, 2] - (lo + 0.74 * (hi - lo))) < 0.007]
    tapered = False
    if len(upper) >= 16:
        limits = np.quantile(upper[:, :2], [0.01, 0.99], axis=0)
        tapered = 0.5 * np.mean(limits[1] - limits[0]) < 0.9 * radius
    levels = (
        (0.49, 0.58, 0.40)
        if not tapered
        else (0.49, 0.46, 0.52, 0.43, 0.55, 0.40, 0.58)
    )
    sections = []
    for level in levels:
        local_radius, local_center = radius, center
        height = lo + level * (hi - lo)
        span = points[np.abs(points[:, 2] - height) < 0.007]
        if tapered and len(span) >= 16:
            bounds = np.quantile(span[:, :2], [0.01, 0.99], axis=0)
            measured = 0.5 * np.mean(bounds[1] - bounds[0])
            if abs(measured - radius) > 0.0012:
                local_radius = round(float(measured), 3)
                local_center = bounds.mean(0)
        sections.append((level, height, local_radius, local_center))
    park = ctrl.ik.fk(ctrl.reference_arm)
    for cant, roll, station in (
        (20, 15, 0.068),
        (10, 8, 0.063),
        (25, 20, 0.075),
    ):
        for level, height, local_radius, local_center in sections:
            key = (local_radius, cant, roll, station)
            if key not in ctrl.wrap_cache:
                ctrl.wrap_cache[key] = wall_fit(ctrl, *key)
            axes, (cx, cz), opened, closed, tight = ctrl.wrap_cache[key]
            shapes = ctrl.link_shapes(closed)
            finger_height = [
                (axes @ shapes[f"right_{d}_elastomer"].mean(0))[1] for d in DIGITS
            ]
            thumb_height = float(finger_height[0] - np.mean(finger_height[1:]))
            for yaw in range(-165, 181, 15):
                az = np.deg2rad(yaw)
                z = np.array([np.cos(az), np.sin(az), 0.0])
                up = np.array([0.0, 0.0, 1.0])
                base = np.column_stack((np.cross(up, z), up, z))
                rot = base @ axes
                turn = Rotation.from_matrix(park[:3, :3].T @ rot).magnitude()
                goal = np.eye(4)
                goal[:3, :3] = rot
                goal[:3, 3] = np.r_[local_center, height] - base @ [cx, 0, cz]
                seat = shifted(goal, 0.003 * base[:, 0])
                for retreat in (0.048, 0.073):
                    pre = shifted(
                        goal, -retreat * rot[:, 0] - 0.012 * rot[:, 2] + [0, 0, 0.008]
                    )
                    # Fingertips extend forward from the workcell. Keep the
                    # approach in front of the arm while retaining all azimuths.
                    score = (
                        0.015 * turn
                        + 0.12 * abs(level - 0.49)
                        + 2.5 * max(0.0, radius - local_radius)
                        + 0.10 * np.linalg.norm(pre[:3, 3] - park[:3, 3])
                        + 0.032 * (1 - np.cos(az + np.pi / 12))
                        + 0.12 * retreat
                        + 0.5 * abs(station - 0.068)
                        + 0.04 * max(0.0, 0.004 - thumb_height)
                    )
                    label = f"wrap/1_Large_Diameter/{yaw}/{cant}/{level:.2f}"
                    yield (
                        score,
                        {
                            "posture": label,
                            "origin": "head_wall_envelope",
                            "q_open": opened,
                            "q_touch": closed,
                            "q_tight": tight,
                            "fit": {
                                "radius_m": local_radius,
                                "waist_xy": local_center.tolist(),
                                "level": level,
                                "yaw_deg": yaw,
                                "cant_deg": cant,
                                "roll_deg": roll,
                                "axis_xz_m": [float(cx), float(cz)],
                                "nominal_thumb_above_four_m": thumb_height,
                            },
                        },
                        {"pre": pre, "grasp": goal, "tight": seat},
                    )


def draw_pads_to_palm(ctrl, initial, distance):
    """Move the opposed pad targets together toward the palm without changing their spacing."""
    sites = []
    palm = np.eye(4)
    shift = np.array([-distance, 0.0, 0.0])
    for digit in DIGITS:
        inner = "PP" if digit == "thumb" else "MP"
        for suffix, weight in (("elastomer", 1.0), (inner, 0.38)):
            link = f"right_{digit}_{suffix}"
            anchor = ctrl.io.model.hand_support[link].mean(0)
            frame, _ = ctrl.hand_contacts.link(link, palm, initial)
            goal = frame[:3, :3] @ anchor + frame[:3, 3] + shift
            sites.append((link, anchor, goal, weight))
    lower, upper = ctrl.io.model.limits["right"].T
    lower = np.maximum(lower, initial - 0.45)
    upper = np.minimum(upper, initial + 0.45)
    angles = initial.copy()
    for _ in range(5):
        jacobians, errors = [], []
        for link, anchor, goal, weight in sites:
            frame, axes = ctrl.hand_contacts.link(link, palm, angles)
            point = frame[:3, :3] @ anchor + frame[:3, 3]
            jacobian = np.zeros((3, len(angles)))
            for index, axis, origin in axes:
                jacobian[:, index] = np.cross(axis, point - origin)
            jacobians.append(weight * jacobian)
            errors.append(weight * (goal - point))
        jacobians.append(0.009 * np.eye(len(angles)))
        errors.append(0.009 * (initial - angles))
        step = np.linalg.lstsq(
            np.vstack(jacobians), np.concatenate(errors), rcond=None
        )[0]
        angles = np.clip(angles + np.clip(step, -0.19, 0.19), lower, upper)
    return angles


def thumb_up_carry(ctrl, palm, hand):
    """Tilt about the enclosed object's pad centroid on the upward stroke."""
    parts = ctrl.link_shapes(hand)
    pads = np.array([parts[f"right_{d}_elastomer"].mean(0) for d in DIGITS])
    delta = palm[:3, :3] @ (pads[0] - pads[1:].mean(0))
    axis = np.cross(delta, [0, 0, 1])
    horizontal = np.linalg.norm(axis)
    if horizontal < 0.01 or delta[2] >= 0.010:
        return palm
    angle = min(np.deg2rad(10.0), (0.010 - delta[2]) / horizontal)
    turn = Rotation.from_rotvec(angle * axis / horizontal).as_matrix()
    target = palm.copy()
    target[:3, :3] = turn @ palm[:3, :3]
    target[:3, 3] += (palm[:3, :3] - target[:3, :3]) @ pads.mean(0)
    return target


def opposing_grasps(ctrl, family, surface, cards, others):
    lo, hi = ctrl.io.model.limits["right"].T
    for score, source, poses in patch_candidates(ctrl, family, surface, cards, others):
        p = poses["grasp"]
        q = np.asarray(source["q_touch"], dtype=float)
        # Rank thumb relative to all four pads in world coordinates. For low
        # objects the longitudinal finger direction may descend towards the table.
        shape = ctrl.link_shapes(q)
        z = np.array([shape[f"right_{d}_elastomer"].mean(0) @ p[2, :3] for d in DIGITS])
        delta = float(z[0] - z[1:].mean())
        if delta < -0.018 or score > 0.034:
            continue
        tight = np.clip(source["q_tight"], lo, hi)
        tight[FLEX] = np.minimum(hi[FLEX], np.maximum(tight[FLEX], q[FLEX] + 0.14))
        a = round(float(np.rad2deg(np.arctan2(p[1, 0], p[0, 0]))) / 15)
        label = f"section/{source['posture']}/{a}"
        yield (
            score + 1.2 * max(0.0, -delta),
            {
                "posture": label,
                "origin": "head_opposed_section",
                "q_open": np.asarray(source["q_open"]),
                "q_touch": q,
                "q_tight": tight,
                "fit": {"pad_distance_cost_m": float(score), "thumb_delta_m": delta},
            },
            poses,
        )


def atlas_grasps(ctrl, family, surface, cards):
    """Align measured opposing cross sections with contact axes within the arm workspace."""
    bands = surface.bands()
    if not bands:
        return
    rotations = ctrl.bank.palm[:, :3, :3]
    wrists = ctrl.bank.palm[:, :3, 3]
    four = family in ("wrap", "handled")
    digits = DIGITS[1:] if four else DIGITS[1:3]
    hi = ctrl.io.model.limits["right"][:, 1]
    by_name = {s["posture"]: s for s in cards}
    extra = (
        "14_Tripod",
        "26_Sphere_4_Finger",
        "9_Palmar_Pinch",
        "6_Prismatic_4_Finger",
    )
    selected = {
        s["posture"]: s
        for s in [*cards[:3], *(by_name[n] for n in extra if n in by_name)]
    }
    for card in selected.values():
        qo, qs = (np.asarray(card[k], dtype=float) for k in ("q_open", "q_tight"))
        for fraction in (0.45, 0.75, 1.0):
            q = qo + fraction * (qs - qo)
            parts = ctrl.link_shapes(q)
            thumb = parts["right_thumb_elastomer"]
            fingers = [parts["right_" + d + "_elastomer"] for d in digits]
            t, finger = thumb.mean(0), np.mean([v.mean(0) for v in fingers], 0)
            axis = finger - t
            axis /= np.linalg.norm(axis)
            t += axis * np.quantile((thumb - t) @ axis, 0.90)
            tip = finger + axis * np.mean(
                [np.quantile((v - finger) @ axis, 0.1) for v in fingers]
            )
            aperture = float((tip - t) @ axis)
            if aperture < 0.003:
                continue
            middle = (t + tip) / 2
            directions = rotations @ axis
            if not len(directions):
                continue
            for band_id, (_, center, across, width, high, low) in enumerate(bands[:3]):
                miss = abs(width - aperture)
                if miss > max(0.019, 0.33 * width):
                    continue
                pos = center - rotations @ middle
                axis_match = np.abs(directions[:, :2] @ across)
                if family in ("compact", "small") and np.ptp(
                    surface.world[:, 0]
                ) < 1.5 * np.ptp(surface.world[:, 1]):
                    axis_match = np.linalg.norm(directions[:, :2], axis=1)
                # Near-vertical opposition on a table would put one contact
                # underneath the object. Radial wrist elevation is independent.
                eligible = (
                    (
                        np.abs(directions[:, 2]) * aperture
                        < min(0.024, 0.75 * (high - low))
                    )
                    & (rotations[:, 2, 1] > -0.20)
                    & (axis_match > 0.68)
                )
                candidates = np.flatnonzero(eligible)
                cost = (
                    np.linalg.norm(pos - wrists, axis=1)
                    + 0.065 * (1 - axis_match)
                    + 0.045 * np.maximum(0.0, directions[:, 2])
                    + 0.02 * np.maximum(0.0, -rotations[:, 2, 1])
                )
                chosen = []
                for k in candidates[np.argsort(cost[candidates])]:
                    if any(
                        np.trace(rotations[k].T @ rotations[j]) > 2.78 for j in chosen
                    ):
                        continue
                    rot = rotations[k]
                    grasp = np.eye(4)
                    grasp[:3, :3], grasp[:3, 3] = rot, pos[k]
                    height = (ctrl.shape(q) @ rot[2]).min() + grasp[2, 3]
                    rise = max(0.0, ctrl.table + 0.0015 - height)
                    if rise <= min(0.009, 0.26 * (high - low)):
                        chosen.append(k)
                        grasp[2, 3] += rise
                        tight = q.copy()
                        tight[FLEX] = np.minimum(hi[FLEX], q[FLEX] + 0.24)
                        for retreat in (
                            (np.array([0.0, 0.0, 0.052]) - 0.018 * rot[:, 0]),
                            (-0.05 * rot[:, 0] + np.array([0.0, 0.0, 0.024])),
                        ):
                            pre = shifted(grasp, retreat)
                            label = f"atlas/{card['posture']}/{band_id}/{k}"
                            yield (
                                miss + 0.20 * cost[k] + 0.7 * rise,
                                {
                                    "posture": label,
                                    "origin": "head_opposed_section",
                                    "q_open": qo,
                                    "q_touch": q,
                                    "q_tight": tight,
                                    "fit": {
                                        "width_m": float(width),
                                        "aperture_m": aperture,
                                        "band": band_id,
                                        "arm_sample": int(k),
                                    },
                                },
                                {"pre": pre, "grasp": grasp, "tight": grasp},
                            )
                    if len(chosen) >= 5:
                        break


def select(ctrl, name, surface, reg, others, previous):
    family, cards = ctrl.shared.query(surface, reg)
    if wants_palm(surface):
        return find_reachable(
            ctrl, wall_grasps(ctrl, surface), surface, others, previous
        )
    if ctrl.lightning is not None:
        try:
            return ctrl.lightning.select(ctrl, name, surface, reg, others, previous)
        except NoPath as exc:
            ctrl.event("mesh_section_search", reason=str(exc), family=family)
    try:
        return find_reachable(
            ctrl,
            opposing_grasps(ctrl, family, surface, cards, others),
            surface,
            others,
            previous,
        )
    except NoPath:
        return find_reachable(
            ctrl, atlas_grasps(ctrl, family, surface, cards), surface, others, previous
        )
