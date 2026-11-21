"""Numpy-only posture-card selection, size interpolation and bbox frame conversion."""
from __future__ import annotations

import numpy as np

from .shapes import visual_shape


def interpolate_joints(table, size):
    knots = table["knots"]
    if not np.isfinite(size) or size <= 0 or not knots:
        raise ValueError("positive feature size and nonempty knots required")
    x = np.array([k["feature_size_m"] for k in knots])
    if not np.isfinite(x).all() or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("knots must be finite, positive, unique and increasing")
    out = {"clamped": bool(size < x[0] - 1e-8 or size > x[-1] + 1e-8)}
    for name in ("q_pre", "q_grasp", "q_squeeze"):
        values = np.asarray([k[name] for k in knots], dtype=float)
        if values.shape != (len(knots), 22) or not np.isfinite(values).all():
            raise ValueError("expected finite SDK-order 22-joint knots")
        out[name] = [float(np.interp(size, x, values[:, j])) for j in range(22)]
    return out


def _lower_bound(passes, trials):
    if not trials:
        return 0.0
    z = 1.96
    p = passes / trials
    return float((p + z*z / (2*trials) - z*np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))) / (1+z*z/trials))


def rank_types(templates, features, purpose=""):
    """Rank distinct-object coverage within comparable visual evidence levels."""
    shape_features = visual_shape(features, templates.get("visual_shape_definition"))
    size = shape_features["feature_size_m"]
    ranked = []
    levels = {"shape_size": 0, "shape": 1, "global": 2, "unobserved": 3}
    for name, card in templates["types"].items():
        if not card["modes"]:
            continue
        rows = [row for row in templates["selection"] if row["grasp_type"] == name]
        shape = [row for row in rows if row["shape_key"] == shape_features["shape_key"]]
        sized = [row for row in shape if row["size_key"] == shape_features["size_key"]
                 and row["size_interval_m"][0] <= size < row["size_interval_m"][1]]
        matched, level = next(((values, label) for values, label in
                               ((sized, "shape_size"), (shape, "shape"), (rows, "global"))
                               if values), ([], "unobserved"))
        objects = set().union(*(row["object_ids"] for row in matched))
        successful_objects = set().union(*(row["successful_object_ids"] for row in matched))
        conservative_objects = set().union(*(row["conservative_successful_object_ids"] for row in matched))
        successes = sum(row["native_passes"] for row in matched)
        total = sum(row["native_trials"] + row["rejected_placements"] for row in matched)
        mode_available = shape_features["shape_key"] in card["validated_shape_keys"]
        ranked.append({"grasp_type": name, "score": _lower_bound(len(successful_objects), len(objects)),
                       "object_trials": len(objects), "object_passes": len(successful_objects),
                       "conservative_object_passes": len(conservative_objects),
                       "successes": successes, "placements": total,
                       "rate": successes / total if total else None, "evidence_level": level,
                       "contact_setting": "moderate", "shape_features": shape_features,
                       "supported_shape_mode": mode_available,
                       "conservative_successes": sum(row["conservative_passes"] for row in matched),
                       "conservative_placements": sum(
                           row["conservative_trials"] + row["conservative_rejected_placements"] for row in matched),
                       "purpose": purpose, "purpose_success_evidence": False})
    return sorted(ranked, key=lambda r: (not r["supported_shape_mode"], levels[r["evidence_level"]],
                                        -r["score"], -(r["rate"] or 0), -r["object_trials"], r["grasp_type"]))


def fixed_power_type(templates):
    available = [(name, card) for name, card in templates["types"].items() if card["power"] and card["modes"]]
    if not available:
        raise ValueError("no power template available")
    return min(available, key=lambda item: (-_lower_bound(item[1]["object_passes"], item[1]["object_trials"]),
                                           -(item[1]["end_to_end_rate"] or 0), item[0]))[0]


class IncompatibleWristMode(ValueError):
    """A candidate mode cannot interpolate a continuous wrist action safely."""


def plan_from_mode(mode, features):
    dims = np.asarray(features["bbox_dims"], float)
    T_obj_bbox = np.asarray(features["T_obj_bbox"], float)
    if (dims.shape != (3,) or np.any(dims <= 0) or not np.isfinite(dims).all()
            or T_obj_bbox.shape != (4, 4) or not np.isfinite(T_obj_bbox).all()):
        raise ValueError("invalid object box frame")
    size = float(min(dims[:2]))
    joints = interpolate_joints(mode, size)
    knots = mode["knots"]
    sizes = np.array([k["feature_size_m"] for k in knots])
    node_dims = np.asarray([k["bbox_dims"] for k in knots], float)
    bbox_clamped = bool(np.any(dims < node_dims.min(axis=0) - 1e-8)
                        or np.any(dims > node_dims.max(axis=0) + 1e-8))
    clamped = joints["clamped"] or bbox_clamped
    if clamped:
        left = right = int(np.argmin(np.linalg.norm(np.log(dims / node_dims), axis=1)))
        fraction = 0.0
        for key in ("q_pre", "q_grasp", "q_squeeze"):
            joints[key] = list(knots[left][key])
    else:
        right = int(np.clip(np.searchsorted(sizes, size), 0, len(knots) - 1))
        left = max(0, right - 1)
        fraction = float(np.clip((size - sizes[left]) / (sizes[right] - sizes[left]), 0, 1)) if right != left else 0.0
    plan = {"clamped": clamped, "size_clamped": joints["clamped"], "bbox_clamped": bbox_clamped,
            "mode_id": mode["mode_id"], "validated_mode": mode["validated"],
            "source_record_ids": [knots[left]["record_id"], knots[right]["record_id"]], "dexonomy_pass": None,
            "extrapolation_policy": "nearest_measured_joints_surface_clearance" if clamped else "within_mode_interpolation"}
    for stage, key in (("pregrasp", "q_pre"), ("grasp", "q_grasp"), ("squeeze", "q_squeeze")):
        a, b = knots[left][stage], knots[right][stage]
        rotation_a, rotation_b = np.asarray(a["rotation"], float), np.asarray(b["rotation"], float)
        angle = np.arccos(np.clip((np.trace(rotation_a.T @ rotation_b) - 1) / 2, -1, 1))
        if left != right and 0 < fraction < 1 and angle > .65:
            raise IncompatibleWristMode("cannot interpolate incompatible wrist orientations")
        offsets = []
        for frame, source_dims in ((a, node_dims[left]), (b, node_dims[right])):
            measured = np.asarray(frame["offset_bbox"]) * source_dims
            anchor = np.clip(measured, -source_dims / 2, source_dims / 2)
            # Move the box surface, not the measured metric clearance from that surface.
            offsets.append(anchor / source_dims * dims + measured - anchor)
        offset = (1 - fraction) * offsets[0] + fraction * offsets[1]
        blend = (1 - fraction) * rotation_a + fraction * rotation_b
        u, _, vt = np.linalg.svd(blend)
        rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        local = np.eye(4)
        local[:3, :3], local[:3, 3] = rotation, offset
        frame = {"T_obj_palm": (T_obj_bbox @ local).tolist(), "q": joints[key]}
        plan[stage] = [frame] if stage == "pregrasp" else frame
        plan[key] = joints[key]
    return plan


def choose_plan(templates, grasp_type, features, T_world_obj=None):
    """Choose a supported discrete orientation, using only true box/table approach geometry."""
    modes = templates["types"][grasp_type]["modes"]
    if not modes:
        raise ValueError(f"no train posture for {grasp_type}")
    dims = np.asarray(features["bbox_dims"], float)
    shape = visual_shape(features, templates.get("visual_shape_definition"))["shape_key"]
    plans = []
    world = np.eye(4) if T_world_obj is None else np.asarray(T_world_obj, float)
    if world.shape != (4, 4) or not np.isfinite(world).all():
        raise ValueError("finite 4x4 world pose required")
    box = world @ np.asarray(features["T_obj_bbox"], float)
    center, up = box[:3, 3], box[2, :3]
    incompatible = 0
    for mode in modes:
        if mode["shape_key"] != shape or np.dot(mode["support_up_bbox"], up) < np.cos(.35):
            continue
        try:
            plan = plan_from_mode(mode, features)
        except IncompatibleWristMode:
            incompatible += 1
            continue
        pre = world @ np.asarray(plan["pregrasp"][0]["T_obj_palm"])
        grasp = world @ np.asarray(plan["grasp"]["T_obj_palm"])
        upper = pre[2, 3] >= center[2] and pre[2, 3] >= grasp[2, 3] - 1e-6
        if not upper:
            continue
        distance = min(np.linalg.norm(np.log(dims / np.asarray(k["bbox_dims"]))) for k in mode["knots"])
        key = (not mode["validated"], plan["clamped"], distance, -mode["support_records"], mode["mode_id"])
        plans.append((key, plan))
    if not plans:
        raise ValueError("no shape/support-compatible upper-hemisphere train mode")
    plan = min(plans, key=lambda item: item[0])[1]
    plan["grasp_type"] = grasp_type
    plan["skipped_incompatible_modes"] = incompatible
    return plan
