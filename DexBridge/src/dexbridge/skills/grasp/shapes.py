"""Numpy-only visual shape evidence; semantic object categories are never features."""
from __future__ import annotations

import json

import numpy as np

from ...paths import CONFIGS


def fit_surface_primitives(vertices, faces, bbox_dims, T_obj_bbox, *, sample_count=2048, seed=20261117):
    """Fit closed primitives to deterministic area-uniform triangle samples, not vertices."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    dims = np.asarray(bbox_dims, dtype=float)
    box = np.asarray(T_obj_bbox, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("finite Nx3 vertices required")
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer)
            or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("nonempty valid integer Mx3 faces required")
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("positive finite bbox dimensions required")
    if box.shape != (4, 4) or not np.isfinite(box).all():
        raise ValueError("finite 4x4 bbox transform required")
    if not isinstance(sample_count, (int, np.integer)) or isinstance(sample_count, bool) or sample_count < 4:
        raise ValueError("at least four surface samples required")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    valid = areas > 0
    triangles, areas = triangles[valid], areas[valid]
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("positive finite triangle surface area required")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = np.searchsorted(np.cumsum(areas) / total_area, rng.random(sample_count), side="right")
    selected = triangles[np.minimum(indices, len(triangles) - 1)]
    uv = rng.random((sample_count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    points = np.einsum("ni,nij->nj", weights, selected)
    local = (points - box[:3, 3]) @ box[:3, :3]
    scores = {"box": float(np.mean(np.min(np.abs(np.abs(local) - dims / 2), axis=1) ** 2))}
    # Use local coordinates for the algebraic sphere fit to avoid global-origin conditioning.
    design = np.column_stack((2 * local, np.ones(sample_count)))
    solution, _, rank, _ = np.linalg.lstsq(design, np.sum(local**2, axis=1), rcond=None)
    radius_sq = solution[3] + solution[:3] @ solution[:3]
    if rank == 4 and radius_sq > 0:
        scores["sphere"] = float(np.mean((np.linalg.norm(local - solution[:3], axis=1) - np.sqrt(radius_sq)) ** 2))
    cylinder_errors, radii = [], []
    for axis in range(3):
        radial_axes = [j for j in range(3) if j != axis]
        radial = np.linalg.norm(local[:, radial_axes], axis=1)
        radius = float(np.mean(dims[radial_axes]) / 2)
        axial = np.abs(local[:, axis])
        # Distances to a finite cylindrical side and finite cap disks, not infinite planes.
        side_sq = (radial - radius)**2 + np.maximum(axial - dims[axis] / 2, 0)**2
        cap_sq = (axial - dims[axis] / 2)**2 + np.maximum(radial - radius, 0)**2
        cylinder_errors.append(float(np.mean(np.minimum(side_sq, cap_sq))))
        radii.append(radius)
    axis = min(range(3), key=lambda i: (cylinder_errors[i], i))
    scores["cylinder"] = cylinder_errors[axis]
    # Radial surface distance to the bbox-centred ellipsoid; retain the original three fits.
    radii_at_sample = np.linalg.norm(local, axis=1)
    ellipsoid_norm = np.linalg.norm(local / (dims / 2), axis=1)
    shell_radius = np.divide(radii_at_sample, ellipsoid_norm,
                             out=np.full(sample_count, float(min(dims) / 2)), where=ellipsoid_norm > 0)
    ellipsoid_error = float(np.mean((radii_at_sample - shell_radius) ** 2))
    return {"primitive": min(scores, key=lambda name: (scores[name], name)),
            "primitive_fit_mse": scores,
            "ellipsoid_fit_mse": ellipsoid_error,
            "cylinder_fit": {"axis_bbox": axis, "radius_m": radii[axis],
                             "height_m": float(dims[axis]), "axis_mse_m2": cylinder_errors},
            "primitive_fit_sampling": {"method": "triangle_area_uniform_sqrt_barycentric",
                                       "sample_count": int(sample_count), "seed": int(seed),
                                       "bit_generator": "PCG64", "surface_area_m2": total_area,
                                       "positive_area_triangles": len(triangles),
                                       "primitive_tie_break": "mse_then_lexicographic_name",
                                       "cylinder_axis_tie_break": "mse_then_bbox_axis_index"}}


def _closed_handle(vertices, faces):
    """Return topology annotation only; genus cannot establish a visible main handle."""
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("finite Nx3 vertices required")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("integer Mx3 faces required")
    if not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        return None
    # Weld exact duplicate surface vertices before counting topology.
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)
    triangles = inverse[faces]
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        return None
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    used = np.unique(triangles)
    chi = int(len(used) - len(unique) + len(triangles))
    if not np.all(counts == 2) or chi > 0 or chi % 2:
        return None
    parent = np.arange(len(inverse))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in unique:
        parent[root(a)] = root(b)
    if len({root(i) for i in used}) != 1:
        return None
    # Every vertex link must be one cycle (not a pinched non-manifold surface).
    links = {}
    for a, b, c in triangles:
        for v, x, y in ((a, b, c), (b, c, a), (c, a, b)):
            links.setdefault(int(v), []).append((int(x), int(y)))
    for pairs in links.values():
        adjacency = {}
        for a, b in pairs:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return None
        seen, pending = set(), [next(iter(adjacency))]
        while pending:
            v = pending.pop()
            if v not in seen:
                seen.add(v)
                pending.extend(adjacency[v] - seen)
        if len(seen) != len(adjacency):
            return None
    return {"closed_surface_euler_characteristic": chi, "genus": (2 - chi) // 2}


def visual_shape(features, definition=None):
    """Classify measurable geometry under the rule frozen into each template index."""
    definition = definition or json.loads((CONFIGS / "skills/visual_shapes.json").read_text())
    dims = np.asarray(features["bbox_dims"], dtype=float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("positive finite bbox dimensions required")
    short, middle, long = np.sort(dims)
    t = definition["thresholds"]
    gating = definition.get("primitive_gating", "winner_first")
    measure = definition.get("size_measure", "max_extent")
    if gating not in ("winner_first", "qualified_fit") or measure not in ("max_extent", "grasp_span"):
        raise ValueError("unknown frozen visual classification rule")
    span = float(min(dims[:2]))
    size_value = span if measure == "grasp_span" else float(long)
    size = ("small" if size_value <= t["small_max_extent_m"] else
            "medium" if size_value <= t["medium_max_extent_m"] else "large")
    observed = features.get("shape_key")
    if observed is not None and observed not in definition["hints_zh"]:
        raise ValueError("unknown observed visual shape key")
    errors = {}
    for name, value in features.get("primitive_fit_mse", {}).items():
        if name in ("box", "sphere", "cylinder") and np.isfinite(value) and value >= 0:
            errors[name] = float(np.sqrt(value) / short)
    projection = features.get("silhouette_features", {})
    views = [view for view in projection.get("views", []) if view["foreground_pixels"] >= 9]
    by_axis = {view["axis_bbox"]: view for view in views}

    def round_projection(axis):
        if not projection:
            return True
        view = by_axis.get(axis)
        return (view is not None
                and view.get("bbox_fill_fraction", 1.) <= t.get("round_max_bbox_fill_fraction", 1.)
                and view.get("central_symmetry_iou", 0.) >= t.get("round_min_central_symmetry_iou", 0.)
                and view["solidity"] >= t.get("round_min_solidity", 0.))

    best = min(errors, key=lambda name: (errors[name], name)) if errors else None
    fitted = best if best and errors[best] <= t["max_fit_rmse_over_short_extent"] else None
    cylinder_axis, cylinder_ratio = None, None
    ratios = [max(np.delete(dims, axis)) / min(np.delete(dims, axis)) for axis in range(3)]
    if fitted == "cylinder" or (gating == "qualified_fit" and "cylinder" in errors):
        fitted_axis = features.get("cylinder_fit", {}).get("axis_bbox")
        if fitted_axis is not None:
            if isinstance(fitted_axis, bool) or not isinstance(fitted_axis, (int, np.integer)) or fitted_axis not in range(3):
                raise ValueError("cylinder bbox axis must be 0, 1 or 2")
            cylinder_axis = int(fitted_axis)
        else:
            cylinder_axis = min(range(3), key=lambda axis: (ratios[axis], axis))
        cylinder_ratio = float(dims[cylinder_axis] / max(np.delete(dims, cylinder_axis)))
    ellipsoid_error = features.get("ellipsoid_fit_mse")
    ellipsoid_rmse = (float(np.sqrt(ellipsoid_error) / short)
                      if ellipsoid_error is not None and np.isfinite(ellipsoid_error) and ellipsoid_error >= 0 else None)
    if gating == "qualified_fit":
        qualified = {name: error for name, error in errors.items()
                     if error <= t["max_fit_rmse_over_short_extent"]
                     and (name != "sphere" or (long / short <= t["sphere_long_short_ratio"]
                                              and all(round_projection(axis) for axis in range(3))))
                     and (name != "cylinder" or (ratios[cylinder_axis] <= t["round_cross_section_ratio"]
                                                and round_projection(cylinder_axis)))}
        if ("rounded" in definition["hints_zh"] and ellipsoid_rmse is not None
                and ellipsoid_rmse <= t["ellipsoid_max_rmse_over_short_extent"]
                and long / short <= t["ellipsoid_long_short_ratio"]
                and all(round_projection(axis) for axis in range(3))):
            qualified["ellipsoid"] = ellipsoid_rmse
        fitted = min(qualified, key=lambda name: (qualified[name], name)) if qualified else None
    key = "small_item" if size == "small" and "small_item" in definition["hints_zh"] else "unknown"
    visible_holes = []
    if "visible_hole_min_area_fraction" in t:
        # Area and width must describe the same visible hole, not unrelated maxima.
        visible_holes = [view for view in views if any(
            hole["area_fraction"] >= t["visible_hole_min_area_fraction"]
            and hole["diameter_ratio"] >= t["visible_hole_min_diameter_ratio"]
            for hole in view["holes"])]
    hole = None
    if observed is None and not projection and "vertices" in features and "faces" in features:
        hole = _closed_handle(np.asarray(features["vertices"], float), np.asarray(features["faces"]))
    if observed is not None:
        key = observed
    elif visible_holes:
        key = "handle_ring"
    elif fitted == "sphere" and long / short <= t["sphere_long_short_ratio"]:
        key = "sphere"
    elif fitted == "cylinder" and ratios[cylinder_axis] <= t["round_cross_section_ratio"]:
        key = "slender_cylinder" if cylinder_ratio >= t["slender_long_middle_ratio"] else "stout_cylinder"
    elif fitted == "ellipsoid":
        key = "rounded"
    elif middle / short >= t["flat_middle_short_ratio"]:
        key = "flat_plate"
    elif fitted == "box":
        key = "box"
    elif "elongated" in definition["hints_zh"] and long / middle >= t["slender_long_middle_ratio"]:
        key = "elongated"
    elif "concave" in definition["hints_zh"] and views and min(v["solidity"] for v in views) <= t["concave_max_solidity"]:
        key = "concave"
    return {"shape_key": key, "size_key": size, "feature_size_m": span,
            "size_measure": measure, "size_measure_m": size_value, "thickness_m": float(short),
            "max_extent_m": float(long), "long_middle_ratio": float(long / middle),
            "middle_short_ratio": float(middle / short), "long_short_ratio": float(long / short),
            "fit_rmse_over_short_extent": errors, "ellipsoid_rmse_over_short_extent": ellipsoid_rmse,
            "hole_evidence": hole, "visible_hole_views": [view["axis_bbox"] for view in visible_holes],
            "cylinder_axis_bbox": cylinder_axis, "cylinder_length_diameter_ratio": cylinder_ratio,
            "automatic_handle_ring_support": bool(visible_holes),
            "shape_source": "caller_visual_observation" if observed is not None else "numeric_geometry",
            "recognition_hint_zh": definition["hints_zh"][key]}
