"""Geometry-only orthographic silhouette measurements, without semantic labels."""
from __future__ import annotations

import numpy as np


def silhouette_features(vertices, faces, bbox_dims, T_obj_bbox, resolution=96):
    """Measure filled triangle projections along each of the three bbox axes.

    ``T_obj_bbox`` maps bbox coordinates to object coordinates, as in library.py;
    vertices and positive bbox dimensions are in metres. Each view uses a square
    ``resolution`` raster with two exterior padding pixels. Bbox endpoints map to
    the first/last interior pixel centres, so the actual horizontal/vertical pixel
    spacing is the corresponding bbox dimension / (resolution - 1). Coordinates
    are rounded to the nearest pixel centre before Pillow fills every nondegenerate
    projected triangle. Unreferenced vertices do not contribute to the silhouette.

    ``views`` is ordered by ``axis_bbox`` (viewing direction 0, 1, 2); ``plane_axes_bbox``
    gives horizontal then vertical axes. Areas are occupied pixel counts times pixel
    area. ``solidity`` is foreground / rasterized convex hull area. The denominator
    of ``largest_hole_area_fraction`` is that same convex hull area: foreground PLUS
    all background inside the hull, not foreground plus the already inclusive hull.
    Enclosed holes are background components disconnected from the padded exterior
    using 8-connectivity; a notch open even diagonally is not a hole. The maximum
    inscribed diameter is twice the largest hole EDT value, with actual per-axis
    metric pixel spacing, divided by the shorter view dimension for its ratio.
    EDT distances are between pixel centres, with pixel-scale boundary uncertainty.

    Aggregate ``max_hole_area_fraction`` and ``max_hole_diameter_ratio`` are the
    maxima of those per-view measurements; ``min_solidity`` is the minimum over
    all three views. Empty/degenerate projections have zero areas, solidity, hole
    counts and diameters (and therefore can lower the aggregate minimum to zero).
    Invalid inputs, nonrigid transforms and vertices outside the bbox raise ValueError.

    These are ground-truth orthographic projections, not a single-view perception
    claim: occlusion can hide openings, subpixel openings can disappear, and raster
    quantization can close narrow notches. Topology/genus and semantic object names
    are never used. No classifier or decision thresholds are selected here. The
    function is deterministic and does not modify inputs or use any random state.
    Pillow and SciPy are imported locally; neither is needed merely to import this module.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    dims = np.asarray(bbox_dims, dtype=float)
    box = np.asarray(T_obj_bbox, dtype=float)
    if vertices.shape == (0,):
        vertices = vertices.reshape(0, 3)
    if faces.shape == (0,):
        faces = np.empty((0, 3), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("finite Nx3 vertices required")
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("valid integer Mx3 faces required")
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("positive finite bbox dimensions required")
    if box.shape != (4, 4) or not np.isfinite(box).all():
        raise ValueError("finite 4x4 bbox transform required")
    rotation = box[:3, :3]
    if (not np.allclose(box[3], [0, 0, 0, 1], rtol=0, atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6)):
        raise ValueError("proper rigid bbox transform required")
    if (not isinstance(resolution, (int, np.integer)) or isinstance(resolution, (bool, np.bool_))
            or resolution < 2):
        raise ValueError("integer resolution >= 2 required")
    resolution = int(resolution)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = ((vertices - box[:3, 3]) @ rotation) / dims + .5
        pixel_size = dims / (resolution - 1)
        pixel_areas = pixel_size[[1, 0, 0]] * pixel_size[[2, 2, 1]]
    if not np.isfinite(normalized).all() or np.any(normalized < -1e-6) or np.any(normalized > 1 + 1e-6):
        raise ValueError("vertices must lie within the bbox")
    if (not np.isfinite(pixel_areas).all() or np.any(pixel_areas <= 0)
            or np.any(pixel_areas > np.finfo(float).max / (resolution * resolution))):
        raise ValueError("bbox dimensions must give representable positive raster areas")

    from PIL import Image, ImageDraw
    from scipy import ndimage
    from scipy.spatial import ConvexHull, QhullError

    padding = 2
    size = resolution + 2 * padding
    pixels = np.rint(np.clip(normalized, 0, 1) * (resolution - 1)).astype(np.int64) + padding
    connectivity = np.ones((3, 3), dtype=bool)
    views = []
    for axis in range(3):
        plane = [j for j in range(3) if j != axis]
        spacing = pixel_size[plane]
        pixel_area = float(pixel_areas[axis])
        image = Image.new("1", (size, size))
        draw = ImageDraw.Draw(image)
        projected = normalized[:, plane][faces]
        edge_a = projected[:, 1] - projected[:, 0]
        edge_b = projected[:, 2] - projected[:, 0]
        twice_area = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
        # Ignore numerical edge-on projections, not physical size-based classes.
        valid = np.abs(twice_area) > 32 * np.finfo(float).eps
        projected_pixels = pixels[:, plane]
        for triangle in faces[valid]:
            draw.polygon([tuple(point) for point in projected_pixels[triangle].tolist()], fill=1)
        foreground = np.asarray(image, dtype=bool)
        foreground_count = int(np.count_nonzero(foreground))
        reflected = foreground[::-1, ::-1]
        reflection_union = int(np.count_nonzero(foreground | reflected))
        symmetry = float(np.count_nonzero(foreground & reflected) / reflection_union) if reflection_union else 0.0
        hull_count = foreground_count
        if foreground_count >= 3:
            points = np.column_stack(np.nonzero(foreground))[:, ::-1]
            try:
                hull = ConvexHull(points)
            except QhullError:
                # A subpixel-width projection can rasterize as a line or point.
                pass
            else:
                hull_image = Image.new("1", (size, size))
                ImageDraw.Draw(hull_image).polygon([tuple(point) for point in points[hull.vertices].tolist()], fill=1)
                hull_count = int(np.count_nonzero(np.asarray(hull_image, dtype=bool) | foreground))
        holes = ndimage.binary_fill_holes(foreground, structure=connectivity) & ~foreground
        labels, hole_count = ndimage.label(holes, structure=connectivity)
        largest_hole_count = 0
        diameter = 0.0
        if hole_count:
            areas = np.bincount(labels.ravel())[1:]
            largest_hole_count = int(areas.max())
            distances = ndimage.distance_transform_edt(holes, sampling=(float(spacing[1]), float(spacing[0])))
            diameter = 2 * float(distances.max())
            diameters = 2 * ndimage.maximum(distances, labels, np.arange(1, hole_count + 1))
            if not np.isfinite(diameter):
                raise ValueError("bbox dimensions must give a representable metric hole diameter")
        views.append({
            "axis_bbox": axis,
            "plane_axes_bbox": plane,
            "view_dims_m": dims[plane].tolist(),
            "pixel_size_m": spacing.tolist(),
            "foreground_pixels": foreground_count,
            "bbox_fill_fraction": foreground_count / (resolution * resolution),
            "central_symmetry_iou": symmetry,
            "foreground_area_m2": foreground_count * pixel_area,
            "convex_hull_area_m2": hull_count * pixel_area,
            "solidity": foreground_count / hull_count if hull_count else 0.0,
            "hole_count": int(hole_count),
            "holes": [{"area_fraction": int(area) / hull_count,
                       "inscribed_diameter_m": float(width),
                       "diameter_ratio": float(width) / float(min(dims[plane]))}
                      for area, width in zip(areas, diameters, strict=True)] if hole_count else [],
            "enclosed_hole_area_m2": int(np.count_nonzero(holes)) * pixel_area,
            "largest_hole_area_m2": largest_hole_count * pixel_area,
            "largest_hole_area_fraction": largest_hole_count / hull_count if hull_count else 0.0,
            "max_hole_inscribed_diameter_m": diameter,
            "max_hole_diameter_ratio": diameter / float(min(dims[plane])),
        })
    return {
        "method": "bbox_orthographic_filled_triangles_v1",
        "resolution": resolution,
        "padding_pixels": padding,
        "hole_area_fraction_denominator": "convex_hull_area_m2",
        "background_connectivity": 8,
        "views": views,
        "max_hole_area_fraction": max(view["largest_hole_area_fraction"] for view in views),
        "max_hole_diameter_ratio": max(view["max_hole_diameter_ratio"] for view in views),
        "min_solidity": min(view["solidity"] for view in views),
    }
