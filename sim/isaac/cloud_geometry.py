"""Depth unprojection, robust tabletop fitting and instance masks."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

_MAX_FIT_POINTS = 4096
_MAX_ITERATIONS = 4096
_MAX_TABLE_TILT_DEGREES = 30.0


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"invalid {name}")
    return value


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside supported bounds")
    return int(value)


def _array(value, name, shape=None, *, finite=True):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if array.dtype.kind not in "iuf" or (shape is not None and array.shape != shape):
        raise ValueError(f"invalid {name} shape or numeric dtype")
    array = np.asarray(array, dtype=float)
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _points(value, minimum):
    points = _array(value, "points_world")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"points_world must contain at least {minimum} finite 3D points")
    return points


def _unit(vector, name):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"degenerate {name}")
    return vector / norm


def _rigid_transform(value):
    transform = _array(value, "T_world_camera", (4, 4))
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError("T_world_camera must be a proper rigid world-from-camera transform")
    return transform


def depth_to_world_points(depth, mask, K, T_world_camera, *, depth_scale=1.0, max_depth_m=5.0):
    """Backproject selected pixels using optical-Z depth, not ray distance.

    Integer OpenCV pixel coordinates identify pixel centers (no half-pixel shift).
    K has positive focal diagonals and conventional last row [0, 0, 1]; its full
    invertible 2D calibration, including skew, is used. Boolean or nonnegative
    finite numeric masks select nonzero entries, including uint8 255 masks.
    Nonfinite/nonpositive/out-of-range selected depths are filtered, not imputed.
    Returns float64 points in raster order, or shape (0, 3) if none are valid.
    """
    depth = _array(depth, "depth", finite=False)
    if depth.ndim != 2 or min(depth.shape) == 0:
        raise ValueError("depth must be a nonempty 2D image")
    mask = np.asarray(mask)
    if (
        mask.shape != depth.shape
        or mask.dtype.kind not in "biuf"
        or not np.isfinite(mask).all()
        or np.any(mask < 0)
    ):
        raise ValueError("mask must be a finite nonnegative image matching depth")
    depth_scale = _scalar(depth_scale, "depth_scale", positive=True)
    max_depth_m = _scalar(max_depth_m, "max_depth_m", positive=True)
    intrinsic = _array(K, "K", (3, 3))
    if (
        intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0)
    ):
        raise ValueError("K must have positive focal lengths and conventional last row")
    if np.linalg.cond(intrinsic) >= 1 / np.finfo(float).eps:
        raise ValueError("K must be numerically invertible")
    transform = _rigid_transform(T_world_camera)
    rows, columns = np.nonzero(mask)
    with np.errstate(over="ignore", invalid="ignore"):
        z = depth[rows, columns] * depth_scale
    valid = np.isfinite(z) & (z > 0) & (z <= max_depth_m)
    pixels = np.column_stack((columns[valid], rows[valid], np.ones(np.count_nonzero(valid))))
    try:
        rays = np.linalg.solve(intrinsic, pixels.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("K must be invertible") from exc
    # Normalization retains the optical-Z convention even within the admitted
    # numerical tolerance on K's last row.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera = rays * (z[valid] / rays[:, 2])[:, None]
        world = camera @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(world).all():
        raise ValueError("calibration produces nonfinite world points")
    return world


def _refine_plane(points, up, minimum_alignment):
    center = points.mean(axis=0)
    centered = points - center
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 3 or singular[1] <= max(singular[0], np.finfo(float).tiny) * 1e-12:
        raise ValueError("table inliers are collinear or coincident")
    normal = axes[-1]
    if normal @ up < 0:
        normal = -normal
    if normal @ up < minimum_alignment:
        raise ValueError("fitted plane is too steep to be a table")
    return np.r_[normal, -normal @ center]


def fit_table_plane(
    points_world, *, up=(0, 0, 1), tolerance_m=0.002, iterations=128, min_inlier_fraction=0.5, seed=0
):
    """Bounded seeded RANSAC, then two inlier-SVD refinements.

    Planes must be within 30 degrees of supplied up. At most 4096 deterministic
    sampled points participate in fitting; all input points participate in final
    inlier count, acceptance and RMS. Iterations must be in [1, 4096]. No plane,
    insufficient support or degenerate inliers raises ValueError, never fallback.
    """
    points = _points(points_world, 3)
    up = _unit(_array(up, "up", (3,)), "up")
    tolerance = _scalar(tolerance_m, "tolerance_m", positive=True)
    fraction = _scalar(min_inlier_fraction, "min_inlier_fraction", positive=True)
    if fraction > 1:
        raise ValueError("min_inlier_fraction must be in (0, 1]")
    iterations = _integer(iterations, "iterations", 1, _MAX_ITERATIONS)
    seed = _integer(seed, "seed", 0)
    generator = np.random.default_rng(seed)
    if len(points) > _MAX_FIT_POINTS:
        indices = np.sort(generator.choice(len(points), _MAX_FIT_POINTS, replace=False))
        fitting = points[indices]
    else:
        fitting = points
    alignment = np.cos(np.deg2rad(_MAX_TABLE_TILT_DEGREES))
    best_plane, best_score = None, None
    for _ in range(iterations):
        a, b, c = fitting[generator.choice(len(fitting), 3, replace=False)]
        ab, ac = b - a, c - a
        normal = np.cross(ab, ac)
        length = np.linalg.norm(normal)
        scale = np.linalg.norm(ab) * np.linalg.norm(ac)
        if not np.isfinite(length) or length <= scale * 1e-12 or length == 0:
            continue
        normal /= length
        if normal @ up < 0:
            normal = -normal
        if normal @ up < alignment:
            continue
        plane = np.r_[normal, -normal @ a]
        residual = np.abs(fitting @ normal + plane[3])
        inliers = residual <= tolerance
        count = int(inliers.sum())
        if count < 3:
            continue
        score = (count, -float(np.mean(residual[inliers] ** 2)))
        if best_score is None or score > best_score:
            best_plane, best_score = plane, score
    if best_plane is None:
        raise ValueError("no nondegenerate table plane found within tilt bound")
    plane = best_plane
    for _ in range(2):
        inliers = np.abs(fitting @ plane[:3] + plane[3]) <= tolerance
        if np.count_nonzero(inliers) < 3:
            raise ValueError("insufficient table inliers for SVD refinement")
        plane = _refine_plane(fitting[inliers], up, alignment)
    residual = np.abs(points @ plane[:3] + plane[3])
    inliers = residual <= tolerance
    count = int(inliers.sum())
    if count < 3 or count / len(points) < fraction:
        raise ValueError("insufficient table inlier fraction")
    diagnostics = {
        "input_count": len(points),
        "fit_point_count": len(fitting),
        "inlier_count": count,
        "inlier_fraction": count / len(points),
        "rms_m": float(np.sqrt(np.mean(residual[inliers] ** 2))),
        "tolerance_m": tolerance,
        "iterations": iterations,
        "seed": seed,
        "maximum_tilt_degrees": _MAX_TABLE_TILT_DEGREES,
    }
    return plane, diagnostics


def instance_mask(instance_ids, labels, prim_path):
    """Select a prim and descendants from uncolored native instance-ID metadata."""
    ids = []
    for identifier, label in labels.items():
        if not isinstance(label, str):
            raise ValueError("instance-ID labels must be prim-path strings")
        if label == prim_path or label.startswith(prim_path + "/"):
            ids.append(int(identifier))
    if not ids:
        raise ValueError(f"no rendered instance for {prim_path}")
    mask = np.isin(instance_ids, ids)
    if not np.any(mask):
        raise ValueError(f"instance has no visible pixels: {prim_path}")
    return mask
