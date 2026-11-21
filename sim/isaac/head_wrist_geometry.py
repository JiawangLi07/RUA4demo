"""Cylinder-local geometry for the two-view development experiment."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def usd_camera_intrinsics(width, height, focal, ha, va, hoff, voff):
    """ROS pixel intrinsics before the half-pixel depth sampling correction."""
    return np.array(
        [
            [width * focal / ha, 0.0, width * (0.5 - hoff / ha)],
            [0.0, height * focal / va, height * (0.5 + voff / va)],
            [0.0, 0.0, 1.0],
        ]
    )


def hand_shapes(scene, q):
    """Native collision support vertices expressed in the right palm for this q."""
    seed = dict.fromkeys(scene.robot.joint_names, 0.0)
    seed.update(zip(scene.hand_names["right"], q))
    inverse = np.linalg.inv(scene.tree.pose("right_hand_C_MC", seed))
    shapes = {}
    for name in scene.hand_bounds["right"]:
        local = inverse @ scene.tree.pose(name, seed)
        shapes[name] = scene.hand_support[name] @ local[:3, :3].T + local[:3, 3]
    return shapes


def separated(shapes, pose, center, radius, table, top, margin):
    """Conservative slab and projected convex-hull test against the observed cylinder."""
    for vertices in shapes.values():
        points = vertices @ pose[:3, :3].T + pose[:3, 3]
        lo, hi = points.min(axis=0), points.max(axis=0)
        if lo[2] < table + margin:
            return False
        if lo[2] > top + margin:
            continue
        delta = np.maximum(0.0, np.maximum(lo[:2] - center[:2], center[:2] - hi[:2]))
        if np.linalg.norm(delta) > radius + margin:
            continue
        poly = ConvexHull(points[:, :2])
        edges = points[poly.vertices, :2]
        vectors = np.roll(edges, -1, axis=0) - edges
        offset = center[:2] - edges
        cross = vectors[:, 0] * offset[:, 1] - vectors[:, 1] * offset[:, 0]
        if np.all(cross >= 0):
            return False
        alpha = np.clip(np.sum(offset * vectors, axis=1) / np.sum(vectors * vectors, axis=1), 0.0, 1.0)
        if np.min(np.linalg.norm(offset - alpha[:, None] * vectors, axis=1)) <= radius + margin:
            return False
    return True


def pad_targets(scene, center, radius, top, table, reference, qopen, qlimit):
    """Place thumb and index-finger pad inner surfaces on the measured side band.

    Search a bounded roll and hand aperture using native FK only. Reject an
    opening unless the whole hand fits outside the inflated cylinder.
    """
    center = np.asarray(center)
    count = 19
    fractions = np.linspace(0.0, 1.0, count)
    joints = [(1 - f) * qopen + f * qlimit for f in fractions]
    shapes = [hand_shapes(scene, q) for q in joints]
    main = np.array(reference["grasp"])[:3, :3]
    height = top - table
    candidates = []
    for roll in (0, -25, 25, -50, 50, -75, 75, -100, 100, -135, 135, 180):
        rot = main @ Rotation.from_euler("z", roll, degrees=True).as_matrix()
        rows = []
        for i, local in enumerate(shapes):
            a = local["right_thumb_elastomer"] @ rot.T
            b = local["right_index_elastomer"] @ rot.T
            u = b[:, :2].mean(axis=0) - a[:, :2].mean(axis=0)
            u /= np.linalg.norm(u)
            thumb = a[np.argmax(a[:, :2] @ u)]
            index = b[np.argmin(b[:, :2] @ u)]
            gap = (index[:2] - thumb[:2]) @ u
            mid = (thumb + index) / 2.0
            low = min(np.min(v @ rot[2]) for v in local.values())
            z = max(center[2] - mid[2], table + 0.004 - low)
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = [*(center[:2] - mid[:2]), z]
            levels = np.array([thumb[2], index[2]]) + z
            band = np.min(np.minimum(levels - table, top - levels))
            rows.append((gap, band, pose))
        for close_idx, (gap, band, goal) in enumerate(rows):
            if not (2 * radius - 0.015 < gap < 2 * radius - 0.003 and band > 0.006):
                continue
            for pre_idx in range(close_idx - 1, -1, -1):
                pgap, _, pre = rows[pre_idx]
                if pgap < 2 * radius + 0.008:
                    continue
                if not separated(shapes[pre_idx], pre, center, radius, table, top, 0.0015):
                    continue
                # Use the closest free opening with favorable contact heights.
                score = (
                    abs(gap - (2 * radius - 0.007))
                    + 0.00014 * abs(roll)
                    + 0.36 * abs(height / 2 - band)
                    + 0.12 * abs(pgap - 2 * radius - 0.010)
                )
                candidates.append((score, pre_idx, close_idx, pre, goal, roll, gap, band))
                break
    for _, i, j, pre, goal, roll, gap, band in sorted(candidates, key=lambda x: x[0]):
        yield (
            pre,
            goal,
            joints[i],
            joints[j],
            {
                "roll_deg": roll,
                "opening_fraction": float(fractions[i]),
                "closure_fraction": float(fractions[j]),
                "unloaded_closed_gap_m": float(gap),
                "pad_edge_margin_m": float(band),
            },
        )
