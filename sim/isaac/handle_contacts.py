"""Measure distinct finger loads and locate their contact points on the observed handle."""

from collections import Counter

import numpy as np

from .head_wrist_trial import FINGERS
from .scene_fixture import pose_matrix


def loaded_digits(links, minimum=0.08):
    """Count distinct digits on the object, including phalanges and elastomer pads."""
    return {
        d
        for d in FINGERS
        if sum(links.get(f"right_{d}_{s}", 0.0) for s in ("PP", "MP", "DP", "elastomer")) >= minimum
    }


def tool_supported(links):
    digits = loaded_digits(links)
    return "thumb" in digits and len(digits) >= 3


def handle_observations(rows, grasp, at_grasp):
    """Move measured world contact points back into the observed handle's table frame."""
    reference = pose_matrix(at_grasp[:3], at_grasp[3:])
    axis = np.array(grasp["handle_axis_xy"])
    transverse = np.array([axis[1], -axis[0]])
    lo, hi = grasp["handle_bounds_m"]
    origin = np.array(grasp["handle_origin_xy"])
    sections = np.asarray(grasp["handle_sections_m"])
    on_handle = Counter()
    opposed_ticks = 0
    minimum = len(FINGERS)
    for row in rows:
        obj = row["object_pose_wxyz"]
        now = pose_matrix(obj[:3], obj[3:])
        back = reference @ np.linalg.inv(now)
        measured = {}
        lateral = {d: [] for d in FINGERS}
        for name, (point, _) in row["contact_surfaces"].items():
            digit = next((d for d in FINGERS if name.startswith("right_" + d + "_")), None)
            if digit is None or not name.endswith(("_PP", "_MP", "_DP", "_elastomer")):
                continue
            observed = back[:3, :3] @ point + back[:3, 3]
            station = axis @ (observed[:2] - origin)
            if not lo <= station <= hi:
                continue
            center, height, half_width, half_height = (
                np.interp(station, sections[:, 0], sections[:, k]) for k in (1, 2, 3, 4)
            )
            x = transverse @ (observed[:2] - origin) - center
            z = observed[2] - height
            radius = np.hypot(x, z)
            normalized = np.hypot(x / half_width, z / half_height)
            # Distance to the fitted cross section along the contact's radial line.
            # Allow 6 mm for RGB-D registration and the convex collider approximation.
            if normalized < 1e-8 or abs(radius - radius / normalized) > 0.006:
                continue
            measured[name] = row["link_normal_N"][name]
            lateral[digit].append(float(x))
        used = loaded_digits(measured)
        minimum = min(minimum, len(used))
        on_handle.update(used)
        # Contacts must occupy opposite sides of the observed handle centerline.
        if "thumb" in used:
            opposed_ticks += int(
                any(
                    sum(any(t * x < 0 and abs(x - t) > 0.010 for x in lateral[d]) for d in used - {"thumb"}) >= 2
                    for t in lateral["thumb"]
                )
            )
    return {
        "on_handle_digit_ticks": dict(on_handle),
        "minimum_digits_on_handle": minimum if rows else 0,
        "opposed_on_handle_ticks": opposed_ticks,
    }
