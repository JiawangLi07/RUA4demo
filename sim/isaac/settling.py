"""Measure table support and contact-free rest before the first observation."""

from collections.abc import Mapping

import numpy as np

SIDES = ("left", "right")


def verify_settled(samples: list[dict], start: float, config: Mapping) -> dict:
    """Require quiet objects without any prior hand contact during bounded initialization."""
    count = round(config["window_seconds"] / config["physics_dt"])
    if count < 2:
        raise ValueError("settling window requires at least two physics samples")
    keys = ("object_linear_speed_m_s", "object_angular_speed_rad_s", "contact_count")
    finite = bool(samples) and all(
        row["finite"]
        and np.isfinite(
            [row["time"], row["hand_table_penetration_m"], *(row[side][key] for side in SIDES for key in keys)]
        ).all()
        for row in samples
    )
    if not finite:
        return {"passed": False, "failures": ["invalid_or_missing_state"], "metrics": {"samples": len(samples)}}
    tail = samples[-count:]
    elapsed = samples[-1]["time"] - start
    metrics = {
        "seconds": elapsed,
        "maximum_linear_speed_m_s": max(row[side]["object_linear_speed_m_s"] for row in tail for side in SIDES),
        "maximum_angular_speed_rad_s": max(
            row[side]["object_angular_speed_rad_s"] for row in tail for side in SIDES
        ),
        "maximum_contact_points": max(row[side]["contact_count"] for row in samples for side in SIDES),
        "maximum_hand_table_penetration_m": max(row["hand_table_penetration_m"] for row in samples),
    }
    failures = []
    times = np.array([row["time"] for row in tail])
    if (
        len(tail) != count
        or elapsed < config["minimum_seconds"] - config["physics_dt"] / 2
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(times) > 1.5 * config["physics_dt"])
    ):
        failures.append("incomplete_settling_window")
    if (
        metrics["maximum_linear_speed_m_s"] > config["maximum_linear_speed_m_s"]
        or metrics["maximum_angular_speed_rad_s"] > config["maximum_angular_speed_rad_s"]
    ):
        failures.append("objects_not_settled")
    if metrics["maximum_contact_points"] > 0:
        failures.append("hand_contact_during_settling")
    if metrics["maximum_hand_table_penetration_m"] > config["maximum_hand_table_penetration_m"]:
        failures.append("hand_table_envelope_overlap")
    return {"passed": not failures, "failures": failures, "metrics": metrics}
