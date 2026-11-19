"""Assess a thumb–index closure from tactile and measured finger geometry."""

from dataclasses import dataclass

import numpy as np


BASELINE_SAMPLES = 8
HOLD_SAMPLES = 3
# Light touch uses a fixed F6 threshold. Firm contact also accounts for baseline noise.
LIGHT_CONTACT_FLOOR = .00001
FIRM_CONTACT_FLOOR = .02
MIN_GAP_REDUCTION_M = .001
HOLD_GAP_VARIATION_M = .001
PINCH_DIGITS = ("thumb", "index")


@dataclass(frozen=True)
class PinchSample:
    frame_ids: tuple[int, int]
    timestamps_s: tuple[float, float]
    force_xyz: tuple[tuple[float, ...], tuple[float, ...]]
    pad_gap_m: float | None
    hand_sequence: int


def judge_pinch(baseline, after_close, *, floor=LIGHT_CONTACT_FLOOR, precontact_gap_m=None):
    """Require distinct frames, measured closure and persistent two-pad contact."""
    if len(baseline) < BASELINE_SAMPLES:
        return {"confirmed": False, "reason": "insufficient_baseline"}

    window = [*baseline, *after_close]
    frame_ids = np.asarray([sample.frame_ids for sample in window])
    timestamps = np.asarray([sample.timestamps_s for sample in window])
    force = np.asarray([sample.force_xyz for sample in window], dtype=float)
    gaps = np.asarray([sample.pad_gap_m for sample in after_close], dtype=float)
    if (force.shape != (len(window), 2, 3) or frame_ids.shape != (len(window), 2)
            or timestamps.shape != frame_ids.shape or not np.isfinite(force).all()
            or not np.isfinite(timestamps).all() or not np.isfinite(gaps).all() or np.any(gaps < 0)):
        return {"confirmed": False, "reason": "invalid_feedback"}
    if np.any(np.diff(timestamps, axis=0) <= 0) or np.any(np.diff(frame_ids, axis=0) == 0):
        return {"confirmed": False, "reason": "stale_tactile_frames"}
    if np.any(np.diff([sample.hand_sequence for sample in window]) == 0):
        return {"confirmed": False, "reason": "stale_hand_feedback"}

    count = len(baseline)
    if precontact_gap_m is None:
        baseline_gaps = np.asarray([sample.pad_gap_m for sample in baseline], dtype=float)
        if not np.isfinite(baseline_gaps).all() or np.any(baseline_gaps < 0):
            return {"confirmed": False, "reason": "invalid_feedback"}
        if np.ptp(baseline_gaps) > HOLD_GAP_VARIATION_M:
            return {"confirmed": False, "reason": "baseline_pad_gap_not_stable"}
        precontact_gap_m = np.median(baseline_gaps)
    if not np.isfinite(precontact_gap_m) or precontact_gap_m < 0:
        return {"confirmed": False, "reason": "invalid_feedback"}
    center = np.median(force[:count], axis=0)
    noise = np.linalg.norm(force[:count] - center, axis=2).max(axis=0)
    threshold = np.full(len(PINCH_DIGITS), floor) if floor == LIGHT_CONTACT_FLOOR else np.maximum(floor, 5 * noise)
    result = {
        "confirmed": False,
        "baseline_pad_gap_m": float(precontact_gap_m),
        "contact_threshold_f6": threshold.tolist(),
    }
    if len(after_close) < HOLD_SAMPLES:
        return {**result, "reason": "awaiting_post_close_feedback"}

    held_gaps = gaps[-HOLD_SAMPLES:]
    reduction = result["baseline_pad_gap_m"] - held_gaps
    held_delta = np.linalg.norm(force[-HOLD_SAMPLES:] - center, axis=2)
    touches = held_delta >= threshold
    contacting = [digit for digit, sustained in zip(PINCH_DIGITS, touches.all(axis=0)) if sustained]
    pattern = "unstable"
    if len(contacting) == len(PINCH_DIGITS):
        pattern = "bilateral"
    elif len(contacting) == 1:
        pattern = "single"
    elif not touches.any():
        pattern = "none"
    result.update(pad_gap_reduction_m=float(np.min(reduction)),
                  minimum_held_force_change_f6=np.min(held_delta, axis=0).tolist(),
                  contact_pattern=pattern, contacting_digits=contacting)
    if pattern == "bilateral":
        return {**result, "confirmed": True, "reason": "bilateral_contact_confirmed"}
    if np.min(reduction) < MIN_GAP_REDUCTION_M:
        return {**result, "reason": "closure_not_measured"}
    if np.ptp(held_gaps) > HOLD_GAP_VARIATION_M:
        return {**result, "reason": "pad_gap_not_stable"}
    return {**result, "reason": "bilateral_contact_not_sustained"}
