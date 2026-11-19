"""Bounded recovery for a measured, unconfirmed pinch."""

import mujoco as mj
import numpy as np

from .pinch_contact import PINCH_DIGITS
from .station import NoPlan


def rank_finger_probes(geometry, arm, hand, joint_names, joint_limits, free, target=None,
                      *, anchor=None, near_table=False, blocked=(), inward_only=False):
    """Rank clearance-checked free-finger moves by target or opposing-pad distance."""
    if free not in PINCH_DIGITS:
        raise ValueError("Unrecognized pinch finger")
    hand = np.asarray(hand)
    pad = geometry.model.geom(f"right_{free}_elastomer_visual").id
    if anchor is not None:
        if anchor not in PINCH_DIGITS or anchor == free:
            raise ValueError("Expected the other pinch finger as the anchor")
        opposing = geometry.model.geom(f"right_{anchor}_elastomer_visual").id

        def distance():
            return mj.mj_geomDistance(geometry.model, geometry.data, pad, opposing, .5, None)
    else:
        point = np.asarray(target, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            return []
        local = geometry.meshes[pad]

        def distance():
            rotation = geometry.data.geom_xmat[pad].reshape(3, 3)
            vertices = local @ rotation.T + geometry.data.geom_xpos[pad]
            return float(np.linalg.norm(vertices - point, axis=1).min())

    geometry.set_pose(arm, hand)
    start = distance()
    candidates = []
    try:
        for i, name in enumerate(joint_names):
            if not name.startswith(f"right_{free}_") or name in blocked:
                continue
            for delta in (-1.5, -.75, .75, 1.5):
                next_pose = hand.copy()
                next_pose[i] += np.deg2rad(delta)
                if not joint_limits[i][0] <= next_pose[i] <= joint_limits[i][1]:
                    continue
                try:
                    for sample in np.linspace(hand, next_pose, 5)[1:]:
                        geometry.pose(arm, sample, near_table=near_table, moving_fingers=(i,))
                except NoPlan:
                    continue
                remaining = distance()
                if np.isfinite(remaining) and (remaining < start or (not inward_only and (anchor or blocked))):
                    candidates.append((remaining, name, delta))
    finally:
        geometry.set_pose(arm, hand)
    candidates.sort(key=lambda proposal: (proposal[0], abs(proposal[2])))
    return [(name, delta) for _, name, delta in candidates]


def retry_pinch(decision, driver, *, select_probe=None):
    """Unload stalled joints and try different clear moves from measured feedback."""
    history = []
    tried, blocked = [], set()
    regrasped = False
    local_search = False
    while not decision["confirmed"]:
        stalled = decision.get("stalled_joints")
        if decision.get("contact_pattern") not in ("none", "single", "unstable") and not stalled:
            break
        if driver is None:
            return {**decision, "retry_blocked": "finger_actuation_unavailable", "retries": history}
        if stalled:
            blocked.update(stalled)
            decision = driver.relieve_stall(decision)
            history.append({"action": "relieve_stall", "joints": stalled, "decision": decision})
            if decision["confirmed"] or decision.get("contact_pattern") not in ("none", "single", "unstable"):
                break

        anchor = next(iter(decision["contacting_digits"]), None)
        local_search = local_search or regrasped or decision.get("contact_pattern") == "unstable"
        if anchor or blocked or local_search:
            next_probe = select_probe or driver.next_probe
            available = [digit for digit in PINCH_DIGITS if digit != anchor]
            available.sort(key=lambda digit: not any(name.startswith(f"right_{digit}_") for name in blocked))
            if local_search and not anchor:
                available.sort(key=lambda digit: sum(name.startswith(f"right_{digit}_") for name, _ in tried))
            for free in available:
                candidate = next_probe(anchor, free, tuple(tried), blocked=blocked)
                if candidate is not None and candidate not in tried:
                    break
            else:
                candidate = None
            if candidate is not None:
                tried.append(candidate)
                try:
                    decision = driver.probe_free_finger(anchor, free, candidate)
                except NoPlan as exc:
                    history.append({"action": "finger_probe_rejected", "candidate": candidate, "reason": str(exc)})
                    continue
                history.append({"action": "finger_probe", "anchor": anchor, "free": free,
                                "candidate": candidate, "decision": decision})
                if (anchor and not blocked and not decision["confirmed"] and not decision.get("stalled_joints")
                        and decision.get("contact_pattern") != "unstable"
                        and decision.get("contacting_digits") != [anchor]):
                    return {**decision, "retry_blocked": "anchor_contact_lost_or_unstable", "retries": history}
                continue
        if anchor or regrasped:
            if not anchor and not blocked:
                break
            return {**decision, "retry_blocked": "no_clear_finger_probe", "retries": history}

        driver.reopen_hand(keep_enabled=True)
        target = driver.relocalize_for_regrasp()
        driver.align_in_place(target)
        decision = driver.close_and_measure()
        history.append({"action": "regrasp_in_place", "decision": decision})
        regrasped = True
        tried.clear()
        blocked.clear()
        local_search = False
    return {**decision, "retries": history}
