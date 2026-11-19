"""Sharpa thumb/index actuation for the live grasp cell."""

from dataclasses import asdict
import json
import time

import numpy as np

from .hand_enable import await_enable_state
from .pinch_contact import (
    BASELINE_SAMPLES, FIRM_CONTACT_FLOOR, HOLD_SAMPLES, LIGHT_CONTACT_FLOOR, PINCH_DIGITS, PinchSample, judge_pinch,
)
from .station import NoPlan, in_bounds
from .station_diagnostics import tcp_report


WAYPOINT_TOLERANCE_RAD = .012
PROGRESS_RAD = .002
NO_PROGRESS_S = .8
TIGHTEN_TRAVEL_M = .004


class LivePinch:
    def __init__(self, robot):
        self.robot = robot
        self.hand = robot.hands["right"]
        self.sdk = self.hand._hand
        self.names = [j.name for j in self.hand.cfg.joints]
        self.digit_indices = {
            digit: [i for i, name in enumerate(self.names) if name.startswith(f"right_{digit}_")]
            for digit in PINCH_DIGITS
        }
        self.bounds = np.array([(j.min_rad, j.max_rad) for j in self.hand.cfg.joints])
        self.owned = False
        self.last = None
        self.baseline = []
        self.precontact_gap_m = None
        self.after = []

    def get(self, call):
        error, value = call()
        self.hand._raise_if_error(error, call.__name__)
        return value

    def set(self, call, *args):
        self.hand._raise_if_error(call(*args), call.__name__)

    def disabled(self):
        return not self.get(self.sdk.get_enable_state)

    def release(self):
        if self.owned:
            self.set(self.sdk.set_enable_state, False)
            await_enable_state(self.disabled, True, lambda: None)
            self.hand._enabled = self.owned = False

    def state(self):
        self.robot.guard()
        state = self.hand.read_state()
        q = np.asarray(state.joint_pos)
        if q.shape != (22,) or not in_bounds(q, self.bounds):
            raise RuntimeError("Invalid Sharpa joint feedback")
        return state, q

    def enable(self, q):
        if not self.disabled():
            raise RuntimeError("Right hand already enabled")
        # Some SDK setters can enable the hand as a side effect. Own cleanup
        # before the first setter, including when a preload fails partway.
        self.hand._enabled = self.owned = True
        mode = self.hand._symbols["ControlMode"].POSITION
        previous = self.get(self.sdk.get_control_mode)

        def set_while_disabled(call, *args):
            self.set(call, *args)
            if not self.disabled():
                raise RuntimeError(f"Sharpa enabled during {call.__name__}")

        if not self.sdk.is_collision_protection_enabled():
            set_while_disabled(self.sdk.enable_collision_protection, True)
        if not self.sdk.is_collision_protection_enabled():
            raise RuntimeError("Sharpa collision protection is unavailable")
        source = self.hand._symbols["ControlSource"].SDK
        if self.get(self.sdk.get_control_source) != source:
            set_while_disabled(self.sdk.set_control_source, source)
        if not 0 < self.get(self.sdk.get_speed_coeff) <= .150001:
            set_while_disabled(self.sdk.set_speed_coeff, .15)
        if not 0 < self.get(self.sdk.get_current_coeff) <= .200001:
            set_while_disabled(self.sdk.set_current_coeff, .20)
        set_while_disabled(self.sdk.set_joint_position, q.tolist(), True)

        def check_preload():
            if np.max(np.abs(self.state()[1] - q)) > .015:
                raise RuntimeError("Sharpa moved away from its measured preload")

        check_preload()
        if not self.disabled():
            raise RuntimeError("Sharpa enabled during preload")
        if previous == mode:
            self.set(self.sdk.set_enable_state, True)
        else:
            self.set(self.sdk.set_control_mode, mode)
        await_enable_state(self.disabled, False, check_preload)

    def pad_gap(self, q):
        arm_deg = self.robot.arm.read_joint_position_deg("B")
        return tcp_report(self.robot.geometry, arm_deg, q, "right", {})["modeled_pad_gap_mm"] / 1000

    def record_open_gap(self):
        self.precontact_gap_m = self.pad_gap(self.state()[1])

    def capture_baseline(self):
        if not self.baseline:
            self.baseline = [self.sample(measure_gap=False) for _ in range(BASELINE_SAMPLES)]
            path = self.robot.out / "tactile_baseline.json"
            path.write_text(json.dumps([asdict(sample) for sample in self.baseline], indent=2, allow_nan=False))

    def sample(self, *, measure_gap=True):
        deadline = time.monotonic() + .8
        while time.monotonic() < deadline:
            self.robot.guard()
            thumb, index = (self.hand.read_tactile_frame(ch, .06) for ch in (0, 1))
            if thumb is None or index is None:
                continue
            if any(len(f.f6) != 6 or not np.isfinite(f.f6).all() for f in (thumb, index)):
                raise RuntimeError("Invalid pinch tactile feedback")
            ids = (thumb.frame_id, index.frame_id)
            stamps = (thumb.sdk_timestamp_s, index.sdk_timestamp_s)
            state, q = self.state()
            if self.last is not None and (state.sequence == self.last.hand_sequence or
                    any(a == b for a, b in zip(ids, self.last.frame_ids)) or
                    any(a <= b for a, b in zip(stamps, self.last.timestamps_s))):
                continue
            gap = self.pad_gap(q) if measure_gap else None
            self.last = PinchSample(ids, stamps, (thumb.f6[:3], index.f6[:3]), gap, state.sequence)
            return self.last
        raise RuntimeError("Fresh thumb/index feedback is unavailable")

    def drive(self, begin, target):
        if not in_bounds(target, self.bounds):
            raise NoPlan("Finger command exceeds native limits")
        arm = self.robot.arm.read_arm_status("B")
        if arm.cur_state != 0 or arm.err_code or self.disabled():
            raise RuntimeError("Unexpected drive state before finger motion")
        arm_deg = self.robot.arm.read_joint_position_deg("B")
        moving = np.flatnonzero(np.abs(target - begin) > 1e-8)
        for q in np.linspace(begin, target, 5):
            self.robot.guard()
            self.robot.geometry.pose(arm_deg, q, near_table=True, moving_fingers=moving)
        self.robot.guard()
        self.hand.send_joint_position_rad(tuple(target))
        self.last_target = target.copy()

    def command(self, begin, target, *, advancing=PINCH_DIGITS, action="waypoint", floor=LIGHT_CONTACT_FLOOR):
        self.drive(begin, target)
        active = {digit: self.digit_indices[digit] for digit in advancing if self.digit_indices[digit]}
        best = {digit: np.max(np.abs(target[joints] - begin[joints])) for digit, joints in active.items()}
        progress_at = dict.fromkeys(active, time.monotonic())
        stalled = []
        samples = [self.sample() for _ in range(HOLD_SAMPLES)]
        while True:
            measured = self.state()[1]
            if np.any(measured < np.minimum(begin, target) - .025) or np.any(measured > np.maximum(begin, target) + .025):
                raise RuntimeError("Sharpa fingers left the checked command interval")
            error = {digit: np.max(np.abs(target[joints] - measured[joints])) for digit, joints in active.items()}
            contact = judge_pinch(self.baseline, samples[-HOLD_SAMPLES:], floor=floor,
                                  precontact_gap_m=self.precontact_gap_m)
            contacted = contact.get("contacting_digits", [])
            lagging = [digit for digit in active if error[digit] > WAYPOINT_TOLERANCE_RAD and digit not in contacted]
            if not lagging:
                break
            now = time.monotonic()
            for digit in lagging:
                if best[digit] - error[digit] >= PROGRESS_RAD:
                    best[digit] = error[digit]
                    progress_at[digit] = now
            stalled = [digit for digit in lagging if now - progress_at[digit] >= NO_PROGRESS_S]
            if stalled:
                break
            samples.append(self.sample())
        stalled_joints = {
            self.names[joint]: float(target[joint] - measured[joint])
            for digit in stalled for joint in active[digit]
            if abs(target[joint] - measured[joint]) > WAYPOINT_TOLERANCE_RAD
        }
        with (self.robot.out / "pinch_commands.jsonl").open("a") as log:
            log.write(json.dumps({"action": action, "target_rad": target.tolist(), "measured_rad": measured.tolist(),
                                  "force_xyz": samples[-1].force_xyz, "contact_floor_f6": floor,
                                  "stalled_digits": stalled, "stalled_joints": stalled_joints}, allow_nan=False) + "\n")
        self.after.extend(samples[-HOLD_SAMPLES:])
        decision = self.decision(floor=floor)
        if stalled:
            decision.update(reason="waypoint_tracking_stalled", stalled_digits=stalled, stalled_joints=stalled_joints)
        elif not decision["confirmed"] and decision.get("contact_pattern") in ("unstable", "bilateral"):
            self.after.extend(self.sample() for _ in range(HOLD_SAMPLES))
            decision = self.decision(floor=floor)
        return decision

    def decision(self, *, floor=LIGHT_CONTACT_FLOOR):
        return judge_pinch(self.baseline, self.after[-HOLD_SAMPLES:], floor=floor,
                           precontact_gap_m=self.precontact_gap_m)

    def prepare_open(self, opened):
        start = self.state()[1]
        if not self.owned:
            self.enable(start)
        count = max(1, int(np.ceil(np.max(np.abs(opened - start)) / np.deg2rad(1.5))))
        held = np.zeros(len(start), dtype=bool)
        for target in np.linspace(start, opened, count + 1)[1:]:
            measured = self.state()[1]
            target[held] = measured[held]
            if np.max(np.abs(target - measured)) < .003:
                continue
            self.drive(measured, target)
            best = np.abs(target - measured)
            progress_at = np.full(len(start), time.monotonic())
            while True:
                measured = self.state()[1]
                error = np.abs(target - measured)
                now = time.monotonic()
                improved = best - error >= PROGRESS_RAD
                best[improved] = error[improved]
                progress_at[improved] = now
                stalled = (~held) & (error > .010) & (now - progress_at >= NO_PROGRESS_S)
                if np.any(stalled):
                    held |= stalled
                    target[held] = measured[held]
                    self.drive(measured, target)
                if not np.any((~held) & (error > .010)):
                    break
                time.sleep(.02)
        self.robot.fingers = self.state()[1]
        self.open_pose = self.robot.fingers.copy()
        return [self.names[i] for i in np.flatnonzero(held)]

    def close_and_measure(self):
        path = self.robot.grasp_plan.finger_path
        self.after = []
        decision = self.decision()
        if decision["reason"] != "awaiting_post_close_feedback":
            return decision
        frozen = set()
        sent = path[0].copy()
        for waypoint in path[1:]:
            current = self.state()[1]
            target = waypoint.copy()
            for digit in frozen:
                joints = self.digit_indices[digit]
                target[joints] = sent[joints]
            try:
                advancing = [digit for digit in PINCH_DIGITS if digit not in frozen]
                decision = self.command(current, target, advancing=advancing)
            except NoPlan:
                if decision.get("contact_pattern") == "single":
                    return decision
                raise
            sent = target
            if decision["reason"] == "waypoint_tracking_stalled":
                return decision
            if decision.get("contact_pattern") == "unstable":
                return decision
            frozen = set(decision.get("contacting_digits", ()))
            if decision["confirmed"] or len(frozen) == len(PINCH_DIGITS):
                return decision
        if decision["reason"] == "pad_gap_not_stable":
            self.after.extend(self.sample() for _ in range(HOLD_SAMPLES))
            decision = self.decision()
        return decision

    def reopen_hand(self, *, keep_enabled=False):
        self.prepare_open(self.open_pose)
        if not keep_enabled:
            self.release()
            self.robot.fingers = self.state()[1]

    def relocalize_for_regrasp(self):
        return self.robot.relocalize_for_regrasp()

    def align_in_place(self, target):
        self.robot.align_in_place(target)

    def relieve_stall(self, decision, *, floor=LIGHT_CONTACT_FLOOR):
        start = self.state()[1]
        hold = start.copy()
        for digit in decision.get("contacting_digits", ()):
            joints = self.digit_indices[digit]
            hold[joints] = self.last_target[joints]
        retreat = hold.copy()
        for name, error in decision["stalled_joints"].items():
            joint = self.names.index(name)
            retreat[joint] -= np.clip(error, -np.deg2rad(1.5), np.deg2rad(1.5))
        retreat = np.clip(retreat, self.bounds[:, 0], self.bounds[:, 1])
        try:
            result = self.command(start, retreat, advancing=decision["stalled_digits"], action="relieve_stall", floor=floor)
        except NoPlan:
            result = None
        if result is not None and result["reason"] != "waypoint_tracking_stalled":
            return result
        # Release the old push even when a retreat direction is unavailable.
        current = self.state()[1]
        for digit in PINCH_DIGITS:
            if digit not in decision.get("contacting_digits", ()):
                joints = self.digit_indices[digit]
                hold[joints] = current[joints]
        return self.command(current, hold, advancing=(), action="hold_after_stall", floor=floor)

    def probe_free_finger(self, anchor, free, candidate):
        self.after.append(self.sample())
        latest = self.decision()
        if latest.get("contact_pattern") == "unstable":
            self.after.extend(self.sample() for _ in range(HOLD_SAMPLES))
            latest = self.decision()
        expected = [anchor] if anchor else []
        if (latest["confirmed"] or latest.get("contact_pattern") not in ("none", "single", "unstable")
                or latest.get("contacting_digits") != expected):
            return latest
        return self.move_probe(free, candidate, hold=expected)

    def move_probe(self, free, candidate, *, hold=(), action="finger_probe", floor=LIGHT_CONTACT_FLOOR):
        name, delta = candidate
        if not name.startswith(f"right_{free}_") or not 0 < abs(delta) <= 1.5:
            raise NoPlan("Probe would move the anchored finger")
        start = self.state()[1]
        goal = start.copy()
        for digit in hold:
            joints = self.digit_indices[digit]
            goal[joints] = self.last_target[joints]
        goal[self.names.index(name)] += np.deg2rad(delta)
        return self.command(start, goal, advancing=(free,), action=action, floor=floor)

    def tighten_and_measure(self, points, select_probe):
        self.after.extend(self.sample() for _ in range(HOLD_SAMPLES))
        light = self.decision()
        if not light["confirmed"]:
            return {**light, "reason": "light_contact_lost_before_tightening"}

        points = np.asarray(points)
        midpoint = points.mean(axis=0)
        offsets = midpoint - points
        half_gap = np.linalg.norm(offsets[0])
        targets = points + offsets * min(.5, TIGHTEN_TRAVEL_M / half_gap)
        tried, blocked = set(), set()
        decision = self.decision(floor=FIRM_CONTACT_FLOOR)
        while not decision["confirmed"]:
            attempted = False
            for digit, target in zip(PINCH_DIGITS, targets):
                firm = decision.get("contacting_digits", ())
                if digit in firm:
                    continue
                candidate = select_probe(None, digit, tried, target=target, blocked=blocked, inward_only=True)
                if candidate is None or candidate in tried:
                    continue
                tried.add(candidate)
                attempted = True
                try:
                    decision = self.move_probe(digit, candidate, hold=firm,
                                               floor=FIRM_CONTACT_FLOOR, action="tighten")
                except NoPlan:
                    continue
                if decision.get("stalled_joints"):
                    blocked.update(decision["stalled_joints"])
                    decision = self.relieve_stall(decision, floor=FIRM_CONTACT_FLOOR)
                if decision["confirmed"]:
                    break
            if not attempted:
                return {**decision, "contact_reason": decision["reason"], "reason": "firm_grip_not_reached"}
        return {**decision, "finished_tightening": True}

    def verify_retention(self):
        self.after.append(self.sample())
        return self.decision(floor=FIRM_CONTACT_FLOOR)["confirmed"]
