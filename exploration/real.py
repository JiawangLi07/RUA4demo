"""Execute a single prepared grasp trial through the installed robot SDKs."""

from contextlib import ExitStack
from pathlib import Path
import fcntl
import json
import time

import numpy as np
from PIL import Image

from hardware import DEPLOYMENT, ROOT, arm_settings, hand_settings, hardware_owners
from rgbd import camera_stream
from .clearance import HandClearance
from .hand_enable import await_enable_state
from .grasp_program import verdict
from .station import CUBE_RESET_TOLERANCE_M, START_TOLERANCE_DEG, NoPlan, Planner, find_red_cube, in_bounds


ARM_STEP_INTERVAL_S = 0.05
TRIAL_SECONDS = 75
REPEAT_CUBE_TOLERANCE_M = 0.012


class TargetMoved(RuntimeError):
    """The observed cube shifted while the arm was still at pregrasp."""


def idle_arm(arm, arm_id, trace):
    """Confirm a native idle transaction while retaining the transport."""
    sdk = arm._robot
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if sdk.clear_set() and sdk.set_state(arm=arm_id, state=0) and sdk.send_cmd():
            before = arm.read_arm_state(arm_id, "idle_requested").frame_serial
            time.sleep(0.1)
            state = arm.read_arm_state(arm_id, "idle")
            if state.frame_serial != before and state.cur_state == 0 and state.err_code == 0:
                trace("idle_confirmed", arm=arm_id, cmd_state=state.cmd_state,
                      frame_serial=state.frame_serial)
                return
        time.sleep(0.1)
    trace("idle_unconfirmed", arm=arm_id)
    raise RuntimeError("Tianji idle was not acknowledged")


class RealTrials:
    def __init__(self, station):
        self.station = station
        self.check_placement = None
        self.grasps = tuple(station.hand_targets)
        self.planner = Planner(station)
        self.detector = None

    def check(self):
        return {"source": "profile_only", "ik": "checked from live state before enabling motion"}

    def run(self, program, out, _layout, *, cancelled):
        from tianji_command_adapter import ReliableTianjiRobot
        from unirobot_infra.robots.end_effector.sharpa import SharpaHand
        from unirobot_infra.utils.sdk_paths import configure_sharpa_runtime, sharpa_manus_sdk_paths

        station = self.station
        from pyrualean.runner import load_policy
        policy_file = out / f"grasp_{station.target}.py"
        source = program.source()
        if policy_file.is_file():
            if policy_file.read_text() != source:
                raise ValueError("Trial program differs from the bounded proposal")
        else:
            policy_file.write_text(source)
        execute = load_policy(policy_file, builtins={})
        if station.target == "shuttlecock" and self.detector is None:
            from .shuttlecock import detector_client
            self.detector = detector_client()
        arm_id = {"left": "A", "right": "B"}[station.arm]
        with ExitStack() as resources:
            log = resources.enter_context((out / "trajectory.jsonl").open("w", buffering=1))

            def trace(event, **data):
                log.write(json.dumps({"monotonic_ns": time.monotonic_ns(), "event": event, **data},
                                     allow_nan=False) + "\n")

            for path in (ROOT / ".capture.lock", DEPLOYMENT / "runs" / ".active.lock"):
                lock = resources.enter_context(path.open("a"))
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            owners = hardware_owners()
            if any(owners.values()):
                raise RuntimeError(f"Devices are in use: {owners}")
            if cancelled():
                raise RuntimeError("Exploration client disconnected")
            read_camera = resources.enter_context(camera_stream(station.serial))
            cfg = arm_settings()
            cfg.vel_ratio = cfg.acc_ratio = 5
            arm = ReliableTianjiRobot(cfg)
            resources.callback(arm.close)
            arm.connect()
            hand_cfg = hand_settings()
            hand_cfg.initialize_control_on_connect = False
            hand_cfg.enable_on_connect = False
            hand_cfg.speed_coeff = 0.15
            hand_cfg.current_coeff = 0.2
            paths = sharpa_manus_sdk_paths(Path(hand_cfg.sdk.vendor_root))
            configure_sharpa_runtime(paths)
            hand = SharpaHand(hand_cfg, side=station.arm, sdk_paths=paths)
            resources.callback(hand.close)
            hand.connect()
            robo = GraspRobot(station, self.planner, arm, arm_id, hand, read_camera,
                             resources, out, trace, cancelled)
            robo.detector = self.detector
            other = "left" if station.arm == "right" else "right"
            parked_hand = SharpaHand(hand_cfg, side=other, sdk_paths=paths)
            resources.callback(parked_hand.close)
            parked_hand.connect()
            robo.parked_hand = parked_hand
            robo.check_start()
            before = robo.observe("before")
            for attempt in range(2):
                if before is not None or station.target != "shuttlecock":
                    break
                # No actuator has been enabled. Preserve the missed frame and take another.
                for suffix in (".png", "_depth.png", "_target.png"):
                    path = out / ("before" + suffix)
                    if path.exists():
                        path.rename(out / f"before_miss_{attempt}{suffix}")
                before = robo.observe("before")
            if before is None or not np.isfinite(before).all() or (station.cube_m is not None
                    and np.linalg.norm(before - station.cube_m) > CUBE_RESET_TOLERANCE_M):
                raise RuntimeError(f"Expected one visible {station.target} at the measured station position")
            if self.check_placement:
                self.check_placement(before)
            robo.before = before
            lifted = []
            try:
                trace("lean_policy_start", program=policy_file.name, target=station.target)
                lifted = execute(robo)
                outcome = verdict(before, lifted, station.up)
                robo.reset()
            except NoPlan as exc:
                outcome = {"success": False, "reason": "path_rejected", "detail": str(exc)}
            except TargetMoved as exc:
                robo.move(robo.paths[0][::-1])
                outcome = {"success": False, "reason": f"{station.target}_shifted_before_grasp",
                           "detail": str(exc)}
            reset = robo.observe("reset")
            return {
                **outcome, "source": "real", "before_m": before.tolist(),
                "lift_samples_m": lifted, "contact": robo.contact,
                "reset_m": None if reset is None else reset.tolist(),
                "reset_ok": reset is not None and bool(
                    (station.cube_m is None or np.linalg.norm(reset - station.cube_m) <= CUBE_RESET_TOLERANCE_M)
                    and np.linalg.norm(reset - before) <= REPEAT_CUBE_TOLERANCE_M),
            }


class GraspRobot:
    def __init__(self, station, planner, arm, arm_id, hand, camera, resources, out, trace, cancelled):
        self.station, self.planner = station, planner
        self.arm, self.arm_id, self.hand = arm, arm_id, hand
        self.parked = "A" if arm_id == "B" else "B"
        self.camera, self.resources, self.out = camera, resources, out
        self.trace, self.cancelled = trace, cancelled
        self.deadline = time.monotonic() + TRIAL_SECONDS
        self.arm_serial, self.hand_serial = None, None
        self.moving = False
        self.contact = {}
        self.parked_origin = None
        self.parked_hand_origin = None
        self.parked_hand_serial = None

    def guard(self):
        if self.cancelled() or time.monotonic() >= self.deadline:
            raise RuntimeError("Exploration client disconnected or the trial timed out")

    def fresh(self, sample, sequence, attr):
        until = time.monotonic() + 0.35
        while time.monotonic() < until:
            self.guard()
            value = sample()
            index = getattr(value, sequence)
            if index != getattr(self, attr):
                setattr(self, attr, index)
                return value
            time.sleep(0.01)
        raise RuntimeError(f"Stale feedback from {attr}")

    def feedback(self):
        state = self.fresh(lambda: self.arm.read_arm_state(self.arm_id, "active"),
                           "frame_serial", "arm_serial")
        expected_mode = 1 if self.moving else 0
        if state.err_code or state.cur_state != expected_mode:
            raise RuntimeError(f"Tianji status error {state.err_code}, mode {state.cur_state}")
        q = np.asarray(self.arm.read_joint_position_deg(self.arm_id))
        parked = np.asarray(self.arm.read_joint_position_deg(self.parked))
        parked_status = self.arm.read_arm_status(self.parked)
        if (not in_bounds(q, self.station.joints_deg) or not np.isfinite(parked).all()
                or parked_status.err_code or parked_status.cur_state != 0
                or (self.parked_origin is not None and np.max(np.abs(parked - self.parked_origin)) > 0.15)):
            raise RuntimeError("Arm feedback outside the taught joint envelope")
        force = np.asarray(state.est_cart_force)
        if not np.isfinite(force).all() or np.any(np.abs(force) > self.station.wrench_limits):
            self.trace("arm_wrench_limit")
            raise RuntimeError("Tianji estimated wrench limit reached")
        pose, _ = self.planner.ik.forward_jacobian(np.deg2rad(q))
        if not in_bounds(pose[:3, 3], self.station.workspace_m):
            raise RuntimeError("Measured TCP outside the station workspace")
        return q

    def hand_feedback(self):
        state = self.fresh(self.hand.read_state, "sequence", "hand_serial")
        q = np.asarray(state.joint_pos)
        torque = np.asarray(state.joint_torque)
        if (not np.isfinite(q).all() or not np.isfinite(torque).all()
                or np.any(np.abs(torque) > self.station.torque_limits)):
            self.trace("hand_torque_limit", measured_rad=q.tolist() if np.isfinite(q).all() else None)
            raise RuntimeError("Sharpa hand torque limit or invalid joint feedback")
        return q, torque

    def check_start(self):
        self.check_hand_disabled()
        parked = np.asarray(self.arm.read_joint_position_deg(self.parked))
        if self.parked_origin is None:
            self.parked_origin = parked.copy()
        q = self.feedback()
        hand, _ = self.hand_feedback()
        if not in_bounds(hand, self.station.hand_joints_rad):
            raise RuntimeError("Current hand outside joint limits")
        self.check_parked_hand()
        return q, hand

    def check_parked_hand(self):
        state = self.fresh(self.parked_hand.read_state, "sequence", "parked_hand_serial")
        measured = np.asarray(state.joint_pos)
        if measured.shape != (22,) or not np.isfinite(measured).all():
            raise RuntimeError("Invalid parked hand feedback")
        if self.parked_hand_origin is None:
            self.parked_hand_origin = measured.copy()
        elif np.max(np.abs(measured - self.parked_hand_origin)) > 0.015:
            raise RuntimeError("Parked hand changed since clearance check")

    def check_hand_disabled(self):
        error, enabled = self.hand._hand.get_enable_state()
        self.hand._raise_if_error(error, "Sharpa enable-state preflight failed")
        if enabled:
            raise RuntimeError("Sharpa hand is still enabled by another session")

    def enable_hand_at_feedback(self):
        self.check_hand_disabled()
        measured, _ = self.hand_feedback()
        self.verify_hand_pose(measured, self.initial_hand)
        self.trace("hand_before_enable", joint_rad=measured.tolist())
        sdk = self.hand._hand
        modes = self.hand._symbols
        error, previous_mode = sdk.get_control_mode()
        self.hand._raise_if_error(error, "Sharpa mode readback failed")
        position_mode = modes["ControlMode"].POSITION
        if not sdk.is_collision_protection_enabled():
            self.hand._raise_if_error(sdk.enable_collision_protection(True), "Sharpa collision protection failed")
        if not sdk.is_collision_protection_enabled():
            raise RuntimeError("Sharpa collision protection was not acknowledged in this session")
        self.check_hand_disabled()
        self.trace("hand_collision_protection_confirmed")
        steps = (
            (sdk.set_control_source, (modes["ControlSource"].SDK,), "select SDK source"),
            (sdk.set_speed_coeff, (float(self.hand.cfg.speed_coeff),), "set speed"),
            (sdk.set_current_coeff, (float(self.hand.cfg.current_coeff),), "set current"),
            (sdk.set_joint_position, (measured.tolist(), True), "preload measured pose"),
        )
        for call, args, label in steps:
            self.guard()
            self.hand._raise_if_error(call(*args), f"Sharpa could not {label}")
        self.guard()
        self.check_hand_disabled()
        before_enable, _ = self.hand_feedback()
        self.verify_hand_pose(before_enable, measured)
        self.trace("hand_enable_readback", joint_rad=before_enable.tolist())
        # Treat even an ambiguous enable acknowledgement as owned for cleanup.
        self.hand._enabled = True
        # The vendor mode setter itself enables the hand before returning.
        if previous_mode == position_mode:
            ack = sdk.set_enable_state(True)
        else:
            ack = sdk.set_control_mode(position_mode)
        self.hand._raise_if_error(ack, "Sharpa enable failed")
        error, observed_mode = sdk.get_control_mode()
        self.hand._raise_if_error(error, "Sharpa mode verification failed")
        if observed_mode != position_mode:
            raise RuntimeError("Sharpa did not enter position mode")
        def read_enabled():
            error, enabled = sdk.get_enable_state()
            self.hand._raise_if_error(error, "Sharpa enable-state confirmation failed")
            return enabled

        def check_hold():
            self.guard()
            actual, _ = self.hand_feedback()
            self.verify_hand_pose(actual, measured)
            self.verify_hand_pose(actual, self.initial_hand)

        await_enable_state(read_enabled, True, check_hold)
        for _ in range(3):
            actual, _ = self.hand_feedback()
            self.trace("hand_after_enable", joint_rad=actual.tolist())
            self.verify_hand_pose(actual, measured)
            self.verify_hand_pose(actual, self.initial_hand)
            time.sleep(0.02)
        self.trace("hand_preloaded_and_enabled", joint_rad=measured.tolist())

    def verify_hand_pose(self, measured, expected):
        if (not in_bounds(measured, self.station.hand_joints_rad)
                or np.max(np.abs(measured - expected)) > 0.015):
            raise RuntimeError("Hand differs from the clearance-checked pose")

    def observe(self, name):
        self.guard()
        snapshot = self.camera()
        meta = {}
        if self.station.target == "shuttlecock":
            from .shuttlecock import monitored_measure
            point, mask, meta = monitored_measure(self.detector, snapshot, self.station, self.check_observing)
            if mask is not None:
                Image.fromarray(mask).save(self.out / f"{name}_target.png")
        else:
            point = find_red_cube(snapshot, self.station)
        Image.fromarray(snapshot.rgb).save(self.out / f"{name}.png")
        Image.fromarray(snapshot.raw_depth).save(self.out / f"{name}_depth.png")
        coordinates = None if point is None else point.tolist()
        self.trace("observe", phase=name, target=self.station.target, target_m=coordinates, detector=meta,
                   **({"cube_m":coordinates} if self.station.target == "cube" else {}),
                   sample_monotonic_ns=snapshot.monotonic_ns,
                   color_sequence=getattr(snapshot, "color_sequence", None),
                   depth_sequence=getattr(snapshot, "depth_sequence", None))
        return point

    def check_observing(self):
        self.guard()
        if self.moving:
            self.feedback()
            self.hand_feedback()

    def prepare(self, offset_m, lift_m, grasp):
        start, self.initial_hand = self.check_start()
        self.expected_hand = self.initial_hand.copy()
        self.paths = self.planner.plan(start, self.before, offset_m, lift_m)
        self.hand_target = self.station.hand_targets[grasp]
        self.clearance = HandClearance(self.station, self.parked_origin, self.parked_hand_origin)
        self.clearance.path(self.paths, self.initial_hand, self.hand_target)
        q, hand = self.check_start()
        if np.max(np.abs(q - start)) > 0.15:
            raise NoPlan("Arm moved during clearance check")
        self.verify_hand_pose(hand, self.initial_hand)
        self.trace("path_clearance_passed", start_deg=start.tolist(), hand_rad=hand.tolist())
        self.guard()
        self.resources.callback(idle_arm, self.arm, self.arm_id, self.trace)
        self.arm.enter_position_mode(self.arm_id)
        self.moving = True
        if np.max(np.abs(self.feedback() - start)) > 0.15:
            raise RuntimeError("Arm moved on enable")
        self.enable_hand_at_feedback()
        self.trace("motion_ready", grasp=grasp, waypoints=[len(path) for path in self.paths])

    def move(self, path):
        if np.max(np.abs(self.feedback() - path[0])) > START_TOLERANCE_DEG:
            raise RuntimeError("Measured arm position differs from the path start")
        self.hand_feedback()
        for target in path[1:]:
            self.guard()
            self.check_parked_hand()
            current_hand, _ = self.hand_feedback()
            self.verify_hand_pose(current_hand, self.expected_hand)
            self.arm.send_joint_position_deg(self.arm_id, tuple(target), blocking=False)
            time.sleep(ARM_STEP_INTERVAL_S)
            until = time.monotonic() + 0.45
            while True:
                q = self.feedback()
                self.hand_feedback()
                if np.max(np.abs(q - target)) <= 0.55:
                    break
                if time.monotonic() >= until:
                    raise RuntimeError("Tianji failed to track the bounded waypoint")
                time.sleep(0.02)
            self.trace("arm_waypoint", target_deg=target.tolist(), measured_deg=q.tolist())

    def hand_path(self, target):
        start, torque = self.hand_feedback()
        count = max(1, int(np.ceil(np.max(np.abs(target - start)) / 0.035)))
        path = np.linspace(start, target, count + 1)[1:]
        if any(not in_bounds(q, self.station.hand_joints_rad) for q in path):
            raise RuntimeError("Finger path exceeds the joint bounds")
        peak = np.abs(torque)
        for q in path:
            self.check_parked_hand()
            self.feedback()
            self.hand.send_joint_position_rad(tuple(q))
            time.sleep(0.05)
            measured, torque = self.hand_feedback()
            self.verify_hand_pose(measured, q)
            peak = np.maximum(peak, np.abs(torque))
            self.trace("hand_waypoint", target_rad=q.tolist(), measured_rad=measured.tolist(),
                       torque=torque.tolist())
        self.expected_hand = target.copy()
        return measured, peak

    def approach(self):
        self.move(self.paths[0])

    def confirm_target(self):
        point = self.observe("at_pregrasp")
        if point is None or np.linalg.norm(point - self.before) > 0.008:
            raise TargetMoved(f"{self.station.target} is missing or shifted by more than 8 mm")

    def descend(self):
        self.move(self.paths[1])

    def close_hand(self):
        if self.station.contact_policy is None:
            measured, peak = self.hand_path(self.hand_target)
        else:
            from .contact import ContactClosure
            opened, baseline = self.hand_feedback()
            controller = ContactClosure(self.station.contact_policy, baseline)
            samples = max(1, int(np.ceil(max(abs(self.hand_target-opened))/.035)))
            peak = abs(baseline)
            measured = opened
            command = opened.copy()
            for target in np.linspace(opened,self.hand_target,samples+1)[1:]:
                self.guard()
                self.feedback()
                command = target.copy()
                for group, held in controller.stopped.items():
                    command[controller.groups[group]] = held
                self.hand.send_joint_position_rad(tuple(command))
                time.sleep(.05)
                measured, baseline = self.hand_feedback()
                peak = np.maximum(peak,abs(baseline))
                command = controller.update(measured,baseline,command)
                self.hand.send_joint_position_rad(tuple(command))
                self.trace("hand_waypoint", target_rad=command.tolist(),
                           measured_rad=measured.tolist(), torque=baseline.tolist())
                if controller.confirmed:
                    break
            self.expected_hand = command.copy()
            self.trace("contact_verdict", confirmed=controller.confirmed, groups=controller.records)
            if not controller.confirmed:
                self.hand_path(self.initial_hand)
                self.move(self.paths[1][::-1])
                self.move(self.paths[0][::-1])
                raise NoPlan("Closure completed without confirmed opposing contacts")
            self.contact["confirmed"] = True
            self.contact["groups"] = controller.records
        after = self.observe("closed")
        self.contact.update({
            "peak_abs_joint_torque": peak.tolist(),
            "closure_lag_rad": (self.hand_target - measured).tolist(),
            f"{self.station.target}_shift_after_close_m": None if after is None else (after - self.before).tolist(),
        })

    def lift(self):
        self.move(self.paths[2])
        measured = []
        for index in range(2):
            time.sleep(0.3)
            self.feedback()
            self.hand_feedback()
            point = self.observe(f"hold_{index}")
            measured.append(None if point is None else point.tolist())
        return measured

    def reset(self):
        self.move(self.paths[2][::-1])
        self.hand_path(self.initial_hand)
        hand, _ = self.hand_feedback()
        if np.max(np.abs(hand - self.initial_hand)) > 0.1:
            raise RuntimeError("Sharpa release could not be verified")
        self.move(self.paths[1][::-1])
        self.move(self.paths[0][::-1])
