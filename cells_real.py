"""One operator-selected arm step for commissioning the Cells motion path."""

from copy import deepcopy
from threading import Event, Lock
import json
import shutil
import time

import numpy as np

from exploration.real import idle_arm
from hardware import SIDES, read_state
from robot import SnapshotBackend


TRACKING_DEG = 0.15


class ArmStepBackend(SnapshotBackend):
    def __init__(self, snapshot, joints, extrinsics, out, arm, hands, camera, step):
        super().__init__(snapshot, joints, extrinsics, out)
        self.arm, self.hands, self.camera = arm, hands, camera
        self.side, joint, delta = step
        self.arm_id, self.parked_id = ("A", "B") if self.side == "left" else ("B", "A")
        self.start = np.array(joints["arm_deg"][self.side], dtype=float)
        self.target = self.start.copy()
        self.target[joint - 1] += delta
        bounds = np.array([(joint.min_deg, joint.max_deg) for joint in arm.cfg.joints])
        if not (np.isfinite(self.start).all() and
                np.all((bounds[:, 0] + TRACKING_DEG <= np.vstack((self.start, self.target))) &
                       (np.vstack((self.start, self.target)) <= bounds[:, 1] - TRACKING_DEG))):
            raise ValueError("Arm start and target need clearance from Tianji joint limits")
        self.used = False
        self.tactile_sides = ()
        self.steps = 0
        self.stop = Event()
        self.action = Lock()
        self.publish_step()

    def publish_step(self):
        self.info["commissioning_step"] = {
            "arm": self.side, "target_joint_deg": self.target.tolist(), "used": self.used,
        }

    def execute_tool(self, name, input_dict):
        if name == "view_env_state":
            return deepcopy(super().execute_tool(name, input_dict))
        if name == "move_joints":
            return self.move(**input_dict)
        if name == "set_hand":
            return {"error": "Commissioning step only drives the selected arm"}
        return super().execute_tool(name, input_dict)

    def env_steps(self):
        return self.steps

    def cancel(self):
        self.stop.set()
        with self.action:
            pass

    def trace(self, event, **values):
        with (self.out / "arm_step.jsonl").open("a") as log:
            log.write(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(),
                                  **values}, allow_nan=False) + "\n")

    def feedback(self, expected_mode):
        if self.stop.is_set():
            raise RuntimeError("Cells session stopped")
        active = self.arm.read_arm_state(self.arm_id, self.side)
        parked = self.arm.read_arm_status(self.parked_id)
        if active.cur_state != expected_mode or parked.cur_state != 0 or active.err_code or parked.err_code:
            raise RuntimeError("Unexpected Tianji mode or controller fault")
        q = np.rad2deg(active.joint_pos)
        self.trace("feedback", frame_serial=active.frame_serial, joint_deg=q.tolist(),
                   joint_torque=active.joint_torque, est_cart_force=active.est_cart_force)
        return q, active.frame_serial

    def move(self, arm, joints_deg):
        requested = np.asarray(joints_deg, dtype=float)
        if (arm != self.side or requested.shape != (7,) or
                not np.allclose(requested, self.target, rtol=0, atol=0.001)):
            return {"error": "Use exactly the commissioning target returned by observe"}
        with self.action:
            if self.used:
                return {"error": "The commissioning step has already been used"}
            self.used = True
            self.publish_step()
            current, serial = self.feedback(0)
            if np.max(np.abs(current - self.start)) > TRACKING_DEG:
                raise RuntimeError("Initial pose changed since the commissioning target was set")
            before = self.out / "before"
            before.mkdir()
            for filename in ("head.png", "head_depth_native.png", "observation.json"):
                shutil.copyfile(self.out / filename, before / filename)
            try:
                self.arm.enter_position_mode(self.arm_id)
                held, held_serial = self.feedback(1)
                if held_serial == serial or np.max(np.abs(held - self.start)) > TRACKING_DEG:
                    raise RuntimeError("Position hold failed its fresh joint tracking check")
                self.trace("target", joint_deg=self.target.tolist())
                self.arm.send_joint_position_deg(self.arm_id, tuple(self.target), blocking=False)
                deadline = time.monotonic() + 2
                while True:
                    measured, serial = self.feedback(1)
                    if (np.any(measured < np.minimum(self.start, self.target) - TRACKING_DEG) or
                            np.any(measured > np.maximum(self.start, self.target) + TRACKING_DEG)):
                        raise RuntimeError("Arm feedback left the commissioned joint interval")
                    if serial != held_serial and np.max(np.abs(measured - self.target)) <= TRACKING_DEG:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Tianji did not track the commissioning target")
                    time.sleep(0.02)
            finally:
                idle_arm(self.arm, self.arm_id, self.trace)

            snapshot = self.camera()
            state = read_state(self.arm, self.hands, snapshot.monotonic_ns, self.tactile_sides)
            final = np.array(state["arm_deg"][arm])
            if np.max(np.abs(final - self.target)) > TRACKING_DEG:
                raise RuntimeError("Post-idle position differs from the commissioning target")
            super().__init__(snapshot, state, self.extrinsics, self.out)
            self.publish_step()
            self.steps = 1
            self.trace("verified", joint_deg=final.tolist(), frame_id=self.frame_id)
            return {"moved": True, "arm": arm, "joints_deg": final.tolist(), "simulated": False,
                    "idle_confirmed": True, "env_steps": 1, "frame_id": self.frame_id}


class ArmTrajectoryBackend(SnapshotBackend):
    """Run repeated, interpolated arm motions inside a local empty-space envelope."""

    # This initial envelope covers nearby motions from the observed pose.
    EXCURSION_DEG = 2.0
    WAYPOINT_DEG = 0.4

    def __init__(self, snapshot, joints, extrinsics, out, arm, hands, camera, *, excursion_deg=2.0):
        super().__init__(snapshot, joints, extrinsics, out)
        self.arm, self.hands, self.camera = arm, hands, camera
        self.excursion_deg = excursion_deg
        self.origin = {side: np.array(joints["arm_deg"][side]) for side in SIDES}
        self.bounds = np.array([(joint.min_deg, joint.max_deg) for joint in arm.cfg.joints])
        self.stop, self.action = Event(), Lock()
        self.tactile_sides = ()
        self.steps = 0
        self.publish_limits()

    def publish_limits(self):
        self.info["motion_limits"] = {
            "origin_deg": {side: q.tolist() for side, q in self.origin.items()},
            "max_sum_absolute_joint_offset_deg": self.excursion_deg,
        }

    def execute_tool(self, name, input_dict):
        if name == "view_env_state":
            return deepcopy(super().execute_tool(name, input_dict))
        if name == "move_joints":
            return self.move(**input_dict)
        if name == "set_hand":
            return {"error": "Arm trajectory mode does not enable hand control"}
        return super().execute_tool(name, input_dict)

    env_steps = ArmStepBackend.env_steps
    cancel = ArmStepBackend.cancel

    def trace(self, event, **values):
        with (self.out / "arm_trajectory.jsonl").open("a") as log:
            log.write(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(),
                                  **values}, allow_nan=False) + "\n")

    def feedback(self, side, mode):
        if self.stop.is_set():
            raise RuntimeError("Cells session stopped")
        active_id, parked_id = ("A", "B") if side == "left" else ("B", "A")
        active = self.arm.read_arm_state(active_id, side)
        parked = self.arm.read_arm_state(parked_id, "parked")
        if active.cur_state != mode or parked.cur_state != 0 or active.err_code or parked.err_code:
            raise RuntimeError("Unexpected Tianji mode or controller fault")
        q = np.rad2deg(active.joint_pos)
        parked_q = np.rad2deg(parked.joint_pos)
        if not np.isfinite(q).all() or not np.isfinite(parked_q).all():
            raise RuntimeError("Nonfinite joint feedback")
        other = "right" if side == "left" else "left"
        if np.max(np.abs(parked_q - self.joints["arm_deg"][other])) > TRACKING_DEG:
            raise RuntimeError("Parked arm moved")
        return q, active.frame_serial

    def inside(self, side, q, margin=0.0):
        return (np.all(q >= self.bounds[:, 0] + TRACKING_DEG - margin)
                and np.all(q <= self.bounds[:, 1] - TRACKING_DEG + margin)
                and (self.excursion_deg is None or np.sum(np.abs(q - self.origin[side])) <= self.excursion_deg + margin))

    def move(self, arm, joints_deg):
        goal = np.asarray(joints_deg, dtype=float)
        if arm not in SIDES or goal.shape != (7,) or not np.isfinite(goal).all():
            return {"error": "Expected left/right and seven finite joint angles in degrees"}
        if not self.inside(arm, goal):
            return {"error": "Target outside this session's empty-space joint envelope"}
        arm_id = "A" if arm == "left" else "B"
        with self.action:
            try:
                return self.run_path(arm, arm_id, goal)
            except BaseException:
                self.stop.set()
                raise

    def run_path(self, side, arm_id, goal, planned_path=None, *, guard=None):
        if guard is not None:
            guard()
        start, serial = self.feedback(side, 0)
        if not self.inside(side, start, margin=TRACKING_DEG):
            raise RuntimeError("Measured pose outside this session's joint envelope")
        path_length = float(np.max(np.abs(start - goal)))
        if path_length <= TRACKING_DEG:
            return {"moved": False, "arm": side, "joints_deg": start.tolist(), "env_steps": 0}
        steps = int(np.ceil(path_length / self.WAYPOINT_DEG))
        path = np.linspace(start, goal, steps + 1) if planned_path is None else np.asarray(planned_path)
        if (path.ndim!=2 or path.shape[1]!=7 or len(path)<2 or not np.isfinite(path).all()
                or np.max(abs(path[0]-start))>TRACKING_DEG or np.max(abs(path[-1]-goal))>1e-6
                or np.max(abs(np.diff(path,axis=0)))>self.WAYPOINT_DEG+1e-8):
            raise RuntimeError("Invalid preplanned arm trajectory")
        steps=len(path)-1
        phase = self.out / f"path-{self.steps:04d}"
        phase.mkdir()
        for filename in ("head.png", "head_depth_native.png", "observation.json"):
            shutil.copyfile(self.out / filename, phase / filename)

        try:
            self.arm.enter_position_mode(arm_id)
            held, held_serial = self.feedback(side, 1)
            if held_serial == serial or np.max(np.abs(held - start)) > TRACKING_DEG:
                raise RuntimeError("Fresh position hold feedback did not match the path start")
            self.trace("path", arm=side, goal_deg=goal.tolist(), waypoints=steps)
            for previous, target in zip(path, path[1:]):
                if guard is not None:
                    guard()
                if self.stop.is_set():
                    raise RuntimeError("Cells session stopped")
                self.arm.send_joint_position_deg(arm_id, tuple(target), blocking=False)
                deadline = time.monotonic() + 0.8
                time.sleep(0.05)
                while True:
                    measured, serial = self.feedback(side, 1)
                    if (np.any(measured < np.minimum(previous, target) - TRACKING_DEG) or
                            np.any(measured > np.maximum(previous, target) + TRACKING_DEG)):
                        raise RuntimeError("Arm left the tracked waypoint interval")
                    if serial != held_serial and np.max(np.abs(measured - target)) <= TRACKING_DEG:
                        held_serial = serial
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Arm did not track the waypoint with fresh feedback")
                    time.sleep(0.02)
                self.trace("waypoint", arm=side, target_deg=target.tolist(),
                           measured_deg=measured.tolist(), frame_serial=serial)
        finally:
            idle_arm(self.arm, arm_id, self.trace)

        snapshot = self.camera()
        state = read_state(self.arm, self.hands, snapshot.monotonic_ns, self.tactile_sides)
        final = np.array(state["arm_deg"][side])
        if not state["joints_aligned"] or np.max(np.abs(final - goal)) > TRACKING_DEG:
            raise RuntimeError("Post-motion camera/joint readback did not verify the path")
        super().__init__(snapshot, state, self.extrinsics, self.out)
        self.publish_limits()
        self.steps += steps
        self.trace("verified", arm=side, joint_deg=final.tolist(), frame_id=self.frame_id)
        return {"moved": True, "arm": side, "joints_deg": final.tolist(), "env_steps": steps,
                "idle_confirmed": True, "simulated": False, "frame_id": self.frame_id}
