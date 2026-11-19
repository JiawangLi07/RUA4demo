"""Station robot with motion primitives and its simulated motion backend.

``StationRobot`` extends :class:`robot.SharpaRobot` (observe / locate /
preview_reach) with the motion primitives a code-cell policy needs
(``move_joints`` / ``set_hand``). The same class runs against two backends:

- :class:`robot.SnapshotBackend` (read-only): every motion call is refused
  with "Unknown tool", so perception-only sessions cannot move the robot;
- :class:`SimMotionBackend`: joints accepted after the configured Tianji
  limits check update a virtual arm/hand state, for offline end-to-end runs
  of the cells loop without hardware.

A real-robot backend only has to implement the same two tools.
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from hardware import SIDES, arm_settings
from robot import SharpaRobot, SnapshotBackend


ROBOT_WORDS = {
    "robot_blurb": "the Tianji dual-arm robot with two Sharpa dexterous hands "
    "(one head D455 RGB-D camera)",
    "cls": "StationRobot",
    "svc": "",
    "svc_nl": "",
    "show_examples": "`robo.show('head')`",
    "scoring": "final",
}


class StationRobot(SharpaRobot):
    """SharpaRobot plus absolute joint-motion primitives.

    Motion is stateful: it counts against the episode budget, attaches the
    head camera image to the feedback (``on-motion`` image policy) and is
    guarded by :class:`~pyrualean._backend.EpisodeFinished` once the episode
    stops. On a read-only backend every motion call raises ``ToolError``.
    """

    NAME = "tianji_sharpa_station"
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = ("head.png",)
    SUMMARY_KEYS = SharpaRobot.SUMMARY_KEYS + ("moved", "arm", "side", "simulated")

    def move_joints(self, arm: str, joints_deg: list) -> dict:
        """Move one arm to absolute joint angles (degrees, 7 joints).

        Refused (ToolError) when the target is outside the configured Tianji
        joint limits; nothing is executed then. With the simulated backend
        the arm teleports and the returned payload carries the virtual state.
        """
        return self._stateful(
            "move_joints", {"arm": arm, "joints_deg": list(joints_deg)}, build=dict
        )

    def set_hand(self, side: str, joints_rad: list) -> dict:
        """Set one Sharpa hand to an absolute 22-joint pose (radians, SDK order)."""
        return self._stateful(
            "set_hand", {"side": side, "joints_rad": list(joints_rad)}, build=dict
        )


class SimMotionBackend(SnapshotBackend):
    """SnapshotBackend plus a virtual arm/hand state for offline motion.

    ``move_joints`` validates the target against the deployed Tianji joint
    configuration and, when accepted, replaces the virtual arm state.
    ``set_hand`` validates finiteness and shape, then replaces the virtual
    hand pose. ``view_env_state`` reports the virtual state, so a policy can
    verify a move by reading the joints back. ``env_steps`` counts simulated
    50 ms ticks from the largest commanded joint delta.
    """

    ARM_JOINTS = 7
    HAND_JOINTS = 22

    def __init__(self, snapshot, joints, extrinsics, out: Path):
        super().__init__(snapshot, joints, extrinsics, out)
        self._arm_deg = {side: list(vals) for side, vals in (joints.get("arm_deg") or {}).items()}
        self._hand_rad = {side: list(vals) for side, vals in (joints.get("hand_rad") or {}).items()}
        self._env_steps = 0

    # -- Backend protocol ----------------------------------------------------

    def env_steps(self):
        return self._env_steps

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        if name == "view_env_state" and input_dict.get("step") in (-1, 0):
            info = dict(self.info)
            joints = dict(info.get("joints") or {})
            joints["arm_deg"] = {side: list(vals) for side, vals in self._arm_deg.items()}
            joints["hand_rad"] = {side: list(vals) for side, vals in self._hand_rad.items()}
            info["joints"] = joints
            return info
        if name == "move_joints":
            return self._move_joints(input_dict["arm"], input_dict["joints_deg"])
        if name == "set_hand":
            return self._set_hand(input_dict["side"], input_dict["joints_rad"])
        return super().execute_tool(name, input_dict)

    # -- motion tools ----------------------------------------------------------

    def _move_joints(self, arm, joints_deg):
        if arm not in SIDES:
            return {"error": f"arm must be one of {SIDES}"}
        target = np.asarray(joints_deg, dtype=float)
        if target.shape != (self.ARM_JOINTS,) or not np.isfinite(target).all():
            return {"error": f"joints_deg must be {self.ARM_JOINTS} finite joint angles in degrees"}
        bounds = np.array([[joint.min_deg, joint.max_deg] for joint in arm_settings().joints])
        if np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]):
            return {
                "error": "target outside the configured Tianji joint limits; nothing was executed",
                "limits_deg": bounds.tolist(),
            }
        current = np.asarray(self._arm_deg.get(arm, np.zeros(self.ARM_JOINTS)), dtype=float)
        delta = float(np.max(np.abs(target - current)))
        self._arm_deg[arm] = target.tolist()
        steps = max(1, int(delta // 10) + 1)
        self._env_steps += steps
        return {
            "moved": True,
            "arm": arm,
            "joints_deg": [round(float(v), 3) for v in target],
            "max_joint_delta_deg": round(delta, 3),
            "simulated": True,
            "env_steps": steps,
        }

    def _set_hand(self, side, joints_rad):
        if side not in SIDES:
            return {"error": f"side must be one of {SIDES}"}
        target = np.asarray(joints_rad, dtype=float)
        if target.shape != (self.HAND_JOINTS,) or not np.isfinite(target).all():
            return {"error": f"joints_rad must be {self.HAND_JOINTS} finite joint angles in radians"}
        self._hand_rad[side] = target.tolist()
        self._env_steps += 1
        return {
            "moved": True,
            "side": side,
            "joints_rad": [round(float(v), 4) for v in target],
            "simulated": True,
            "env_steps": 1,
        }
