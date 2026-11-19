"""Measured cube-station settings, perception and bounded IK planning."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np
from scipy import ndimage

from hardware import DEPLOYMENT, SIDES, arm_settings, hand_settings, load_extrinsics
from rgbd import HEIGHT, WIDTH
from .grasp_program import vector


MAX_CART_STEP_M = 0.0025
MAX_JOINT_STEP_DEG = 1.5
START_TOLERANCE_DEG = 1.0
CUBE_RESET_TOLERANCE_M = 0.025


class NoPlan(ValueError):
    """A proposed path failed before it could issue a motion command."""


@dataclass(frozen=True)
class Station:
    serial: str
    mount_from_color: np.ndarray
    arm: str
    start_deg: np.ndarray | None
    parked_deg: np.ndarray | None
    joints_deg: np.ndarray
    workspace_m: np.ndarray
    up: np.ndarray
    cube_m: np.ndarray | None
    tcp_offset_m: np.ndarray
    roi: tuple[int, int, int, int]
    hand_open: np.ndarray | None
    hand_targets: dict
    hand_joints_rad: np.ndarray
    torque_limits: np.ndarray
    wrench_limits: np.ndarray
    target: str = "cube"
    clearance: dict | None = None
    contact_policy: dict | None = None

    @classmethod
    def read(cls, path: Path):
        d = json.loads(path.read_text())
        target = d.get("target", "cube")
        if target not in ("cube", "shuttlecock"):
            raise ValueError("Unknown grasp station target")
        if target == "shuttlecock":
            d["cube_m"] = d.get("target_m")
            d["red_cube_roi_xyxy"] = d.get("target_roi_xyxy")
        if "rig" in d:
            validate_rig(d["camera_serial"], d["rig"])
        required = {
            "camera_serial", "calibration", "arm", "joints_deg",
            "workspace_m", "up_axis_mount", "cube_m", "tcp_offset_m", "red_cube_roi_xyxy",
            "hand_grasps_rad", "hand_max_abs_torque", "arm_max_abs_est_wrench",
        }
        if target == "shuttlecock":
            required.remove("cube_m")
        missing = sorted(key for key in required if d.get(key) is None)
        if missing:
            if target == "shuttlecock":
                missing = [{"cube_m":"target_m","red_cube_roi_xyxy":"target_roi_xyxy"}.get(k,k) for k in missing]
            raise ValueError(f"Fill the measured station fields: {', '.join(missing)}")
        if d["arm"] not in SIDES:
            raise ValueError("station arm must be left or right")
        serial = d["camera_serial"]
        extrinsics = load_extrinsics(path.parent / d["calibration"], serial)
        joints = np.array([vector(pair, 2, "joints_deg") for pair in d["joints_deg"]])
        limits = arm_limits_deg()
        if (joints.shape != (7, 2) or np.any(joints[:, 0] >= joints[:, 1])
                or np.any(joints[:, 0] < limits[:, 0]) or np.any(joints[:, 1] > limits[:, 1])):
            raise ValueError("joints_deg must define seven boxes within the Tianji limits")
        start = None if d.get("start_deg") is None else np.array(vector(d["start_deg"], 7, "start_deg"))
        parked = None if d.get("parked_deg") is None else np.array(vector(d["parked_deg"], 7, "parked_deg"))
        if (start is not None and not in_bounds(start, joints)) or (parked is not None and not in_bounds(parked, limits)):
            raise ValueError("Start or parked pose is outside its joint bounds")
        workspace = np.array([vector(pair, 2, "workspace_m") for pair in d["workspace_m"]])
        if workspace.shape != (3, 2) or np.any(workspace[:, 0] >= workspace[:, 1]):
            raise ValueError("workspace_m must contain one [min, max] pair per mount axis")
        up = np.array(vector(d["up_axis_mount"], 3, "up_axis_mount"))
        if not np.isclose(np.linalg.norm(up), 1, atol=1e-4):
            raise ValueError("up_axis_mount must be a measured unit vector")
        cube = None if d.get("cube_m") is None else np.array(vector(d["cube_m"], 3, "target_m"))
        offset = np.array(vector(d["tcp_offset_m"], 3, "tcp_offset_m"))
        roi = tuple(d["red_cube_roi_xyxy"])
        if (len(roi) != 4 or any(type(value) is not int for value in roi)
                or not (0 <= roi[0] < roi[2] <= WIDTH and 0 <= roi[1] < roi[3] <= HEIGHT)):
            raise ValueError("red_cube_roi_xyxy must fit the 640x480 RGB frame")
        opened = None if d.get("hand_open_rad") is None else np.array(vector(d["hand_open_rad"], 22, "hand_open_rad"))
        targets = {name: np.array(vector(values, 22, f"hand_grasps_rad.{name}"))
                   for name, values in d["hand_grasps_rad"].items()}
        if not targets or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", name) for name in targets):
            raise ValueError("Set at least one named hand grasp pose")
        hand_limits = np.array([[j.min_rad, j.max_rad] for j in hand_settings().joints])
        if any(not in_bounds(q, hand_limits) for q in [opened, *targets.values()] if q is not None):
            raise ValueError("A Sharpa hand pose exceeds the configured joint bounds")
        torques = np.array(vector(d["hand_max_abs_torque"], 22, "hand_max_abs_torque"))
        wrench = np.array(vector(d["arm_max_abs_est_wrench"], 6, "arm_max_abs_est_wrench"))
        if np.any(torques <= 0) or np.any(wrench <= 0):
            raise ValueError("Set positive limits in the native torque and wrench feedback units")
        return cls(serial, extrinsics, d["arm"], start, parked, joints, workspace, up, cube,
                   offset, roi, opened, targets, hand_limits, torques, wrench, target, d.get("clearance"), d.get("contact_policy"))


def validate_rig(head_serial, rig):
    """Verify the recorded identity against the configured SDKs and URDF, offline."""
    cameras = rig["camera_serials"]
    if (set(cameras) != {"head", "left_wrist", "right_wrist"}
            or cameras["head"] != head_serial or len(set(cameras.values())) != 3):
        raise ValueError("Camera roles do not match the bound teleop rig")
    hands = hand_settings().hands
    serials = {side: str(hands[side].expected_sn) for side in SIDES}
    if rig["sharpa_serials"] != serials or rig["tianji_ip"] != str(arm_settings().robot_ip):
        raise ValueError("Station identity differs from the deployed Tianji / Sharpa configuration")
    urdf = DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf"
    if rig["urdf_sha256"] != sha256(urdf.read_bytes()).hexdigest():
        raise ValueError("Station URDF differs from the teleop recording model")


def arm_limits_deg():
    return np.array([[joint.min_deg, joint.max_deg] for joint in arm_settings().joints])


def in_bounds(values, limits):
    return bool(np.all(np.isfinite(values)) and np.all(values >= limits[:, 0])
                and np.all(values <= limits[:, 1]))


def find_red_cube(snapshot, station):
    x1, y1, x2, y2 = station.roi
    r, g, b = snapshot.rgb[y1:y2, x1:x2].astype(np.int16).transpose(2, 0, 1)
    labels, _ = ndimage.label((r > 100) & (r - g > 50) & (r - b > 50))
    sizes = np.bincount(labels.ravel())
    good = np.flatnonzero(sizes[1:] >= 48) + 1
    if len(good) != 1:
        return None
    mask = ndimage.binary_erosion(labels == good[0], iterations=1)
    return mask_surface_point(snapshot, station, mask)


def mask_samples(snapshot, station, mask):
    x1, y1, x2, y2 = station.roi
    depth_options = {key: getattr(station, key) for key in ("min_depth_m", "max_depth_m")
                     if hasattr(station, key)}
    pixels, points = snapshot.geometry.register(snapshot.raw_depth, **depth_options)
    pixels = np.rint(pixels).astype(int)
    u, v = pixels.T
    inside = (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
    keep = mask[v[inside] - y1, u[inside] - x1]
    return pixels[inside][keep], points[inside][keep]


def mask_surface_point(snapshot, station, mask):
    _, matched = mask_samples(snapshot, station, mask)
    if len(matched) < 20:
        return None
    median = np.median(matched, axis=0)
    if np.median(np.abs(matched[:, 2] - median[2])) > 0.02:
        return None
    return (station.mount_from_color @ np.r_[median, 1])[:3]


class Planner:
    def __init__(self, station):
        from async_policy import FastLeftIK
        from kinematics import URDFKinematics

        self.station = station
        robot = URDFKinematics(DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf")
        self.ik = FastLeftIK(robot, station.joints_deg, station.arm)

    def plan(self, seed_deg, cube_m, offset_m, lift_m):
        station = self.station
        if not in_bounds(seed_deg, station.joints_deg):
            raise NoPlan("Measured start outside station joint bounds")
        q = np.deg2rad(seed_deg)
        tcp, _ = self.ik.forward_jacobian(q)
        start = tcp[:3, 3].copy()
        goal = cube_m + station.tcp_offset_m + np.asarray(offset_m)
        clearance = float((start - goal) @ station.up)
        if not lift_m + 0.01 <= clearance <= 0.22:
            raise NoPlan("Measured start height does not provide the requested vertical clearance")
        targets = [start, goal + clearance * station.up, goal, goal + lift_m * station.up]
        if any(not in_bounds(point, station.workspace_m) for point in targets):
            raise NoPlan("TCP waypoint outside the measured station workspace")
        paths = []
        for origin, target in zip(targets, targets[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(target - origin) / MAX_CART_STEP_M)))
            if count > 160:
                raise NoPlan("Path segment exceeds 0.4 m")
            positions = [np.rad2deg(q)]
            for point in np.linspace(origin, target, count + 1)[1:]:
                tcp[:3, 3] = point
                try:
                    solved = self.ik.solve(tcp, q)
                except ValueError as exc:
                    raise NoPlan("IK failed before motion") from exc
                if np.max(np.abs(np.rad2deg(solved - q))) > MAX_JOINT_STEP_DEG:
                    raise NoPlan("IK discontinuity before motion")
                measured, _ = self.ik.forward_jacobian(solved)
                if (not in_bounds(np.rad2deg(solved), station.joints_deg)
                        or not in_bounds(measured[:3, 3], station.workspace_m)
                        or np.linalg.norm(measured[:3, 3] - point) > 0.0002):
                    raise NoPlan("IK solution outside the station bounds")
                q = solved
                positions.append(np.rad2deg(q))
            paths.append(positions)
        return paths
