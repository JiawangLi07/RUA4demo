"""Bind the right-wrist RGB-D frame to the robot pose at capture time."""

import json

import numpy as np

from hardware import DEPLOYMENT, MOUNT_FRAME, read_state
from robot import SnapshotBackend


class WristView:
    def __init__(self, camera, calibration, serial):
        from kinematics import URDFKinematics

        self.camera, self.serial = camera, serial
        if (calibration["camera_serial"] != serial or calibration["target_frame"] != "TCP_Link_R"
                or calibration["source_frame"] != "right_wrist_color_optical_native"
                or calibration["translation_unit"] != "m"):
            raise ValueError("Right wrist calibration does not match the camera or TCP frame")
        self.tcp_from_color = np.asarray(calibration["T_tcp_from_color"], dtype=float)
        matrix = self.tcp_from_color
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
            raise ValueError("Invalid right wrist rigid transform")
        self.intrinsics = calibration["native_intrinsics"]
        self.rotation_quarters = calibration.get("recorded_image_rotation_ccw_deg", 0) // 90
        self.model = URDFKinematics(DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf")
        self.joint_names = [f"Joint{i}_R" for i in range(1, 8)]

    def capture(self, arm, hands, out):
        before = np.asarray(arm.read_joint_position_deg("B"))
        snapshot = self.camera()
        if snapshot.serial != self.serial:
            raise RuntimeError("Unexpected right-wrist camera serial")
        joints = read_state(arm, hands, snapshot.monotonic_ns)
        current = np.asarray(joints["arm_deg"]["right"])
        status = joints["arm_status"]["right"]
        if (not joints["joints_aligned"] or status["cur_state"] != 0 or status["err_code"]
                or not np.isfinite(current).all() or not np.isfinite(before).all()
                or np.max(np.abs(before - current)) > .15):
            raise RuntimeError("Right-wrist image lacks a stationary, time-aligned arm pose")
        intrinsics = snapshot.geometry.metadata()["color"]
        for key in ("width", "height", "model"):
            if intrinsics[key] != self.intrinsics[key]:
                raise RuntimeError("Right-wrist camera profile differs from calibration")
        if not np.allclose([intrinsics[k] for k in ("fx", "fy", "ppx", "ppy")],
                           [self.intrinsics[k] for k in ("fx", "fy", "ppx", "ppy")], atol=.1, rtol=1e-3):
            raise RuntimeError("Right-wrist camera intrinsics differ from calibration")
        mount_from_tcp = self.model.forward(np.deg2rad(current[None, :]), self.joint_names,
                                            MOUNT_FRAME, "TCP_Link_R")[0]
        out.mkdir()
        return SnapshotBackend(snapshot, joints, mount_from_tcp @ self.tcp_from_color, out,
                               camera_frame="right_wrist_color_optical_native")


def open_wrist_view(resources, config, profile, open_camera):
    calibration_path = config.get("right_wrist_calibration")
    if not calibration_path:
        return None, "right_wrist_calibration_not_configured"
    serial = config["rig"]["camera_serials"]["right_wrist"]
    calibration = json.loads((profile.parent / calibration_path).read_text())
    try:
        camera = resources.enter_context(open_camera(serial))
    except (OSError, RuntimeError) as exc:
        return None, str(exc)
    return WristView(camera, calibration, serial), None
