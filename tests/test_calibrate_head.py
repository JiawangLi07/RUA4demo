import json
import sys

import h5py
import imageio.v2 as imageio
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrate_head import cv2, main, recorded_samples, solve_mount_from_color
from hardware import DEPLOYMENT, MOUNT_FRAME, ROOT, load_extrinsics
from kinematics import URDFKinematics
from rgbd import CameraGeometry, HEAD_SERIAL, REPLAY_PROFILE, rs


def synthetic_recording(episode):
    cv2.setNumThreads(1)
    intrinsics = CameraGeometry.from_file(REPLAY_PROFILE, HEAD_SERIAL).color
    profile = json.loads((ROOT / "config/cube_station.json").read_text())
    q0 = np.asarray(profile["start_deg"])
    offsets = np.array([
        [0, 0, 0, 0, 0, 0, 0], [5, 1, 0, 0, 4, -19, 4],
        [-5, 0, 2, -1, 0, 18, 1], [0, 5, -5, 0, 4, 8, 18],
        [0, -5, 5, 0, -4, -7, -18], [2, 0, 0, -3, 14, 2, 10],
        [-3, 2, 4, 0, -9, 16, -6], [-5, 4, 1, 4, 0, -12, 10],
    ])
    q = q0 + offsets
    model = URDFKinematics(DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf")
    mount_from_tcp = model.forward(np.deg2rad(q), [f"Joint{i}_R" for i in range(1, 8)],
                                   MOUNT_FRAME, "TCP_Link_R")
    tcp_from_board = np.eye(4)
    tcp_from_board[:3, 3] = [0.02, -0.015, 0.04]
    nominal_color_from_board = np.eye(4)
    nominal_color_from_board[:3, 3] = [-0.09, -0.045, 0.72]
    expected = mount_from_tcp[0] @ tcp_from_board @ np.linalg.inv(nominal_color_from_board)
    color_from_mount = np.linalg.inv(expected)

    square_m = 0.025
    supersample = 4

    def polygon(color_from_board, x, y, width, height):
        board = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])
        corners = np.column_stack((board * square_m, np.zeros(4), np.ones(4)))
        xyz = (color_from_board @ corners.T).T[:, :3]
        pixels = [rs.rs2_project_point_to_pixel(intrinsics, point.tolist()) for point in xyz]
        return np.rint(np.asarray(pixels) * supersample).astype(np.int32)

    (episode / "main").mkdir(parents=True)
    video_path = episode / "main" / "color_take001.mp4"
    with imageio.get_writer(video_path, fps=4, macro_block_size=1, ffmpeg_log_level="error") as out:
        for tcp in mount_from_tcp:
            color_from_board = color_from_mount @ tcp @ tcp_from_board
            shape = (intrinsics.height * supersample, intrinsics.width * supersample, 3)
            image = np.full(shape, 165, dtype=np.uint8)
            cv2.fillConvexPoly(image, polygon(color_from_board, -2, -2, 11, 8), (255,) * 3)
            for y in range(-1, 5):
                for x in range(-1, 8):
                    if (x + y) % 2 == 0:
                        cv2.fillConvexPoly(image, polygon(color_from_board, x, y, 1, 1),
                                           (0,) * 3, lineType=cv2.LINE_AA)
            image = cv2.resize(image, (intrinsics.width, intrinsics.height),
                               interpolation=cv2.INTER_AREA)
            out.append_data(image)
    times_s = np.arange(len(q)) * 0.5 + 10
    log = episode / "main" / "timestamps_take001.jsonl"
    log.write_text("".join(json.dumps({"i": i + 1, "mono": t}) + "\n"
                           for i, t in enumerate(times_s)))
    with h5py.File(episode / "robot_states.h5", "w") as f:
        arm = f.create_group("robot_state/pico_teleop")
        samples_ns = np.rint((times_s[:, None] + np.array([-0.03, 0, 0.03])) * 1e9)
        arm["sys_time_ns"] = samples_ns.astype(np.int64).reshape(-1)
        arm["real_right_deg"] = np.repeat(q, 3, axis=0)
    return expected


def test_exports_mount_direction_from_recorded_images_and_measured_joints(tmp_path, monkeypatch):
    episode = tmp_path / "synthetic"
    expected = synthetic_recording(episode)
    output = tmp_path / "extrinsics.json"
    monkeypatch.setattr(sys, "argv", [
        "calibrate-head", "--episode", str(episode), "--fit-frames", "1", "2", "3", "4", "5", "6",
        "--check-frames", "7", "8", "--output", str(output),
    ])
    main()
    fitted = load_extrinsics(output, HEAD_SERIAL)
    report = json.loads(output.read_text())
    angle = Rotation.from_matrix(expected[:3, :3].T @ fitted[:3, :3]).magnitude()
    assert np.linalg.norm(fitted[:3, 3] - expected[:3, 3]) < 0.02
    assert angle < np.deg2rad(2)
    assert max(report["reprojection_rms_px"].values()) < 1.0
    checks = report["held_out_tcp_from_board_errors"]
    assert set(checks) == {"7", "8"}
    assert max(error["translation_mm"] for error in checks.values()) < 5


def test_refuses_repeated_or_moving_calibration_poses(tmp_path):
    episode = tmp_path / "synthetic"
    synthetic_recording(episode)
    samples, _ = recorded_samples(episode, [1], "right", 0.025)
    with pytest.raises(ValueError, match="distinct axes"):
        solve_mount_from_color([samples[1]] * 3, [samples[1]] * 2)
    with h5py.File(episode / "robot_states.h5", "r+") as f:
        f["robot_state/pico_teleop/real_right_deg"][1, 0] += 2
    with pytest.raises(ValueError, match="missing or moving"):
        recorded_samples(episode, [1], "right", 0.025)
