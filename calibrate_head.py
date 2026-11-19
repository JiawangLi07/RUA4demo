"""Solve the fixed D455 camera transform using a chessboard teleop recording."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

import h5py
import imageio.v2 as imageio
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

from hardware import DEPLOYMENT, MOUNT_FRAME
from kinematics import URDFKinematics
from rgbd import CameraGeometry, HEAD_SERIAL, REPLAY_PROFILE

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
try:
    import cv2
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    # Reuse the OpenCV wheel already deployed for OpenWAM. Keep the robot
    # interpreter and its dependencies first on the import path.
    sys.path.append(str(DEPLOYMENT / "envs/openwam/lib/python3.10/site-packages"))
    import cv2

cv2.ocl.setUseOpenCL(False)

from unirobot_infra.calibration.hand_eye import CalibrationSample, solve_hand_eye


def board_pose(rgb, intrinsics, square_m):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (8, 5))
    if not found:
        raise ValueError("8 x 5 checkerboard corners not found")

    # The D455 records an inverse-Brown model. Use its native undistortion
    # on color rays before passing normalized points to PnP.
    normalized = np.array([
        rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], 1.0)[:2]
        for x, y in corners.reshape(-1, 2)
    ], dtype=float)
    points = np.zeros((40, 3))
    points[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * square_m
    found, rvec, tvec = cv2.solvePnP(points, normalized, np.eye(3), None)
    if not found or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        raise ValueError("Checkerboard pose solve failed")
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    projected = cv2.projectPoints(points, rvec, tvec, np.eye(3), None)[0].reshape(-1, 2)
    errors = (projected - normalized) * [intrinsics.fx, intrinsics.fy]
    return transform, float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))


def recorded_samples(episode, frames, side, square_m):
    folder = episode / "main"
    rows = (json.loads(line) for line in (folder / "timestamps_take001.jsonl").read_text().splitlines())
    frame_times_ns = {row["i"]: round(row["mono"] * 1e9) for row in rows if "i" in row}
    with h5py.File(episode / "robot_states.h5") as recording:
        arms = recording["robot_state/pico_teleop"]
        stamps = arms["sys_time_ns"][:]
        measured_deg = arms[f"real_{side}_deg"][:]
    intrinsics = CameraGeometry.from_file(REPLAY_PROFILE, HEAD_SERIAL).color
    model = URDFKinematics(DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf")
    suffix = side[0].upper()
    joint_names = [f"Joint{i}_{suffix}" for i in range(1, 8)]
    video_path = folder / "color_take001.mp4"

    samples, residuals = {}, {}
    with imageio.get_reader(video_path, format="FFMPEG") as video:
        for frame in sorted(frames):
            try:
                dt = np.abs(stamps - frame_times_ns[frame])
                nearby = measured_deg[dt <= 100_000_000]
                if (len(nearby) < 2 or not np.isfinite(nearby).all()
                        or np.max(np.ptp(nearby, axis=0)) > 0.25):
                    raise ValueError("Arm pose is missing or moving at image time")
                # Use native joint feedback and forward kinematics directly in
                # dual_arm_mount. No SDK-base-to-URDF transform is assumed.
                mount_from_tcp = model.forward(np.deg2rad(measured_deg[[int(np.argmin(dt))]]),
                                               joint_names, MOUNT_FRAME, f"TCP_Link_{suffix}")[0]
                rgb = video.get_data(frame - 1)
                if rgb.shape != (intrinsics.height, intrinsics.width, 3):
                    raise ValueError("RGB frame does not match the saved D455 profile")
                color_from_board, residuals[frame] = board_pose(rgb, intrinsics, square_m)
                samples[frame] = CalibrationSample(frame, video_path, mount_from_tcp,
                                                   color_from_board)
            except (KeyError, IndexError, ValueError, cv2.error) as exc:
                raise ValueError(f"Frame {frame}: {exc}") from exc
    return samples, residuals


def solve_mount_from_color(fit, check):
    rotations = np.array([sample.base_from_ee[:3, :3] for sample in fit])
    relative = rotations[0].T @ rotations
    axes = Rotation.from_matrix(relative).as_rotvec()
    if np.linalg.matrix_rank(axes, tol=np.deg2rad(3)) < 2:
        raise ValueError("Fit poses need rotations about at least two distinct axes")

    result = solve_hand_eye(tuple(fit), mode="eye-to-hand", method="park")
    # The legacy solver returns color_from_mount for the eye-to-hand case.
    mount_from_color = np.linalg.inv(result.transform)
    rotation = mount_from_color[:3, :3]
    if (not np.isfinite(mount_from_color).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("Hand-eye solver did not produce a rigid transform")

    def tcp_from_board(sample):
        return np.linalg.inv(sample.base_from_ee) @ mount_from_color @ sample.camera_from_board

    mount_fit = np.array([tcp_from_board(sample) for sample in fit])
    expected_position = mount_fit[:, :3, 3].mean(axis=0)
    expected_rotation = Rotation.from_matrix(mount_fit[:, :3, :3]).mean()
    errors = {}
    for sample in check:
        pose = tcp_from_board(sample)
        errors[sample.record_index] = {
            "translation_mm": float(1000 * np.linalg.norm(pose[:3, 3] - expected_position)),
            "rotation_deg": float((expected_rotation.inv() *
                                   Rotation.from_matrix(pose[:3, :3])).magnitude() * 180 / np.pi),
        }
    return mount_from_color, errors


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--side", choices=("left", "right"), default="right",
                        help="Arm the board was rigidly mounted on")
    parser.add_argument("--square-mm", default=25.0, type=float)
    parser.add_argument("--fit-frames", required=True, type=int, nargs="+")
    parser.add_argument("--check-frames", required=True, type=int, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frames = args.fit_frames + args.check_frames
    if (len(args.fit_frames) < 3 or len(args.check_frames) < 2 or min(frames) < 1
            or len(set(frames)) != len(frames)):
        parser.error("Use 3+ fit and 2+ independent check frames, with distinct positive indices")
    if not np.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error("--square-mm must be the positive measured square width")
    cv2.setNumThreads(1)
    try:
        samples, reprojection = recorded_samples(args.episode, frames, args.side,
                                                 args.square_mm / 1000)
        transform, errors = solve_mount_from_color([samples[i] for i in args.fit_frames],
                                                   [samples[i] for i in args.check_frames])
        payload = {
            "camera_serial": HEAD_SERIAL,
            "target_frame": MOUNT_FRAME,
            "T_mount_from_color": transform.tolist(),
            "source_episode": str(args.episode.resolve()),
            "arm": args.side,
            "square_mm": args.square_mm,
            "fit_frames": args.fit_frames,
            "check_frames": args.check_frames,
            "reprojection_rms_px": reprojection,
            "held_out_tcp_from_board_errors": errors,
        }
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        with args.output.open("x") as file:
            file.write(text)
        print(text, end="")
    except (ValueError, OSError, np.linalg.LinAlgError, cv2.error) as exc:
        parser.exit(1, f"Calibration failed: {exc}\n")


if __name__ == "__main__":
    main()
