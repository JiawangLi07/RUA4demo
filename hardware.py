"""Reuse the existing Tianji and Sharpa deployment for observation and IK."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

import h5py
import numpy as np
from omegaconf import OmegaConf

from rgbd import camera_stream, video_users


ROOT = Path(__file__).resolve().parent
DEPLOYMENT = Path(os.environ.get(
    "DEXHUMANID_RUA_DEPLOYMENT", ROOT.parent / "real_experiment" / "deployment",
)).expanduser().resolve()
CONFIG = DEPLOYMENT / "vendor/universal_robot_infra/configs/robot_hardware"
MOUNT_FRAME = "dual_arm_mount"
SIDES = ("left", "right")
STATE_MAX_SKEW_NS = 100_000_000
TACTILE_BOARD_CONFIG = ROOT / "config" / "sharpa_tactile_board.json"


def arm_settings():
    return OmegaConf.load(CONFIG / "arm/tianji.yaml")


def hand_settings():
    return OmegaConf.load(CONFIG / "end_effector/sharpa.yaml")


def hardware_owners():
    hand_cfg = hand_settings()
    endpoints = [arm_settings().robot_ip, *(hand_cfg.hands[s].expected_ip for s in SIDES)]
    sockets = subprocess.run(
        ["ss", "-tnHp", "state", "established"], check=True, capture_output=True,
        text=True, timeout=5,
    ).stdout
    return {
        "camera_pids": video_users(),
        "robot_connections": [
            row.strip() for row in sockets.splitlines()
            if any(f" {endpoint}:" in row for endpoint in endpoints)
        ],
    }


@contextmanager
def observe_hardware(serial: str, *, camera_only: bool, tactile_sides=()):
    """Acquire one snapshot and then release the devices."""
    with (ROOT / ".capture.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = hardware_owners()
        if owners["camera_pids"] or (owners["robot_connections"] and not camera_only):
            raise RuntimeError(f"Devices are in use: {owners}")
        with ExitStack() as resources:
            arm, hands = (None, {}) if camera_only else open_state_sources(resources, tactile_sides=tactile_sides)
            read_camera = resources.enter_context(camera_stream(serial))
            for hand in hands.values():
                hand.read_state()
            snapshot = read_camera()
            state = {} if camera_only else read_state(arm, hands, snapshot.monotonic_ns, tactile_sides)
            yield snapshot, state


def open_state_sources(resources, arm=None, *, tactile_sides=()):
    from unirobot_infra.robots.arm.tianji import TianjiRobot

    if arm is None:
        arm = TianjiRobot(arm_settings())
    resources.callback(arm.close)
    arm.connect()
    return arm, open_hand_sources(resources, SIDES, tactile_sides=tactile_sides)


def open_hand_sources(resources, sides, *, tactile_sides=()):
    from unirobot_infra.robots.end_effector.sharpa import SharpaHand
    from unirobot_infra.utils.sdk_paths import configure_sharpa_runtime, sharpa_manus_sdk_paths

    cfg = hand_settings()
    cfg.initialize_control_on_connect = False
    cfg.enable_on_connect = False
    paths = sharpa_manus_sdk_paths(Path(cfg.sdk.vendor_root))
    configure_sharpa_runtime(paths)
    hands = {}
    for side in sides:
        hand_cfg = cfg.copy()
        tactile_config_file = None
        if side in tactile_sides:
            hand_cfg.skip_tactile = False
            tactile_config_file = str(TACTILE_BOARD_CONFIG)
        hand = SharpaHand(hand_cfg, side=side, sdk_paths=paths,
                          tactile_config_file=tactile_config_file)
        resources.callback(hand.close)
        hand.connect()
        hands[side] = hand
    return hands


def read_state(arm, hands, camera_ns, tactile_sides=()):
    arm_deg, hand_rad, stamps, arm_status, hand_sequence = {}, {}, {}, {}, {}
    for side, arm_id in zip(SIDES, ("A", "B")):
        stamps[f"arm_{side}"] = time.monotonic_ns()
        arm_deg[side] = list(arm.read_joint_position_deg(arm_id))
        status = arm.read_arm_status(arm_id)
        arm_status[side] = vars(status)
        stamps[f"hand_{side}"] = time.monotonic_ns()
        hand = hands[side].read_state()
        hand_rad[side] = list(hand.joint_pos)
        hand_sequence[side] = hand.sequence
    state = aligned_state(arm_deg, hand_rad, stamps, camera_ns)
    if tactile_sides:
        state["tactile"] = {}
        for side in tactile_sides:
            rows = []
            for channel in (range(5) if side == "right" else range(5, 10)):
                sample = hands[side].read_tactile_frame(channel, .02)
                received = time.monotonic_ns()
                if sample is None:
                    continue
                force = np.asarray(sample.f6)
                if force.shape == (6,) and np.isfinite(force).all():
                    rows.append({"finger":sample.finger, "f6":force.tolist(),
                                 "frame_id":sample.frame_id, "sdk_timestamp_s":sample.sdk_timestamp_s,
                                 "receipt_offset_ms":(received-camera_ns)/1e6})
            state["tactile"][side] = {"source":"live_sdk_f6", "samples":rows}
    return {**state, "arm_status": arm_status, "hand_sequence": hand_sequence}


def read_recorded_state(episode: Path, camera_ns: int):
    with h5py.File(episode / "robot_states.h5") as file:
        root = file["robot_state"]
        arms = root["pico_teleop"]
        hands = root["sharpa_joints"]
        arm_stamps = arms["sys_time_ns"][:]
        hand_stamps = hands["sys_time_ns"][:]
        arm_index = int(np.argmin(np.abs(arm_stamps - camera_ns)))
        hand_index = int(np.argmin(np.abs(hand_stamps - camera_ns)))
        arm_deg = {side: arms[f"real_{side}_deg"][arm_index].tolist() for side in SIDES}
        hand_rad = {side: hands[f"{side}_rad"][hand_index].tolist() for side in SIDES}
        stamps = {"arms": int(arm_stamps[arm_index]), "hands": int(hand_stamps[hand_index])}
        state = aligned_state(arm_deg, hand_rad, stamps, camera_ns)
        tactile = {}
        for group in root.get("sharpa_tactile", {}).values():
            times = group["sys_time_ns"][:]
            if not len(times):
                continue
            index = int(np.argmin(np.abs(times.astype(np.int64) - camera_ns)))
            skew = int(times[index]) - camera_ns
            side = group["side"].asstr()[index]
            force = group["force"][index]
            if side not in SIDES or force.shape != (3,) or not np.isfinite(force).all():
                continue
            samples = tactile.setdefault(side, {"source":"recorded_force_xyz", "samples":[]})["samples"]
            if abs(skew) <= STATE_MAX_SKEW_NS:
                samples.append({"finger":group["finger"].asstr()[index], "force_xyz":force.tolist(),
                                "sample_monotonic_ns":int(times[index]), "camera_offset_ms":skew/1e6})
        state["tactile"] = tactile
        return state


def aligned_state(arm_deg, hand_rad, stamps, camera_ns):
    skew_ns = max(abs(stamp - camera_ns) for stamp in stamps.values())
    return {
        "joints_aligned": skew_ns <= STATE_MAX_SKEW_NS,
        "max_camera_joint_skew_ms": skew_ns / 1e6,
        "joint_monotonic_ns": stamps,
        "arm_deg": arm_deg,
        "hand_rad": hand_rad,
    }


def load_extrinsics(path: Path | None, serial: str):
    if path is None:
        return None
    data = json.loads(path.read_text())
    if (data["camera_serial"], data["target_frame"]) != (serial, MOUNT_FRAME):
        raise ValueError("Calibration must map this color camera to dual_arm_mount")
    matrix = np.asarray(data["T_mount_from_color"], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError("Invalid homogeneous camera transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("Camera transform must contain a proper 3D rotation")
    return matrix


def preview_reach(side, point_mount_m, measured_deg):
    from async_policy import FastLeftIK
    from kinematics import URDFKinematics

    reference = URDFKinematics(DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf")
    limits = np.array([[joint.min_deg, joint.max_deg] for joint in arm_settings().joints])
    solver = FastLeftIK(reference, limits, side)
    seed = np.deg2rad(np.asarray(measured_deg, dtype=float))
    if (seed.shape != (7,) or not np.isfinite(seed).all()
            or np.any(seed < solver.bounds[:, 0]) or np.any(seed > solver.bounds[:, 1])):
        raise ValueError("Measured arm state is outside the configured IK limits")
    current, _ = solver.forward_jacobian(seed)
    target = current.copy()
    target[:3, 3] = point_mount_m
    q = solver.solve(target, seed)
    attained, _ = solver.forward_jacobian(q)
    return {
        "arm": side,
        "frame": MOUNT_FRAME,
        "target_tcp_m": target[:3, 3].tolist(),
        "current_tcp_m": current[:3, 3].tolist(),
        "target_joint_deg": np.rad2deg(q).tolist(),
        "position_residual_m": float(np.linalg.norm(attained[:3, 3] - target[:3, 3])),
        "status": "kinematic_preview",
    }
