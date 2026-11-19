"""Capture measured station fields from one RGB-D frame and a recorded TCP.

On-site, with the red cube placed at its initial pose, one live frame plus a
rough pixel box around the cube produce the ``cube_m`` / ``red_cube_roi_xyxy``
values for ``config/cube_station.json``. With ``--calibration`` (the output of
``calibrate-head``) the surface point is reported in ``dual_arm_mount``;
otherwise in the color optical frame.

``--tcp-episode`` / ``--tcp-mono`` compute the arm TCP from a recorded robot
state at the moment the operator aligned the hand with the cube, so
``tcp_offset_m = aligned_tcp_m - cube_m`` comes from measurement instead of
estimation.

Everything is read-only: the frame comes from one snapshot and the TCP from
files; no device is held or moved.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import h5py
import numpy as np

from cli import json_print, new_run, source
from hardware import DEPLOYMENT, MOUNT_FRAME, load_extrinsics
from kinematics import URDFKinematics
from rgbd import HEAD_SERIAL


def surface_point(snapshot, bbox, min_samples=20):
    """Median registered depth point inside the box's central region."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    pixels, points = snapshot.geometry.register(snapshot.raw_depth)
    dx, dy = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    u, v = pixels.T
    kept = points[(u >= x1 + dx) & (u < x2 - dx) & (v >= y1 + dy) & (v < y2 - dy)]
    if len(kept) < min_samples:
        raise ValueError(f"Only {len(kept)} depth samples inside the box; widen it")
    median = np.median(kept, axis=0)
    mad = float(np.median(np.abs(kept[:, 2] - median[2])))
    return median, mad, len(kept)


def suggest_roi(bbox, width, height, margin=0.6):
    """Pixel box dilated so it still covers the cube after a vertical lift."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    roi = (
        max(0, int(round(x1 - margin * w))),
        max(0, int(round(y1 - margin * h))),
        min(width, int(round(x2 + margin * w))),
        min(height, int(round(y2 + margin * h))),
    )
    if not (roi[0] < roi[2] and roi[1] < roi[3]):
        raise ValueError("Suggested ROI is empty; use a wider initial box")
    return list(map(int, roi))


def aligned_tcp_m(episode: Path, side: str, mono_ns: int):
    """TCP pose in dual_arm_mount from the recorded joints nearest to a time."""
    with h5py.File(episode / "robot_states.h5") as file:
        arms = file["robot_state/pico_teleop"]
        stamps = arms["sys_time_ns"][:]
        index = int(np.argmin(np.abs(stamps - mono_ns)))
        measured_deg = arms[f"real_{side}_deg"][index].tolist()
    model = URDFKinematics(DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf")
    suffix = side[0].upper()
    joint_names = [f"Joint{i}_{suffix}" for i in range(1, 8)]
    mount_from_tcp = model.forward(np.deg2rad(np.asarray(measured_deg)[None, :]), joint_names,
                                   MOUNT_FRAME, f"TCP_Link_{suffix}")[0]
    return mount_from_tcp[:3, 3].tolist(), measured_deg


def main():
    parser = ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--live", action="store_true")
    source_group.add_argument("--episode", type=Path, default=None)
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--serial", default=HEAD_SERIAL)
    parser.add_argument("--bbox", type=float, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"), help="Rough box around the cube")
    parser.add_argument("--calibration", type=Path,
                        help="calibrate-head JSON; without it the point stays in the camera frame")
    parser.add_argument("--tcp-episode", type=Path,
                        help="Recording that holds the manual-alignment robot state")
    parser.add_argument("--tcp-side", choices=("left", "right"), default="right")
    parser.add_argument("--tcp-mono", type=int,
                        help="Monotonic timestamp (ns) of the aligned state in --tcp-episode")
    args = parser.parse_args()
    if args.episode is None and not args.live:
        from cli import REFERENCE_EPISODE
        args.episode = REFERENCE_EPISODE
    if (args.tcp_mono is None) != (args.tcp_episode is None):
        parser.error("--tcp-episode and --tcp-mono must be given together")
    if args.camera_only and not args.live:
        parser.error("--camera-only is for live acquisition")
    if args.frame < 1:
        parser.error("--frame starts at 1")
    transform = load_extrinsics(args.calibration, args.serial) \
        if args.calibration else None

    with source(args) as (snapshot, _state):
        out = new_run()
        point, mad, samples = surface_point(snapshot, args.bbox)
        roi = suggest_roi([int(v) for v in args.bbox], snapshot.rgb.shape[1], snapshot.rgb.shape[0])
    patch = {
        "calibration": str(args.calibration) if args.calibration else None,
        "red_cube_roi_xyxy": roi,
        "cube_m": None,
        "tcp_offset_m": None,
    }
    frame = "dual_arm_mount" if transform is not None else "head_color_optical"
    if transform is not None:
        patch["cube_m"] = (transform @ np.r_[point, 1])[:3].round(4).tolist()
    report = {
        "run": str(out),
        "source": snapshot.source,
        "frame": frame,
        "surface_point_m": point.round(4).tolist(),
        "depth_mad_m": round(mad, 4),
        "samples": samples,
        "config_patch": patch,
    }
    if args.tcp_episode is not None:
        tcp, measured_deg = aligned_tcp_m(args.tcp_episode, args.tcp_side, args.tcp_mono)
        report["aligned_tcp_m"] = [round(v, 4) for v in tcp]
        if patch["cube_m"] is not None:
            patch["tcp_offset_m"] = (np.asarray(tcp) - np.asarray(patch["cube_m"])).round(4).tolist()
        report["aligned_tcp_side"] = args.tcp_side
        report["aligned_tcp_joints_deg"] = [round(v, 2) for v in measured_deg]
    (out / "station_capture.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    json_print(report)


if __name__ == "__main__":
    main()
