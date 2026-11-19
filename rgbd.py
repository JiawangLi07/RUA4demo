"""RGB and native depth from the D455, with depth-to-color registration."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import pyrealsense2 as rs


HEAD_SERIAL = "241122305532"
WIDTH, HEIGHT, FPS = 640, 480, 30
REPLAY_PROFILE = Path(__file__).resolve().parent / "config" / "d455_replay_profile.json"


def camera_devices():
    return {
        device.get_info(rs.camera_info.serial_number): device
        for device in rs.context().query_devices()
    }


def video_users():
    nodes = sorted(Path("/dev").glob("video[0-9]*"))
    if not nodes:
        return []
    result = subprocess.run(
        ["fuser", *map(str, nodes)], capture_output=True, text=True, timeout=5
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip())
    return sorted(set(int(pid) for pid in result.stdout.split()))


def inventory():
    return {
        "cameras": {
            serial: device.get_info(rs.camera_info.name)
            for serial, device in camera_devices().items()
        },
        "head_serial": HEAD_SERIAL,
        "video_users": video_users(),
    }


@dataclass(frozen=True)
class CameraGeometry:
    color: object
    depth: object
    depth_to_color: object
    depth_scale_m: float

    @classmethod
    def from_profiles(cls, color, depth, depth_scale_m):
        return cls(
            color.get_intrinsics(),
            depth.get_intrinsics(),
            depth.get_extrinsics_to(color),
            float(depth_scale_m),
        )

    @classmethod
    def from_file(cls, path, serial):
        data = json.loads(path.read_text())
        if data["camera_serial"] != serial:
            raise ValueError("Replay camera serial does not match the recorded depth profile")

        return cls.from_metadata(data)

    @classmethod
    def from_metadata(cls, data):
        def intrinsics(source):
            profile = rs.intrinsics()
            for key in ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs"):
                setattr(profile, key, source[key])
            profile.model = getattr(rs.distortion, source["model"].split(".")[-1])
            return profile

        transform = rs.extrinsics()
        transform.rotation = data["depth_to_color_rotation"]
        transform.translation = data["depth_to_color_translation_m"]
        return cls(intrinsics(data["color"]), intrinsics(data["depth"]),
                   transform, data["depth_scale_m"])

    def metadata(self):
        def intrinsics(profile):
            return {
                "fx": profile.fx, "fy": profile.fy, "ppx": profile.ppx, "ppy": profile.ppy,
                "width": profile.width, "height": profile.height,
                "model": str(profile.model), "coeffs": list(profile.coeffs),
            }

        return {
            "color": intrinsics(self.color),
            "depth": intrinsics(self.depth),
            "depth_scale_m": self.depth_scale_m,
            "depth_to_color_rotation": list(self.depth_to_color.rotation),
            "depth_to_color_translation_m": list(self.depth_to_color.translation),
        }

    def register(self, raw_depth, *, min_depth_m=.2, max_depth_m=2.5):
        """Return native depth points in the color optical frame and their RGB pixels."""
        meters = np.asarray(raw_depth, dtype=np.float64) * self.depth_scale_m
        rows, cols = np.nonzero((meters >= min_depth_m) & (meters <= max_depth_m))
        z = meters[rows, cols]
        depth = self.depth
        if not any(depth.coeffs):
            points = np.column_stack(((cols - depth.ppx) * z / depth.fx,
                                      (rows - depth.ppy) * z / depth.fy, z))
        else:
            points = np.array([
                rs.rs2_deproject_pixel_to_point(depth, [float(u), float(v)], float(d))
                for u, v, d in zip(cols, rows, z)
            ], dtype=float).reshape(-1, 3)
        transform = self.depth_to_color
        rotation = np.asarray(transform.rotation, dtype=float).reshape(3, 3, order="F")
        color_points = points @ rotation.T + transform.translation
        color_points = color_points[color_points[:, 2] > 0]
        pixels = np.array([
            rs.rs2_project_point_to_pixel(self.color, xyz) for xyz in color_points.tolist()
        ], dtype=float).reshape(-1, 2)
        return pixels, color_points


@dataclass
class Snapshot:
    rgb: np.ndarray
    raw_depth: np.ndarray
    geometry: CameraGeometry
    serial: str
    stamp_ns: int
    monotonic_ns: int
    source: str
    color_sequence: int | None = None
    depth_sequence: int | None = None
    archived_frame_id: str | None = None

    def __post_init__(self):
        if self.rgb.shape != (self.geometry.color.height, self.geometry.color.width, 3):
            raise ValueError("RGB dimensions do not match the camera profile")
        if self.raw_depth.shape != (self.geometry.depth.height, self.geometry.depth.width):
            raise ValueError("Depth dimensions do not match the camera profile")


def archived_observation(directory, frame_id):
    """Restore one completed hardware capture without opening a device."""
    summary = json.loads((directory / "result.json").read_text())
    if summary.get("input_policy") != "own_trials_v1" or summary.get("controller") == "SimMotionBackend":
        raise ValueError("Archive must come from an autonomous hardware session")
    info = json.loads((directory / "observation.json").read_text())
    if info["source"] != "live" or not frame_id or info["frame_id"] != frame_id:
        raise ValueError("Archived frame identity differs from the physical trial")
    with Image.open(directory / "head.png") as color, Image.open(directory / "head_depth_native.png") as depth:
        snap = Snapshot(np.asarray(color.convert("RGB")), np.array(depth, dtype=np.uint16),
                        CameraGeometry.from_metadata(info["depth_geometry"]), info["camera_serial"],
                        info["captured_at_ns"], info["captured_monotonic_ns"], "archived_live_replay",
                        archived_frame_id=frame_id)
    transform = info.get("T_mount_from_color")
    return snap, {**info["joints"], "tactile":info.get("tactile", {})}, (
        np.asarray(transform) if transform is not None else None)


def replay(episode: Path, frame: int, serial: str):
    """Pair a 1-based RGB frame with the native-depth PNG from the same log row."""
    folder = episode / "main"
    rows = [json.loads(line) for line in (folder / "timestamps_take001.jsonl").read_text().splitlines()]
    row = next((row for row in rows if row.get("i") == frame), None)
    if not row or not row["depth"]:
        raise ValueError(f"Recorded RGB frame {frame} has no paired depth")
    geometry = CameraGeometry.from_file(REPLAY_PROFILE, serial)
    with imageio.get_reader(folder / "color_take001.mp4", format="FFMPEG") as video:
        rgb = video.get_data(frame - 1)
    with Image.open(folder / row["depth"]) as depth_image:
        raw_depth = np.array(depth_image, dtype=np.uint16)
    return Snapshot(rgb, raw_depth, geometry, serial, round(row["wall"] * 1e9),
                    round(row["mono"] * 1e9), "replay")


@contextmanager
def camera_stream(serial: str):
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    camera = rs.pipeline()
    profile = camera.start(config)
    try:
        for _ in range(10):
            camera.wait_for_frames(2000)
        geometry = CameraGeometry.from_profiles(
            profile.get_stream(rs.stream.color).as_video_stream_profile(),
            profile.get_stream(rs.stream.depth).as_video_stream_profile(),
            profile.get_device().first_depth_sensor().get_depth_scale(),
        )

        def read():
            frames = camera.wait_for_frames(2000)
            rgb, depth = frames.get_color_frame(), frames.get_depth_frame()
            stamp_ns = time.time_ns()
            monotonic_ns = time.monotonic_ns()
            if not rgb or not depth:
                raise RuntimeError(f"RealSense {serial} did not return a paired RGB-D frame")
            return Snapshot(np.asanyarray(rgb.get_data()).copy(),
                            np.asanyarray(depth.get_data()).copy(), geometry, serial,
                            stamp_ns, monotonic_ns, "live",
                            int(rgb.get_frame_number()), int(depth.get_frame_number()))

        yield read
    finally:
        camera.stop()


def capture(serial: str):
    with camera_stream(serial) as read:
        return read()
