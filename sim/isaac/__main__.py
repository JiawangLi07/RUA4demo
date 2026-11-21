"""Run a head-D455 / wrist-D405 grasp episode in native Isaac Sim."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .paths import asset_path

CASES = {
    "bottle": "bottle_balance",
    "tube": "tube_pivot",
    "shuttle": "shuttle_side",
    "box": "box_adaptive",
    "screwdriver": "driver_retry",
}


def profile_config(case, assets):
    """Compose versioned simulation parameters with the deployment's hand postures."""
    directory = Path(__file__).with_name("config")
    station = json.loads((directory / "station.json").read_text())
    obj = json.loads((directory / "objects.json").read_text())[case]
    settings = station["head_wrist_validation"]
    settings.update(obj.pop("head_wrist_validation"))
    hand_file = assets / "hand_postures.json"
    if hand_file.exists():
        library = json.loads(hand_file.read_text())
    else:
        # Older replay bundles store the same pose library inside each case.
        exported = assets / "cases" / (CASES[case] + ".json")
        if not exported.is_file():
            raise FileNotFoundError(f"provide {hand_file} or the exported case {exported}")
        library = json.loads(exported.read_text())["head_wrist_validation"]
    for key in ("posture", "pinch_postures"):
        settings[key] = library[key]
    return {**station, **obj}


def load_case(path, case=None):
    from .scene_meshes import load_mesh_bundle

    config = profile_config(case, Path(os.environ["RUA_SIM_ASSETS"])) if case else json.loads(path.read_text())
    settings = config["head_wrist_validation"]
    if not settings.get("use_wrist") or not settings.get("direct_pregrasp"):
        raise ValueError("expected a two-camera direct pregrasp case")
    if case == "screwdriver" or settings.get("object_label") in ("driver", "screwdriver"):
        settings["multi_finger_handle"] = True
    if settings.get("multi_finger_handle"):
        settings.setdefault("max_grasp_attempts", 3)
        if not 1 <= settings["max_grasp_attempts"] <= 8:
            raise ValueError("a handle grasp requires one to eight multi-finger attempts")
        settings["record_link_forces"] = True
    source = config["source_name"]
    asset_path(source, config["scene_file"])
    for name in (
        "scenario.json",
        "initial_pose.json",
        "hand_motor_config.json",
        "robot_cad.urdf",
        "tube_texture.png",
    ):
        asset_path(source, config["metadata_directory"], name)
    for obj in config.get("added_objects", []):
        if obj.get("kind") == "mesh":
            load_mesh_bundle(obj["mesh"])
    if settings.get("metric_shape_prior"):
        load_mesh_bundle(settings["metric_shape_prior"])
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Isaac Lab options such as --headless and --device cuda:0 are forwarded to the simulator.",
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--case", choices=CASES)
    choice.add_argument("--config", type=Path, help="full station case JSON")
    parser.add_argument("--assets", type=Path, default=os.environ.get("RUA_SIM_ASSETS"))
    parser.add_argument("--out", type=Path, help="new episode directory")
    parser.add_argument("--check", action="store_true", help="validate external inputs on CPU and exit")
    parser.add_argument("--video", action="store_true", help="encode native camera frames with ffmpeg")
    known, native_args = parser.parse_known_args(argv)
    if not known.assets:
        parser.error("--assets or RUA_SIM_ASSETS is required")
    os.environ["RUA_SIM_ASSETS"] = str(known.assets.expanduser().resolve())
    path = known.config
    config = load_case(path, known.case)
    if known.check:
        parser.parse_args(argv)
        print(f"Verified station assets and meshes for {known.case or path}")
        return 0
    if not known.out:
        parser.error("--out is required for a simulation run")
    out = known.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Kit can exit the interpreter during shutdown. The parent owns videos and status.
    child = subprocess.run([sys.executable, "-m", "sim.isaac.native_runner", str(out), *native_args])
    if child.returncode or (out / "failure.json").is_file() or not (out / "result.json").is_file():
        raise RuntimeError(f"Isaac process failed. See {out} (exit {child.returncode}).")
    if known.video:
        for folder in ("observer_frames", "frames"):
            if (out / folder).is_dir():
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-framerate",
                        str(config["images"]["fps"]),
                        "-i",
                        str(out / folder / "%06d.png"),
                        "-c:v",
                        "libx264",
                        "-crf",
                        "17",
                        "-threads",
                        "2",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(out / (folder + ".mp4")),
                    ],
                    check=True,
                )
    from .score import score

    return 0 if score(out)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
