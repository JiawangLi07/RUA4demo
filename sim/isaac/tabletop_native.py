"""Run one continuous RGB-D-only tabletop trial in a native Isaac process."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument(
        "--lift-height-m",
        type=float,
        default=0.05,
        help="required object rise and bottom clearance",
    )
    parser.add_argument(
        "--target",
        action="append",
        help="visit a named target while all scene objects remain active",
    )
    parser.add_argument(
        "--template-comparison",
        action="store_true",
        help="replay the fixed-posture comparison",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.enable_cameras = True
    args.kit_args += (
        f' --portable --portable-root="{out}/kit" --allow-root --/log/file="{out}/kit.log"'
        " --/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=4"
        " --/persistent/physics/numThreads=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"
    )
    app = AppLauncher(args).app
    try:
        from .scene_isaac import NativeBimanualScene
        from .tabletop_io import connect

        scene = NativeBimanualScene(
            json.loads(args.scene.read_bytes()), out, args.device
        )
        io, audit = connect(scene, out)
        home = io.commanded_joints()
        for _ in range(round(1.5 / io.dt)):
            io.step(home[io.arm], home[io.hand])
        head = io.camera("head")
        np.savez_compressed(out / "initial_head.npz", **head)
        (out / "joint_names.json").write_text(json.dumps(io.names))
        # The auditor has no return channel into the controller.
        try:
            if not args.observe_only:
                from .tabletop_catalog import load_registry
                from .tabletop_control import run

                if args.library is None:
                    parser.error("--library is required for a grasp trial")
                spec, shared = load_registry(args.library, args.template_comparison)
                (out / "trial_settings.json").write_text(
                    json.dumps(
                        {
                            "lift_height_m": args.lift_height_m,
                            "grasp_inputs": "rgbd_mesh_commanded_kinematics",
                            "joint_observation": "issued_targets",
                        }
                    )
                )
                run(
                    io,
                    spec,
                    out,
                    args.target,
                    args.template_comparison,
                    shared,
                    args.lift_height_m,
                )
        finally:
            audit.save()
            (out / "frames.json").write_text(json.dumps(scene.frame_times))
        (out / "complete.json").write_text(
            json.dumps({"frames": scene.frame_number, "sim_seconds": scene.time})
        )
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
