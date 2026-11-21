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
        home = io.q()
        for _ in range(round(1.5 / io.dt)):
            io.step(home[io.arm], home[io.hand])
        head = io.camera("head")
        np.savez_compressed(out / "initial_head.npz", **head)
        (out / "joint_names.json").write_text(json.dumps(io.names))
        # The auditor has no return channel into the controller.
        try:
            if not args.observe_only:
                from .tabletop_control import run

                run(io, json.loads(args.library.read_bytes()), out)
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
