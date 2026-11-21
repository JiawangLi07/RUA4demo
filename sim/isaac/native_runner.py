"""Launch the native Kit process for one fully resolved grasp episode."""

import argparse
import json
import traceback
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    out = args.out
    config = json.loads((out / "config.json").read_text())
    args.kit_args += (
        f' --portable --portable-root="{out}/kit" --allow-root --/log/file="{out}/kit.log"'
        " --/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=4"
        " --/persistent/physics/numThreads=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"
    )
    app = AppLauncher(args).app
    try:
        from .scene_isaac import NativeBimanualScene
        from .head_wrist_trial import run_trial

        run_trial(NativeBimanualScene(config, out, args.device))
    except Exception:
        (out / "failure.json").write_text(json.dumps({"traceback": traceback.format_exc()}))
        raise
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
