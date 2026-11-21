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
    parser.add_argument(
        "--adaptive-grasps",
        action="store_true",
        help="use mesh-conditioned wall envelopes, oblique opposition and hand reassignment",
    )
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--camera-mode", choices=("head", "head-wrist"), default="head")
    parser.add_argument(
        "--wrist-hardware",
        choices=("removed", "mounted"),
        help="defaults to removed for head-only operation, mounted for head-wrist",
    )
    parser.add_argument("--hands", choices=("right", "left", "both"), default="right")
    parser.add_argument(
        "--lightning-candidates",
        type=Path,
        help="directory of native Sharpa Lightning Grasp banks",
    )
    parser.add_argument(
        "--wrist-mount", type=Path, help="registered 002.3dm fixture bundle"
    )
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
    hardware = args.wrist_hardware or (
        "removed" if args.camera_mode == "head" else "mounted"
    )
    if hardware == "removed" and (
        args.camera_mode != "head" or args.wrist_mount is not None
    ):
        parser.error(
            "removed wrist hardware requires head camera mode and no wrist fixture path"
        )
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

        config = json.loads(args.scene.read_bytes())
        config["active_hands"] = args.hands
        config["check_forearm_table"] = args.adaptive_grasps
        config["camera_mode"] = args.camera_mode
        config["wrist_hardware"] = hardware
        if args.wrist_mount:
            config["wrist_mount"] = str(args.wrist_mount.resolve())
        scene = NativeBimanualScene(config, out, args.device)
        sides = ["right", "left"] if args.hands == "both" else [args.hands]
        io, audit = connect(scene, out, sides[0])
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
                from .tabletop_control import Controller
                from .tabletop_lightning import LightningBank

                if args.library is None:
                    parser.error("--library is required for a grasp trial")
                spec, shared = load_registry(
                    args.library,
                    args.template_comparison,
                    load_postures=args.adaptive_grasps
                    or args.lightning_candidates is None,
                )
                if args.adaptive_grasps and shared is None:
                    parser.error(
                        "--adaptive-grasps requires a metric registry and its shared posture catalog"
                    )
                (out / "trial_settings.json").write_text(
                    json.dumps(
                        {
                            "lift_height_m": args.lift_height_m,
                            "grasp_inputs": "rgbd_mesh_commanded_kinematics",
                            "joint_observation": "issued_targets",
                            "closure_telemetry": "completed_command",
                            "hands": sides,
                            "camera_mode": args.camera_mode,
                            "wrist_hardware": hardware,
                            "solver": "adaptive_mesh_and_lightning"
                            if args.adaptive_grasps
                            else "lightning_grasp"
                            if args.lightning_candidates
                            else "mesh_fit",
                            "mount_installed": scene.fixture_path is not None,
                        }
                    )
                )
                controllers = []
                for i, side in enumerate(sides):
                    device = io if i == 0 else connect(scene, out, side, audit)[0]
                    bank = (
                        LightningBank(args.lightning_candidates, side)
                        if args.lightning_candidates
                        else None
                    )
                    controllers.append(
                        Controller(
                            device,
                            spec,
                            out,
                            args.template_comparison,
                            shared,
                            args.lift_height_m,
                            bank,
                            args.adaptive_grasps,
                        )
                    )
                targets = list(args.target or spec)
                if len(controllers) == 1:
                    controllers[0].execute(targets)
                else:
                    (out / "observations").mkdir(exist_ok=True)
                    seen = controllers[0].survey()
                    # Assign from the two initial palm poses and segmented head depth.
                    # No simulator identities or world object transforms enter this choice.
                    palms = [io.fk(side + "_hand_C_MC")[:2, 3] for side in sides]
                    groups = [[], []]
                    for name in targets:
                        group = (
                            0
                            if name not in seen
                            else int(
                                np.argmin(
                                    [
                                        np.linalg.norm(seen[name]["center"][:2] - palm)
                                        for palm in palms
                                    ]
                                )
                            )
                        )
                        groups[group].append(name)
                    for i, ctrl in enumerate(controllers):
                        ctrl.event("hand_assignment", targets=groups[i])
                        ctrl.execute(
                            groups[i], finish=i == 1 and not args.adaptive_grasps
                        )
                        if i == 0 and groups[i]:
                            ctrl.leave_low_pose()
                    if args.adaptive_grasps:
                        # Give unexecuted targets to the opposite arm after its
                        # partner has cleared the view. Obtain fresh head depth.
                        for i in (1, 0):
                            partner = controllers[1 - i]
                            closed = {
                                e["name"]
                                for e in partner.events
                                if e["phase"] == "closure_completed"
                            }
                            waiting = [n for n in groups[1 - i] if n not in closed]
                            if waiting:
                                if i == 0:
                                    controllers[1].leave_low_pose()
                                controllers[i].event(
                                    "hand_reassignment", targets=waiting
                                )
                                controllers[i].execute(waiting, finish=False)
                        controllers[0].event("episode_end")
        finally:
            audit.save()
            (out / "sensor_usage.json").write_text(
                json.dumps(
                    {
                        "controller_reads": scene.sensor_calls,
                        "active_wrist_devices": list(scene.wrist_devices),
                    }
                )
            )
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
