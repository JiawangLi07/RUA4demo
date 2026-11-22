"""Run one continuous RGB-D-only tabletop trial in a native Isaac process."""

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np


def grasp_source(args):
    if args.library is None:
        raise ValueError("--library is required for a grasp trial")
    if args.mesh_grasps or args.template_comparison:
        if args.lightning_candidates:
            raise ValueError(
                "a Lightning bank conflicts with an explicit mesh or template comparison"
            )
        if args.template_comparison:
            if args.adaptive_grasps:
                raise ValueError(
                    "the fixed-template comparison uses the historical executor"
                )
            return "template_comparison", None
        return ("adaptive_mesh" if args.adaptive_grasps else "mesh_fit"), None
    if args.hybrid_grasps and not args.adaptive_grasps:
        raise ValueError("--hybrid-grasps requires --adaptive-grasps")
    bank = args.lightning_candidates
    if bank is None:
        folders = [
            args.library.parent / "native",
            args.library.parent / "assets" / "native",
        ]
        if base := os.environ.get("RUA_SIM_ASSETS"):
            folders.append(Path(base) / "native")
        bank = next((directory for directory in folders if directory.is_dir()), None)
    if bank is None or not bank.is_dir():
        raise ValueError(
            "the Lightning Grasp bank directory was not found; pass --lightning-candidates"
        )
    return (
        "adaptive_mesh_and_lightning" if args.hybrid_grasps else "lightning_grasp"
    ), bank.resolve()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--adaptive-grasps",
        action="store_true",
        help="enable tabletop registration, path checks, local replanning and hand reassignment",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--hybrid-grasps",
        action="store_true",
        help="reproduce the mixed Lightning/mesh solver used in the historical comparison",
    )
    source.add_argument(
        "--mesh-grasps", action="store_true", help="replay the geometric mesh solver"
    )
    parser.add_argument(
        "--placement-chart",
        type=Path,
        help="reuse a saved arm chart when exporting the checked native placement domain",
    )
    parser.add_argument(
        "--export-lightning-placements",
        type=Path,
        help="use head depth and arm IK to prepare fixed placements for the native contact solver",
    )
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument(
        "--allow-hand-table-contact",
        action="store_true",
        help="allow Sharpa links to touch the physical table during head-only grasp planning",
    )
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
        "--scene-candidates",
        type=int,
        choices=range(1, 5),
        help="request at most this many fresh native contact solves per target and hand in this episode",
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
    source.add_argument(
        "--template-comparison",
        action="store_true",
        help="replay the fixed-posture comparison",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.allow_hand_table_contact and args.camera_mode != "head":
        parser.error("--allow-hand-table-contact requires head-only planning")
    if args.scene_candidates and (
        not args.adaptive_grasps
        or args.observe_only
        or args.export_lightning_placements
        or args.mesh_grasps
        or args.template_comparison
        or args.hybrid_grasps
        or args.camera_mode != "head"
    ):
        parser.error(
            "--scene-candidates requires a head-only adaptive Lightning grasp trial"
        )
    if args.placement_chart and not args.export_lightning_placements:
        parser.error("--placement-chart requires --export-lightning-placements")
    if args.export_lightning_placements and (
        not args.adaptive_grasps
        or args.hands == "both"
        or args.camera_mode != "head"
        or not args.target
        or args.observe_only
        or args.hybrid_grasps
        or args.mesh_grasps
        or args.template_comparison
    ):
        parser.error(
            "placement export requires one hand, head mode, --adaptive-grasps and an explicit target"
        )
    if not args.observe_only:
        try:
            if args.export_lightning_placements and args.library:
                solver, bank_path = "lightning_grasp", None
            else:
                solver, bank_path = grasp_source(args)
        except ValueError as error:
            parser.error(str(error))
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
        config["record_hand_table_contact"] = args.allow_hand_table_contact
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

                spec, shared = load_registry(
                    args.library,
                    args.template_comparison,
                    load_postures=solver != "lightning_grasp"
                    or bool(args.scene_candidates),
                )
                if (
                    solver in ("adaptive_mesh", "adaptive_mesh_and_lightning")
                    and shared is None
                ):
                    parser.error(
                        "adaptive mesh synthesis requires a metric registry and its shared posture catalog"
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
                            "solver": solver,
                            "scene_candidate_limit": args.scene_candidates,
                            "allow_hand_table_contact": args.allow_hand_table_contact,
                            "adaptive_grasps": args.adaptive_grasps,
                            "mount_installed": scene.fixture_path is not None,
                        }
                    )
                )
                controllers = []
                for i, side in enumerate(sides):
                    device = io if i == 0 else connect(scene, out, side, audit)[0]
                    bank = LightningBank(bank_path, side) if bank_path else None
                    if args.scene_candidates:
                        from .tabletop_scene_refit import LiveSceneBank

                        bank = LiveSceneBank(
                            bank_path,
                            side,
                            out / "scene_generation",
                            args.scene_candidates,
                        )
                    if args.export_lightning_placements:
                        from .tabletop_lightning_reach import PlacementExport

                        bank = PlacementExport(
                            args.export_lightning_placements,
                            args.placement_chart,
                            args.lightning_candidates,
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
                            hybrid_grasps=args.hybrid_grasps,
                            allow_hand_table_contact=args.allow_hand_table_contact,
                        )
                    )
                targets = list(args.target or spec)
                if args.export_lightning_placements:
                    ctrl = controllers[0]
                    (out / "observations").mkdir(exist_ok=True)
                    observed = ctrl.survey()
                    for name in targets:
                        if name not in observed:
                            raise ValueError("requested object not found in head RGB-D")
                        ctrl.name, ctrl.attempt = name, 1
                        ctrl.geometry(name, observed[name], {"prior_postures": []})
                elif len(controllers) == 1:
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
