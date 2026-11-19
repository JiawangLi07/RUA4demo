"""Deployment entry point on 4090-zhenji."""

from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import json
import os
import select
import sys
import tempfile

from pyrualean.mcp_server import ArmMcpServer
from pyrualean.runner import load_policy, policy_sha256, run_policy

from hardware import DEPLOYMENT, ROOT, hardware_owners, load_extrinsics, observe_hardware, read_recorded_state
from rgbd import HEAD_SERIAL, inventory, replay
from robot import RobotTools, SnapshotBackend


REFERENCE_EPISODE = DEPLOYMENT / "data/episodes_badminton/no/episode_0001"
DEPLOYED_TASKS = {"badminton_into_tube": "easywam_v1"}


def json_print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


@contextmanager
def source(args):
    if args.live:
        with observe_hardware(args.serial, camera_only=args.camera_only,
                              tactile_sides=getattr(args, "tactile_side", ())) as observation:
            yield observation
    elif args.episode is None:
        from autonomous_scene import observation
        yield observation(args.frame - 1)
    else:
        snapshot = replay(args.episode, args.frame, args.serial)
        yield snapshot, read_recorded_state(args.episode, snapshot.monotonic_ns)


def new_run():
    parent = ROOT / "runs"
    parent.mkdir(mode=0o700, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def serve(tools, args):
    server = ArmMcpServer(tools, port=args.port, annotations={
        "readOnlyHint": True, "destructiveHint": False, "openWorldHint": False,
    })
    try:
        url = server.start()
        print("PYRUA_SHARPA_READY " + json.dumps({"url": url, "run": str(tools.out)}), flush=True)
        select.select([sys.stdin], [], [], args.seconds)
    finally:
        server.stop()


def launch_deployed_task(task, execute):
    if execute:
        owners = hardware_owners()
        if any(owners.values()):
            raise RuntimeError(f"Devices are in use: {owners}")
    # The existing launcher owns motion, the GPU check and shutdown.
    python = str(DEPLOYMENT / "envs/robot/bin/python")
    os.execv(python, [
        python, str(DEPLOYMENT / "scripts/launch.py"), "--profile", DEPLOYED_TASKS[task],
        "--run" if execute else "--check",
    ])


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Inventory and device ownership")
    deployed = commands.add_parser("deployed-task", help="Use an existing commissioned controller")
    deployed.add_argument("task", choices=DEPLOYED_TASKS)
    deployed.add_argument("--execute", action="store_true", help="Start physical motion")
    for name in ("observe", "locate", "skill", "serve"):
        command = commands.add_parser(name)
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--live", action="store_true", help="Acquire from available hardware")
        mode.add_argument("--episode", type=Path, default=None)
        command.add_argument("--camera-only", action="store_true")
        command.add_argument("--frame", type=int, default=1)
        command.add_argument("--serial", default=HEAD_SERIAL)
        command.add_argument("--calibration", type=Path, help="Measured T_mount_from_color JSON")
        if name in ("locate", "skill"):
            command.add_argument("--label", required=True)
            command.add_argument("--bbox", type=float, nargs=4, required=True)
        if name == "skill":
            command.add_argument("program", type=Path, help="Operator-reviewed Python skill")
        if name == "serve":
            command.add_argument("--port", type=int, default=0)
            command.add_argument("--seconds", type=int, default=180, choices=range(10, 301),
                                 metavar="10..300")
    args = parser.parse_args()
    if args.command == "check":
        json_print({"devices": inventory(), "owners": hardware_owners()})
        return 0
    if args.command == "deployed-task":
        return launch_deployed_task(args.task, args.execute)
    if args.calibration and not args.live and args.episode is None:
        parser.error("Physical mount calibration cannot be applied to a procedural layout")
    if args.command == "serve" and args.episode:
        parser.error("Autonomous observation uses a procedural scene or the live workspace")
    if args.camera_only and not args.live:
        parser.error("--camera-only is for live acquisition")
    if args.frame < 1:
        parser.error("--frame starts at 1")
    transform = load_extrinsics(args.calibration, args.serial)
    with source(args) as (snapshot, state):
        out = new_run()
        backend = SnapshotBackend(snapshot, state, transform, out)
    tools = RobotTools(backend)
    if args.command == "serve":
        serve(tools, args)
        return 0
    if args.command == "skill":
        policy = partial(load_policy(args.program), label=args.label, bbox_xyxy=args.bbox)
        report = run_policy(policy, tools.robot, timeout_s=20)
        result = {**report.to_dict(), "sha256": policy_sha256(args.program)}
        (out / "policy_outcome.json").write_text(json.dumps(result, indent=2) + "\n")
        tools.ledger.write_jsonl(out / "calls.jsonl")
        json_print({"run": str(out), **result})
        return 0 if report.status == "completed" else 1
    inputs = ({"frame_id": backend.frame_id, "label": args.label, "bbox_xyxy": args.bbox}
              if args.command == "locate" else {})
    result = tools.execute_tool(args.command, inputs).result
    json_print({"run": str(out), **result})
    return int("error" in result or result.get("found") is False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
