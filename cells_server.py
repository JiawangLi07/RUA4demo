"""Cells-mode server: GPT-written Python cells against StationRobot.

Two subcommands:

``serve``  exposes the upstream code arm (``python`` cell tool + ``finish``)
through an in-process MCP server, like ``cli.py serve`` does for the
read-only tools. The local launcher (``cells_agent.py``) forwards a port and
attaches a Codex/GPT session to it.

``script`` runs the same toolkit in-process from a JSON file::

    {"task": "...", "cells": ["code", ...], "finish": {"status": "success", "summary": "..."}}

so the full cells loop (persistent session, sandbox, motion feedback,
budget) is verifiable end to end without a model or SSH.

Motion comes from ``--motion``: ``sim`` (virtual joints, default) or
``none`` (read-only backend; motion cells are refused).
"""

from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path
import fcntl
import json
import math
import select
import signal
import sys

from pyrualean._backend import Ledger
from pyrualean.arms import ArmToolkit
from pyrualean.mcp_server import ArmMcpServer

from cells_robot import SimMotionBackend, StationRobot
from cli import new_run, source
from hardware import DEPLOYMENT, ROOT, SIDES, arm_settings, hardware_owners, load_extrinsics, open_state_sources, read_state
from rgbd import HEAD_SERIAL, camera_stream
from robot import SnapshotBackend
from physical_evidence import cells_record


def json_print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def build_toolkit(args, resources):
    """One robot + cells toolkit for this session; returns (robot, toolkit, out)."""
    transform = load_extrinsics(args.calibration, args.serial)
    out = new_run()
    if args.motion in ("real", "trajectory"):
        from cells_real import ArmStepBackend, ArmTrajectoryBackend
        from tianji_command_adapter import ReliableTianjiRobot

        for path in (ROOT / ".capture.lock", DEPLOYMENT / "runs" / ".active.lock"):
            lock = resources.enter_context(path.open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = hardware_owners()
        if any(owners.values()):
            raise RuntimeError(f"Devices are in use: {owners}")
        cfg = arm_settings()
        cfg.vel_ratio = cfg.acc_ratio = 5
        arm, hands = open_state_sources(resources, ReliableTianjiRobot(cfg), tactile_sides=args.tactile_side)
        camera = resources.enter_context(camera_stream(args.serial))
        for hand in hands.values():
            hand.read_state()
        snapshot = camera()
        state = read_state(arm, hands, snapshot.monotonic_ns, args.tactile_side)
        if args.motion == "real":
            backend = ArmStepBackend(snapshot, state, transform, out, arm, hands, camera, args.arm_step)
        else:
            if not state["joints_aligned"]:
                raise RuntimeError("Trajectory session requires aligned live feedback")
            backend = ArmTrajectoryBackend(snapshot, state, transform, out, arm, hands, camera)
        backend.tactile_sides = args.tactile_side
        resources.callback(backend.cancel)
    else:
        factory = SimMotionBackend if args.motion == "sim" else SnapshotBackend
        with source(args) as (snapshot, state):
            backend = factory(snapshot, state, transform, out)
    robot = StationRobot(backend, ledger=Ledger())
    toolkit = ArmToolkit(
        robot,
        arm=args.execution,
        program_timeout_s=90 if args.execution == "program" else None,
        workspace=out / "workspace",
        images="on-demand" if args.execution == "program" else "on-motion",
        feedback="rich",
        max_turns=args.max_turns,
    )
    return robot, toolkit, out


def serve(args, robot, toolkit, out):
    server = ArmMcpServer(toolkit, port=args.port, annotations={
        "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False,
    })
    try:
        url = server.start()
        from cells_agent import build_prompt
        source = "live" if args.live else f"procedural tabletop layout {args.frame}"
        prompt = build_prompt(args.task, source, args.seconds, args.max_turns, args.motion, args.execution)
        print("PYRUA_CELLS_READY " + json.dumps({"url":url,"run":str(out),"prompt":prompt}),flush=True)
        select.select([sys.stdin], [], [], args.seconds)
    finally:
        robot._backend.cancel()
        server.stop()
        save_results(toolkit, robot, out, args.task, autonomous=True)


def save_results(toolkit, robot, out, task="", *, autonomous=False):
    robot._ledger.write_jsonl(out / "calls.jsonl")
    summary = {
        "run": str(out),
        "task": task,
        "turns": toolkit.turns,
        "controller": type(robot._backend).__name__,
        "input_policy": "own_trials_v1" if autonomous else "script_or_diagnostic",
        "execution": toolkit.arm,
        "finish": toolkit.finish,
        "sandbox_flags": toolkit.sandbox_flags,
    }
    (out / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if robot._backend.info.get("source") == "live":
        cells_record(out, ROOT / ".local/autonomous_library")
    return summary


def run_script(toolkit, robot, script: dict, out: Path) -> dict:
    """Execute a scripted cells episode; also used by tests."""
    (out / "script.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    key, tool = ("programs","run_program") if toolkit.arm == "program" else ("cells","python")
    if not script.get(key):
        raise ValueError(f"Script requires nonempty {key}")
    failed = None
    for index, code in enumerate(script[key], 1):
        result = toolkit.execute_tool(tool, {"code": code})
        json_print({"cell": index, "result": result.result})
        if result.result.get("exception") or result.result.get("error") or result.result.get("status") in ("error", "timeout"):
            failed = result.result.get("exception") or result.result.get("error") or result.result["status"]
            break
        if toolkit.finish is not None:
            break
    finish = {"status":"stuck","summary":str(failed)[:500]} if failed else script.get("finish")
    if finish and toolkit.finish is None:
        toolkit.execute_tool("finish", finish)
    return save_results(toolkit, robot, out, script.get("task", ""))


def script(args, robot, toolkit, out):
    summary = run_script(toolkit, robot, json.loads(Path(args.script).read_text()), out)
    json_print(summary)
    return 2 if (summary.get("finish") or {}).get("status") == "stuck" else 0


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "script"):
        command = commands.add_parser(name)
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--live", action="store_true", help="Acquire from available hardware")
        mode.add_argument("--episode", type=Path, help="Path to a RealSense recording on the server",
                          default=None)
        command.add_argument("--tactile-side", action="append", default=[], choices=SIDES,
                             help="Also sample live F6 when acquiring this hand snapshot")
        command.add_argument("--camera-only", action="store_true")
        command.add_argument("--frame", type=int, default=1)
        command.add_argument("--serial", default=HEAD_SERIAL)
        command.add_argument("--calibration", type=Path, help="Measured T_mount_from_color JSON")
        command.add_argument(
            "--motion", choices=("sim", "none", "real", "trajectory"), default="sim",
        )
        command.add_argument("--arm-step", nargs=3, metavar=("SIDE", "JOINT", "DELTA_DEG"),
                             help="Commissioning only: one joint step between 0.5 and 1 degree")
        command.add_argument("--max-turns", type=int, default=None)
        command.add_argument("--execution", choices=("cells","program"), default="cells")
        if name == "serve":
            command.add_argument("--task", default="")
            command.add_argument("--port", type=int, default=0)
            command.add_argument("--seconds", type=int, default=180, choices=range(10, 301),
                                 metavar="10..300")
        else:
            command.add_argument("script", help="JSON file: {task, cells, finish?}")
    args = parser.parse_args()
    if args.execution == "program" and args.motion in ("real","trajectory"):
        parser.error("Program mode has been validated on replay/sim only")
    if args.tactile_side and (not args.live or args.camera_only or args.motion == "sim"):
        parser.error("--tactile-side requires live hand state and a physical or read-only backend")
    if args.camera_only and not args.live:
        parser.error("--camera-only is for live acquisition")
    if args.frame < 1:
        parser.error("--frame starts at 1")
    if args.motion in ("real", "trajectory"):
        if not args.live or args.camera_only or not args.calibration:
            parser.error("hardware motion requires --live and --calibration with full state feedback")

        def interrupt(_signum, _frame):
            raise KeyboardInterrupt("Cells session interrupted")

        signal.signal(signal.SIGTERM, interrupt)
    if args.motion == "real":
        if not args.arm_step:
            parser.error("real requires --arm-step")
        side, joint, delta = args.arm_step
        try:
            joint, delta = int(joint), float(delta)
        except ValueError:
            parser.error("--arm-step expects SIDE, JOINT and DELTA_DEG")
        if side not in SIDES or joint not in range(1, 8) or not 0.5 <= abs(delta) <= 1 or not math.isfinite(delta):
            parser.error("--arm-step requires left/right, joint 1..7 and absolute delta 0.5..1 degree")
        args.arm_step = side, joint, delta

    elif args.arm_step:
        parser.error("--arm-step is for --motion real")
    if args.calibration and not args.live:
        parser.error("Physical mount calibration applies to live hardware observations")
    if args.episode:
        parser.error("Autonomous Cells accepts live observations or the procedural scene")
    with ExitStack() as resources:
        robot, toolkit, out = build_toolkit(args, resources)
        entry = serve if args.command == "serve" else script
        return entry(args, robot, toolkit, out)


if __name__ == "__main__":
    raise SystemExit(main())
