"""Run the cells arm locally against DexHumanid-RUA on 4090-zhenji.

Like ``agent.py`` (fixed-parameter observation tools) but the remote server
exposes the upstream code arm: a single ``python(code)`` MCP tool whose
cells run in a persistent sandboxed session with ``robo`` (a StationRobot)
injected, so GPT drives the robot by writing Python cells. With
``--motion sim`` (the server default) joint motion updates a virtual state
for offline runs; the read-only backend refuses motion.

Modes:

- default: SSH-forward ``run.sh cells-serve`` and attach a local Codex
  (GPT-6.2 Astra) session with the cells contract and the live API reference;
- ``--script``: run a scripted cells episode on the server without a model
  (``cells_server.py script``), for offline verification of the whole loop.
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import shlex
import subprocess
import sys
import tempfile

from agent import HOST, MODEL_FLAGS, MODEL_TOOL_FLAGS, REMOTE, free_port, model_environment, wait_until_ready


READY = "PYRUA_CELLS_READY "


def build_prompt(task: str, source: str, budget_s: int, max_turns, motion="sim", execution="cells") -> str:
    from pyrualean.arms import arm_contract
    from pyrualean.prompt import api_reference
    from cells_robot import ROBOT_WORDS, StationRobot

    contract = arm_contract(
        execution,
        budget_s=budget_s,
        max_env_steps=10000,
        max_programs=None,
        images="on-demand" if execution == "program" else "on-motion",
        feedback="rich",
        budget_mode="calls" if max_turns else "wall",
        robot=ROBOT_WORDS,
    )
    reference = api_reference(StationRobot)
    user = "\n".join([
        "# Task card",
        "- robot: tianji dual arm + two Sharpa hands (StationRobot, `robo`)",
        f"- observation source: {source}",
        "- head camera: `robo.image('head')` (640x480 RGB); the same frame's "
        "measured depth gives metric points via `robo.locate(...)`",
        "- task instruction (authoritative): " + repr(task),
        "",
        ("Work in short cells: " if execution == "cells" else "Submit a complete run(robo) program: ") +
        "observe, act, verify from the returned state, "
        "then proceed. Call finish when the instruction is satisfied or you "
        "cannot make progress. Use Chinese for the final answer.",
    ])
    if execution == "program":
        user += "\nTo view a frame, call robo.show('head') inside run(robo). The host attaches its image on return."
    if motion == "real":
        user += ("\nCommissioning permits one arm target. First read `robo.observe()`, then "
                 "call `robo.move_joints` using the exact side and target from "
                 "`obs['commissioning_step']['arm']` and "
                 "`obs['commissioning_step']['target_joint_deg']`, only when they match the "
                 "requested action. Use another cell to read the final joint state. Report success "
                 "only when the returned motion result confirms movement and idle.")
    if motion == "trajectory":
        user += ("\nUse the live observation and `obs['motion_limits']`. Each `robo.move_joints` "
                 "target must remain inside that session envelope. The controller interpolates "
                 "and checks each waypoint and returns a fresh camera frame. Move only in visibly "
                 "clear space. Use the latest feedback to choose the next motion.")
    user += "\nExplore from your own current observations and action feedback. Choose visual prompts yourself. Use no demonstrations, saved boxes, or operator trajectory."
    return contract + "\n\n# API reference (the `robo` object)\n\n" + reference + "\n\n" + user


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Natural-language robot task")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="One fresh RGB-D snapshot from hardware")
    mode.add_argument("--episode", help="Path to a RealSense recording on the server")
    parser.add_argument("--frame", default=1, type=int)
    parser.add_argument("--calibration", help="Camera-to-robot transform on the server")
    parser.add_argument("--motion", choices=("sim", "none", "real", "trajectory"), default="sim")
    parser.add_argument("--execution", choices=("cells","program"), default="cells")
    parser.add_argument("--tactile-side", action="append", default=[], choices=("left","right"),
                        help="Also sample live Sharpa F6 for this hand")
    parser.add_argument("--arm-step", nargs=3, metavar=("SIDE", "JOINT", "DELTA_DEG"),
                        help="Operator-selected single joint commissioning step")
    parser.add_argument("--research-loop", action="store_true",
                        help="Have a researcher plan this Cells experiment from archived physical trials")
    parser.add_argument("--local", action="store_true", help="Start the robot server on this host")
    parser.add_argument("--max-turns", type=int, default=None, help="Decision budget per episode")
    parser.add_argument("--seconds", type=int, default=240, choices=range(10, 301),
                        metavar="10..300", help="Server session lifetime")
    parser.add_argument("--script", help="Run a scripted cells JSON on the server instead of a model")
    parser.add_argument("--codex", default="codex", help="Local Codex CLI")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "local_runs")
    args = parser.parse_args()
    if args.episode:
        parser.error("Exploration starts from a fresh scene. Recorded episodes are excluded.")
    if args.motion == "real" and not (args.live and args.calibration and args.arm_step):
        parser.error("real requires --live, --calibration and --arm-step")
    if args.motion == "trajectory" and not (args.live and args.calibration):
        parser.error("trajectory requires --live and --calibration")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y%m%dT%H%M%S-")
    out = Path(tempfile.mkdtemp(prefix=prefix, dir=args.output.resolve()))

    plan = None
    if args.research_loop:
        if args.script:
            parser.error("--research-loop uses the model-driven Cells client")
        from task import model_reply
        path = Path(__file__).with_name("physical_evidence.py") if args.local else Path(REMOTE)/"physical_evidence.py"
        history = [sys.executable if args.local else "python3", str(path), "--history", args.task]
        command = history if args.local else ["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=8",
                                              HOST,"exec " + shlex.join(history)]
        archived = json.loads(subprocess.check_output(command, text=True, timeout=20))
        facts = {"execution":"live snapshot" if args.live else "procedural tabletop scene", "motion":args.motion,
                 "execution_mode":args.execution, "live_f6_sides":args.tactile_side,
                 "tactile":"robo.tactile(frame_id, side) returns available snapshot samples; synthetic scenes have no physical touch data",
                 "hand_control":"virtual" if args.motion == "sim" else "disabled",
                 "perception":"read observation for available SAM, SigLIP and registered depth"}
        request = {"task":args.task, "environment_facts":facts, "physical_trials":archived,
                   "available_implementations":[], "round":0}
        plan = model_reply(args.codex, "research", request, out / "researcher")
        if plan["action"] == "stop":
            print(f"Researcher stopped before the device session: {plan['instruction']}\n{out}", flush=True)
            return 2

    port = free_port()
    remote = [
        str(Path(__file__).resolve().with_name("run.sh")) if args.local else f"{REMOTE}/run.sh",
        "cells-serve" if args.script is None else "cells-script",
        "--frame", str(args.frame), "--motion", args.motion, "--execution", args.execution,
    ]
    if args.script is None:
        remote += ["--port", str(port), "--seconds", str(args.seconds), "--task", args.task]
    if args.max_turns:
        remote += ["--max-turns", str(args.max_turns)]
    if args.live:
        remote.append("--live")
    for side in args.tactile_side:
        remote.extend(["--tactile-side", side])
    if args.episode:
        remote += ["--episode", args.episode]
    if args.calibration:
        calibration = str(Path(args.calibration).resolve()) if args.local else str(Path(REMOTE) / args.calibration)
        remote += ["--calibration", calibration]
    if args.arm_step:
        remote += ["--arm-step", *args.arm_step]
    if args.script:
        remote.append(args.script)

    server_command = remote if args.local else [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=8",
        "-L", f"127.0.0.1:{port}:127.0.0.1:{port}", HOST, "exec " + shlex.join(remote),
    ]
    ssh_stdout, ssh_stderr = out / "ssh.stdout", out / "ssh.stderr"
    if args.script:
        with ssh_stdout.open("w") as stdout, ssh_stderr.open("w") as stderr:
            subprocess.run(server_command, stdin=subprocess.DEVNULL, stdout=stdout,
                           stderr=stderr, timeout=args.seconds, check=True)
        print(ssh_stdout.read_text().strip())
        return
    with ssh_stdout.open("w") as stdout, ssh_stderr.open("w") as stderr:
        ssh = subprocess.Popen(server_command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        try:
            ready = wait_until_ready(ssh, ssh_stdout, ssh_stderr, ready_prefix=READY)
            print(f"Server run directory {ready['run']}", flush=True)
            prompt = ready["prompt"]
            if plan:
                prompt += "\nExperiment plan for the engineer, subject to the task and backend limits:\n" + json.dumps(plan, ensure_ascii=False)
            model_command = [
                args.codex, "-a", "never", "exec", "--ignore-user-config", "--ignore-rules", *MODEL_TOOL_FLAGS,
                "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                *MODEL_FLAGS,
                "-c", f"mcp_servers.pyrua_cells.url={json.dumps(ready['url'])}",
                "-c", "mcp_servers.pyrua_cells.required=true",
                "-c", "project_doc_max_bytes=0", "--json",
                "-o", str(out / "answer.md"), "-",
            ]
            with (out / "model.jsonl").open("w") as events, (out / "model.stderr").open("w") as errors:
                result = subprocess.run(
                    model_command, input=prompt, text=True, stdout=events, stderr=errors,
                    cwd=out, env=model_environment(), timeout=args.seconds - 15,
                )
            if result.returncode:
                raise RuntimeError(f"Codex exited with code {result.returncode}. See {out}")
            print((out / "answer.md").read_text().strip())
            print(f"Local model logs {out}", flush=True)
        finally:
            ssh.stdin.close()
            try:
                ssh.wait(timeout=8)
            except subprocess.TimeoutExpired:
                ssh.terminate()
                ssh.wait(timeout=8)


if __name__ == "__main__":
    raise SystemExit(main())
