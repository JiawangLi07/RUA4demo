"""Use the local GPT-6.2 Astra session to propose DexHumanid-RUA grasp programs."""

from argparse import ArgumentParser, SUPPRESS
from contextlib import suppress
from pathlib import Path
import base64
import json
import os
import select
import shlex
import subprocess
import tempfile
import time

from task_library import atomic_json

from agent import HOST, MODEL_FLAGS, MODEL_REPLY_TIMEOUT_S, MODEL_TOOL_FLAGS, REMOTE, model_environment


def send_proposal(codex, request, directory):
    directory.mkdir()
    request = dict(request)
    images = []
    for phase, data in request.pop("own_trial_images", {}).items():
        if phase not in ("before", "hold_1"):
            raise ValueError("Unexpected trial image")
        image = directory / f"{phase}.jpg"
        image.write_bytes(base64.b64decode(data, validate=True))
        images.extend(["--image", str(image)])
    result = directory / "proposal.json"
    schema = directory / "proposal.schema.json"
    schema.write_text(json.dumps({
        "type": "object",
        "additionalProperties": False,
        "required": ["program", "reason"],
        "properties": {
            "reason": {"type": "string"},
            "program": {
                "type": "object", "additionalProperties": False,
                "required": ["offset_m", "lift_m", "grasp", "recheck"],
                "properties": {
                    "offset_m": {"type": "array", "items": {"type": "number"}},
                    "lift_m": {"type": "number"},
                    "grasp": {"type": "string", "enum": request["grasp_names"]},
                    "recheck": {"type": "boolean"},
                },
            },
        },
    }))
    prompt = (
        "Propose the next grasp Python program for the requested object via the output schema. "
        "The host compiles your proposal into prepare, approach, optional confirm_target, "
        "descend, close_hand and lift. offset_m has exactly three entries in the rig's "
        "mount frame, bounded per axis by offset_limit_m. lift_m must be within lift_range_m. "
        "Use the previous plan on round zero as a starting point. For a revision choose "
        "one meaningful change supported by evidence. In history, active may refer to a "
        "rolled-back plan, so use each trial's own program when interpreting its evidence. "
        "Avoid a plan already tried. The result is judged independently by measured object "
        "height over time. Never claim a successful grasp in your proposal. "
        "Use only the data below and return JSON.\n\n"
        + json.dumps(request, ensure_ascii=False)
    )
    if request.get("proposal_format") == "python":
        schema.write_text(json.dumps({
            "type":"object", "additionalProperties":False, "required":["source","reason"],
            "properties":{"source":{"type":"string"},"reason":{"type":"string"}},
        }))
        single_cell = request.get("single_cell", False)
        first_call = "Call robo.locate_target() first. " if single_cell else ""
        last_line = "After the function definition, write exactly result = run(robo). " if single_cell else ""
        prompt = (
            "You are the engineer. Write an executable Python grasp program for the REAL experiment. "
            "Use only the task, API, and autonomous trial evidence. Set a literal dict PLAN with "
            "offset_m, lift_m, grasp and recheck within the supplied bounds. Define def run(robo). "
            + first_call
            + 'Call robo.prepare(PLAN["offset_m"], PLAN["lift_m"], PLAN["grasp"]). '
            "Call robo.approach(); if the literal recheck is True, include an unconditional "
            "robo.confirm_target() here. Then call robo.descend(), robo.close_hand(), and "
            "return robo.lift(). "
            + last_line
            + "Use one statement per call, no imports or other statements. "
            "Any attached before and hold images come from your own preceding physical trial. "
            "Tune the supported parameters in response to feedback. Host checks the frozen code "
            "and runs it with Lean against the measured station. Return source and reason as JSON.\n"
            + json.dumps(request, ensure_ascii=False)
        )
    (directory / "prompt.txt").write_text(prompt)
    command = [
        codex, "-a", "never", "exec", "--ignore-user-config", "--ignore-rules", *MODEL_TOOL_FLAGS, "--ephemeral",
        "--skip-git-repo-check", "--sandbox", "read-only", *MODEL_FLAGS, "-c", "project_doc_max_bytes=0",
        "--output-schema", str(schema), "-o", str(result), "--json", *images, "-",
    ]
    with (directory / "codex.jsonl").open("w") as events, (directory / "codex.stderr").open("w") as log:
        completed = subprocess.run(command, input=prompt, text=True, cwd=directory,
                                   stdout=events, stderr=log, env=model_environment(), timeout=MODEL_REPLY_TIMEOUT_S)
    if completed.returncode or not result.is_file():
        raise RuntimeError(f"Learner call failed. See {directory}")
    return result.read_text()


def messages(pipe):
    buffer = b""
    deadline = time.monotonic() + 100
    while True:
        line, newline, buffer_after_line = buffer.partition(b"\n")
        if newline:
            buffer = buffer_after_line
            yield json.loads(line)
            deadline = time.monotonic() + 240  # a serial batch can contain two trials
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([pipe], [], [], remaining)[0]:
            raise TimeoutError("Timed out waiting for the robot host")
        chunk = os.read(pipe.fileno(), 65536)
        if not chunk:
            if buffer:
                raise RuntimeError("Incomplete robot-host message")
            return
        buffer += chunk
        if len(buffer) > 262144:
            raise RuntimeError("Robot-host response too large")


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--single-cell", action="store_true")
    parser.add_argument("--rounds", type=int, choices=range(1, 7), default=3, metavar="1..6")
    parser.add_argument("--run-directory", type=Path, help=SUPPRESS)
    parser.add_argument("--local", action="store_true", help="Run the host process on the current machine")
    parser.add_argument("--target", choices=("cube","shuttlecock"), default="cube")
    parser.add_argument("--arm", choices=("right","left"))
    parser.add_argument("--real", action="store_true", help="Execute on the measured grasp station")
    parser.add_argument("--profile", help="Absolute path to the measured profile on the robot host")
    parser.add_argument("--verify-campaign", help="Absolute server path of the pending physical campaign")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "local_runs")
    args = parser.parse_args(argv)
    if args.real != bool(args.profile) or (args.profile and not args.profile.startswith("/")):
        parser.error("--real requires --profile with an absolute server path")

    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    out = args.run_directory or Path(tempfile.mkdtemp(prefix=args.target+"-", dir=args.output.resolve()))
    atomic_json(out / "status.json", {"phase":"connecting", "target":args.target})
    remote = [str(Path(__file__).resolve().with_name("run.sh")) if args.local else f"{REMOTE}/run.sh", "explore", "--interactive", "--rounds", str(args.rounds)]
    if args.real:
        remote += ["--real", "--profile", args.profile, "--target", args.target]
    elif args.target != "cube":
        parser.error("Shuttlecock grasp requires the physical backend")
    if args.calibrate:
        if not args.real: parser.error("--calibrate requires --real")
        remote.append("--calibrate")
    if args.single_cell:
        if args.calibrate or args.verify_campaign:
            parser.error("--single-cell cannot run calibration or campaign verification")
        remote.append("--single-cell")
    if args.arm:
        remote += ["--arm", args.arm]
    if args.verify_campaign:
        if not args.real or not Path(args.verify_campaign).is_absolute():
            parser.error("--verify-campaign requires --real and an absolute server campaign path")
        remote += ["--verify-campaign", args.verify_campaign]
    command = remote if args.local else [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3", HOST, "exec " + shlex.join(remote),
    ]
    print(f"Local exploration logs {out}", flush=True)
    with (out / "host.log").open("w") as log:
        ssh = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        result = None
        host_run = None
        try:
            for message in messages(ssh.stdout):
                kind = message.pop("kind")
                if kind == "started":
                    host_run = message["run"]
                    atomic_json(out / "status.json", {"phase":"started", **message})
                    print(f"Server run directory {message['run']}", flush=True)
                elif kind == "progress":
                    atomic_json(out / "status.json", {"run":host_run, **message})
                elif kind == "diagnostic_request":
                    from task import model_reply
                    turn = message["turn"]
                    atomic_json(out / "status.json", {"phase":"diagnosing","turn":turn,"run":host_run})
                    decision = model_reply(args.codex, "diagnostic", message, out / f"diagnostic-{turn}")
                    # The independent hardware monitor can finish while inference is running.
                    with suppress(BrokenPipeError):
                        ssh.stdin.write(json.dumps(decision, ensure_ascii=False).encode() + b"\n")
                        ssh.stdin.flush()
                elif kind == "research_request":
                    atomic_json(out / "status.json", {"phase":"planning", "round":message["round"], "run":host_run})
                    from task import model_reply
                    folder = out / f"research-{message['round']}-{message.get('draft', 0)}"
                    decision = model_reply(args.codex, "research", message, folder)
                    ssh.stdin.write(json.dumps(decision, ensure_ascii=False).encode() + b"\n")
                    ssh.stdin.flush()
                elif kind == "proposal_request":
                    atomic_json(out / "status.json", {"phase":"writing_program", "round":message["round"], "run":host_run})
                    folder = out / f"proposal-{message['round']}-{message.get('draft', 0)}"
                    from task import model_reply
                    proposal = model_reply(args.codex, "cube", message, folder)
                    compact = json.dumps(proposal, ensure_ascii=False).encode()
                    ssh.stdin.write(compact + b"\n")
                    ssh.stdin.flush()
                    print(f"Submitted program for round {message['round']}", flush=True)
                elif kind in ("result", "error"):
                    result = message
                else:
                    raise RuntimeError(f"Unknown robot-host message {kind}")
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            result = {"status":"interrupted", "run":host_run, "error":str(exc)}
        finally:
            with suppress(BrokenPipeError):
                ssh.stdin.close()
            ssh.stdout.close()
            try:
                ssh.wait(timeout=10)
            except subprocess.TimeoutExpired:
                ssh.terminate()
                ssh.wait(timeout=10)
        if result is None:
            result = {"status":"interrupted", "run":host_run, "error":"Host exited without a result"}
        atomic_json(out / "result.json", result)
        atomic_json(out / "status.json", {"phase":"finished", "run":host_run, "status":result.get("status", result.get("reason"))})
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return ssh.returncode or int(result.get("status") == "interrupted")


if __name__ == "__main__":
    raise SystemExit(main())
