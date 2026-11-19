"""Run an autonomous task on the offline router or physical grasp station."""

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import base64
import fcntl
import json
import sys
import re
import shlex
import subprocess
import tempfile

from agent import HOST, MODEL_FLAGS, MODEL_REPLY_TIMEOUT_S, MODEL_TOOL_FLAGS, REMOTE, model_environment
from explore import messages, send_proposal
from task_library import atomic_json


RESOLVE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["kind", "arm", "object", "values", "reason"],
    "properties": {
        "kind": {"type": "string", "enum": ["joint_delta", "joint_target", "hand_target",
                                                "locate", "grasp_cube", "unsupported"]},
        "arm": {"type": "string", "enum": ["left", "right", "none"]},
        "object": {"type": "string", "enum": ["cube", "shuttlecock", "tube", "none"]},
        "values": {"type": "array", "items": {"type": "number"}},
        "reason": {"type": "string"},
    },
}
SOURCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["source", "reason"],
    "properties": {"source": {"type": "string"}, "reason": {"type": "string"}},
}


def model_attempt(codex, kind, request, out):
    if kind == "cube":
        return json.loads(send_proposal(codex, request, out))
    out.mkdir()
    if kind == "diagnostic":
        request = dict(request)
        image = request.pop("image_base64", None)
        if image:
            (out / "head.jpg").write_bytes(base64.b64decode(image, validate=True))
        calibration = request.get("calibration_authorized",False)
        schema = {"type":"object","additionalProperties":False,"required":["action","source","reason"],
                  "properties":{"action":{"type":"string","enum":["cell","stop","proceed"]},
                                "source":{"type":"string"},"reason":{"type":"string"}}}
        calibration_prompt = ("Physical calibration is requested. Inspect the attached fresh image for people and ongoing work in the robot area. If occupied, stop and ask for a clear station. Once clear, call robo.calibrate() "
                              "as its own cell to open a monitored session and perform an initial unloaded index step. The station will ask you for subsequent steps in the same session. "
                              "It cannot command contact; honor SDK refusals. Recapture after the attempt. "
                              "Use inspect('pad_measurement', [thumb_box then index_box, eight numbers]) if both pads visible; "
                              "use inspect('preview_approach', target_box) for incremental pregrasp IK and geometry. "
                              "Use the feedback to decide what can be verified and whether physical assistance is needed. "
                              if calibration else "")
        prompt = (calibration_prompt + "You are the REAL agent on the physical grasp station. Write Python for a Lean "
                  "cell and use measured results to choose your next experiment. Code consists only of one "
                  "to four literal robo.inspect calls, one per line. Available tools: 'profile' checks calibration, "
                  "'capture' acquires RGB-D and joints, 'clearance' checks measured geometry on that capture, "
                  "'locate' takes a second argument [x1,y1,x2,y2] in 640x480 RGB (use the attached live image), "
                  "Optional tools: 'fixed_arm_escape' searches stationary-wrist finger paths, 'tcp' examines model versus measured grasp offset, "
                  "'contact_stop' for static force plus live tactile diagnostics, and 'workspace' with a second [x1,y1,x2,y2] box "
                  "covering the target, approach corridor and surrounding table. 'clearance' gives all links including the minimum. "
                  "Compare local workspace observations to unobserved areas; never call a model-only escape executable or box depth a validated TCP. "
                  "Choose tools relevant to your hypothesis. A failed candidate is evidence for another experiment within the task budget. "
                  "Use prior observations to select the next cell. Missing physical calibration cannot be invented. "
                  "With a valid station profile you may proceed directly to the grasp planner. It checks the proposed path and fresh state before any motion. No diagnostic checklist is required. Choose stop when further "
                  "read-only diagnostics cannot resolve the blockers. At last_turn summarize. Use Chinese for reason, "
                  "cite specific measurements and distinguish surface point from TCP. No external shell. JSON only.\n"
                  + json.dumps(request,ensure_ascii=False))
        if request.get("session_active"):
            prompt = ("You control an already enabled unloaded right hand. Read the last measured step, current image, "
                      "and sensor feedback. The left arm stays fixed. The right arm is available via robo.arm_step('Joint1_R' through 'Joint7_R', signed_degrees), each proposed arm path is interpolated and checked against geometry and native joint limits. Prefer robo.move_relative([dx,dy,dz]) in meters along dual_arm_mount axes, or robo.move_pose([x,y,z,qx,qy,qz,qw]) with unit xyzw quaternion. A 3-number move_pose preserves current orientation. tcp_pose_mount gives current 4x4 pose. Server computes IK and executes all interpolated points locally. Choose a meaningful pregrasp approach target using measured object and modeled pad center, with observed free space. Each proposal receives a structured path rejection or actual pose feedback. Do not keep using half-degree joint probes when Cartesian motion is available. Choose one informative next action relative to CURRENT "
                      "feedback, using exactly robo.step('listed_joint_name', signed_degrees), hand commands at most 2 degrees per command, following the deployed hand step limiter. "
                      "You can call robo.measure([x1,y1,x2,y2]) on the fresh image to obtain target surface and modeled finger center "
                      "in meters in dual_arm_mount. Use before/after measurements to test whether an approach reduces distance. "
                      "Keep the original shuttlecock grasp objective. An executed calibration step is not task completion. "
                      "Continue relevant reachable geometry and grasp-pose exploration when the tool can still provide evidence. "
                      "Use hand geometry changes and target measurements to test grip offsets in free space before contact. "
                      "A shortfall can be explored by another bounded step or reversing and measuring repeatability. "
                      "The station monitors hardware during inference. Do not repeatedly drive against an obstruction or "
                      "attempt contact. Stop for a concrete unresolved failure or when you have identified the specific missing evidence for grasping and another non-contact action has no justified way to resolve it. Repeating opening/reversing a finger is not grasp progress once its response is measured. Describe the specific target visibility, alignment, or contact feedback needed. If the remaining task requires unavailable contact "
                      "calibration, choose stop (also on last_turn). Explain in Chinese; distinguish actual from requested "
                      "motion. Use robo.grasp with a literal plan containing offset_m (3 meters offsets, each within 0.012), lift_m (0.06 to 0.08 meters), grasp (configured target name), recheck (bool) to request contact execution. The server validates commissioned contact limits, freezes this program, executes closure/lift/reset and automatically attempts independent replays. A request is not success. Return JSON with action cell or stop, source and reason.\n"+json.dumps(request,ensure_ascii=False))
    elif kind == "research":
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["action", "hypothesis", "instruction", "expected_observation", "stop_when", "skill_references"],
                  "properties": {"action": {"type": "string", "enum": ["experiment", "stop"]},
                                 "skill_references":{"type":"array","items":{"type":"string"}},
                                 "stop_when":{"type":"string"},
                                 **{k: {"type": "string"} for k in
                                    ("hypothesis", "instruction", "expected_observation")}}}
        execution = request["environment_facts"]["execution"]
        prompt = (f"You are the researcher for a {execution} robot experiment. Select one testable "
                  "improvement for the engineer using prior experiments and measured outcomes. "
                  "Select relevant implementation ids from available_implementations or []. Specify an observable stopping condition. Keep the original task and independent verifier fixed. Select stop only when "
                  "the available backend cannot test a further relevant hypothesis. "
                  "Use only the current backend and distinguish archived observations from verified object outcomes. Return JSON.\n" + json.dumps(request, ensure_ascii=False))
    elif kind == "resolve":
        schema = {"type": "object", "additionalProperties": False, "required": ["steps"],
                  "properties": {"steps": {"type": "array", "items": RESOLVE_SCHEMA}}}
        prompt = (
            "Translate this robot instruction into its exact independent goal contract. "
            "Supported offline tasks: joint_delta (seven degree increments, zero for unchanged "
            "joints), joint_target (seven absolute degree targets), hand_target (22 explicitly "
            "specified radian targets), locate (one shuttlecock or tube in the procedural tabletop scene), "
            "and grasp_cube (right-hand stable lift in the cube fixture). "
            "Use arm none for locate. Unused object is none and unused values is []. "
            "Preserve requested side, joint number and sign. Require an explicit joint number "
            "and magnitude for joint_delta, not just 'move right arm'. Do not invent hand "
            "angles. Split compound tasks into the ordered steps required by the instruction. Never replace an unsupported manipulation with an easier "
            "supported subtask. Those tasks must return unsupported with a reason. "
            "A right-hand shuttlecock grasp, moving objects into containers, and physical "
            "two-hand manipulation currently have no offline goal checker. Return JSON.\n\n"
            + request["task"]
        )
    else:
        schema = SOURCE_SCHEMA
        request = json.loads(json.dumps(request))
        encoded = request["context"].pop("image_base64", None)
        if encoded and (kind == "discovery" or request["contract"]["kind"] == "locate"):
            (out / "head.jpg").write_bytes(base64.b64decode(encoded, validate=True))
        prompt = (
            "Write one reusable Python skill: def run(robo, params). Return source and reason "
            "as JSON. The source must contain exactly this one function, with no decorators, "
            "annotations, default arguments or module-level imports. Imports inside the "
            "function are governed by the existing PyRUA sandbox. Tool replies are mappings: use keys, never getattr, hasattr or introspection. Use the documented robo "
            "API and read fresh observations. params is the task contract below. Numerical "
            "goals are runtime parameters; do not hardcode a joint start state, target vector "
            "or frame ID. For joint_delta add params['values'] to freshly observed arm angles. "
            "For joint_target and hand_target use their absolute requested values. For locate "
            "inspect the current image to find the object at runtime. Object positions vary across trials; use image processing or SAM with a prompt derived from the current scene, then call "
            "robo.locate with the current frame ID. Return the selected locate or locate_mask result dictionary for the caller. Use failure evidence to revise the previous "
            "program. The host checks actual virtual joint readback or current depth "
            "and tests frozen code on fresh starts and held-out parameters. All execution is "
            "offline. You can call segment, locate_mask and rank_labels when perception_backends is available. A claim of success does not satisfy the goal checker.\n\n"
            "The supplied API and state are complete. Return JSON directly without shell "
            "commands or filesystem inspection.\n\n"
            + json.dumps(request, ensure_ascii=False)
        )
        if kind == "discovery":
            prompt = ("You are the engineer. Write one plain def run(robo, params) implementing "
                      "the researcher hypothesis against the offline image, state and listed API. "
                      "The requested task lacks a success verifier. Use only the current observation and autonomous trials where "
                      "available. Always obtain the frame_id from robo.observe() in this run. When context.archive is present, the supplied image and depth replay that exact physical capture read-only. Do not claim physical grasp or completion. Return source and reason "
                      "directly as JSON.\n" + json.dumps(request, ensure_ascii=False))
    schema_path = out / "schema.json"
    schema_path.write_text(json.dumps(schema))
    (out / "prompt.txt").write_text(prompt)
    result = out / "reply.json"
    command = [codex, "-a", "never", "exec", "--ignore-user-config", "--ignore-rules", *MODEL_TOOL_FLAGS,
               "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", *MODEL_FLAGS, "-c", "project_doc_max_bytes=0",
               "--output-schema", str(schema_path), "-o", str(result), "--json"]
    if (out / "head.jpg").is_file():
        command += ["--image", str(out / "head.jpg")]
    command.append("-")
    with (out / "model.jsonl").open("w") as events, (out / "model.stderr").open("w") as log:
        completed = subprocess.run(command, input=prompt, text=True, cwd=out,
                                   stdout=events, stderr=log, env=model_environment(), timeout=MODEL_REPLY_TIMEOUT_S)
    if completed.returncode or not result.is_file():
        raise RuntimeError(f"Task model call failed; inspect {out}")
    reply = json.loads(result.read_text())
    if kind == "resolve":
        return {"kind": "sequence", "steps": reply["steps"]}
    return reply


def model_reply(codex, kind, request, out):
    for attempt in range(2):
        directory = out if attempt == 0 else out.with_name(out.name + "-retry")
        try:
            return model_attempt(codex, kind, request, directory)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
            if attempt == 1:
                raise RuntimeError(f"Model request failed after retry; inspect {directory}") from exc
            print(f"Retrying stalled model request; logs {directory}", flush=True)


def physical_goal(task):
    """Resolve only complete supported grasp instructions before starting the hardware client."""
    text = task.strip().rstrip("。.!！").lower()
    match = re.fullmatch(r"(?:请)?(?:用|使用)?(左手|右手)?(?:把|将)?(?:桌面上|桌上|桌面)?的?(羽毛球|红色方块|红方块)(?:抓起|抓起来|拿起)", text)
    if not match:
        match = re.fullmatch(r"(?:请)?(?:用|使用)?(左手|右手)?(?:去)?(?:抓取|抓起|抓|拿起)(?:桌面上|桌上|桌面)?的?(?:一个)?(羽毛球|红色方块|红方块)", text)
    if not match:
        raise ValueError("Unregistered physical goal. Supported tasks are a tabletop shuttlecock or red cube grasp.")
    side, noun = match.groups()
    return ("left" if side == "左手" else "right", "shuttlecock" if noun == "羽毛球" else "cube")


def background_grasp(options, output):
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="physical-job-", dir=output.resolve()))
    command = [sys.executable, str(Path(__file__).resolve().with_name("explore.py")),
               *options, "--run-directory", str(directory)]
    with (directory / "driver.log").open("w") as log, (directory / "driver.lock").open("a") as lock:
        # Only the driver inherits this descriptor. Its exit releases the receipt even after a crash.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True, pass_fds=(lock.fileno(),))
    atomic_json(directory/"job.json", {"pid":process.pid})
    print(json.dumps({"job":str(directory),"pid":process.pid}, ensure_ascii=False), flush=True)
    return 0


def job_status(directory):
    with (directory / "driver.lock").open("r") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            running = False
        except BlockingIOError:
            running = True
        result = directory / "result.json"
        if result.exists():
            row = json.loads(result.read_text())
            return {"phase":"finished", **{k:row[k] for k in (
                "status","success","reason","error","candidate","skill","run","last_trial") if k in row}}
        path = directory / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {"phase":"starting"}
        if not running:
            return {"phase":"interrupted","last_state":state, "log":str(directory/"driver.log")}
        return state


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?")
    parser.add_argument("--background", action="store_true", help="Detach the physical task driver and print its job directory")
    parser.add_argument("--status", type=Path, help="Read the local receipt of a background task")
    parser.add_argument("--real", action="store_true", help="Run a supported physical manipulation task")
    parser.add_argument("--calibrate", action="store_true", help="Allow a bounded right-hand calibration trial before grasp planning")
    parser.add_argument("--profile", help="Measured autonomous station on the robot host")
    parser.add_argument("--verify-campaign", help="Validate the frozen revision of a pending real campaign")
    parser.add_argument("--research-loop", action=BooleanOptionalAction, default=True, help="Plan experiments with a separate researcher role")
    parser.add_argument("--local", action="store_true", help="Run the offline host on this machine")
    parser.add_argument("--library", help="Persistent library directory on the robot host")
    parser.add_argument("--rounds", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--scripted", action="store_true", help="Use deterministic integration proposals")
    parser.add_argument("--contract", help="Goal JSON path on the robot host")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "local_runs")
    args = parser.parse_args()
    if args.status:
        if args.task or args.real or args.background:
            parser.error("--status reads one job directory")
        print(json.dumps(job_status(args.status), ensure_ascii=False, indent=2))
        return 0
    if not args.task:
        parser.error("Specify a task")
    if args.background and not args.real:
        parser.error("--background is for physical tasks")
    if args.calibrate and not args.real:
        parser.error("--calibrate requires --real")
    if args.real:
        if args.scripted or args.contract or args.library or not args.research_loop:
            parser.error("Real tasks use physical goal checks, the autonomous library and the research loop")
        try:
            arm, target = physical_goal(args.task)
        except ValueError as exc:
            parser.error(str(exc))
        from explore import main as physical_main
        profile = args.profile or str(Path(REMOTE) / "config" / f"{arm}_{target}_station.json")
        options = ["--real","--arm",arm,"--target",target,"--profile",profile,
                   "--rounds",str(args.rounds),"--codex",args.codex,"--output",str(args.output)]
        if args.calibrate:
            options.append("--calibrate")
        if args.verify_campaign:
            options += ["--verify-campaign", args.verify_campaign]
        if args.local:
            options.append("--local")
        if args.background:
            return background_grasp(options, args.output)
        return physical_main(options)
    if args.profile or args.verify_campaign:
        parser.error("--profile and --verify-campaign require --real")
    if args.scripted and not args.contract:
        parser.error("--scripted requires --contract")
    args.output.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="task-", dir=args.output.resolve()))
    remote = [str(Path(__file__).resolve().with_name("run.sh")) if args.local else f"{REMOTE}/run.sh",
              "task-server", "--interactive", "--rounds", str(args.rounds)]
    remote.append("--research-loop" if args.research_loop else "--no-research-loop")
    if args.library:
        remote += ["--library", args.library]
    if args.contract:
        remote += ["--contract", args.contract]
    if args.scripted:
        remote.append("--scripted")
    remote.append(args.task)
    command = remote if args.local else [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
        HOST, "exec " + shlex.join(remote),
    ]
    calls = 0
    result = None
    host_run = None
    print(f"Local task logs {out}", flush=True)
    with (out / "host.log").open("w") as log:
        host = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        try:
            for packet in messages(host.stdout):
                kind = packet.pop("kind")
                if kind == "started":
                    host_run = packet["run"]
                    print(f"Server run directory {packet['run']}", flush=True)
                elif kind in ("resolve_request", "proposal_request"):
                    format = "resolve" if kind == "resolve_request" else packet["format"]
                    request = packet if format == "resolve" else packet["request"]
                    reply = model_reply(args.codex, format, request, out / f"model-{calls}")
                    calls += 1
                    host.stdin.write(json.dumps(reply, ensure_ascii=False).encode() + b"\n")
                    host.stdin.flush()
                elif kind == "result":
                    result = packet
                else:
                    raise RuntimeError(f"Unknown task-host message {kind}")
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            result = {"status": "failed", "error": str(exc), "run": host_run}
        finally:
            host.stdin.close()
            host.stdout.close()
            try:
                host.wait(timeout=8)
            except subprocess.TimeoutExpired:
                host.terminate()
                host.wait(timeout=8)
    if result is None:
        raise RuntimeError(f"Host exited without a task result; inspect {out}")
    result["model_calls"] = calls
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result.get(key) for key in
                      ("status", "route", "contract", "model_calls", "skill", "run", "reason", "error")},
                     indent=2, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "success" and host.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
