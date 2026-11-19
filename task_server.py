"""Offline task routing protocol used by task.py. This entry has no live mode."""

from argparse import ArgumentParser, BooleanOptionalAction
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import tempfile

from exploration.offline import scripted_learner
from hardware import ROOT
from task_library import atomic_json
from task_router import TaskRouter


PROTOCOL = sys.stdout


def message(kind, **data):
    packet = {"kind": kind, **data}
    text = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    # Full shared-state traces can exceed the client's single-message budget.
    # The complete result has already been saved before this notification.
    if kind == "result" and len(text.encode()) > 240000:
        packet["trials"] = [{k: v for k, v in row.items()
                             if k in ("success", "reason", "case", "completed_steps", "run")}
                            for row in data.get("trials", [])]
        packet["detail_path"] = str(Path(data["run"]) / "result.json")
        text = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    print(text, file=PROTOCOL, flush=True)


def ask(kind, **data):
    message(kind, **data)
    line = sys.stdin.readline(262145)
    if not line or len(line) > 262144:
        raise RuntimeError("Task client disconnected or response exceeded 256 KiB")
    return json.loads(line)


def scripted_proposal(format, request):
    if format == "discovery":
        return {"source": "def run(robo, params):\n    return robo.observe()\n",
                "reason": "Probe available offline state"}
    if format == "cube":
        return scripted_learner(request)
    kind = request["contract"]["kind"]
    if kind in ("joint_delta", "joint_target"):
        target = ("[q + d for q, d in zip(obs['joints']['arm_deg'][params['arm']], params['values'])]"
                  if kind == "joint_delta" else "params['values']")
        source = ("def run(robo, params):\n    obs = robo.observe()\n"
                  f"    target = {target}\n"
                  "    robo.move_joints(params['arm'], target)\n    return robo.observe()\n")
    elif kind == "hand_target":
        source = ("def run(robo, params):\n"
                  "    robo.set_hand(params['arm'], params['values'])\n    return robo.observe()\n")
    else:
        source = ("def run(robo, params):\n    import numpy as np\n    obs=robo.observe()\n"
                  "    rgb=robo.image('head')\n"
                  "    pixels=(rgb[:,:,0]>195) if params['object']=='shuttlecock' else (rgb[:,:,2]>rgb[:,:,0].astype(float)+75)\n"
                  "    y,x=np.nonzero(pixels)\n"
                  "    i=int(np.argmin((x-np.median(x))**2+(y-np.median(y))**2))\n    box=[int(x[i])-4,int(y[i])-4,int(x[i])+5,int(y[i])+5]\n"
                  "    return robo.locate(obs['frame_id'],params['object'],box)\n")
    return {"source": source, "reason": "Deterministic offline integration fixture"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--research-loop", action=BooleanOptionalAction, default=True)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--contract", type=Path, help="Explicit host-side goal JSON")
    parser.add_argument("--library", type=Path, default=ROOT / ".local/autonomous_library")
    parser.add_argument("--rounds", type=int, choices=range(1, 7), default=3)
    args = parser.parse_args()
    if not args.interactive and not (args.scripted and args.contract):
        parser.error("Use the task client, or --scripted with an explicit --contract")
    (ROOT / "runs").mkdir(exist_ok=True)
    prefix = "task-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-")
    out = Path(tempfile.mkdtemp(prefix=prefix, dir=ROOT / "runs"))
    message("started", run=str(out), source="offline")
    router = TaskRouter(
        args.library / "fixtures" if args.scripted else args.library, out,
        resolve=lambda task: ask("resolve_request", task=task),
        propose=scripted_proposal if args.scripted else
        lambda format, request: ask("proposal_request", format=format, request=request),
        researcher=(lambda request: {
            "action": "experiment", "hypothesis": "Task feedback identifies the needed correction",
            "instruction": "Use the recorded error and the independent goal checker",
            "expected_observation": "Measured goal holds on held-out trials",
        } if args.scripted else ask("proposal_request", format="research", request=request))
        if args.research_loop else None,
        rounds=args.rounds,
    )
    try:
        contract = json.loads(args.contract.read_text()) if args.contract else None
        # RSIJudge prints human-readable progress during trials.
        with redirect_stdout(sys.stderr):
            result = router.run(args.task, contract)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        result = {"status": "failed", "error": str(exc), "run": str(out)}
        atomic_json(out / "result.json", result)
    message("result", **result)
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
