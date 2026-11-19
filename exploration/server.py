"""Run an object-grasp RSIJudge campaign on the robot host."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import select
import signal
import sys
import tempfile
import time

from agent import MODEL_REPLY_TIMEOUT_S
from hardware import ROOT
from .campaign import GraspCampaign, save_json
from .grasp_program import Program
from .offline import OfflineTrials, scripted_learner


def new_run(target="cube"):
    parent = ROOT / "runs"
    parent.mkdir(mode=0o700, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime(f"{target}-%Y%m%dT%H%M%S-")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


class Channel:
    def __init__(self, interactive):
        self.interactive = interactive
        if interactive:
            # Preserve a JSON-only channel even if a native SDK logs to stdout.
            sys.stdout.flush()
            self.protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(self, kind, data):
        if self.interactive:
            print(json.dumps({"kind": kind, **data}, ensure_ascii=False, allow_nan=False),
                  file=self.protocol, flush=True)
        else:
            print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), flush=True)

    def cancelled(self):
        # During a trial, the client sends no data. Readability here is EOF or an
        # out-of-order reply, either of which ends motion.
        return self.interactive and bool(select.select([sys.stdin], [], [], 0)[0])

    def research(self, request):
        return self.request("research_request", request)

    def propose(self, request):
        if not self.interactive:
            return scripted_learner(request)
        return self.request("proposal_request", request)

    def disconnected(self):
        poller = select.poll()
        poller.register(sys.stdin, select.POLLHUP | select.POLLERR)
        return bool(poller.poll(0))

    def request(self, kind, request, watch=None):
        self.send(kind, request)
        deadline=time.monotonic()+(55 if watch else 2 * MODEL_REPLY_TIMEOUT_S + 30)
        while True:
            if watch:
                watch()
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise RuntimeError("Learner reply timed out")
            if select.select([sys.stdin], [], [], min(.02,remaining) if watch else remaining)[0]:
                break
        reply = sys.stdin.readline(8192)
        if not reply.endswith("\n"):
            raise RuntimeError("Learner reply missing or too large")
        return json.loads(reply)


def interrupt(_number, _frame):
    raise KeyboardInterrupt("Exploration interrupted")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("cube","shuttlecock"), default="cube")
    parser.add_argument("--arm", choices=("right","left"), help="Match the requested physical task to the station arm")
    parser.add_argument("--real", action="store_true", help="Execute on the measured grasp station")
    parser.add_argument("--profile", type=Path, help="Measured cube-station profile")
    parser.add_argument("--check-profile", action="store_true", help="Validate profile and nominal IK offline")
    parser.add_argument("--verify-campaign", type=Path, help="Check a frozen candidate at one newly staged physical position")
    parser.add_argument("--program", type=Path, help="Run one previously exported grasp program")
    parser.add_argument("--interactive", action="store_true", help="Request model proposals over stdio")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--single-cell", action="store_true", help="Run the live right-hand shuttlecock cell")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Optional SAM3 search region containing the full shuttlecock")
    parser.add_argument("--clear-view-first", action="store_true",
                        help="From a checked live scene, clear the right-hand view before locating")
    parser.add_argument("--rounds", type=int, choices=range(1, 7), default=3, metavar="1..6")
    args = parser.parse_args()
    live_cell = args.single_cell and args.real and args.target == "shuttlecock" and args.arm in (None, "right")
    if args.single_cell and (not live_cell or args.calibrate or args.program or args.verify_campaign):
        parser.error("--single-cell runs a standalone live right-hand shuttlecock task")
    if args.bbox is not None and not live_cell:
        parser.error("--bbox requires the live single-cell task")
    if args.clear_view_first and not live_cell:
        parser.error("--clear-view-first requires the live single-cell task")
    if args.target != "cube" and not (args.real or args.check_profile):
        parser.error("The offline cube fixture cannot execute a shuttlecock grasp")
    if bool(args.profile) != (args.real or args.check_profile):
        parser.error("--profile is required with --real or --check-profile")
    if args.check_profile and (args.real or args.program or args.interactive):
        parser.error("--check-profile runs standalone")
    if args.verify_campaign and (not args.real or args.program):
        parser.error("--verify-campaign requires --real and a calibrated profile")
    if args.real and not (args.interactive or args.program or args.verify_campaign or live_cell):
        parser.error("Real exploration uses the local model client or an exported program")
    if args.calibrate and not (args.real and args.interactive):
        parser.error("--calibrate requires an interactive physical task")
    channel = Channel(args.interactive)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    out = None
    research = None
    try:
        if live_cell:
            from .live_cell import run_live_cell

            config = json.loads(args.profile.read_text())
            if config["arm"] != "right" or config["target"] != args.target:
                raise ValueError("Requested object or arm differs from the station identity")
            out = new_run(args.target)
            channel.send("started", {"run": str(out), "source": "real", "phase": "executing_cell"})
            result = run_live_cell(config, args.profile, out, channel.cancelled, args.bbox,
                                   clear_view_first=args.clear_view_first)
            save_json(out / "result.json", result)
            save_json(out / "status.json", {"phase": "finished", "status": result["status"]})
            channel.send("result", result)
            return 0 if result["success"] else 2
        if args.real and args.interactive:
            from .diagnostics import Diagnostics, diagnose
            out = new_run(args.target)
            channel.send("started", {"run":str(out), "source":"real", "phase":"diagnostics"})
            result = diagnose(Diagnostics(args.profile, args.arm or "right", args.target, out, calibration=args.calibrate, cancelled=channel.disconnected), channel, out)
            if result is not None:
                save_json(out / "result.json", result)
                save_json(out / "status.json", {"phase":"finished","status":result["status"]})
                channel.send("result", result)
                return 2
        if args.profile:
            from .real import RealTrials
            from .station import Station

            if args.real:
                from .physical import check_profile_origin
                check_profile_origin(json.loads(args.profile.read_text()))
            station = Station.read(args.profile.resolve())
            if station.target != args.target or (args.arm and args.arm != station.arm):
                raise ValueError("Requested arm or object differs from the measured physical station")
            backend = RealTrials(station)
            if args.check_profile:
                channel.send("result", backend.check())
                return 0
        else:
            backend = OfflineTrials()
        if out is None:
            out = new_run(args.target)
            channel.send("started", {"run": str(out), "source": "real" if args.real else "offline_fixture"})
        if args.profile:
            save_json(out / "station.json", json.loads(args.profile.read_text()))
            save_json(out / "mount_from_color.json", {
                "camera_serial": backend.station.serial, "target_frame": "dual_arm_mount",
                "T_mount_from_color": backend.station.mount_from_color.tolist(),
            })
        physical = None
        propose = channel.propose
        if args.real:
            from .physical import PhysicalTrials, environment
            from research_loop import ResearchLoop
            from task_library import SkillLibrary

            library = SkillLibrary(ROOT / ".local/autonomous_library")
            physical = PhysicalTrials(library, environment(args.profile, backend.station), backend.station)
            if args.interactive and not (args.verify_campaign or args.program):
                research = ResearchLoop(library, out, physical.task, physical.contract, physical.env,
                                        channel.research, lambda kind, req: channel.propose(req))
                propose = lambda req: research.propose("cube", req)
        campaign = GraspCampaign(out, backend, propose, channel.cancelled, physical=physical,
                                 notify=lambda data: channel.send("progress", data),
                                 rounds=args.rounds, layouts=[0] if args.real else [0, 1],
                                 verification=[] if args.real else [2, 3])
        cached = (physical.library.lookup(physical.contract, physical.env)
                  if physical and not args.verify_campaign else None)
        if physical and args.program and (not cached or args.program.resolve() != Path(cached["path"]).resolve()):
            raise ValueError("Physical skill reuse requires a revision validated in the autonomous library")
        if out.joinpath("grasp-request.json").is_file() and not args.verify_campaign:
            from .free_grasp import requested_program, finish_and_verify
            program = requested_program(json.loads((out/"grasp-request.json").read_text()), backend.station)
            result = finish_and_verify(campaign,physical,program,out/"free-grasp")
        elif args.verify_campaign:
            result = physical.verify_next(campaign, args.verify_campaign)
        elif args.program or cached:
            program = Program.read(args.program or Path(cached["path"]), backend.grasps)
            result = campaign.trial(program, "run-skill", 0)
            if cached and not result["success"]:
                # A failed deployed revision is evidence for the next proposal.
                result = campaign.train()
            elif cached:
                result["route"] = "reused_physical_skill"
        else:
            result = campaign.train()
        save_json(out / "result.json", result)
        save_json(out / "status.json", {"phase":"finished", "status":result.get("status", result.get("reason")),
                                        "result":str(out / "result.json")})
        if research:
            research.finish({**result, "status": result.get("status", result.get("reason"))})
        channel.send("result", {"run": str(out), **result})
        return 0 if result.get("status") in ("validated", "awaiting_new_placement") or result.get("success") else 2
    except (ValueError, TypeError, KeyError, RuntimeError, OSError, KeyboardInterrupt) as exc:
        error = {"status": "aborted", "type": type(exc).__name__, "error": str(exc),
                 "run": None if out is None else str(out)}
        if out is not None:
            save_json(out / "aborted.json", error)
            save_json(out / "status.json", {"phase":"interrupted", **error})
        if research:
            research.finish(error)
        channel.send("error", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
