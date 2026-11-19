"""Record measured cube outcomes and validate frozen code on new physical placements."""
from dataclasses import asdict
from hashlib import sha256
import fcntl
import json
import math
from pathlib import Path

from physical_evidence import put_record, read_rows
from task_library import atomic_json, atomic_text, digest
from .grasp_program import MIN_OBJECT_LIFT_M, Program, verdict

# Separate planar positions by at least the existing station staging tolerance.
PLACEMENT_SEPARATION_M = 0.025


def check_profile_origin(settings):
    controls = settings.get("input_sources", {})
    legacy = settings.get("provenance", {})
    if (controls.get("initial_pose") not in ("station_measurement", "live_feedback")
            or any(k in legacy for k in ("demonstration_origin","recording","hand_grasp_row","arm_row"))):
        raise ValueError("Autonomous trials need an independently measured start and no demonstration inputs")
    if controls.get("hand_targets") not in ("hardware_primitive", "own_trial"):
        raise ValueError("Missing hand target with hardware_primitive or own_trial provenance")


def environment(profile, station):
    settings = json.loads(profile.read_text())
    for key in ("calibration", "cube_m", "target_m", "provenance"):
        settings.pop(key, None)
    from hardware import ROOT
    files = ["exploration/clearance.py", "config/station_geometry.xml", "exploration/real.py", "exploration/station.py", "exploration/grasp_program.py",
             "exploration/physical.py", "rgbd.py", "hardware.py"]
    if station.target == "shuttlecock":
        files += ["exploration/shuttlecock.py", "upstream/src/pyrualean/hosts/sam3_http.py"]
    return {"input_policy":"own_trials_v1", "backend": "physical-" + station.target, "station": digest([settings, station.mount_from_color.tolist()]),
            "controller": digest([sha256((ROOT / name).read_bytes()).hexdigest() for name in files]),
            "camera_serial": station.serial, "arm": station.arm}


def finite_vector(point, length=3):
    return isinstance(point, (list, tuple)) and len(point) == length and all(
        type(x) in (float, int) and math.isfinite(x) for x in point)


def independent_measurement(out, axis, target="cube"):
    events = read_rows(out / "trajectory.jsonl")
    phases = {row["phase"]: (i, row) for i, row in enumerate(events) if row["event"] == "observe"}
    actions = {name: sum(r["event"] == name for r in events)
               for name in ("arm_waypoint", "hand_waypoint", "idle_confirmed")}
    measured = {"success": False, "reason": "incomplete_sensor_trace", "measured_actions": actions}
    if not all(key in phases for key in ("before", "closed", "hold_0", "hold_1", "reset")):
        return measured
    order = [phases[k] for k in ("before","closed","hold_0","hold_1","reset")]
    if any(row.get("target", "cube") != target for _, row in order):
        return {**measured, "reason":"wrong_target_trace"}
    if target == "shuttlecock" and any(
        row.get("detector", {}).get("reason") != "surface_measured"
        or not (out / f"{row['phase']}_target.png").is_file()
        for _,row in order if row["phase"] != "closed"
    ):
        return {**measured, "reason":"shuttlecock_unobservable"}
    indices = [i for i, _ in order]
    stamps = [r["sample_monotonic_ns"] for _, r in order]
    if (indices != sorted(indices) or not all(type(s) is int and s > 0 for s in stamps)
            or any(b <= a for a, b in zip(stamps, stamps[1:]))):
        return {**measured, "reason": "stale_or_unordered_samples"}
    for stream in ("color_sequence","depth_sequence"):
        sequence = [row.get(stream) for _,row in order]
        if not all(type(n) is int for n in sequence) or any(b <= a for a,b in zip(sequence,sequence[1:])):
            return {**measured, "reason": "stale_or_unordered_samples"}
    if stamps[3] - stamps[2] < 250_000_000:
        return {**measured, "reason": "hold_interval_too_short"}
    before_i, closed_i, hold_i, _, reset_i = indices
    if (not any(r["event"] == "hand_waypoint" and finite_vector(r.get("measured_rad"), 22)
                and finite_vector(r.get("torque"), 22) for r in events[before_i+1:closed_i])
            or not any(r["event"] == "arm_waypoint" and finite_vector(r.get("measured_deg"), 7)
                       for r in events[closed_i+1:hold_i])
            or not any(r["event"] == "idle_confirmed" for r in events[reset_i+1:])):
        return measured
    def position(row):
        return row.get("target_m", row.get("cube_m") if target == "cube" else None)
    before = position(order[0][1])
    lifted = [position(r) for _, r in order[2:4]]
    if not all(finite_vector(p) for p in [before, *lifted]):
        return {**measured, "reason": target + "_unobservable"}
    reset = position(order[-1][1])
    from .real import REPEAT_CUBE_TOLERANCE_M
    if not finite_vector(reset) or math.dist(reset, before) > REPEAT_CUBE_TOLERANCE_M:
        return {**measured, "reason": "reset_unverified"}
    outcome = verdict(before, lifted, axis)
    if target == "shuttlecock" and outcome["success"]:
        observations = [order[i][1]["detector"] for i in (0,2,3)]
        heights = [d.get("low_height_m") for d in observations]
        areas = [d.get("pixels",0) for d in observations]
        if (not all(type(h) in (int,float) and math.isfinite(h) for h in heights)
                or min(areas[1:]) < .5 * areas[0] or areas[0] <= 0):
            outcome.update(success=False, reason="shuttlecock_visibility_insufficient")
        else:
            clearance = [h-heights[0] for h in heights[1:]]
            outcome["lower_surface_lift_m"] = clearance
            if min(clearance) < MIN_OBJECT_LIFT_M:
                outcome.update(success=False, reason="shuttlecock_not_clear_of_table")
    outcome["reason"] = outcome["reason"].replace("cube_", target + "_", 1)
    return {**measured, **outcome, "before_m": before, "lift_samples_m": lifted}


def new_placement(point, previous, axis):
    if not finite_vector(point) or not previous or not all(finite_vector(p) for p in previous):
        return False
    for old in previous:
        diff = [x-y for x, y in zip(point, old)]
        height = sum(d*a for d, a in zip(diff, axis))
        if math.sqrt(sum((d-height*a)**2 for d, a in zip(diff, axis))) < PLACEMENT_SEPARATION_M:
            return False
    return True


class PlacementNotChanged(ValueError):
    """No motion was issued because the same placement is still staged."""


class PhysicalTrials:
    def __init__(self, library, env, station):
        self.library, self.env, self.station = library, env, station
        self.target = getattr(station, "target", "cube")
        label = "red cube" if self.target == "cube" else self.target
        self.task = f"{station.arm} hand stable lift of a {label}"
        self.contract = {"kind":"grasp_" + self.target, "arm":station.arm, "object":self.target, "values":[]}
        self.skill = self.contract["kind"] + ".py"

    def record(self, out, result):
        measured = independent_measurement(out, self.station.up, self.target)
        if self.station.contact_policy is not None:
            contact = [r for r in read_rows(out/"trajectory.jsonl") if r.get("event")=="contact_verdict"]
            if not contact or contact[-1].get("confirmed") is not True:
                measured.update(success=False, reason="contact_unconfirmed")
        if result.get("reset_ok") is not True:
            measured.update(success=False, reason="reset_unverified")
        elif not measured["success"] and measured["reason"] == "incomplete_sensor_trace":
            measured["controller_reason"] = result["reason"]
        result.update(measured)
        record = {"input_policy":"own_trials_v1", "task": self.task, "run": str(out.resolve()), "source": f"real_{self.target}_controller",
                  "environment": self.env, "object_verdict": result["reason"],
                  "program": result["program"], "program_sha256": sha256((out/self.skill).read_bytes()).hexdigest(),
                  **{k:result.get(k) for k in ("success","before_m","lift_samples_m","measured_actions","contact","controller_reason")}}
        atomic_json(out / "physical_evidence.json", record)
        put_record(self.library.root, self.task, record)

    def record_interruption(self, out, exc):
        record = {"input_policy":"own_trials_v1", "task": self.task, "run": str(out.resolve()), "source": f"real_{self.target}_controller",
                  "environment": self.env, "object_verdict": "interrupted", "error": str(exc),
                  "program_sha256": sha256((out / self.skill).read_bytes()).hexdigest(),
                  "program": asdict(Program.read(out / self.skill, self.station.hand_targets)),
                  "trajectory": str(out / "trajectory.jsonl")}
        rows = read_rows(out / "trajectory.jsonl")
        record["measured_actions"] = {event:sum(row["event"] == event for row in rows)
                                      for event in ("arm_waypoint","hand_waypoint","idle_confirmed")}
        atomic_json(out / "physical_evidence.json", record)
        put_record(self.library.root, self.task, record)

    def freeze(self, campaign, program, report, state):
        source = program.source()
        path = campaign.root / "pending" / self.skill
        atomic_text(path, source)
        development = [json.loads((Path(run)/"result.json").read_text())
                       for row in state["rounds"] for run in row["runs"]]
        accepted = next(r for r in reversed(development) if r["success"] and
                        (Path(r["run_dir"])/self.skill).read_text() == source)
        report.update(status="awaiting_new_placement", candidate=str(path), environment=self.env,
                      program_sha256=sha256(source.encode()).hexdigest(),
                      development_positions=[r["before_m"] for r in development if finite_vector(r.get("before_m"))],
                      trials=[{**accepted, "case":0}])

    def verify_next(self, campaign, parent):
        parent = Path(parent).resolve()
        state_path = parent / "result.json"
        with (parent / ".verification.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = json.loads(state_path.read_text())
            if state["status"] == "validated" and state.get("environment") == self.env:
                return state
            if (state["status"] not in ("awaiting_new_placement", "verification_in_progress")
                    or state.get("environment") != self.env):
                raise ValueError("Verification requires a pending campaign on the same calibrated rig")
            path = parent / "pending" / self.skill
            program = Program.read(path, campaign.backend.grasps)
            if sha256(path.read_bytes()).hexdigest() != state["program_sha256"]:
                raise ValueError("Frozen candidate changed")
            prior = state["development_positions"] + [r["before_m"] for r in state["trials"][1:]]
            case = len(state["trials"])
            if state["status"] == "verification_in_progress":
                # Recover a completed trial after a host exit without repeating any motion.
                directory = Path(state["verification_run"]) / "trials" / f"held-out-{case}"
                if (not (directory/"result.json").is_file()
                        or sha256((directory/self.skill).read_bytes()).hexdigest() != state["program_sha256"]):
                    raise ValueError(f"Incomplete verification. Inspect the controller log in {directory}")
                trial = json.loads((directory/"result.json").read_text())
            else:
                if self.station.cube_m is not None:
                    self.require_placement(self.station.cube_m.tolist(), prior)
                campaign.backend.check_placement = lambda measured: self.require_placement(measured.tolist(), prior)
                state.update(status="verification_in_progress", verification_run=str(campaign.root))
                atomic_json(state_path, state)
                try:
                    trial = campaign.trial(program, f"held-out-{case}", case)
                except PlacementNotChanged:
                    state["status"] = "awaiting_new_placement"
                    atomic_json(state_path, state)
                    raise
                except BaseException:
                    state["status"] = "verification_interrupted"
                    atomic_json(state_path, state)
                    raise
            if trial.get("success") and not new_placement(trial.get("before_m"), prior, self.station.up):
                trial = {**trial, "success":False, "reason":"placement_repeated"}
            state["trials"].append({**trial, "case":case})
            if not trial["success"]:
                state["status"] = "verification_failed"
            elif case == 2:
                entry = self.library.publish(self.task, self.contract, self.env, program.source(),
                                             state["trials"], str(parent))
                state.update(status="validated", skill=entry["path"])
            else:
                state["status"] = "awaiting_new_placement"
            atomic_json(state_path, state)
            return state

    def require_placement(self, point, prior):
        if not new_placement(point, prior, self.station.up):
            raise PlacementNotChanged("A distinct measured tabletop position at least 25 mm from prior trials is required")
