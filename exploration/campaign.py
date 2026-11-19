"""Adapt RSIJudge's campaign to serial trials on a single grasp station."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from dataclasses import asdict
from pathlib import Path
import threading

from pyrualean.learn.campaign import Campaign

from .grasp_program import LIFT_RANGE_M, OFFSET_LIMIT_M, Program


TASK = "grasp_cube"
SKILL = "grasp_cube.py"


def save_json(path: Path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


class GraspCampaign(Campaign):
    """Use upstream learn, library, rollback and round persistence."""

    def __init__(self, root, backend, propose, cancelled, *, rounds, layouts, verification,
                 physical=None, notify=None, single_cell=False):
        # Upstream configuration and runners are tied to RoboDojo/RoboCasa clients.
        self.root = root
        self.config = {
            "rounds": rounds,
            "dev": {"layouts": layouts},
            "learning": {"rollback_if_worse_by": 1},
        }
        self.log_lock = threading.Lock()
        self.backend = backend
        self.physical = physical
        self.notify = notify
        self.single_cell = single_cell
        self.task = physical.contract["kind"] if physical else TASK
        self.skill = self.task + ".py"
        self.propose = propose
        self.cancelled = cancelled
        self.verification = verification
        self.max_rounds = rounds
        self.baseline = root / "zero_shot.py"

    def _proposal(self, round_index, previous, evidence, history, tried=()):
        request = {
            "round": round_index,
            "task": self.physical.task if self.physical else "Pick up one cube and hold it stably above its initial position.",
            "grasp_names": list(self.backend.grasps),
            "offset_limit_m": OFFSET_LIMIT_M,
            "lift_range_m": LIFT_RANGE_M,
            "previous": previous,
            "proposal_format": "python" if self.physical else "parameters",
            "evidence": evidence,
            "history": history,
        }
        if self.single_cell:
            request["single_cell"] = True
        if self.physical and evidence:
            from PIL import Image
            trial_dir = Path(evidence[-1]["run_dir"])
            frames = {}
            for phase in ("before", "hold_1"):
                image_path = trial_dir / f"{phase}.png"
                if image_path.is_file():
                    with Image.open(image_path) as image:
                        image.thumbnail((320,240))
                        encoded = BytesIO()
                        image.convert("RGB").save(encoded, format="JPEG", quality=65)
                        frames[phase] = base64.b64encode(encoded.getvalue()).decode("ascii")
            request["own_trial_images"] = frames
        for draft in range(2 if self.physical and not self.single_cell else 1):
            if self.cancelled():
                raise RuntimeError("Exploration client disconnected")
            tag = f"proposal-{round_index}" + (f"-repair-{draft}" if draft else "")
            request["draft"] = draft
            save_json(self.root / f"{tag}-request.json", request)
            reply = self.propose(request)
            save_json(self.root / f"{tag}-reply.json", reply)
            try:
                field = "source" if self.physical else "program"
                if (not isinstance(reply, dict) or set(reply) != {field, "reason"}
                        or not isinstance(reply["reason"], str)):
                    raise ValueError(f"Learner must return {field} and a reason")
                if self.physical:
                    program = Program.from_proposal(reply[field], self.backend.grasps,
                                                    single_cell=self.single_cell)
                else:
                    program = Program.parse(reply[field], self.backend.grasps)
                if program in tried:
                    raise ValueError("Learner proposed an already tested program")
                return program, reply["reason"][:400]
            except ValueError as exc:
                if not self.physical or self.single_cell or draft:
                    raise
                request["rejected_proposal"] = reply
                request["validation_error"] = str(exc)

    def _program(self, task, revision):
        path = self.library(task).view(revision, task) / "skills" / self.skill
        if path.is_file():
            return Program.read(path, self.backend.grasps)
        if not self.baseline.exists():
            initial = {"offset_m": [0, 0, 0], "lift_m": 0.065,
                       "grasp": self.backend.grasps[0], "recheck": False}
            program, _ = self._proposal(0, initial, [], [])
            self.baseline.write_text(program.source())
        return Program.read(self.baseline, self.backend.grasps)

    def _improve(self, task, round_index, state):
        rounds = state["rounds"]
        previous = rounds[-1]
        parent = previous["active"]
        history = [{key: row[key] for key in ("round", "revision", "active", "dev")}
                   for row in rounds]
        evidence = [json.loads((Path(path) / "result.json").read_text())
                    for path in previous["runs"]]
        current = self._program(task, parent)
        tried = [self._program(task, row["revision"]) for row in rounds]
        candidate, reason = self._proposal(round_index, asdict(current), evidence, history, tried)
        revision = self._freeze(task, candidate, parent, f"r{round_index}: {reason}")
        return {"round": round_index, "parent": parent, "revision": revision, "reason": reason}

    def _freeze(self, task, program, parent, note):
        candidate = self.root / "candidate" / "tasks" / task / "skills"
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / self.skill).write_text(program.source())
        return self.library(task).freeze(self.root / "candidate", parent=parent, note=note)

    def progress(self, **data):
        if self.physical:
            save_json(self.root / "status.json", data)
            if self.notify:
                self.notify(data)

    def trial(self, program, name, layout):
        out = self.root / "trials" / name
        out.mkdir(parents=True)
        (out / self.skill).write_text(program.source(single_cell=self.single_cell))
        self.progress(phase="executing", trial=str(out), target=self.task)
        try:
            result = self.backend.run(program, out, layout, cancelled=self.cancelled)
        except (ValueError, RuntimeError, OSError, KeyboardInterrupt) as exc:
            save_json(out / "aborted.json", {"error": str(exc), "type": type(exc).__name__})
            if self.physical:
                self.physical.record_interruption(out, exc)
            raise
        result.update(layout=layout, program=asdict(program), run_dir=str(out))
        if self.physical:
            self.physical.record(out, result)
        save_json(out / "result.json", result)
        self.progress(phase="trial_complete", trial=str(out), success=result["success"],
                      reason=result["reason"], measured_actions=result.get("measured_actions", {}))
        if not result["reset_ok"]:
            raise RuntimeError(f"Grasp station reset could not be verified. See {out}")
        return result

    def run_single_cell(self):
        previous = {"offset_m": [0, 0, 0], "lift_m": 0.065,
                    "grasp": self.backend.grasps[0], "recheck": True}
        program, _ = self._proposal(0, previous, [], [])
        return self.trial(program, "single-cell", 0)

    def batch(self, split, task, revision, repeat=0):
        if split != "dev":
            raise ValueError("The grasp campaign runs validation separately")
        program = self._program(task, revision)
        entries = []
        for layout in self.config["dev"]["layouts"]:
            result = self.trial(program, f"r{repeat}-layout{layout}", layout)
            entries.append({"success": result["success"], "run_dir": result["run_dir"],
                            "end": result["reason"], "claim": None})
        return entries

    def train(self):
        state_path = self.root / "tasks" / self.task / "learn.json"
        for rounds in range(1, self.max_rounds + 1):
            self.config["rounds"] = rounds
            active = self.learn(self.task)
            state = json.loads(state_path.read_text())
            latest = state["rounds"][-1]["dev"]
            if latest["successes"] == len(self.config["dev"]["layouts"]):
                break
        report = {"rounds": state["rounds"], "active": active, "status": "not_validated"}
        complete = any(row["revision"] == active and row["dev"]["successes"] ==
                       len(self.config["dev"]["layouts"]) for row in state["rounds"])
        if self.physical and not complete:
            attempted = Path(state["rounds"][-1]["runs"][-1])
            report.update(status="failed", reason="trial_budget_exhausted",
                          candidate=str(attempted / self.skill), last_trial=str(attempted / "result.json"))
        if complete:
            program = self._program(self.task, active)
            if self.physical:
                self.physical.freeze(self, program, report, state)
                save_json(self.root / "result.json", report)
                return report
            results = [self.trial(program, f"verify-{index}", layout)
                       for index, layout in enumerate(self.verification)]
            report["verification"] = results
            if all(result["success"] for result in results):
                library = self.library(self.task)
                if active == self.v0(self.task):
                    active = self._freeze(self.task, program, active, "Validated zero-shot program")
                library.set_active(active)
                library.record({"event": "validated", "revision": active,
                                "runs": [r["run_dir"] for r in results]})
                target = self.root / "validated" / self.skill
                target.parent.mkdir()
                target.write_text(program.source())
                report.update(active=active, status="validated", skill=str(target))
        save_json(self.root / "result.json", report)
        return report
