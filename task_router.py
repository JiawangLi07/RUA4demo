"""Look up a skill, execute and check it, or explore and publish a revision."""

from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys

from exploration.campaign import GraspCampaign
from exploration.grasp_program import Program
from exploration.offline import OfflineTrials
from hardware import CONFIG, ROOT
from perception_backend import capabilities
from physical_evidence import recent
from research_loop import ResearchLoop, ResearchStop
from task_library import SkillLibrary, atomic_json, digest
from task_trials import contract_from, describe, validate_source
from skill_composition import compose


def execution_environment():
    # A virtual-joint success cannot match a hardware execution environment.
    files = [ROOT / name for name in (
        "cells_robot.py", "robot.py", "perception_backend.py", "perception/worker.py", "task_trials.py", "autonomous_scene.py",
        "exploration/offline.py", "exploration/grasp_program.py",
    )]
    files += [CONFIG / "arm/tianji.yaml", CONFIG / "end_effector/sharpa.yaml"]
    return {"backend": "autonomous-procedural-v1", "validator": "procedural-goals-v1",
            "input_policy": "own_trials_v1",
            "perception": capabilities(),
            "dependencies": digest([sha256(p.read_bytes()).hexdigest() for p in files])}


class TaskRouter:
    def __init__(self, library, out, *, resolve, propose, rounds=3, legacy_root=ROOT,
                 trial_timeout=120, researcher=None):
        self.library = SkillLibrary(library)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.resolve = resolve
        self.propose = propose
        self.rounds = rounds
        self.legacy_root = Path(legacy_root)
        self.trial_timeout = trial_timeout
        self.environment = execution_environment()
        self.evidence = []
        self.archive = None
        self.context = None
        self.researcher = researcher
        self.research = None

    def trial(self, source, contract, name, case=0, *, discovery=False):
        out = self.out / name
        out.mkdir()
        request = {"source": source, "contract": contract, "case": case}
        if discovery and self.archive:
            request["archive"] = self.archive
        atomic_json(out / "request.json", request)
        try:
            with (out / "worker.log").open("w") as log:
                completed = subprocess.run(
                    [sys.executable, str(ROOT / ("discovery_trial.py" if discovery else "task_trials.py")), str(out / "request.json")],
                    stdout=log, stderr=subprocess.STDOUT, timeout=self.trial_timeout,
                    check=False,
                )
            path = out / "result.json"
            if completed.returncode or not path.is_file():
                result = {"success": False, "reason": "worker_failed",
                          "returncode": completed.returncode}
            else:
                result = json.loads(path.read_text())
        except subprocess.TimeoutExpired:
            result = {"success": False, "reason": "worker_timeout"}
        result = {**result, "run": str(out)}
        atomic_json(out / "result.json", result)
        self.evidence.append(result)
        return result

    def validate(self, source, contract, prefix):
        results = []
        for case in range(3):
            result = self.trial(source, contract, f"{prefix}-check-{case}", case)
            results.append(result)
            if not result["success"] or result.get("reset_ok") is False:
                break
        return results

    def legacy_skill(self, contract):
        return self.library.previous_candidate(contract, self.environment)

    def result(self, task, contract, route, status, **fields):
        report = {"task": task, "contract": contract, "route": route, "status": status,
                  "environment": self.environment, "run": str(self.out),
                  "trials": self.evidence, **fields}
        if self.research:
            self.research.finish(report)
            report["research_report"] = str(self.out / "research_report.json")
        atomic_json(self.out / "result.json", report)
        return report

    def run(self, task, contract=None):
        try:
            value = contract or self.library.resolve_alias(task, self.environment)
            if value is None:
                value = self.resolve(task)
            steps = value.get("steps", [value]) if isinstance(value, dict) else []
            missing = [s.get("reason", "Missing goal verifier") for s in steps
                       if isinstance(s, dict) and s.get("kind") == "unsupported"] if isinstance(steps, list) else []
            if self.researcher and missing:
                return self.investigate(task, missing)
            contract = contract_from(value)
        except ValueError as exc:
            return self.result(task, None, "blocked", "blocked", reason=str(exc))
        atomic_json(self.out / "task.json", contract)

        try:
            cached = self.library.lookup(contract, self.environment)
        except (OSError, ValueError, KeyError) as exc:
            cached = None
            self.evidence.append({"success": False, "reason": "library_entry_invalid",
                                  "error": str(exc)})
        previous = None
        if cached:
            previous = cached["source"]
            check = self.trial(previous, contract, "reuse")
            if check["success"]:
                self.library.remember(task, contract, self.environment)
                return self.result(task, contract, "reused", "success",
                                   skill=cached["path"], revision=cached["revision"])

        legacy = self.legacy_skill(contract) if cached is None else None
        if legacy:
            previous = legacy.read_text()
            checks = self.validate(previous, contract, "import")
            if len(checks) == 3 and all(row["success"] for row in checks):
                entry = self.library.publish(task, contract, self.environment, previous,
                                             checks, str(legacy))
                return self.result(task, contract, "imported", "success",
                                   skill=entry["path"], revision=entry["revision"])

        if contract["kind"] == "sequence":
            return self.compose_task(task, contract)

        if self.researcher:
            self.research = ResearchLoop(self.library, self.out, task, contract, self.environment,
                                         self.researcher, self.propose)
        try:
            if contract["kind"] == "grasp_cube":
                source, checks = self.explore_cube(previous)
            else:
                source, checks = self.explore_cells(task, contract, previous)
        except ResearchStop as exc:
            return self.result(task, contract, "explored", "blocked", reason=str(exc))
        if source is None:
            return self.result(task, contract, "explored", "failed",
                               reason="No candidate passed the independent goal checks")
        entry = self.library.publish(task, contract, self.environment, source, checks, str(self.out))
        return self.result(task, contract, "learned", "success", skill=entry["path"],
                           revision=entry["revision"])

    def investigate(self, task, missing):
        # Scope empirical notes to this exact unsupported goal, not all unknown tasks.
        goal = {"kind": "investigate", "arm": "none", "object": task, "values": [],
                "missing_verifiers": missing}
        self.research = ResearchLoop(self.library, self.out, task, goal, self.environment,
                                     self.researcher, self.propose)
        for record in recent(self.library.root, task, autonomous=True):
            if (record["source"] == "live_cells" and
                    all((Path(record["run"])/f).is_file() for f in ("head.png","head_depth_native.png","observation.json"))):
                self.archive = {k:record[k] for k in ("run","frame_id")}
                break
        context = describe(self.archive)
        previous = None
        reason = "; ".join(missing)
        for index in range(self.rounds):
            request = {"task": task, "contract": goal, "round": index, "context": context,
                       "previous_source": previous, "evidence": self.evidence}
            try:
                reply = self.research.propose("discovery", request)
                previous = reply["source"]
                validate_source(previous)
            except ResearchStop as exc:
                reason = str(exc)
                break
            except (SyntaxError, ValueError, KeyError, TypeError) as exc:
                self.evidence.append({"success": False, "reason": "invalid_proposal", "error": str(exc)})
                continue
            (self.out / f"investigation-{index}.py").write_text(previous)
            self.trial(previous, goal, f"investigation-{index}", discovery=True)
        return self.result(task, goal, "investigated", "blocked", reason=reason)

    def compose_task(self, task, contract):
        entries, dependencies = [], []
        for i, step in enumerate(contract["steps"]):
            child = TaskRouter(self.library.root, self.out / f"step-{i}", resolve=self.resolve,
                               propose=self.propose, rounds=self.rounds, legacy_root=self.legacy_root,
                               trial_timeout=self.trial_timeout, researcher=self.researcher)
            label = f"{step['kind']}({step['arm']}, {step['object']}, {step['values']})"
            report = child.run(label, step)
            dependencies.append({key: report.get(key) for key in ("route", "status", "run", "revision")})
            if report["status"] != "success":
                return self.result(task, contract, "composition", report["status"],
                                   reason=f"Step {i+1}: {report.get('reason')}", dependencies=dependencies)
            # Use exactly the child artifact verified by this invocation, even
            # when a concurrent run promotes another revision of the same skill.
            source = Path(report["skill"]).read_text()
            validate_source(source)
            entries.append(({"source": source, "revision": report["revision"]}, None))
        # A verified component can still fail after another component changes the scene.
        # Refine that component within this sequence; keep its published atomic skill intact.
        for attempt in range(self.rounds):
            source = compose(entries)
            checks = self.validate(source, contract, f"composed-{attempt}")
            if len(checks) == 3 and all(row["success"] for row in checks):
                break
            failure = checks[-1]
            stages = failure.get("steps", [])
            if attempt + 1 == self.rounds or not stages:
                return self.result(task, contract, "composition", "failed",
                                   reason=failure["reason"], dependencies=dependencies)
            i = len(stages) - 1
            step = contract["steps"][i]
            feedback = stages[-1]
            if step["kind"] == "joint_delta":
                start = feedback["before"]["arm_deg"][step["arm"]]
                target = [q + d for q, d in zip(start, feedback["contract"]["values"])]
                from hardware import arm_settings
                if any(not j.min_deg <= v <= j.max_deg for v, j in zip(target, arm_settings().joints)):
                    return self.result(task, contract, "composition", "failed",
                                       reason=f"Step {i+1} exceeds joint limits in the shared state",
                                       dependencies=dependencies)
            request = {"task": task, "contract": step, "round": attempt+1,
                       "previous_source": entries[i][0]["source"], "context": describe(),
                       "evidence": [feedback], "sequence_step": i, "sequence": contract}
            if self.researcher and self.research is None:
                self.research = ResearchLoop(self.library, self.out, task, contract, self.environment,
                                             self.researcher, self.propose)
            try:
                reply = (self.research.propose if self.research else self.propose)("cells", request)
                revised = reply["source"]
                validate_source(revised)
            except ResearchStop as exc:
                return self.result(task, contract, "composition", "blocked", reason=str(exc),
                                   dependencies=dependencies)
            except (KeyError, TypeError, SyntaxError, ValueError) as exc:
                return self.result(task, contract, "composition", "failed", reason=str(exc),
                                   dependencies=dependencies)
            revision = sha256(revised.encode()).hexdigest()
            entries[i] = ({"source": revised, "revision": revision}, None)
            dependencies[i]["embedded_revision"] = revision
        entry = self.library.publish(task, contract, self.environment, source, checks, str(self.out))
        return self.result(task, contract, "composed", "success", dependencies=dependencies,
                           skill=entry["path"], revision=entry["revision"])

    def explore_cube(self, previous):
        def propose(request):
            if previous is not None and request["round"] == 0:
                path = self.out / "previous_cube.py"
                path.write_text(previous)
                try:
                    plan = Program.read(path, OfflineTrials.grasps)
                    request = {**request, "previous": {
                        "offset_m": list(plan.offset_m), "lift_m": plan.lift_m,
                        "grasp": plan.grasp, "recheck": plan.recheck,
                    }, "evidence": self.evidence}
                except (ValueError, OSError):
                    pass
            return (self.research.propose if self.research else self.propose)("cube", request)

        learning = self.out / "learning"
        learning.mkdir()
        campaign = GraspCampaign(learning, OfflineTrials(), propose, lambda: False,
                                 rounds=self.rounds, layouts=[0, 1], verification=[2, 3])
        report = campaign.train()
        if report["status"] != "validated":
            return None, []
        source = Path(report["skill"]).read_text()
        checks = self.validate(source, {"kind": "grasp_cube", "arm": "right",
                                        "object": "cube", "values": []}, "candidate")
        if len(checks) != 3 or not all(row["success"] for row in checks):
            return None, checks
        return source, checks

    def explore_cells(self, task, contract, previous):
        self.context = describe()
        for index in range(self.rounds):
            request = {"task": task, "contract": contract, "round": index,
                       "previous_source": previous, "evidence": self.evidence,
                       "context": self.context}
            reply = (self.research.propose if self.research else self.propose)("cells", request)
            try:
                source = reply["source"]
                validate_source(source)
                (self.out / f"candidate-{index}.py").write_text(source)
                checks = self.validate(source, contract, f"candidate-{index}")
            except (KeyError, TypeError, ValueError, SyntaxError) as exc:
                self.evidence.append({"success": False, "reason": "invalid_proposal",
                                      "error": str(exc)})
                continue
            if len(checks) == 3 and all(row["success"] for row in checks):
                return source, checks
            previous = source
        return None, []
