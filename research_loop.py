"""Local researcher/engineer orchestration over the offline PyRUA task router."""
import json

from task_library import atomic_json, skill_key
from experience import gather
from physical_evidence import recent


class ResearchStop(RuntimeError):
    pass


def trial_notes(rows):
    fields = ("success", "reason", "case", "object_lift_m", "contact", "run", "run_dir", "before_m", "lift_samples_m", "measured_actions", "sandbox_flags", "error")
    notes = []
    for row in rows[-6:]:
        note = {key: row[key] for key in fields if key in row}
        execution = row.get("execution", {})
        if execution:
            note["feedback"] = {k: str(execution[k])[:1200] for k in
                                ("stdout", "value", "exception") if execution.get(k)}
        notes.append(note)
    return notes


class ResearchLoop:
    def __init__(self, library, out, task, contract, environment, researcher, engineer):
        key = skill_key(contract, environment)
        self.directory = library.root / "research" / key
        self.out = out
        self.task = task
        self.contract = contract
        self.researcher = researcher
        self.engineer = engineer
        self.rounds = []
        self.facts, memories, self.skills = gather(library, environment, contract)
        self.history = [m["report"] for m in memories]
        self.references = [m["path"] for m in memories]
        self.environment = environment
        self.library = library

    def propose(self, kind, request):
        evidence = trial_notes(request.get("evidence", []))
        physical = recent(self.library.root, self.task, autonomous=True)
        if isinstance(self.environment, dict) and self.environment.get("backend") in ("physical-cube", "physical-shuttlecock"):
            physical = [r for r in physical if r.get("environment") == self.environment]
        brief = {"physical_trials": physical,"task": self.task, "contract": self.contract, "prior_experiments": self.history,
                 "current_experiments": self.rounds, "latest_observations": evidence,
                 "environment_facts": self.facts, "experience_paths": self.references,
                 "available_implementations": [{"id": k, "contract": v["contract"], "revision": v["revision"]}
                                               for k,v in self.skills.items()],
                 "survey": request.get("survey", self.contract),
                 "previous_program": request.get("previous", request.get("previous_source")),
                 "round": request["round"], "draft":request.get("draft", 0),
                 "validation_error":request.get("validation_error")}
        plan = self.researcher(brief)
        if (not isinstance(plan, dict) or plan.get("action") not in ("experiment", "stop")
                or not all(isinstance(plan.get(k), str) and plan[k].strip()
                           for k in ("hypothesis", "instruction", "expected_observation"))):
            raise ValueError("Invalid researcher decision")
        row = {"round": request["round"], "researcher": plan, "evidence": evidence,
               "physical_trial_paths": [p["run"] for p in physical]}
        self.rounds.append(row)
        atomic_json(self.out / "research_rounds.json", self.rounds)
        if plan["action"] == "stop":
            raise ResearchStop(plan["instruction"])
        selected = plan.get("skill_references", [])
        if not isinstance(selected, list) or any(k not in self.skills for k in selected):
            raise ValueError("Researcher selected an unknown implementation")
        implementations = [{k:v for k,v in self.skills[key].items() if k in
                            ("revision","source","contract","path")} for key in dict.fromkeys(selected)]
        candidate = self.engineer(kind, {**request, "research_plan": plan, "physical_trials": physical,
                                         "implementations": implementations})
        row["engineer_reason"] = candidate.get("reason", "")
        atomic_json(self.out / "research_rounds.json", self.rounds)
        return candidate

    def finish(self, result):
        report = {"task": self.task, "contract": self.contract, "run": str(self.out),
                  "status": result["status"], "reason": result.get("reason", result.get("error")),
                  "environment": self.environment, "experience_paths": self.references,
                  "rounds": self.rounds, "verification": trial_notes(result.get("trials", []))}
        atomic_json(self.out / "research_report.json", report)
        atomic_json(self.directory / f"{self.out.name}.json", report)
