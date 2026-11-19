"""Host-side evidence shared between Lean Cells and the research loop."""
from collections import Counter
import json
from pathlib import Path

from task_library import alias_key, atomic_json, digest


def read_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def put_record(library, task, record):
    if not task.strip():
        return
    section = "own_evidence" if record.get("input_policy") == "own_trials_v1" else "physical_evidence"
    folder = Path(library) / section / digest(alias_key(task))
    atomic_json(folder / (digest(record["run"]) + ".json"), record)


def recent(library, task, *, autonomous=False):
    folder = Path(library) / ("own_evidence" if autonomous else "physical_evidence") / digest(alias_key(task))
    paths = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return [json.loads(p.read_text()) for p in paths[:5]]


def cells_record(out, library, goal=None):
    observation = json.loads((out / "observation.json").read_text())
    result = json.loads((out / "result.json").read_text())
    if observation["source"] != "live" or result.get("controller") == "SimMotionBackend":
        return None
    calls = read_rows(out / "calls.jsonl")
    log = read_rows(out / "arm_step.jsonl") + read_rows(out / "arm_trajectory.jsonl")
    modes = {"target", "path"}
    idle = None
    commanded = False
    verified = []
    for row in log:
        if row["event"] in modes:
            commanded, idle = True, None
        elif row["event"] == "idle_confirmed" and commanded:
            idle = row.get("arm")
        elif row["event"] == "verified" and idle in ("A", "B"):
            verified.append({**{k:row[k] for k in ("frame_id","joint_deg") if k in row},
                             "arm": {"A":"left","B":"right"}[idle]})
            commanded, idle = False, None
    reads = [{k:c.get(k) for k in ("tool","kwargs","summary","ok","error")} for c in calls
             if c["tool"] in ("locate","locate_mask","rank_labels","segment","preview_reach","tactile")]
    task = result.get("task") or ""
    record = {"input_policy": result.get("input_policy","unverified"), "source": "live_cells", "task": task, "run": str(out.resolve()),
              "camera_serial": observation["camera_serial"], "frame_id": observation["frame_id"],
              "controller": result.get("controller","legacy_unspecified"),
              "tools": dict(Counter(c["tool"] for c in calls)), "perception": reads[-6:],
              "measured_arm_moves": verified[-6:], "motion_pending_verification": commanded,
              "model_finish": result.get("finish"), "object_verdict": "unverified"}
    atomic_json(out / "physical_evidence.json", record)
    put_record(library, goal or task, record)
    return record


if __name__ == "__main__":
    from argparse import ArgumentParser
    root = Path(__file__).resolve().parent
    parser = ArgumentParser(description="Import a completed hardware Cells record without opening devices")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("run", nargs="?", type=Path)
    selection.add_argument("--history", help="Read the latest physical trials associated with a task")
    parser.add_argument("--goal", help="Associate this observation with an explicit parent task")
    parser.add_argument("--library", type=Path, default=root / ".local/autonomous_library")
    args = parser.parse_args()
    record = recent(args.library, args.history, autonomous=True) if args.history else cells_record(args.run, args.library, args.goal)
    print(json.dumps(record, ensure_ascii=False, indent=2))
