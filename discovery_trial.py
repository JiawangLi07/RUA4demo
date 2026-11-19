"""Collect observations for an offline hypothesis without a task success oracle."""
import json
import sys
from pathlib import Path

from pyrualean._backend import Ledger
from pyrualean.arms import ArmToolkit
from cells_robot import StationRobot
from task_trials import ObservedSimBackend, initial_state, validate_source
from task_library import atomic_json


def observe_candidate(request, out):
    source = request["source"]
    validate_source(source)
    archived = request.get("archive")
    if archived:
        from rgbd import archived_observation
        from robot import SnapshotBackend
        snapshot, joints, transform = archived_observation(Path(archived["run"]), archived["frame_id"])
        backend = SnapshotBackend(snapshot, joints, transform, out)
    else:
        snapshot, joints = initial_state()
        backend = ObservedSimBackend(snapshot, joints, None, out)
    robot = StationRobot(backend, ledger=Ledger())
    toolkit = ArmToolkit(robot, arm="cells", workspace=out/"workspace", max_turns=1)
    execution = toolkit.execute_tool("python", {
        "code": source + f"\nrun(robo, {request['contract']!r})\n"
    }).result
    robot._ledger.write_jsonl(out/"calls.jsonl")
    error = execution.get("exception") or execution.get("error")
    return {"success": False, "reason": "candidate_error" if error else "goal_unverified",
            "source": snapshot.source if archived else "procedural_scene", "archive": archived,
            "execution": execution,
            "sandbox_flags": toolkit.sandbox_flags, "case": request["case"]}


if __name__ == "__main__":
    path = Path(sys.argv[1])
    atomic_json(path.parent/"result.json", observe_candidate(json.loads(path.read_text()), path.parent))
