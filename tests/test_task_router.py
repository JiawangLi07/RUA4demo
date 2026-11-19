"""Task routing regressions: measured validation, reuse, fallback and publication."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from exploration.grasp_program import Program
from exploration.offline import OfflineTrials
from task_library import SkillLibrary
from task_router import TaskRouter
from task_server import scripted_proposal


JOINT = {"kind": "joint_delta", "arm": "right", "object": "none",
         "values": [0, 0, 0, 0, 0, 0, 3]}
CUBE = {"kind": "grasp_cube", "arm": "right", "object": "cube", "values": []}
SOURCE = (
    "def run(robo, params):\n"
    "    obs = robo.observe()\n"
    "    target = [q + d for q, d in zip(obs['joints']['arm_deg'][params['arm']], params['values'])]\n"
    "    robo.move_joints(params['arm'], target)\n"
)


def unexpected(*_args):
    raise AssertionError("A cache hit must not call the resolver or learner")


def router(tmp_path, name, **kwargs):
    return TaskRouter(tmp_path / "library", tmp_path / name,
                      resolve=kwargs.pop("resolve", lambda _task: JOINT),
                      propose=kwargs.pop("propose", scripted_proposal),
                      legacy_root=tmp_path / "legacy", **kwargs)


def test_cold_start_checks_frozen_code_then_reuses_without_model(tmp_path):
    first = router(tmp_path, "first").run("让右臂第七关节正向转动3度")
    assert (first["status"], first["route"]) == ("success", "learned")
    assert len(first["trials"]) == 3
    assert len({tuple(t["before"]["arm_deg"]["right"]) for t in first["trials"]}) == 3
    assert first["trials"][0]["contract"]["values"] != first["trials"][1]["contract"]["values"]
    second = router(tmp_path, "second", resolve=unexpected, propose=unexpected).run(
        "让右臂第七关节正向转动3度")
    assert second["route"] == "reused" and len(second["trials"]) == 1
    assert first["revision"] == second["revision"]


def test_paraphrase_and_new_numeric_goal_reuse_parameterized_skill(tmp_path):
    first = router(tmp_path, "first").run("right joint seven +3")
    second = router(tmp_path, "second", propose=unexpected).run(
        "右臂末关节正转", {**JOINT, "values": [0, 0, 0, 0, 0, 0, 4]})
    assert second["route"] == "reused" and second["revision"] == first["revision"]
    trial = second["trials"][0]
    assert trial["after"]["arm_deg"]["right"][6] - trial["before"]["arm_deg"]["right"][6] == 4


def test_success_claim_cannot_publish_without_goal_feedback(tmp_path):
    run = router(tmp_path, "claims", propose=lambda *_: {
        "source": "def run(robo, params):\n    return {'success': True}\n",
        "reason": "Claim success without any action",
    }, rounds=1)
    result = run.run("right joint +3")
    assert result["status"] == "failed"
    assert run.library.lookup(JOINT, run.environment) is None
    assert run.library.resolve_alias("right joint +3", run.environment) is None


def test_failed_candidate_gets_evidence_and_cannot_replace_active_skill(tmp_path):
    limited = SOURCE.replace("    obs =", "    if abs(params['values'][6]) > 5:\n        return\n    obs =")
    first = router(tmp_path, "limited", propose=lambda *_: {"source": limited})
    original = first.run("small move")
    assert original["status"] == "success"
    goal = {**JOINT, "values": [0, 0, 0, 0, 0, 0, 6]}
    failed = router(tmp_path, "failed", propose=lambda *_: {"source": "def run(robo, params):\n    pass\n"},
                    rounds=1)
    assert failed.run("larger move", goal)["status"] == "failed"
    assert failed.library.lookup(JOINT, failed.environment)["revision"] == original["revision"]

    def improve(_format, request):
        assert request["previous_source"] == limited
        assert request["evidence"][0]["success"] is False
        return {"source": SOURCE}

    fixed = router(tmp_path, "fixed", propose=improve)
    result = fixed.run("larger move", goal)
    assert result["route"] == "learned" and result["revision"] != original["revision"]
    assert result["trials"][0]["success"] is False


def test_heldout_start_rejects_memorized_joint_vector(tmp_path):
    from task_trials import initial_state
    _, joints = initial_state()
    target = joints["arm_deg"]["right"][:]
    target[6] += 3
    memorized = f"def run(robo, params):\n    robo.move_joints('right', {target!r})\n"

    def propose(_format, request):
        if request["round"] == 0:
            return {"source": memorized}
        assert any(row["case"] == 1 and not row["success"] for row in request["evidence"])
        return {"source": SOURCE}

    result = router(tmp_path, "heldout", propose=propose, rounds=2).run("move right")
    assert result["status"] == "success" and len(result["trials"]) == 5


def test_unprovenanced_cube_skill_is_excluded(tmp_path):
    legacy = tmp_path / "legacy/runs/cube-old/validated"
    legacy.mkdir(parents=True)
    program = Program.parse({"offset_m": [0.008, 0, 0], "lift_m": 0.065,
                             "grasp": "pinch", "recheck": True}, OfflineTrials.grasps)
    (legacy / "grasp_cube.py").write_text(program.source())
    (legacy.parent / "result.json").write_text(json.dumps({
        "status": "validated", "verification": [
            {"source": "offline_fixture", "success": True},
            {"source": "offline_fixture", "success": True},
        ],
    }))
    result = router(tmp_path, "import", resolve=lambda _: CUBE).run("抓起方块")
    assert result["route"] == "learned" and len(result["trials"]) == 3
    assert router(tmp_path, "reuse", resolve=unexpected, propose=unexpected).run("抓起方块")["route"] == "reused"


def test_cube_miss_uses_existing_rsijudge_campaign_then_publishes(tmp_path):
    result = router(tmp_path, "cube", resolve=lambda _: CUBE).run("抓住方块")
    assert result["route"] == "learned"
    learning = tmp_path / "cube/learning/result.json"
    rounds = json.loads(learning.read_text())["rounds"]
    assert [r["dev"]["successes"] for r in rounds] == [0, 2]


def test_legacy_localization_is_not_an_exploration_seed(tmp_path):
    from hardware import ROOT
    legacy = tmp_path / "legacy/skills"
    legacy.mkdir(parents=True)
    original = (ROOT / "skills/locate_object.py").read_text()
    (legacy / "locate_object.py").write_text(original)
    goal = {"kind": "locate", "arm": "none", "object": "shuttlecock", "values": []}
    result = router(tmp_path, "old-locate", resolve=lambda _: goal).run("定位球")
    assert result["route"] == "learned"
    from pathlib import Path
    assert Path(result["skill"]).read_text() != original


@pytest.mark.parametrize("kind", ["joint_target", "hand_target"])
def test_absolute_arm_and_hand_goals_are_checked_and_reused(tmp_path, kind):
    from hardware import arm_settings, hand_settings
    settings = hand_settings() if kind == "hand_target" else arm_settings()
    values = [(j.min_rad + j.max_rad) / 2 if kind == "hand_target"
              else (j.min_deg + j.max_deg) / 2 for j in settings.joints]
    goal = {"kind": kind, "arm": "left", "object": "none", "values": values}
    result = router(tmp_path, "absolute", resolve=lambda _: goal).run(kind)
    assert result["status"] == "success"
    assert router(tmp_path, "reuse", resolve=unexpected, propose=unexpected).run(kind)["route"] == "reused"


@pytest.mark.parametrize("obj", ["shuttlecock", "tube"])
def test_localization_uses_procedural_geometry_at_three_placements(tmp_path, obj):
    contract = {"kind": "locate", "arm": "none", "object": obj, "values": []}
    result = router(tmp_path, "locate", resolve=lambda _: contract).run(f"定位{obj}")
    assert result["status"] == "success"
    assert all(row["reason"] == "measured_target_localized" for row in result["trials"])


def test_wrong_object_and_unsupported_goal_do_not_enter_library(tmp_path):
    wrong = {"kind": "locate", "arm": "none", "object": "shuttlecock", "values": []}
    source = ("def run(robo, params):\n    obs=robo.observe()\n"
              "    return robo.locate(obs['frame_id'], 'shuttlecock', [194,260,245,371])\n")
    result = router(tmp_path, "wrong", propose=lambda *_: {"source": source}, rounds=1).run("球", wrong)
    assert result["status"] == "failed"
    result = router(tmp_path, "unsupported", resolve=lambda _: {
        "kind": "unsupported", "reason": "No offline shuttlecock contact verifier",
    }).run("右手把羽毛球抓起来")
    assert result["status"] == "blocked" and not result["trials"]


def test_library_keeps_backends_separate_and_handles_concurrent_publish(tmp_path):
    store = SkillLibrary(tmp_path / "library")
    checks = [{"success": True, "case": i} for i in range(3)]

    def publish(side):
        return store.publish(side, {**JOINT, "arm": side}, {"backend": "offline"},
                             SOURCE, checks, "test")

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert len(list(pool.map(publish, ("left", "right")))) == 2
    assert store.lookup(JOINT, {"backend": "offline"})
    assert store.lookup(JOINT, {"backend": "real"}) is None
    assert len(json.loads((store.root / "index.json").read_text())["skills"]) == 2


def test_corrupted_program_is_detected_before_execution(tmp_path):
    store = SkillLibrary(tmp_path / "library")
    checks = [{"success": True, "case": i} for i in range(3)]
    entry = store.publish("move", JOINT, {"backend": "offline"}, SOURCE, checks, "test")
    from pathlib import Path
    Path(entry["path"]).write_text("def run(robo, params):\n    pass\n")
    with pytest.raises(ValueError, match="checksum"):
        store.lookup(JOINT, {"backend": "offline"})


def test_worker_timeout_cannot_publish_or_leave_trial_running(tmp_path):
    run = router(tmp_path, "timeout", propose=lambda *_: {
        "source": "def run(robo, params):\n    while True:\n        pass\n",
    }, rounds=1, trial_timeout=2)
    result = run.run("move right")
    assert result["status"] == "failed" and result["trials"][0]["reason"] == "worker_timeout"


def test_transient_model_timeout_retries_without_reusing_partial_output(tmp_path, monkeypatch):
    import subprocess
    import task
    calls = []

    def attempt(_codex, _kind, _request, directory):
        calls.append(directory)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(["codex"], 120)
        return {"source": SOURCE}

    monkeypatch.setattr(task, "model_attempt", attempt)
    result = task.model_reply("codex", "cells", {}, tmp_path / "model-0")
    assert result["source"] == SOURCE
    assert [p.name for p in calls] == ["model-0", "model-0-retry"]
