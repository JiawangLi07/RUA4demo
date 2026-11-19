import json
from pathlib import Path

import pytest

from exploration.campaign import TASK, GraspCampaign
from exploration.grasp_program import Program, verdict
from exploration.offline import OfflineTrials, scripted_learner


BASE = {"offset_m": [0, 0, 0], "lift_m": 0.065, "grasp": "pinch", "recheck": False}


def test_single_cell_writes_complete_code_then_runs_one_trial(tmp_path):
    proposals = []

    def propose(request):
        proposals.append(request)
        return {"program": {**BASE, "offset_m": [0.008, 0, 0], "recheck": True},
                "reason": "Matched fixture pinch"}

    campaign = GraspCampaign(tmp_path, OfflineTrials(single_cell=True), propose, lambda: False,
                             rounds=3, layouts=[0, 1], verification=[2, 3], single_cell=True)
    result = campaign.run_single_cell()
    assert len(proposals) == 1 and proposals[0]["single_cell"]
    assert result["success"]
    assert result["commands"] == [
        "locate_target", "prepare", "approach", "confirm_target", "descend", "close_hand", "lift",
    ]
    source = (Path(result["run_dir"]) / campaign.skill).read_text()
    assert source.endswith("result = run(robo)\n")
    Program.from_proposal(source, campaign.backend.grasps, single_cell=True)
    with pytest.raises(ValueError):
        Program.from_proposal(source + "robo.approach()\n", campaign.backend.grasps, single_cell=True)
    assert len(list((tmp_path / "trials").iterdir())) == 1


def test_learns_python_from_empty_library_and_checks_frozen_program(tmp_path):
    campaign = GraspCampaign(tmp_path, OfflineTrials(), scripted_learner, lambda: False,
                             rounds=3, layouts=[0, 1], verification=[2, 3])
    result = campaign.train()
    assert [row["dev"]["successes"] for row in result["rounds"]] == [0, 2]
    assert result["status"] == "validated"
    assert {row["layout"] for row in result["verification"]} == {2, 3}
    observed = [json.loads(path.read_text())["before_m"]
                for path in (tmp_path / "trials").glob("*/result.json")]
    assert len({tuple(point) for point in observed}) == 4
    library = campaign.library(TASK)
    assert library.manifest(campaign.v0(TASK))["parent"] is None
    assert library.manifest(result["active"])["parent"] == campaign.v0(TASK)
    artifact = Path(result["skill"])
    assert Program.read(artifact, campaign.backend.grasps).offset_m == (0.008, 0, 0)
    assert "robo.confirm_target()" in artifact.read_text()


def test_uses_upstream_rollback_when_candidate_loses(tmp_path):
    def propose(request):
        return {"program": {**BASE, "offset_m": [0.008, 0, 0],
                            "grasp": "pinch" if request["round"] == 0 else "wide"},
                "reason": "Test a wider hand pose"}

    campaign = GraspCampaign(tmp_path, OfflineTrials(), propose, lambda: False,
                             rounds=2, layouts=[0, 1], verification=[2, 3])
    first = campaign.learn(TASK)
    state = json.loads((tmp_path / "tasks" / TASK / "learn.json").read_text())
    assert state["rounds"][1]["rolled_back_to"] == first == campaign.v0(TASK)
    assert [row["dev"]["successes"] for row in state["rounds"]] == [2, 0]


def test_stops_after_unverified_reset(tmp_path):
    class BadReset(OfflineTrials):
        def run(self, program, out, layout, *, cancelled):
            return {**super().run(program, out, layout, cancelled=cancelled), "reset_ok": False}

    campaign = GraspCampaign(tmp_path, BadReset(), scripted_learner, lambda: False,
                             rounds=3, layouts=[0, 1], verification=[2, 3])
    with pytest.raises(RuntimeError, match="reset"):
        campaign.train()
    assert len(list((tmp_path / "trials").iterdir())) == 1
    assert not (tmp_path / "validated").exists()


def test_refuses_motion_outside_proposal_and_arbitrary_program(tmp_path):
    with pytest.raises(ValueError, match="offset_m"):
        Program.parse({**BASE, "offset_m": [0, 0, 0.13]}, OfflineTrials.grasps)
    path = tmp_path / "policy.py"
    original = Program.parse(BASE, OfflineTrials.grasps).source()
    path.write_text(original + "robo.other_action()\n")
    with pytest.raises(ValueError, match="primitive sequence"):
        Program.read(path, OfflineTrials.grasps)


def test_verdict_rejects_slip_and_occlusion_with_measured_up_axis():
    before = (0, 0, 0)
    up = (0, -1, 0)
    assert verdict(before, [(0, -0.062, 0), (0, -0.061, 0)], up)["success"]
    assert verdict(before, [(0, -0.062, 0), before], up)["reason"] == "cube_slipped"
    assert verdict(before, [(0, -0.062, 0), None], up)["reason"] == "cube_unobservable"
