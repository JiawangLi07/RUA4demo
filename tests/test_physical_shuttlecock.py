from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autonomous_scene import observation
from exploration.cork_pinch import fit_cork
from exploration.shuttlecock import cork_mask, measure
from exploration.physical import independent_measurement
from task import physical_goal


def test_visible_cork_cap_recovers_center_behind_observed_depth():
    center = np.array([.13, -.64, .014])
    radius = .013
    xz = np.linspace(-.007, .007, 13)
    x, z = np.meshgrid(xz, xz)
    visible = center + np.c_[x.ravel(), np.sqrt(radius**2 - x.ravel()**2 - z.ravel()**2), z.ravel()]
    cork = fit_cork(visible, np.array([0., 0., 1.]), table=0.)
    np.testing.assert_allclose(cork.center, center, atol=1e-6)
    assert cork.radius == pytest.approx(radius, abs=1e-6)
    assert np.all(cork.samples[:, 1] > cork.center[1])


@pytest.mark.parametrize("text,expected", [
    ("右手抓取桌面上的羽毛球", ("right","shuttlecock")),
    ("抓取桌面上的羽毛球", ("right","shuttlecock")),
    ("用左手把桌面上的红方块拿起", ("left","cube")),
])
def test_one_task_selects_requested_arm_and_object(text, expected):
    assert physical_goal(text) == expected


def test_compound_goal_is_not_reduced_to_grasp():
    with pytest.raises(ValueError):
        physical_goal("右手抓起羽毛球然后放入球桶")


def test_semantic_mask_only_supplies_its_own_depth_and_rejects_ambiguity():
    snap, _ = observation()
    station = SimpleNamespace(roi=(31,20,622,459), mount_from_color=np.eye(4), up=np.array([0.,0.,1.]))
    x,y,r,b = station.roi
    # A generated scene supplies a stand-in for the semantic model; exercise actual depth projection.
    mask = snap.raw_depth[y:b,x:r] == 792
    class Client:
        last_candidates = [{"score":.91, "mask":mask}]
        def segment(self, image, **args):
            assert image.shape[:2] == mask.shape and args["text_prompt"] == "badminton shuttlecock"
            return SimpleNamespace(found=True)
    client = Client()
    point, observed, info = measure(client, snap, station)
    assert point[2] == pytest.approx(.792) and info["reason"] == "surface_measured"
    assert np.array_equal(observed, mask)
    client.last_candidates = [{"score": .523, "mask": mask}]
    point, observed, info = measure(client, snap, station, cork=True)
    assert point is None and observed is None and info["reason"] == "weak_shuttlecock_candidate"
    client.last_candidates[0]["score"] = .91
    client.last_candidates *= 2
    assert measure(client, snap, station)[0] is None


def test_cube_evidence_cannot_validate_a_shuttlecock(tmp_path):
    from test_physical_evidence import trace
    trace(tmp_path, 0, .063)
    verdict = independent_measurement(tmp_path, [0,0,1], "shuttlecock")
    assert not verdict["success"] and verdict["reason"] == "wrong_target_trace"


def test_sam3_cork_filters_whole_object_and_ambiguous_parts():
    image = np.zeros((90, 140, 3), dtype=np.uint8)
    whole = np.zeros(image.shape[:2], dtype=bool)
    whole[30:55, 40:90] = True

    class Client:
        def segment(self, crop, **kwargs):
            assert crop.shape[:2] == (53, 78)
            full = np.zeros(crop.shape[:2], dtype=bool)
            full[14:39, 14:64] = True
            cork = np.zeros_like(full)
            cork[20:33, 50:64] = True
            if self.fallback and kwargs["text_prompt"] == "cork tip of a badminton shuttlecock":
                self.last_candidates = [{"mask": full, "score": .95}]
                return
            self.last_candidates = [{"mask": full, "score": .95}, {"mask": cork, "score": .8}]
            if self.ambiguous:
                self.last_candidates.append({"mask": np.fliplr(cork), "score": .75})

    client = Client()
    client.ambiguous = False
    client.fallback = False
    mask, score = cork_mask(client, image, whole)
    assert score == .8 and np.array_equal(mask[36:49, 76:90], np.ones((13, 14), dtype=bool))
    assert mask.sum() == 13 * 14
    client.fallback = True
    assert np.array_equal(cork_mask(client, image, whole)[0], mask)
    client.fallback = False
    client.ambiguous = True
    assert cork_mask(client, image, whole) == (None, None)


def test_bad_draft_repaired_before_any_trial_and_budget_failure_retains_source(tmp_path):
    from exploration.campaign import GraspCampaign
    from exploration.physical import PhysicalTrials
    from exploration.grasp_program import Program
    from task_library import SkillLibrary
    from test_physical_evidence import trace
    station = SimpleNamespace(arm="right", up=[0,0,1], target="shuttlecock", cube_m=None,
                              contact_policy=None)
    physical = PhysicalTrials(SkillLibrary(tmp_path/"lib"), {"backend":"unit-test"}, station)
    plan = {"offset_m":[0,0,0], "lift_m":.065, "recheck":False, "grasp":"pinch"}
    order = []
    def propose(request):
        order.append("propose")
        if not request["draft"]:
            return {"source":"import os", "reason":"invalid draft"}
        assert request["validation_error"]
        return {"source":Program.parse(plan, ["pinch"]).source(), "reason":"bounded revision"}
    class Backend:
        grasps = ["pinch"]
        def run(self, program, out, layout, *, cancelled):
            order.append("execute")
            return trace(out, 0, 0)
    root = tmp_path/"campaign"
    root.mkdir()
    campaign = GraspCampaign(root, Backend(), propose, lambda:False, rounds=1, layouts=[0],
                             verification=[], physical=physical)
    result = campaign.train()
    assert result["status"] == "failed" and order == ["propose", "propose", "execute"]
    assert Path(result["candidate"]).name == "grasp_shuttlecock.py"
    assert not physical.library.lookup(physical.contract, physical.env)


def test_background_receipt_tracks_local_status_without_hardware(tmp_path, monkeypatch):
    import json
    import task
    import os
    launch = []
    def start(command, **kwargs):
        launch.append((command, kwargs, os.dup(kwargs["pass_fds"][0])))
        return SimpleNamespace(pid=987123)
    monkeypatch.setattr(task.subprocess,"Popen",start)
    task.background_grasp(["--target","shuttlecock"],tmp_path)
    job = next(tmp_path.glob("physical-job-*"))
    assert launch[0][1]["start_new_session"] and "--run-directory" in launch[0][0]
    (job/"status.json").write_text(json.dumps({"phase":"executing"}))
    assert task.job_status(job)["phase"] == "executing"
    os.close(launch[0][2])
    assert task.job_status(job)["phase"] == "interrupted"
    (job/"result.json").write_text(json.dumps({"status":"failed","reason":"trial_budget_exhausted"}))
    assert task.job_status(job) == {"phase":"finished","status":"failed","reason":"trial_budget_exhausted"}


def test_tilted_shuttlecock_on_table_does_not_pass_as_a_grasp(tmp_path):
    import json
    from test_physical_evidence import trace
    trace(tmp_path, 0, .064, "shuttlecock")
    path = tmp_path / "trajectory.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row.get("phase", "").startswith("hold"):
            row["detector"]["low_height_m"] = .003
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert independent_measurement(tmp_path, [0,0,1], "shuttlecock")["reason"] == "shuttlecock_not_clear_of_table"
