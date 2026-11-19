import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exploration.campaign import GraspCampaign
from exploration.physical import PhysicalTrials, independent_measurement
from physical_evidence import cells_record, recent
from research_loop import ResearchLoop
from task_library import SkillLibrary, atomic_json


def test_archived_cells_reach_both_roles_without_promoting_model_claim(tmp_path):
    library = SkillLibrary(tmp_path / "library")
    run = tmp_path / "run"
    run.mkdir()
    atomic_json(run/"result.json", {"task":"look", "finish":{"status":"success"}})
    observation = {"source":"replay","frame_id":"frame","camera_serial":"D455"}
    atomic_json(run/"observation.json", observation)
    assert cells_record(run, library.root, "grasp") is None
    atomic_json(run/"observation.json", {**observation, "source":"live"})
    atomic_json(run/"result.json", {"task":"look", "controller":"SimMotionBackend"})
    assert cells_record(run, library.root, "grasp") is None
    atomic_json(run/"result.json", {"task":"look", "finish":{"status":"success"}})
    cells_record(run, library.root, "grasp")
    record = cells_record(run, library.root, "grasp")
    assert record["object_verdict"] == "unverified" and not record["measured_arm_moves"]
    assert recent(library.root,"grasp") == [record] and not recent(library.root,"throw")
    assert not recent(library.root, "grasp", autonomous=True)
    atomic_json(run/"result.json", {"task":"look", "input_policy":"own_trials_v1"})
    record = cells_record(run, library.root, "grasp")
    decision = {"action":"experiment","hypothesis":"object stationary","instruction":"measure after lift",
                "expected_observation":"new object height"}
    seen = []
    def researcher(brief):
        seen.append(brief["physical_trials"])
        return decision
    def engineer(kind, request):
        seen.append(request["physical_trials"])
        return {"reason":"test"}
    loop = ResearchLoop(library, run, "grasp", {"kind":"investigate","arm":"none","object":"grasp"},
                        {"backend":"offline"}, researcher, engineer)
    loop.propose("discovery", {"round":0})
    assert seen == [[record], [record]]
    assert not (library.root/"index.json").exists()


def trace(out, x, height, target="cube"):
    p = [x,0,0]
    seq = []
    def observation(phase, i, z):
        if target == "shuttlecock":
            from PIL import Image
            Image.new("1", (12,12),1).save(out / f"{phase}_target.png")
        seq.append({"event":"observe", "phase":phase, "target":target, "target_m":[x,0,z], "color_sequence":i,
                    "detector":{"reason":"surface_measured","low_height_m":z, "pixels":144},
                    "depth_sequence":i+2, "sample_monotonic_ns":i*300_000_000})
    observation("before", 1, 0)
    seq.append({"event":"hand_waypoint", "measured_rad":[.1]*22, "torque":[.05]*22})
    observation("closed", 2, 0)
    seq.append({"event":"arm_waypoint", "measured_deg":[1.]*7})
    observation("hold_0", 3, height)
    observation("hold_1", 4, height)
    observation("reset", 5, 0)
    seq.append({"event":"idle_confirmed"})
    (out/"trajectory.jsonl").write_text("".join(json.dumps(r)+"\n" for r in seq))
    return {"source":"real", "before_m":p, "success":True, "reason":"claimed_lift", "reset_ok":True}


def test_camera_sequence_and_holdout_motion_cannot_be_reused(tmp_path):
    trace(tmp_path, 0, .061)
    assert independent_measurement(tmp_path, [0,0,1])["success"]
    path = tmp_path/"trajectory.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[5]["depth_sequence"] = rows[4]["depth_sequence"]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert independent_measurement(tmp_path, [0,0,1])["reason"] == "stale_or_unordered_samples"
    trace(tmp_path, 0, 0)
    assert independent_measurement(tmp_path, [0,0,1])["reason"] == "cube_did_not_lift"


@pytest.mark.parametrize("target", ["cube","shuttlecock"])
def test_frozen_physical_program_checked_across_distinct_placements(tmp_path, target):
    library = SkillLibrary(tmp_path/"library")
    env = {"backend":"physical-"+target,"station":"test"}
    station = SimpleNamespace(arm="right", up=np.array([0,0,1]), cube_m=np.zeros(3), target=target)
    physical = PhysicalTrials(library, env, station)
    class Backend:
        grasps = ["pinch"]
        check_placement = None
        def run(self, program, out, layout, *, cancelled):
            if self.check_placement:
                self.check_placement(station.cube_m)
            return trace(out, float(station.cube_m[0]), .061 if program.recheck else 0., target)
    calls = []
    def propose(req):
        calls.append(req)
        from exploration.grasp_program import Program
        program = Program.parse({**req["previous"], "recheck":bool(req["round"])}, ["pinch"])
        return {"reason":"check before closing", "source":program.source()}
    def campaign(name):
        root = tmp_path/name
        root.mkdir()
        return GraspCampaign(root, Backend(), propose, lambda:False, physical=physical,
                             rounds=2, layouts=[0], verification=[])
    first = campaign("development")
    pending = first.train()
    assert calls[1]["evidence"][0]["reason"] == target + "_did_not_lift"
    assert pending["status"] == "awaiting_new_placement" and not library.lookup(physical.contract, env)
    with pytest.raises(ValueError, match="distinct"):
        physical.verify_next(campaign("repeat"), first.root)
    for x,expected in [(0.04,"awaiting_new_placement"),(.08,"validated")]:
        station.cube_m[0] = x
        checked = physical.verify_next(campaign(str(x)), first.root)
        assert checked["status"] == expected
    active = library.lookup(physical.contract, env)
    assert active["revision"] == pending["program_sha256"]
    assert [r["before_m"][0] for r in checked["trials"]] == [0,.04,.08]
    assert Path(checked["skill"]).read_text() == active["source"]
    assert not library.lookup(physical.contract, {**env, "station":"other"})


def test_recorded_tactile_cell_uses_camera_timestamp(tmp_path):
    from pyrualean._backend import Ledger
    from pyrualean.arms import ArmToolkit
    from cli import REFERENCE_EPISODE
    from hardware import read_recorded_state
    from rgbd import HEAD_SERIAL, replay
    from robot import SnapshotBackend
    from cells_robot import StationRobot
    snap = replay(REFERENCE_EPISODE, 1, HEAD_SERIAL)
    state = read_recorded_state(REFERENCE_EPISODE, snap.monotonic_ns)
    robot = StationRobot(SnapshotBackend(snap, state, None, tmp_path), ledger=Ledger())
    reply = robot.tactile(robot.observe()["frame_id"], "right")
    assert reply["available"] and reply["source"] == "recorded_force_xyz"
    assert all(abs(r["camera_offset_ms"]) <= 100 and len(r["force_xyz"]) == 3 for r in reply["samples"])
    with pytest.raises((RuntimeError, ValueError), match="frame_id"):
        robot.tactile("old","right")
    cells = ArmToolkit(robot, arm="cells", workspace=tmp_path/"work")
    result = cells.execute_tool("python",{"code":"obs=robo.observe()\nrobo.tactile(obs['frame_id'],'right')['available']"}).result
    assert result["value"] == "True"


def test_live_f6_snapshot_keeps_only_valid_sdk_samples(monkeypatch):
    from hardware import read_state
    sides = ("left","right")
    arm = SimpleNamespace(read_joint_position_deg=lambda _: [0]*7,
                          read_arm_status=lambda _: SimpleNamespace(cur_state=0))
    seen = []
    def sample(channel, timeout):
        seen.append(channel)
        if channel > 1:
            return None
        return SimpleNamespace(finger="thumb", frame_id=7, sdk_timestamp_s=1.2,
                               f6=[1]*6 if channel == 0 else [float("nan")]*6)
    hands = {s:SimpleNamespace(read_state=lambda:SimpleNamespace(joint_pos=[0]*22,sequence=10),
                               read_tactile_frame=sample) for s in sides}
    monkeypatch.setattr("hardware.time.monotonic_ns", lambda:1_000_000)
    state = read_state(arm, hands, 500_000, ["right"])
    assert seen == list(range(5)) and list(state["tactile"]) == ["right"]
    data = state["tactile"]["right"]
    assert data["source"] == "live_sdk_f6" and data["samples"] == [
        {"finger":"thumb","frame_id":7,"f6":[1.0]*6,"sdk_timestamp_s":1.2,"receipt_offset_ms":.5}]


def test_archived_snapshot_keeps_its_geometry_identity_and_never_moves(tmp_path):
    from autonomous_scene import observation
    from discovery_trial import observe_candidate
    from rgbd import HEAD_SERIAL, archived_observation, replay
    from robot import SnapshotBackend
    captured = tmp_path / "captured"
    work = tmp_path / "work"
    captured.mkdir()
    work.mkdir()
    snap, joints = observation(3)
    snap.source = "live"  # Synthetic stored capture, only inside the test directory.
    old = SnapshotBackend(snap, joints, np.eye(4), captured)
    archived = {"run":str(captured), "frame_id":old.frame_id}
    atomic_json(captured/"result.json", {"input_policy":"script_or_diagnostic"})
    with pytest.raises(ValueError,match="autonomous"):
        archived_observation(captured, old.frame_id)
    atomic_json(captured/"result.json", {"input_policy":"own_trials_v1", "controller":"SnapshotBackend"})
    restored, _, transform = archived_observation(captured, old.frame_id)
    assert np.array_equal(snap.raw_depth,restored.raw_depth) and np.array_equal(snap.rgb,restored.rgb)
    assert np.array_equal(transform,np.eye(4))
    with pytest.raises(ValueError,match="identity"):
        archived_observation(captured,"elsewhere")
    result = observe_candidate({"archive":archived, "case":0,"contract":{},
                                "source":"def run(robo, params):\n    return robo.observe()['frame_id']\n"},work)
    assert result["source"] == "archived_live_replay" and old.frame_id in result["execution"]["value"]
    assert not result["success"]
