from pathlib import Path
import json

import pytest
from autonomous_scene import observation, object_boxes
from exploration.physical import check_profile_origin
from physical_evidence import put_record, recent
from task_library import SkillLibrary
from task_router import TaskRouter


def test_default_observation_paths_never_open_a_recording(monkeypatch, tmp_path):
    import cli
    from types import SimpleNamespace
    def forbidden(*args):
        pytest.fail("opened an expert recording during autonomous exploration")
    monkeypatch.setattr(cli, "replay", forbidden)
    monkeypatch.setattr(cli, "read_recorded_state", forbidden)
    monkeypatch.setattr(cli, "REFERENCE_EPISODE", tmp_path/"missing")
    with cli.source(SimpleNamespace(live=False,episode=None,frame=1)) as (snap, state):
        assert snap.source == "procedural_scene" and snap.serial == "synthetic"
        assert not state.get("tactile")
    assert object_boxes(0) != object_boxes(1)
    lib = SkillLibrary(tmp_path/"library")
    put_record(lib.root,"goal",{"source":"live_cells","run":"/old", "program":"DEMO_SENTINEL"})
    assert recent(lib.root, "goal", autonomous=True) == []
    def researcher(request):
        assert "DEMO_SENTINEL" not in json.dumps(request)
        return {"action":"experiment","hypothesis":"observe","instruction":"inspect current RGB",
                "expected_observation":"new scene"}
    def engineer(kind, req):
        assert req["context"]["source"] == "procedural_scene" and req["context"]["archive"] is None
        assert req["physical_trials"] == [] and req["implementations"] == []
        return {"source":"def run(robo, params):\n    return robo.observe()['source']\n"}
    r = TaskRouter(lib.root,tmp_path/"run",resolve=None,propose=engineer,researcher=researcher,rounds=1)
    result = r.run("goal", {"kind":"unsupported","reason":"no goal checker"})
    assert result["trials"][0]["source"] == "procedural_scene"
    assert not result.get("skill")


def test_station_requires_independent_primitives_and_start():
    settings = {"input_sources":{"initial_pose":"station_measurement","hand_targets":"hardware_primitive"}}
    check_profile_origin(settings)
    for invalid in ({}, {**settings, "provenance":{"hand_grasp_row":123}}):
        with pytest.raises(ValueError,match="independently measured"):
            check_profile_origin(invalid)
