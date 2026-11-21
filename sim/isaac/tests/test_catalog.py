import json
import sys

import numpy as np
import pytest

from sim.isaac import catalog
from sim.isaac.catalog import palm_case, palm_contacts, select
from sim.isaac.paths import RIGHT_JOINTS
from sim.isaac.score import score


def test_requested_contact_includes_the_palm_and_distinct_proximal_fingers():
    contacts = ["right_hand_C_MC", "right_index_PP", "right_index_MP", "right_thumb_PP"]
    assert palm_contacts({"intended_contacts": contacts}) == (False, 1, True)
    contacts += ["right_middle_PP", "right_middle_MP"]
    assert palm_contacts({"intended_contacts": contacts}) == (True, 2, True)
    assert not palm_contacts({"intended_contacts": contacts[1:]})[0]


def test_native_case_uses_the_shortlist_and_preserves_the_starting_profile():
    base = {"head_wrist_validation": {"pinch_postures": {"original": [1] * 22}}}
    candidates = [{"posture": name, "grasp": {"q_rad": [v] * 22}} for name, v in (("one", 0.2), ("two", 0.4))]
    selection = {"goal": "palm_wrap", "shape": "stout_cylinder", "candidates": candidates}
    updated = palm_case(base, selection, 1)["head_wrist_validation"]
    assert updated["wrap_postures"] == ["one"] and updated["max_grasp_attempts"] == 1
    assert updated["require_palm_hold"]
    assert updated["pinch_postures"]["one"] == [0.2] * 22
    assert list(base["head_wrist_validation"]["pinch_postures"]) == ["original"]


def test_scoring_requires_palm_contact_at_every_hold_step(tmp_path):
    def save(name, data):
        (tmp_path / (name + ".json")).write_text(json.dumps(data))

    save("config", {"physics_dt": 1 / 240, "head_wrist_validation": {"require_palm_hold": True}})
    save(
        "result", {"passed": True, "reason": "held", "frames": 1, "maximum_slip_m": 0, "maximum_slip_rad": 0}
    )
    save("frames", [{"frame": 0}])
    save("events", [{"stage": "refined_grasp", "power_wrap": True}])
    loads = {"right_hand_C_MC": 3, "right_thumb_MC": 2, "right_index_PP": 2, "right_middle_MP": 2}
    hold = [
        {"time": i / 240, "stage": "hold", "rise_m": 0.1, "clearance_m": 0.08, "link_normal_N": dict(loads)}
        for i in range(480)
    ]
    save("physics", hold)
    assert score(tmp_path)["envelope_steps"] == 480
    hold[200]["link_normal_N"]["right_hand_C_MC"] = 0
    save("physics", hold)
    with pytest.raises(ValueError, match="central palm"):
        score(tmp_path)


def test_registered_world_frame_and_all_joint_stages(monkeypatch):
    source = pytest.importorskip("dexbridge.skills.grasp.postures")
    ranked = [
        {"grasp_type": "tip", "shape_features": {"shape_key": "stout_cylinder"}},
        {"grasp_type": "palm", "shape_features": {"shape_key": "stout_cylinder"}},
    ]
    monkeypatch.setattr(source, "rank_types", lambda *_: ranked)
    local = np.eye(4)
    local[:3, 3] = [0.03, 0.02, 0.01]

    def mode(_, name, features, world):
        assert name == "palm"
        poses = [{"T_obj_palm": local, "q": [v] * 22} for v in (0.1, 0.5, 0.7)]
        return {
            "mode_id": "m",
            "clamped": False,
            "source_record_ids": ["r"],
            "pregrasp": poses[:1],
            "grasp": poses[1],
            "squeeze": poses[2],
        }

    monkeypatch.setattr(source, "choose_plan", mode)
    templates = {
        "joint_names": RIGHT_JOINTS,
        "types": {
            "tip": {"intended_contacts": ["right_thumb_DP", "right_index_DP"]},
            "palm": {"intended_contacts": ["right_hand_C_MC", "right_index_MP", "right_ring_PP"]},
        },
    }
    world = np.eye(4)
    world[:2, :2] = [[0, -1], [1, 0]]
    world[:3, 3] = [0.6, -0.2, 0.8]
    result = select(templates, {"bbox_dims": np.ones(3)}, world, "palm_wrap")
    assert result["types_inspected"] == 2 and list(result["rejected"]) == ["tip"]
    candidate = result["candidates"][0]
    for stage, angle in (("pregrasp", 0.1), ("grasp", 0.5), ("squeeze", 0.7)):
        np.testing.assert_allclose(candidate[stage]["T_world_palm"], world @ local)
        assert candidate[stage]["q_rad"] == [angle] * 22


def test_standalone_glb_uses_metric_scale_and_world_pose(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    source = pytest.importorskip("dexbridge.skills.grasp.templates")
    mesh = tmp_path / "box.glb"
    trimesh.creation.box(extents=(2, 3, 4)).export(mesh)
    world = np.eye(4)
    world[:3, 3] = [0.3, -0.2, 0.8]
    pose = tmp_path / "pose.json"
    pose.write_text(json.dumps(world.tolist()))
    out = tmp_path / "output"
    monkeypatch.setattr(source, "load_templates", lambda _: {})

    def inspect(_, features, transform, goal):
        np.testing.assert_allclose(np.sort(features["bbox_dims"]), [0.02, 0.03, 0.04], atol=1e-6)
        np.testing.assert_allclose(transform, world)
        return {"shape": "boxy", "types_inspected": 1, "candidates": [{"posture": "test"}]}

    monkeypatch.setattr(catalog, "select", inspect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "catalog",
            "--index",
            "unused",
            "--mesh",
            str(mesh),
            "--meters-per-unit",
            "0.01",
            "--world-from-mesh",
            str(pose),
            "--out",
            str(out),
        ],
    )
    assert catalog.main() == 0
    assert json.loads((out / "candidates.json").read_bytes())["mesh"]["meters_per_unit"] == 0.01
    assert not (out / "case.json").exists()
