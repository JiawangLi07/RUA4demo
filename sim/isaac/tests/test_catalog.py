import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.isaac import catalog
from sim.isaac.catalog import opposed_case, palm_case, palm_contacts, select
from sim.isaac.direct_pregrasp import DirectGrasp, Grasp
from sim.isaac.paths import RIGHT_JOINTS
from sim.isaac.score import score


def test_requested_contact_includes_the_palm_and_distinct_proximal_fingers():
    contacts = ["right_hand_C_MC", "right_index_PP", "right_index_MP", "right_thumb_PP"]
    assert palm_contacts({"intended_contacts": contacts}) == (False, 1, True)
    contacts += ["right_middle_PP", "right_middle_MP"]
    assert palm_contacts({"intended_contacts": contacts}) == (True, 2, True)
    assert not palm_contacts({"intended_contacts": contacts[1:]})[0]


def test_native_case_uses_the_shortlist_and_preserves_the_starting_profile():
    base = {"head_wrist_validation": {"pinch_postures": {"original": [1] * 22}, "catalog_grasps": {"old": {}}}}
    candidates = [{"posture": name, "grasp": {"q_rad": [v] * 22}} for name, v in (("one", 0.2), ("two", 0.4))]
    selection = {"goal": "palm_wrap", "shape": "stout_cylinder", "candidates": candidates}
    updated = palm_case(base, selection, 1)["head_wrist_validation"]
    assert updated["wrap_postures"] == ["one"] and updated["max_grasp_attempts"] == 1
    assert updated["require_palm_hold"]
    assert updated["record_link_forces"] and "catalog_grasps" not in updated
    assert updated["pinch_postures"]["one"] == [0.2] * 22
    assert list(base["head_wrist_validation"]["pinch_postures"]) == ["original"]


def test_handle_bridge_requires_several_fingers_and_wrist_refits_the_local_pose():
    head = np.eye(4)
    head[:3, 3] = [0.5, -0.2, 0.7]
    local = np.eye(4)
    local[:3, 3] = [0.01, 0.02, 0.05]
    candidates = []
    for i, digits in enumerate((("thumb", "index"), ("thumb", "index", "middle"), ("thumb", "index", "middle", "ring"))):
        candidates.append({
            "posture": f"p{i}", "mode_id": f"m{i}", "intended_contacts": [f"right_{d}_DP" for d in digits],
            "grasp": {"T_world_palm": (head @ local).tolist()},
            "pregrasp": {"q_rad": [0.2] * 22}, "squeeze": {"q_rad": [0.6] * 22},
        })
    selection = {"goal": "any", "T_world_mesh": head.tolist(), "candidates": candidates}
    base = {"head_wrist_validation": {"multi_finger_handle": True, "gesture_order": ["old"], "pinch_postures": {"old": []}}}
    settings = opposed_case(base, selection, 2)["head_wrist_validation"]
    assert list(settings["catalog_grasps"]) == ["p2", "p1"]
    assert list(settings["pinch_postures"]) == ["p2", "p1"]
    assert "gesture_order" not in settings and base["head_wrist_validation"]["gesture_order"] == ["old"]
    np.testing.assert_allclose(settings["catalog_grasps"]["p2"]["T_mesh_palm"], local)
    wrist = np.eye(4)
    wrist[:2, :2] = [[0, -1], [1, 0]]
    wrist[:3, 3] = [0.53, -0.20, 0.71]
    trial = DirectGrasp.__new__(DirectGrasp)
    trial.registration = {"pose_world_rotation": wrist[:3, :3], "pose_world_translation_m": wrist[:3, 3]}
    trial.cfg, trial.family, trial.used_postures = settings, "side_wall", {"p1"}
    trial.scene = SimpleNamespace(limits={"right": np.tile([-1, 2], (22, 1))})

    def side_options(role, reference, opened, bounds):
        np.testing.assert_allclose(reference["grasp"], wrist @ local)
        return [Grasp(reference["grasp"], reference["grasp"], opened, opened, opened, {})]

    trial.side_options = side_options
    grips = trial.catalog_options("wrist", SimpleNamespace(details={"posture": "p2"}))
    assert [g.details for g in grips] == [{"posture": "p2", "catalog_mode_id": "m2"}]


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
    source = pytest.importorskip(f"{catalog.GRASP_PACKAGE}.postures")
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
    source = pytest.importorskip(f"{catalog.GRASP_PACKAGE}.templates")
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


def test_bundled_catalog_references_every_cached_component():
    import hashlib

    index = json.loads(catalog.BUNDLED_INDEX.read_bytes())
    cards = index["types"]
    refs = [card["modes_file"] for card in cards.values()]
    refs.extend((index["selection_file"], index["visual_shape_evidence_file"]))
    assert len(cards) == 36
    assert len({ref["sha256"] for ref in refs}) == 38
    for ref in refs:
        payload = (catalog.BUNDLED_INDEX.parents[2] / ref["cache_path"]).read_bytes()
        assert len(payload) == ref["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == ref["sha256"]


def test_shuttle_example_registers_both_views_without_station_assets(tmp_path, monkeypatch):
    example = catalog.REPO_ROOT / "sim/isaac/examples/shuttle"
    out = tmp_path / "selection"
    monkeypatch.delenv("RUA_SIM_ASSETS", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "catalog", "--prior", "shuttle", "--head-mesh", str(example / "head_mesh.npz"),
        "--wrist-mesh", str(example / "wrist_mesh.npz"), "--table-z", "0.7165", "--out", str(out),
    ])
    assert catalog.main() == 0
    selection = json.loads((out / "candidates.json").read_bytes())
    assert selection["types_inspected"] == 36 and len(selection["candidates"]) == 22
    assert selection["mesh"]["file"] == "shuttle_side_prior.npz"
    assert len(selection["registration"]) == 2
    assert selection["joint_names"] == RIGHT_JOINTS
    for candidate in selection["candidates"]:
        for stage in ("pregrasp", "grasp", "squeeze"):
            pose, angles = (np.asarray(candidate[stage][field]) for field in ("T_world_palm", "q_rad"))
            assert pose.shape == (4, 4) and angles.shape == (22,)
            assert np.isfinite(pose).all() and np.isfinite(angles).all()
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            np.testing.assert_allclose(pose[3], [0, 0, 0, 1])
    assert not (out / "case.json").exists()
