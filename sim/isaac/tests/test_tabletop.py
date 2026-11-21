import json

import numpy as np

from sim.isaac import tabletop_layout, tabletop_score


def cube():
    return np.array(
        [
            [x, y, z]
            for x in (-0.012, 0.012)
            for y in (-0.012, 0.012)
            for z in (-0.01, 0.01)
        ]
    )


def test_layout_keeps_sources_unposed_and_reproducible(monkeypatch):
    mesh = {"collisions": [(cube(), None)]}
    monkeypatch.setattr(tabletop_layout, "load_mesh_bundle", lambda _: mesh)
    inputs = [
        {
            "name": n,
            "mesh": {},
            "prim_path": "/World/" + n,
            "quaternion_wxyz": [1, 0, 0, 0],
        }
        for n in ("one", "two", "three")
    ]
    base = {"images": {}}
    first = tabletop_layout.make_scene(base, inputs, 41)
    again = tabletop_layout.make_scene(base, inputs, 41)
    other = tabletop_layout.make_scene(base, inputs, 73)
    assert first == again and first["added_objects"] != other["added_objects"]
    assert all("position" not in original for original in inputs)
    assert len(first["initial_joint_positions"]) == 14


def test_offline_score_requires_an_executed_closure_and_two_loaded_seconds(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        tabletop_score, "load_mesh_bundle", lambda _: {"collisions": [(cube(), None)]}
    )
    dt = 0.01
    n = 480
    obj = np.zeros((n, 1, 7))
    obj[:, :, 2] = 0.011
    obj[:, :, 3] = 1.0
    palm = np.zeros((n, 7))
    palm[:, 3] = 1.0
    obj[140:390, 0, 2] += 0.105
    palm[140:390, 2] += 0.105
    forces = np.zeros((n, 1))
    forces[140:390, 0] = 1.0
    recording = {
        "object_names": ["obj"],
        "pose_wxyz": obj,
        "hand_load_N": forces,
        "palm_wxyz": palm,
        "dt": dt,
        "table_z": 0.0,
    }
    np.savez(tmp_path / "physics_audit.npz", **recording)
    scene = {
        "added_objects": [{"name": "obj", "mesh": {"file": "test.npz"}}],
        "tabletop_seed": 4,
    }
    events = []
    for attempt, stages in enumerate(
        [
            [
                (0.01, "attempt_start"),
                (0.03, "close"),
                (0.03, "lower_and_release"),
                (0.1, "attempt_end"),
            ],
            [
                (0.2, "attempt_start"),
                (0.3, "close"),
                (1.5, "hold_start"),
                (3.6, "hold_end"),
                (4.05, "lower_and_release"),
                (4.4, "attempt_end"),
            ],
        ],
        1,
    ):
        for t, stage in stages:
            events.append(
                {
                    "name": "test",
                    "time": t,
                    "phase": stage,
                    "attempt": attempt,
                    "posture": "tripod",
                }
            )
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (tmp_path / "decisions.json").write_text(
        json.dumps({"test": {"outcomes": ["arm IK", "vision_pass"]}})
    )
    result = tabletop_score.evaluate(scene, tmp_path)
    assert result["stable"] == 1 and result["closed"] == 1
    assert result["raised"] == 1 and result["held_through_end"] == 1
    assert result["confirmed_closures"] == 0
    assert result["per_object"]["test"][1]["closure_evidence"] == "elapsed_only"
    assert not result["per_object"]["test"][0]["closure"]
    # Reassignment changes the physical load channel and keeps unique attempt IDs.
    (tmp_path / "decisions.json").write_text(
        json.dumps(
            {
                "test": {
                    "outcomes": ["arm IK", "vision_pass"],
                    "attempt_hands": ["right", "left"],
                }
            }
        )
    )
    recording.update(
        left_hand_load_N=forces.copy(), right_hand_load_N=np.zeros_like(forces)
    )
    np.savez(tmp_path / "physics_audit.npz", **recording)
    (tmp_path / "trial_settings.json").write_text(
        json.dumps({"lift_height_m": 0.05, "closure_telemetry": "completed_command"})
    )
    assert tabletop_score.evaluate(scene, tmp_path)["closed"] == 0
    events.append(
        {"name": "test", "time": 1.25, "phase": "closure_completed", "attempt": 2}
    )
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    rows = tabletop_score.evaluate(scene, tmp_path)["per_object"]["test"]
    assert [(r["hand"], r["closure"]) for r in rows] == [
        ("right", False),
        ("left", True),
    ]
    assert rows[1]["held_through_end"]
    assert rows[1]["closure_evidence"] == "completed_command"
    assert not rows[1]["full_palm_held_2s"]
    digit_loads = np.ones((n, 5, 1))
    palm_loads = np.ones((n, 1, 1))
    recording.update(
        digit_names=["thumb", "index", "middle", "ring", "pinky"],
        digit_load_N=digit_loads,
        left_link_names=["left_hand_C_MC"],
        left_link_load_N=palm_loads,
    )
    np.savez(tmp_path / "physics_audit.npz", **recording)
    assert tabletop_score.evaluate(scene, tmp_path)["full_palm_held_objects"] == 1
    digit_loads[350, 3, 0] = 0.0
    np.savez(tmp_path / "physics_audit.npz", **recording)
    assert tabletop_score.evaluate(scene, tmp_path)["full_palm_held_objects"] == 0
    del recording["left_hand_load_N"], recording["right_hand_load_N"]
    # A 6 cm raise clears the current 5 cm acceptance height, but misses the earlier goal.
    obj[140:390, 0, 2] -= 0.045
    palm[140:390, 2] -= 0.045
    np.savez(tmp_path / "physics_audit.npz", **recording)
    (tmp_path / "trial_settings.json").write_text(json.dumps({"lift_height_m": 0.05}))
    assert tabletop_score.evaluate(scene, tmp_path)["stable"] == 1
    assert tabletop_score.evaluate(scene, tmp_path, 0.09)["stable"] == 0
    next(e for e in events if e["phase"] == "hold_end")["time"] = 4.0
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    slipped = tabletop_score.evaluate(scene, tmp_path)
    assert slipped["stable"] == 1 and slipped["held_through_end"] == 0
    forces[140:390, 0] = 0.0
    np.savez(tmp_path / "physics_audit.npz", **recording)
    assert tabletop_score.evaluate(scene, tmp_path)["stable"] == 0
