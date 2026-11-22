import hashlib
import json
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from sim.isaac import tabletop_scene_refit
from sim.isaac.head_wrist_surface import RegisteredSurface, free_space_error
from sim.isaac.tabletop_control import Controller
from sim.isaac.tabletop_scene_refit import LiveSceneBank


def test_admit_hand_contact_preserves_fixture_and_neighbor_clearance():
    surface = RegisteredSurface.__new__(RegisteredSurface)
    surface.table, surface.bounds = 0.0, []
    hand = np.array([[0.0, 0.0, -0.002], [0.001, 0.0, -0.002], [0.0, 0.001, -0.002]])
    pose = np.eye(4)
    assert not surface.clear({"right_index_elastomer": hand}, pose, 0.0005)
    surface.allow_hand_table_contact = True
    assert surface.clear({"right_index_elastomer": hand}, pose, 0.0005)
    assert not surface.clear({"tool_flange": hand}, pose, 0.0005)
    ctrl = Controller.__new__(Controller)
    ctrl.lightning, ctrl.table = None, 0.0
    ctrl.io = SimpleNamespace(model=SimpleNamespace(fixture_palm={}))
    assert not ctrl.vertices_clear(pose, hand, None)
    ctrl.allow_hand_table_contact = True
    assert ctrl.vertices_clear(pose, hand, None)
    assert not ctrl.vertices_clear(pose, hand, cKDTree(hand))


def test_one_head_request_and_same_response_on_a_retry(tmp_path, monkeypatch):
    # The live episode stays at the observation timestamp while the external
    # worker answers, then retries address rows in the exact same bank.
    priors = tmp_path / "priors"
    priors.mkdir()
    eye = np.eye(4)
    q = np.zeros((3, 22))
    fields = {
        k: np.zeros((3, 5, 3))
        for k in ("target_pos", "target_normal", "contact_pos", "contact_normal")
    }
    fields.update(
        q=q, q_mask=np.ones_like(q, bool), contact_link_id=np.ones((3, 5), int)
    )
    np.savez(priors / "part.right.npz", **fields)
    live = LiveSceneBank(priors, "right", tmp_path / "episode", 2)
    data = {"q": q, "link_names": []}
    provenance = {
        "bank_sha256": "parent",
        "input_sha256": "mesh",
        "hand_sha256": "hand",
        "upstream": "revision",
    }
    monkeypatch.setattr(live, "read", lambda *_: (data, provenance))
    chart = {"object_pose": np.stack([eye] * 3)}
    monkeypatch.setattr(
        tabletop_scene_refit, "placements_from_prior", lambda *_: (chart, np.arange(3))
    )
    calls = []

    def select(
        bank, ctrl, name, surface, reg, others, previous, *, placement_limit=None
    ):
        if placement_limit is not None:
            calls.append(placement_limit)
            return [(1, i, eye, q[i], np.zeros(7)) for i in range(placement_limit)]
        return bank.path, previous

    monkeypatch.setattr(tabletop_scene_refit.LightningBank, "select", select)
    surface = SimpleNamespace(camera_samples={})
    ctrl = SimpleNamespace(
        library={
            "part": {
                "prior": {"sha256": "geom", "lightning_input_sha256": {"right": "mesh"}}
            }
        },
        arm=np.zeros(7),
        table=0.7,
        shared=None,
        event=lambda *a, **k: None,
    )
    folder = tmp_path / "episode/right/part"

    def answer(_):
        with np.load(folder / "request.npz") as request:
            assert len(request["parent"]) == 2
            np.testing.assert_array_equal(request["parent"], [0, 1])
        response = folder / "part.right.npz"
        np.savez(response, q=q[:2])
        report = {
            "candidates": 2,
            "issued": 2,
            "request_sha256": hashlib.sha256(
                (folder / "request.npz").read_bytes()
            ).hexdigest(),
            "bank_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
        }
        (folder / "result.json").write_text(json.dumps(report))

    monkeypatch.setattr(tabletop_scene_refit.time, "sleep", answer)
    assert live.select(ctrl, "part", surface, eye, None, []) == (folder, [])
    assert live.select(ctrl, "part", surface, eye, None, ["LG-0"]) == (folder, ["LG-0"])
    assert calls == [2]


def test_depth_conflict_distinguishes_a_free_ray_from_an_occluded_surface():
    k = np.array([[100.0, 0, 1], [0, 100.0, 1], [0, 0, 1]])
    frame = {"pose": np.eye(4), "K": k, "depth": np.ones((3, 3))}
    # One point in front of the measurement, one behind an occluder, one outside the image.
    points = np.array([[0, 0, 0.990], [0.010, 0, 1.010], [1.0, 0, 1.0]])
    np.testing.assert_allclose(free_space_error(points, frame), (0.010 - 0.0025) / 2)
    frame["depth"].fill(np.nan)
    assert free_space_error(points, frame) == 0.0
