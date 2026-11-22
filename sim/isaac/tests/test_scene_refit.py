import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
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


@pytest.mark.parametrize("sensor", ["head", "left_observer"])
def test_one_view_request_and_same_response_on_a_retry(tmp_path, monkeypatch, sensor):
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
    measured = np.array([[0.53, -0.23, 0.8], [0.55, -0.21, 0.81]])
    surface = SimpleNamespace(camera_samples={sensor + "_depth": measured}, primary_depth=sensor + "_depth")
    ctrl = SimpleNamespace(
        camera_roles=(sensor,),
        sampling_seed=751 if sensor == "left_observer" else None,
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
            if sensor == "left_observer":
                assert "head_local" not in request
                assert (
                    json.loads(str(request["receipt"]))["observation_source"] == sensor
                )
                np.testing.assert_allclose(request["observed_local"], measured)
            else:
                np.testing.assert_allclose(request["head_local"], measured)
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


def test_long_tool_placements_offer_centered_contacts_and_keep_original():
    # Opposed horizontal pad sites 34 mm from the mass centre on a 140 mm rod.
    x, y, z = np.meshgrid(
        np.linspace(-0.07, 0.07, 141), [-0.006, 0.006], [-0.002, 0.002]
    )
    samples = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    reg = np.eye(4)
    reg[:3, 3] = [0.66, -0.14, 0.71]
    surface = SimpleNamespace(
        local=samples,
        tree=cKDTree(samples),
        camera_samples={},
        world_parts=[samples + reg[:3, 3]],
    )
    targets = np.array([[[0.034, -0.006, 0.002], [0.034, 0.006, 0.002]]])
    data = dict(
        object_pose=np.eye(4)[None],
        q=np.zeros((1, 22)),
        contact_link_id=np.array([[0, 1]]),
        target_pos=targets,
        target_normal=np.zeros_like(targets),
        source=np.array(["official"]),
        source_id=np.array([0]),
        link_names=["thumb", "index"],
    )
    chart, parents = tabletop_scene_refit.placements_from_prior(data, surface, reg)
    offsets = chart["object_pose"][:, 0, 3]
    assert np.isclose(offsets[0], 0) and np.isclose(offsets, 0.034).any()
    assert np.all(parents == 0)


def test_seated_pressure_keeps_a_loaded_pad_and_closes_the_opposed_gap():
    from sim.isaac.tabletop_meshfit import retain_squeeze

    sites = []
    for j, (name, sign) in enumerate((("thumb", -1), ("middle", 1))):
        sites.append(
            dict(
                link=name,
                joint=j,
                normal=np.array([sign, 0.0, 0.0]),
                goal=np.array([sign * 0.02, 0.0, 0.01]),
            )
        )

    def point_jac(site, palm, q):
        j, direction = site["joint"], site["normal"]
        jac = np.zeros((3, 8))
        jac[:, 6 + j] = direction
        return site["goal"] + q[j] * direction, None, jac, None

    initial = np.array([-0.012, 0.002])
    path, _, gaps = retain_squeeze(
        SimpleNamespace(point_jac=point_jac),
        np.eye(4),
        initial,
        sites,
        np.array([[-0.05, 0.05]] * 2),
        0.0,
    )
    np.testing.assert_allclose(path[:, 0], initial[0])
    assert np.all(np.diff(path[:, 1]) <= 0) and -0.0031 < gaps["middle"] < -0.0028


def test_probe_detects_rotation_about_table_before_accepting_rgbd_pixel_votes(
    monkeypatch,
):
    from sim.isaac import tabletop_control, tabletop_tracking

    ctrl = Controller.__new__(Controller)
    ctrl.io = SimpleNamespace(fk=lambda link: np.eye(4))
    ctrl.table, ctrl.camera_roles, ctrl.tracks = 0.7, ("head",), {"head": object()}
    ctrl.adaptive, ctrl.template_comparison, ctrl.lift_height = True, False, 0.05
    ctrl.event = lambda *a, **k: None
    ctrl.snapshot = lambda *a: None
    skin = SimpleNamespace(local=np.zeros((1, 3)), world=np.zeros((1, 3)))
    track = dict(
        texture_inliers=10,
        texture_span_m=0.0083,
        tracked_rise_m=0.00807,
        tracked_bottom_rise_m=-0.00003,
    )
    monkeypatch.setattr(tabletop_tracking, "measure", lambda *a: track)
    monkeypatch.setattr(
        tabletop_control,
        "match_height",
        lambda *a: {"carried_pixels": 67, "static_pixels": 36},
    )
    assert (
        ctrl.vote(skin, np.eye(4), np.eye(4), [], "test_observation")
        == "pivot_on_table"
    )
    track["tracked_bottom_rise_m"] = 0.007
    assert (
        ctrl.vote(skin, np.eye(4), np.eye(4), [], "test_observation_repeat")
        == "carried"
    )


def test_visual_interruption_resumes_from_last_issued_joints():
    ctrl = Controller.__new__(Controller)
    commands = []
    ctrl.io = SimpleNamespace(
        dt=0.01,
        model=SimpleNamespace(limits={"right": np.array([[-1, 1]])}),
        step=lambda arm, hand: commands.append((arm.copy(), hand.copy())),
    )
    ctrl.adaptive, ctrl.lightning, ctrl.table = False, None, None
    ctrl.arm, ctrl.hand = np.zeros(1), np.zeros(1)
    ctrl.command(
        [ctrl.arm, np.array([0.15])], np.array([0.8]), 0.65, observe=lambda f: True
    )
    np.testing.assert_array_equal(ctrl.arm, commands[-1][0])
    np.testing.assert_array_equal(ctrl.hand, commands[-1][1])
    assert 0 < ctrl.hand[0] < 0.8 and 0 < ctrl.arm[0] < 0.15
    ctrl.command([ctrl.arm, np.array([0.20])], np.array([0.7]), 0.20)
    np.testing.assert_allclose(ctrl.arm, [0.20])
    np.testing.assert_allclose(ctrl.hand, [0.7])
