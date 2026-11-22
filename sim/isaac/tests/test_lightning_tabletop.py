import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.isaac.paths import RIGHT_JOINTS
from sim.isaac.tabletop_control import Controller, NoPath
from sim.isaac.tabletop_lightning import LightningBank, central_palm_samples
from sim.isaac.tabletop_lightning_banks import merge


def test_remove_camera_and_mount_preserves_hand_flange():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    from sim.isaac.scene_wrist_mount import remove_wrist_hardware

    stage = Usd.Stage.CreateInMemory()
    for suffix, name in (("L", "wrist2_camera"), ("R", "wrist_camera")):
        root = "/Robot/TCP_Link_" + suffix
        UsdGeom.Xform.Define(stage, root)
        UsdGeom.Camera.Define(stage, root + "/" + name)
        for part in ("WristCameraFixture/housing", "collisions/cad_adapter_plate"):
            UsdPhysics.CollisionAPI.Apply(
                UsdGeom.Mesh.Define(stage, root + "/" + part).GetPrim()
            )
    for _ in range(2):
        report = remove_wrist_hardware(stage, "/Robot")
        for side, row in report.items():
            assert row["active_wrist_cameras"] == []
            assert row["fixture_colliders"] == row["fixture_prims"] == 0
            assert len(row["retained_tcp_colliders"]) == 1
            assert row["retained_tcp_colliders"][0].endswith("cad_adapter_plate")


def test_head_only_lift_initializes_only_head_observation(monkeypatch):
    from sim.isaac import tabletop_tracking

    controller = Controller.__new__(Controller)
    controller.io = object()
    controller.camera_roles = ("head",)
    calls = []
    controller.snapshot = lambda role, tag: calls.append((role, tag))
    monkeypatch.setattr(tabletop_tracking, "initialize", lambda *args: None)
    controller.begin_lift(SimpleNamespace(world=np.zeros((4, 3))))
    assert calls == [("head", "before_lift")]
    assert set(controller.tracks) == {"head"}


def test_bank_preserves_native_indices_and_rejects_mismatched_object(tmp_path):
    def source(folder, input_id):
        folder.mkdir()
        path = folder / "part.right.npz"
        rho = {
            "upstream": "rev",
            "input_sha256": input_id,
            "hand_sha256": "hand",
            "prior_sha256": "registered",
            "source_frame": "object_to_palm",
        }
        data = {
            key: np.zeros((1, 5, 3))
            for key in ("target_pos", "contact_pos", "target_normal", "contact_normal")
        }
        np.savez(
            path,
            **data,
            q=np.ones((1, 22)),
            q_mask=np.ones((1, 22)),
            object_pose=np.eye(4)[None],
            link_names=["right_index_elastomer"],
            joint_names=RIGHT_JOINTS,
            contact_link_id=np.array([[0, -1, -1, -1, -1]]),
            receipt=json.dumps(rho),
        )
        return path

    a = source(tmp_path / "native", "same")
    b = source(tmp_path / "power", "same")
    out = tmp_path / "banks" / a.name
    merge([a, b], out)
    data, _ = LightningBank(out.parent, "right").read("part", "registered")
    assert data["source"].tolist() == ["native", "power"]
    assert data["source_id"].tolist() == [0, 0]
    assert np.array_equal(data["q"], np.ones((2, 22)))
    bad = source(tmp_path / "wrong", "different")
    with pytest.raises(ValueError, match="incompatible"):
        merge([a, bad], out)


def test_lightning_registry_needs_only_metric_mesh(tmp_path):
    from sim.isaac.tabletop_catalog import load_registry

    registry = tmp_path / "registry.json"
    objects = {
        "example": {
            "prior": {"source": "fused_metric", "file": "object.npz", "sha256": "a"}
        }
    }
    registry.write_text(
        json.dumps({"schema": "rua.tabletop-geometries.v1", "objects": objects})
    )
    assert load_registry(registry, load_postures=False) == (objects, None)


def test_default_entry_requires_lightning_bank_and_accepts_explicit_comparisons(
    tmp_path, monkeypatch
):
    from sim.isaac.tabletop_native import grasp_source

    monkeypatch.delenv("RUA_SIM_ASSETS", raising=False)
    args = SimpleNamespace(
        library=tmp_path / "registry.json",
        lightning_candidates=None,
        adaptive_grasps=True,
        hybrid_grasps=False,
        mesh_grasps=False,
        template_comparison=False,
    )
    with pytest.raises(ValueError, match="Lightning Grasp bank"):
        grasp_source(args)
    bank = tmp_path / "native"
    bank.mkdir()
    assert grasp_source(args) == ("lightning_grasp", bank)
    args.hybrid_grasps = True
    assert grasp_source(args) == ("adaptive_mesh_and_lightning", bank)
    args.hybrid_grasps = False
    args.mesh_grasps = True
    assert grasp_source(args) == ("adaptive_mesh", None)
    args.lightning_candidates = bank
    with pytest.raises(ValueError, match="conflicts"):
        grasp_source(args)


@pytest.mark.parametrize("name", ["bottle", "carrot"])
def test_adaptive_execution_uses_only_lightning_through_retry(name, monkeypatch):
    from sim.isaac import tabletop_adaptive

    def select(ctrl, target, surface, reg, obstacles, previous):
        assert target == name
        if previous:
            raise NoPath("native candidates exhausted")
        return {"posture": "LG-4"}, {}

    monkeypatch.setattr(
        tabletop_adaptive, "select", lambda *args: pytest.fail("geometric fallback")
    )
    ctrl = Controller.__new__(Controller)
    ctrl.lightning = SimpleNamespace(select=select)
    ctrl.adaptive = True
    ctrl.hybrid_grasps = False
    assert (
        ctrl.fit_registered(name, None, None, None, [], [], None)[0]["posture"]
        == "LG-4"
    )
    with pytest.raises(NoPath, match="native candidates exhausted"):
        ctrl.fit_registered(name, None, None, None, [], ["LG-4"], None)


@pytest.mark.parametrize("hybrid", [False, True])
def test_adaptive_mesh_and_explicit_hybrid_still_use_mesh_selector(hybrid, monkeypatch):
    from sim.isaac import tabletop_adaptive

    ctrl = Controller.__new__(Controller)
    ctrl.lightning = object() if hybrid else None
    ctrl.hybrid_grasps = hybrid
    ctrl.adaptive = True
    monkeypatch.setattr(tabletop_adaptive, "select", lambda *args: ("comparison", {}))
    assert ctrl.fit_registered("item", None, None, None, [], [], None) == (
        "comparison",
        {},
    )


def test_canonical_hand_allows_upright_park_and_rejects_inverted_transit():
    from scipy.spatial.transform import Rotation

    controller = Controller.__new__(Controller)
    controller.lightning = object()
    palm = np.eye(4)
    assert controller.approach_orientation_clear(palm)
    palm[:3, :3] = Rotation.from_euler("x", 65, degrees=True).as_matrix()
    assert controller.approach_orientation_clear(palm)
    palm[:3, :3] = Rotation.from_euler("x", -65, degrees=True).as_matrix()
    assert not controller.approach_orientation_clear(palm)
    palm[:3, :3] = Rotation.from_euler("x", 150, degrees=True).as_matrix()
    assert not controller.approach_orientation_clear(palm)


def test_palm_gap_samples_exclude_back_and_flange():
    from itertools import product

    cad = np.array(list(product((-0.008, 0.011), (-0.04, 0.04), (0.0, 0.10))))
    samples = central_palm_samples(cad)
    assert len(samples) >= 9
    assert np.allclose(samples[:, 0], 0.011)
    assert np.max(np.abs(samples[:, 1])) < 0.024
    assert samples[:, 2].min() > 0.030 and samples[:, 2].max() < 0.082


def test_static_palm_contact_joins_native_contact_search_in_object_frame():
    from scipy.spatial.transform import Rotation
    from sim.isaac.tabletop_lightning_generate import palm_condition_in_object_frame

    poses = np.eye(4)[None]
    poses[0, :3, :3] = Rotation.from_euler("zy", [73, -38], degrees=True).as_matrix()
    poses[0, :3, 3] = [0.07, 0.019, -0.032]
    obj_point = np.array([[[0.008, -0.037, 0.028]]])
    obj_out = np.array([[[0, -1.0, 0]]])
    rot = poses[0, :3, :3]
    condition = dict(
        extra_contact_pos=obj_point @ rot.T + poses[0, :3, 3],
        extra_contact_normal=-obj_out @ rot.T,
        extra_contact_mask=np.ones((1, 1)),
    )
    out = palm_condition_in_object_frame(poses, condition)
    assert np.allclose(out["extra_contact_pos"], obj_point)
    assert np.allclose(out["extra_contact_normal"], -obj_out)


def test_lightning_lift_uses_actual_seating_branch():
    c = Controller.__new__(Controller)
    c.lift_travel = 0.067
    calls = []

    def solve(pose, seed):
        calls.append((pose[:3, 3].copy(), seed.copy()))
        return seed + 1

    c.ik = SimpleNamespace(solve=solve)
    c.lightning_lift_ik(np.eye(4), np.zeros(7))
    assert np.allclose(
        [p for p, _ in calls], [[-0.009, 0, 0], [0, 0, 0], [0, 0, 0.022], [0, 0, 0.067]]
    )
    assert all(np.all(seed == index) for index, (_, seed) in enumerate(calls))


def test_native_collision_integer_flags_select_passing_pose_rows():
    torch = pytest.importorskip("torch")
    from sim.isaac.tabletop_lightning_generate import native_collision_mask

    flags = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    assert torch.arange(4)[native_collision_mask(flags, 4)].tolist() == [1, 3]
    with pytest.raises(ValueError, match="number of rows"):
        native_collision_mask(flags, 5)


def test_same_transform_different_native_joints_keep_their_parent_rows():
    from sim.isaac.tabletop_lightning_yaw import distinct_pose_batches

    a, b = np.eye(4), np.eye(4)
    b[0, 3] = 0.02
    rows = [(4, 10, a), (8, 10, a), (3, 10, b)]
    groups = list(distinct_pose_batches(rows, 96))
    assert [[r[0] for r in g] for g in groups] == [[4, 3], [8]]


def test_yaw_reprojection_compares_native_inward_normal_to_scan_surface():
    from scipy.spatial.transform import Rotation
    from sim.isaac.tabletop_lightning_yaw import yaw_candidates

    angle = np.arange(36) * np.pi / 18
    normals = np.c_[np.cos(angle), np.sin(angle), np.zeros_like(angle)]
    mesh = SimpleNamespace(vertices=0.032 * normals, vertex_normals=normals)
    indices = [0, 9, 18, 27]
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    pose[:3, 3] = [0.055, 0.012, 0.062]
    seeds = dict(
        object_pose=pose[None],
        contact_link_id=np.array([[0, 1, 2, 3]]),
        link_names=np.array(["right_thumb_elastomer", "index", "middle", "ring"]),
        target_pos=(mesh.vertices[indices] @ pose[:3, :3].T + pose[:3, 3])[None],
        target_normal=(-normals[indices] @ pose[:3, :3].T)[None],
    )
    candidates = list(yaw_candidates(mesh, seeds, np.zeros((1, 3)), 90))
    assert [row[1] for row in candidates] == [90, 180, 270]
    for _, _, rotated, points, outward, *_ in candidates:
        assert np.allclose(
            points @ rotated[:3, :3].T + rotated[:3, 3], seeds["target_pos"][0]
        )
        assert np.allclose(-outward @ rotated[:3, :3].T, seeds["target_normal"][0])


def test_oblique_knuckles_keep_thumb_upright():
    from scipy.spatial.transform import Rotation

    controller = Controller.__new__(Controller)
    controller.adaptive = True
    p = np.eye(4)
    p[:3, :3] = Rotation.from_euler("y", 133, degrees=True).as_matrix()
    assert controller.approach_orientation_clear(p)
    p[:3, :3] = Rotation.from_euler("x", -60, degrees=True).as_matrix()
    assert not controller.approach_orientation_clear(p)


def test_forearm_may_pass_in_front_of_table_but_must_clear_its_edge():
    controller = Controller.__new__(Controller)
    controller.table = 0.72
    controller.table_front_x = 0.49
    vertices = np.array([[0.45, 0.0, 0.69], [0.52, 0.0, 0.74]])
    controller.io = SimpleNamespace(
        tree=SimpleNamespace(pose=lambda name, q: np.eye(4)),
        model=SimpleNamespace(forearm_support={"Link6_R": vertices}),
    )
    assert controller.arm_above_table(np.zeros(7))
    vertices[1, 2] = 0.719
    assert not controller.arm_above_table(np.zeros(7))


def test_thumb_up_carry_preserves_the_enclosed_pad_center():
    from sim.isaac.tabletop_adaptive import DIGITS, thumb_up_carry

    pads = np.array([[0.07, 0, -0.008]] + [[0.01, 0.01 * j, 0] for j in range(4)])
    parts = {f"right_{d}_elastomer": p[None] for d, p in zip(DIGITS, pads)}
    controller = SimpleNamespace(link_shapes=lambda hand: parts)
    palm = np.eye(4)
    palm[:3, 3] = [0.61, -0.22, 0.81]
    target = thumb_up_carry(controller, palm, None)
    moved = pads @ target[:3, :3].T + target[:3, 3]
    assert np.allclose(moved.mean(0), pads.mean(0) + palm[:3, 3])
    assert moved[0, 2] > moved[1:, 2].mean()


def test_carry_ik_draws_five_opposed_pads_together():
    from scipy.spatial.transform import Rotation
    from sim.isaac.tabletop_adaptive import DIGITS, draw_pads_to_palm

    support = {
        f"right_{digit}_{part}": np.array([[0.0, 0.012 * i, 0.055]])
        for i, digit in enumerate(DIGITS)
        for part in ("elastomer", "PP" if digit == "thumb" else "MP")
    }

    def link(name, palm, angles):
        i = DIGITS.index(name.split("_")[1])
        frame = palm.copy()
        frame[:3, :3] = Rotation.from_euler("y", angles[i]).as_matrix()
        return frame, [(i, np.array([0, 1, 0]), np.zeros(3))]

    limits = np.tile([-0.5, 0.5], (len(DIGITS), 1))
    model = SimpleNamespace(hand_support=support, limits={"right": limits})
    ctrl = SimpleNamespace(
        io=SimpleNamespace(model=model), hand_contacts=SimpleNamespace(link=link)
    )
    angles = draw_pads_to_palm(ctrl, np.zeros(len(DIGITS)), 0.009)
    for digit in DIGITS:
        name = f"right_{digit}_elastomer"
        pad = support[name][0]
        rotation = link(name, np.eye(4), angles)[0][:3, :3]
        assert np.allclose(rotation @ pad - pad, [-0.009, 0.0, 0.0], atol=0.001)


def test_head_reacquisition_after_a_roll_requires_a_unique_mesh_fit():
    ctrl = Controller.__new__(Controller)
    ctrl.name, ctrl.table = "test", 0.72
    ctrl.event = lambda *args, **kwargs: None
    ref = dict(center=np.zeros(3), color=np.ones(4), dims=np.array([0.07, 0.065, 0.20]))
    new = dict(
        ref,
        center=np.array([0.10, 0.0, 0.0]),
        dims=ref["dims"][[2, 0, 1]],
        xyz=np.zeros((60, 3)),
    )
    ctrl.head_pieces = [new]
    fit = {"median_residual_m": 0.001, "p90_residual_m": 0.004}
    ctrl.models = SimpleNamespace(fit=lambda *args: (None, None, fit))
    assert ctrl.reacquire_head(ref) is None
    assert ctrl.reacquire_head(ref, after_release=True) is not None
    ctrl.head_pieces = [new, new.copy()]
    assert ctrl.reacquire_head(ref, after_release=True) is None
    ctrl.head_pieces = [new]
    fit["p90_residual_m"] = 0.012
    assert ctrl.reacquire_head(ref, after_release=True) is None


def test_tapered_wall_uses_the_radius_at_each_grasp_height(monkeypatch):
    from sim.isaac import tabletop_adaptive as adaptive

    def wall_fit(ctrl, radius, cant, roll, station):
        q = np.zeros(22)
        return np.eye(3), (radius, station), q, q, q

    monkeypatch.setattr(adaptive, "wall_fit", wall_fit)
    shapes = {f"right_{d}_elastomer": np.zeros((1, 3)) for d in adaptive.DIGITS}
    ctrl = SimpleNamespace(
        ik=SimpleNamespace(fk=lambda q: np.eye(4)),
        reference_arm=np.zeros(7),
        wrap_cache={},
        link_shapes=lambda q: shapes,
    )
    z, theta = np.meshgrid(
        np.linspace(0, 0.16, 81), np.linspace(0, 2 * np.pi, 96, endpoint=False)
    )

    def radii_by_level(radius):
        world = np.stack((radius * np.cos(theta), radius * np.sin(theta), z), axis=-1)
        surface = SimpleNamespace(world=world.reshape(-1, 3))
        return {
            card["fit"]["level"]: card["fit"]["radius_m"]
            for _, card, _ in adaptive.wall_grasps(ctrl, surface)
        }

    cylinder = radii_by_level(np.full_like(z, 0.030))
    assert cylinder == {0.40: 0.030, 0.49: 0.030, 0.58: 0.030}
    tapered = radii_by_level(0.031 - 0.026 * np.clip((z / 0.16 - 0.35) / 0.65, 0, 1))
    assert {0.46, 0.52, 0.55} <= tapered.keys()
    assert tapered[0.58] < tapered[0.49]
