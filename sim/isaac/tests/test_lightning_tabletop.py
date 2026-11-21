import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.isaac.paths import RIGHT_JOINTS
from sim.isaac.tabletop_control import Controller
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
