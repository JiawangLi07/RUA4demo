import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.isaac.paths import RIGHT_JOINTS
from sim.isaac.tabletop_control import Controller
from sim.isaac.tabletop_lightning import LightningBank
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
