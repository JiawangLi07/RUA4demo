import json
import sys

import numpy as np
import pytest
from scipy.spatial import ConvexHull

trimesh = pytest.importorskip("trimesh")
from sim.isaac import tabletop_fused_assets as converter


def test_decomposition_preserves_a_compact_objects_opening():
    pytest.importorskip("vhacdx")
    ring = trimesh.creation.annulus(r_min=0.045, r_max=0.060, height=0.035, sections=16)

    def contains(parts, point):
        for vertices, _ in parts:
            planes = ConvexHull(vertices).equations
            if np.all(planes[:, :3] @ point + planes[:, 3] <= 1e-7):
                return True
        return False

    solid = converter.collision_hulls(ring.vertices, ring.faces, decompose=False)
    hollow = converter.collision_hulls(ring.vertices, ring.faces, decompose=True)
    assert contains(solid, [0, 0, 0])
    assert not contains(hollow, [0, 0, 0])
    assert contains(hollow, [0.052, 0, 0])


def test_converter_reuses_reference_mass_for_a_hash_named_bundle(tmp_path, monkeypatch):
    (tmp_path / "evaluation.json").write_text(json.dumps({
        "objects": 1, "objects_metrics": [{"id": "box"}],
    }))
    reference = tmp_path / "scene.json"
    reference.write_text(json.dumps({"added_objects": [{
        "name": "box", "mesh": {"file": "fused_example.npz"}, "mass_kg": 0.075,
    }]}))
    monkeypatch.setattr(sys, "argv", [
        "tabletop_fused_assets", "--input", str(tmp_path), "--out", str(tmp_path),
        "--mass-reference", str(reference),
    ])

    def prepare(path, record, out, mass, *, decompose):
        assert mass == 0.075 and not decompose
        return "fused_example", {}, {}, {"faces": 12}

    monkeypatch.setattr(converter, "prepare", prepare)
    converter.main()
