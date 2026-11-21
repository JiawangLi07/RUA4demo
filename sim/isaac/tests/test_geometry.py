"""Regression tests for the calibrated two-view can probe."""

import numpy as np

from sim.isaac.head_wrist_geometry import usd_camera_intrinsics
from sim.isaac.head_wrist_trial import cylinder


def test_native_wrist_offaxis_intrinsics():
    K = usd_camera_intrinsics(480, 640, 24, 33.005703, 43.45813, 2.7501652, -2.349328)
    expected = [[349.0306, 0, 200.0045], [0, 353.4436, 285.4019], [0, 0, 1]]
    np.testing.assert_allclose(K, expected, atol=0.0001, rtol=0)


def test_wrist_refits_xy_and_radius_on_a_partial_measured_side():
    angles, heights = np.meshgrid(np.linspace(-2.8, -1.0, 30), np.linspace(0.73, 0.82, 20))
    radius = 0.033
    surface = np.column_stack(
        [(0.62 + radius * np.cos(angles)).ravel(), (-0.225 + radius * np.sin(angles)).ravel(), heights.ravel()]
    )
    head = {"center": [0.618, -0.226, 0.77], "radius_m": 0.031, "top_m": 0.83}
    wrist = cylinder(surface, 0.71, head)
    np.testing.assert_allclose(wrist["center"], [0.62, -0.225, 0.77], atol=1e-6)
    np.testing.assert_allclose(wrist["radius_m"], radius, atol=1e-6)


def test_registered_prior_recovers_a_visible_offcenter_box():
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    from sim.isaac.head_wrist_surface import RegisteredSurface

    x, y = np.meshgrid(np.linspace(-0.045, 0.045, 25), np.linspace(-0.02, 0.02, 17))
    xwall, zwall = np.meshgrid(np.linspace(-0.045, 0.045, 25), np.linspace(-0.015, 0.015, 15))
    ywall, zside = np.meshgrid(np.linspace(-0.02, 0.02, 17), np.linspace(-0.015, 0.015, 15))
    side = np.c_[np.full(ywall.size, -0.045), ywall.ravel(), zside.ravel()]
    front = np.c_[xwall.ravel(), np.full(xwall.size, -0.02), zwall.ravel()]
    top = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.015)]
    vertices = np.vstack([side, front, top, -side, -front, -top])
    hull = ConvexHull(vertices)
    prior = RegisteredSurface(
        {"visual": (vertices, hull.simplices, None), "collisions": [(vertices[hull.vertices], None)]}
    )
    rot = Rotation.from_euler("z", 0.42).as_matrix()
    xyz = np.array([0.64, -0.21, 0.735])
    points = np.vstack([side, front, top]) @ rot.T + xyz
    fit = prior.register(points, table=0.72)
    assert fit["median_residual_m"] < 0.002
    np.testing.assert_allclose(fit["center"], xyz, atol=0.004)
    assert any(0.034 < row[3] < 0.044 for row in prior.bands())


def test_direct_grasp_family_follows_surface_geometry_after_repositioning():
    from scipy.spatial.transform import Rotation

    from sim.isaac.direct_pregrasp import family

    angles, levels = np.meshgrid(np.linspace(0, 2 * np.pi, 96, endpoint=False), np.linspace(0, 1, 41))
    for rx, ry, height, taper, expected in (
        (0.03, 0.03, 0.12, 0.0, "side_wall"),
        (0.045, 0.04, 0.02, 0.0, "over_edge"),
        (0.06, 0.015, 0.028, 0.0, "across_handle"),
        (0.04, 0.04, 0.07, 0.0, "round_wrap"),
        (0.035, 0.035, 0.11, 0.85, "narrow_tip"),
    ):
        scale = 1 - taper * levels
        points = np.c_[
            (rx * scale * np.cos(angles)).ravel(), (ry * scale * np.sin(angles)).ravel(), (height * levels).ravel()
        ]
        for yaw in (0.0, 0.72, -1.21):
            rotated = Rotation.from_euler("z", yaw).apply(points)
            assert family(rotated + [0.62, -0.21, 0.71]) == expected


def test_direct_pregrasp_visibility_includes_the_preshaped_hand():
    from sim.isaac.direct_pregrasp import WristVisibility

    K = np.array([[349, 0, 200], [0, 353, 285], [0, 0, 1]])
    pad = np.array([[-0.008, -0.008, 0.06], [0.008, -0.008, 0.06], [0.008, 0.008, 0.06], [-0.008, 0.008, 0.06]])
    vision = WristVisibility(np.eye(4), K, {"pad": pad})
    target = np.array([[0, 0, 0.13], [0.05, 0, 0.13], [0, 0, -0.10], [0.4, 0, 0.13]])
    assert vision.fraction(np.eye(4), target) == 0.25
    shifted = np.eye(4)
    shifted[:3, 3] = [0.60, -0.21, 0.78]
    assert vision.fraction(shifted, target + shifted[:3, 3]) == 0.25


def test_direct_grasp_uses_assigned_mesh_and_requires_the_second_exposure(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from sim.isaac import scene_meshes
    from sim.isaac.direct_pregrasp import DirectGrasp
    from sim.isaac.head_wrist_trial import ARM, Rejected

    expected = {"file": "target.npz"}
    vertices = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]])

    def load(prior):
        assert prior == expected
        return {"visual": (vertices, faces, None), "collisions": [(vertices, faces)]}

    monkeypatch.setattr(scene_meshes, "load_mesh_bundle", load)
    config = {
        "head_wrist_validation": {"direct_pregrasp": True, "use_wrist": False},
        "assignment": {"right": "target"},
        "added_objects": [{"name": "other", "mesh": {"file": "other.npz"}}, {"name": "target", "mesh": expected}],
        "physics_dt": 1 / 240,
    }
    scene = SimpleNamespace(
        config=config, out=tmp_path, arm_ids={k: i for i, k in enumerate(ARM)}, ids={"right": []}
    )
    trial = DirectGrasp(scene)
    np.testing.assert_array_equal(trial.surface.parts[0], vertices)
    with pytest.raises(Rejected, match="fresh wrist observation"):
        trial.execute()
