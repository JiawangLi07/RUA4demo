from itertools import product

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from sim.isaac.tabletop_collision_ik import ConvexBody, constraints, point_jacobian


def test_rotated_link_moves_away_from_points_inside_its_surface():
    body = ConvexBody(np.array(list(product((-0.01, 0.01), repeat=3))), 1)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    poses[1, :3, 3] = [0.03, 0, 0.02]
    samples = cKDTree([[0.03, 0.009, 0.02], [1, 1, 1]])
    plane = (np.array([0, 0, 1]), 0)
    (correction,) = constraints([body], poses, samples, [], plane)
    link, other, _, displacement, _ = correction
    assert link == 1 and other is None
    assert np.allclose(displacement, [0, -0.0016, 0])
    poses[1, :3, 3] += displacement
    assert not constraints([body], poses, samples, [], plane)


def test_floor_and_self_collision_have_distinct_relative_velocities():
    verts = np.array(list(product((-0.01, 0.01), repeat=3)))
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, 2, 3] = 0.008
    poses[1, :3, 3] += [0.017, 0.0005, 0.0005]
    bodies = [ConvexBody(verts, i) for i in (0, 1)]
    samples = cKDTree([[1, 1, 1]])
    plane = (np.array([0, 0, 1]), 0.0)
    rows = constraints(bodies, poses, samples, [(0, 1)], plane)
    assert sum(other is None for _, other, _, _, _ in rows) == 2
    for link, other, _, displacement, _ in rows:
        if other is None:
            assert displacement[2] > 0
        else:
            assert np.dot(displacement, poses[link, :3, 3] - poses[other, :3, 3]) > 0
    assert any(other is not None for _, other, _, _, _ in rows)


def test_point_jacobian_matches_link_rotation_around_offset_point():
    jac = np.zeros((6, 1))
    jac[5, 0] = 1
    offset = np.array([0.02, 0.03, -0.005])
    numerical = (Rotation.from_rotvec([0, 0, 1e-6]).apply(offset) - offset) / 1e-6
    np.testing.assert_allclose(point_jacobian(jac, offset)[:, 0], numerical, atol=2e-8)
