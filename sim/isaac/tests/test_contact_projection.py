import numpy as np
import pytest
from scipy.spatial import cKDTree

from sim.isaac.tabletop_scene_solver import MIN_NORMAL_DOT, match_surface_patches


@pytest.mark.parametrize("outer_z,expected_matches", [(0.0055, 1), (0.010, 0)])
def test_dense_thin_wall_uses_only_aligned_surface_within_metric_tolerance(
    outer_z, expected_matches
):
    y = np.linspace(-0.001, 0.001, 32)
    inner = np.column_stack((np.zeros_like(y), y, np.zeros_like(y)))
    vertices = np.vstack((inner, [0, 0, outer_z]))
    normals = np.tile([0.0, 0.0, -1.0], (len(vertices), 1))
    normals[-1] *= -1
    # One contact asks for the outer wall and the other for the inner wall.
    # The initial 24 nearest vertices of both are all on the inner wall.
    sites = np.zeros((2, 3))
    outward = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    indices, gaps, alignment, recovered = match_surface_patches(
        cKDTree(vertices), normals, sites, outward
    )
    assert recovered == expected_matches
    assert (alignment[0] >= MIN_NORMAL_DOT) == bool(expected_matches)
    if expected_matches:
        assert indices[0] == len(vertices) - 1
        assert gaps[0] == pytest.approx(outer_z)
    assert indices[1] < len(inner)
    assert alignment[1] == 1 and gaps[1] < 0.001
