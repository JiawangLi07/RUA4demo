"""Rigid transforms and collision support in station coordinates."""

from itertools import product
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def _array(value: object, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat):
        raise ValueError(f"{name}: numeric array of shape {shape} required")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: finite values required")
    return array


def rigid_transform(value: object, name: str = "transform") -> NDArray[np.float64]:
    matrix = _array(value, (4, 4), name)
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
        or abs(np.linalg.det(rotation) - 1) > 1e-5
    ):
        raise ValueError(f"{name}: proper rigid transform required; no implicit rotation repair")
    return matrix


def pose_matrix(position, quaternion_wxyz):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    pose[:3, 3] = position
    return pose


def box_corners(bounds):
    lower, upper = np.asarray(bounds, float)
    if not np.isfinite([lower, upper]).all() or np.any(lower > upper):
        raise ValueError("invalid bounding envelope")
    return np.array(list(product(*zip(lower, upper))))


def transform_points(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
