"""Sample Lightning Grasp placements and fingertip domains above a registered head-depth plane."""

import numpy as np
from scipy.spatial.transform import Rotation


def basis(normal):
    tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    tangent /= np.linalg.norm(tangent)
    return np.column_stack((tangent, np.cross(normal, tangent), normal))


def placements(api, torch, robot, field, tree, static, pts, normals, obj, args):
    # Sample wrist tilt independently of object yaw. Each proposal goes through native
    # contact domain search, contact optimization, finger IK and collision filtering.
    count = args.outer
    n, h = np.asarray(args.room["normal"]), args.room["offset"]
    obj_center = 0.5 * (obj.mesh.vertices.max(0) + obj.mesh.vertices.min(0))
    indices = np.random.randint(len(args.parents), size=count)
    prior = args.parents[indices]
    centers = prior[:, :3, :3] @ obj_center + prior[:, :3, 3]
    centers += np.random.normal(0.0, [0.006, 0.006, 0.009], size=centers.shape)
    centers[:, 2] += np.random.uniform(0.001, 0.027, size=count)
    axis = -centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    up = axis + np.random.normal(0.0, 0.65, (count, 3))
    up[:, 1] = np.maximum(up[:, 1], -0.11)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    if getattr(args, "patch_prior", None) is not None:
        other = args.patch_prior
        choice = np.random.randint(len(other["up"]), size=count)
        up = other["up"][choice] + np.random.normal(0.0, 0.18, (count, 3))
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        centers = other["center"][choice] + np.random.normal(
            0.0, [0.004, 0.004, 0.006], (count, 3)
        )
    yaws = Rotation.from_euler("z", np.random.uniform(-np.pi, np.pi, count)).as_matrix()
    R = np.stack([basis(row) for row in up]) @ yaws @ basis(n).T
    a = np.tile(np.eye(4), (count, 1, 1))
    a[:, :3, :3] = R
    a[:, :3, 3] = centers - R @ obj_center
    if getattr(args, "patch_prior", None) is not None:
        required = other["floor"][choice] + np.random.uniform(-0.003, 0.004, count)
        current = h - np.einsum("ij,ij->i", up, a[:, :3, 3])
        a[:, :3, 3] += (current - required)[:, None] * up
    good = (
        np.einsum("ij,kj->ik", up, args.fixture).min(axis=1)
        - np.einsum("ij,ij->i", up, a[:, :3, 3])
        + h
    ) > 0.004
    return torch.as_tensor(a[good], device=pts.device, dtype=pts.dtype), {}


def tabletop_points(torch, pts, room):
    n = np.asarray(room["normal"])
    t = basis(n)
    x, y = np.meshgrid(np.linspace(-0.098, 0.098, 36), np.linspace(-0.098, 0.098, 36))
    sheets = [
        np.c_[x.ravel(), y.ravel(), np.full(x.size, -room["offset"] - d)] @ t.T
        for d in (0.0, 0.009, 0.021)
    ]
    return torch.as_tensor(np.concatenate(sheets), device=pts.device, dtype=pts.dtype)
