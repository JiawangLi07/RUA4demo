"""Hash-pinned visual meshes and convex collider parts; no source code execution."""

from __future__ import annotations

import hashlib
import io
from pathlib import PurePosixPath

import numpy as np

from .paths import asset_path


def _mesh(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 4
        or vertices.dtype.kind not in "fi"
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("mesh vertices must be finite Nx3 real coordinates in metres")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("mesh faces must contain valid triangle indices")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        vertices = vertices.astype(np.float32).astype(float)
        triangles = vertices[faces]
        areas = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
    if not np.isfinite(vertices).all() or not np.isfinite(areas).all() or np.any(areas == 0):
        raise ValueError("mesh triangles collapse or overflow in USD float32 coordinates")
    return vertices, faces.astype(np.int32)


def load_mesh_bundle(spec, *, assets=None):
    """Read prepared metre-space arrays; callers must retain source attribution.

    NPZ keys: visual_vertices, visual_faces, visual_colors, collision_count, and
    collision_NNN_vertices/faces. Colours are constant or per-face linear RGB.
    Collider parts are authored as separate convex hulls, never one hull spanning
    the complete multi-part object. Preparation owns convex decomposition/units.
    """
    source, relative, expected = spec["source"], PurePosixPath(spec["file"]), spec["sha256"]
    if (
        not isinstance(source, str)
        or len(PurePosixPath(source).parts) != 1
        or source in (".", "..")
        or PurePosixPath(source).is_absolute()
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
    ):
        raise ValueError("mesh bundle must be relative to one third-party source")
    root = asset_path(source, base=assets).resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("mesh bundle escapes its third-party source")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("mesh bundle SHA256 mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        vertices, faces = _mesh(archive["visual_vertices"], archive["visual_faces"])
        colors = np.asarray(archive["visual_colors"], dtype=float)
        if (
            colors.shape not in ((1, 3), (len(faces), 3))
            or not np.isfinite(colors).all()
            or np.any((colors < 0) | (colors > 1))
        ):
            raise ValueError("visual colours must be constant or per-triangle RGB in [0,1]")
        count = archive["collision_count"]
        if count.shape != () or count.dtype.kind not in "iu" or not 1 <= int(count) <= 128:
            raise ValueError("collision_count must be an integer in [1,128]")
        collisions = []
        for index in range(int(count)):
            part = _mesh(archive[f"collision_{index:03d}_vertices"], archive[f"collision_{index:03d}_faces"])
            if np.linalg.matrix_rank(part[0] - part[0].mean(axis=0)) != 3:
                raise ValueError("convex collision part must have nonzero volume")
            collisions.append(part)
    return {"visual": (vertices, faces, colors), "collisions": collisions}
