"""Table, object proposals and lift observations derived solely from calibrated RGB-D and robot CAD."""

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .head_wrist_surface import RegisteredSurface
from .scene_meshes import load_mesh_bundle


def world_image(frame):
    depth, k, pose = frame["depth"], frame["K"], frame["pose"]
    v, u = np.indices(depth.shape)
    good = np.isfinite(depth) & (depth > 0.025) & (depth < 2.2)
    z = np.where(good, depth, 0)
    cam = np.stack(
        ((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z), axis=-1
    )
    return cam @ pose[:3, :3].T + pose[:3, 3], good


def outside_robot(points, io, q):
    valid = np.ones(len(points), bool)
    params = dict(zip(io.names, q))
    for name, bounds in io.bounds.items():
        pose = io.tree.pose(name, params)
        # Broad-phase before transforming potentially hundreds of thousands of depth points.
        center = pose[:3, :3] @ bounds.mean(0) + pose[:3, 3]
        extent = np.abs(pose[:3, :3]) @ (bounds[1] - bounds[0]) / 2 + 0.003
        indices = np.flatnonzero(
            valid & np.all(np.abs(points - center) <= extent, axis=1)
        )
        if len(indices):
            local = (points[indices] - pose[:3, 3]) @ pose[:3, :3]
            valid[indices] &= ~np.all(
                (local >= bounds[0] - 0.0025) & (local <= bounds[1] + 0.0025), axis=1
            )
    return valid


def foreground(frame, io, table=None):
    world, valid = world_image(frame)
    roi = (
        valid
        & (world[:, :, 0] > 0.49)
        & (world[:, :, 0] < 1.12)
        & (world[:, :, 1] > -0.67)
        & (world[:, :, 1] < 0.40)
    )
    z = world[:, :, 2]
    if table is None:
        heights = z[roi & (z > 0.66) & (z < 0.77)]
        if len(heights) < 200:
            raise ValueError("table is not observable")
        count, edges = np.histogram(heights, bins=130, range=(0.655, 0.785))
        t = edges[count.argmax()] + 0.0005
        table = float(np.median(heights[np.abs(heights - t) < 0.0025]))
    mask = roi & (z > table + 0.003) & (z < table + 0.36)
    v, u = np.nonzero(mask)
    xyz = world[v, u]
    keep = outside_robot(xyz, io, frame["q"])
    return (
        float(table),
        xyz[keep],
        frame["rgb"][v[keep], u[keep]],
        np.c_[u[keep], v[keep]],
    )


def dimensions(xyz, table):
    _, axes = np.linalg.eigh(np.cov(xyz[:, :2].T))
    uv = xyz[:, :2] @ axes
    xy = np.sort(np.ptp(np.quantile(uv, [0.015, 0.985], axis=0), axis=0))[::-1]
    return np.maximum([*xy, np.quantile(xyz[:, 2], 0.985) - table], 0.004)


def appearance(rgb):
    color = np.median(rgb, axis=0).astype(float) / 255
    return np.r_[color / (0.035 + color.sum()), color.mean()]


def proposals(frame, io, table=None):
    table, xyz, rgb, uv = foreground(frame, io, table)
    if not len(xyz):
        return table, []
    cell = 0.0035
    grid = np.floor((xyz[:, :2] - [0.48, -0.68]) / cell).astype(int)
    occupied = np.zeros(tuple(grid.max(0) + 3), bool)
    occupied[tuple(grid.T)] = True
    labels, _ = ndimage.label(
        ndimage.binary_dilation(occupied, iterations=1), structure=np.ones((3, 3))
    )
    ids = labels[tuple(grid.T)]
    items = []
    for i in np.unique(ids):
        use = ids == i
        if use.sum() < 14:
            continue
        pts = xyz[use]
        dims = dimensions(pts, table)
        if dims[0] < 0.01 or dims[0] > 0.42 or dims[2] > 0.30:
            continue
        items.append(
            {
                "xyz": pts,
                "rgb": rgb[use],
                "uv": uv[use],
                "dims": dims,
                "color": appearance(rgb[use]),
                "center": np.median(pts, axis=0),
            }
        )
    return table, items


class Models:
    def __init__(self, library):
        self.library = library
        self.bundles = {
            name: load_mesh_bundle(row["prior"]) for name, row in library.items()
        }
        self.features = {}
        for name, bundle in self.bundles.items():
            v, _, _ = bundle["visual"]
            self.features[name] = (
                dimensions(v, v[:, 2].min()),
                np.asarray(library[name]["color"]),
            )

    def identify(self, components):
        if not components:
            return {}
        names = list(self.features)
        costs = []
        for det in components:
            row = []
            for n in names:
                dims, color = self.features[n]
                diff = np.log(det["dims"] / (dims * [0.94, 0.83, 0.98]))
                row.append(
                    np.linalg.norm(diff * [0.95, 0.63, 1.2])
                    + 0.62 * np.linalg.norm(det["color"][:3] - color[:3])
                    + 0.75 * abs(det["color"][3] - color[3])
                )
            costs.append(row)
        rows, cols = linear_sum_assignment(costs)
        return {
            names[j]: dict(components[i], match_cost=float(costs[i][j]))
            for i, j in zip(rows, cols)
            if costs[i][j] < 2.0
        }

    def fit(self, name, points, table, previous=None):
        surface = RegisteredSurface(self.bundles[name])
        if previous is not None:
            surface.fit = previous.fit.copy()
        info = surface.register(points, table)
        pose = np.eye(4)
        pose[:3, :3] = info["pose_world_rotation"]
        pose[:3, 3] = info["pose_world_translation_m"]
        return surface, pose, info


def wrist_points(frame, io, surface, table, margin=0.015):
    _, pts, _, _ = foreground(frame, io, table)
    if not len(pts):
        return pts
    low, high = surface.world.min(0) - margin, surface.world.max(0) + margin
    pts = pts[np.all((pts >= low) & (pts <= high), axis=1)]
    if len(pts):
        dist = cKDTree(surface.world).query(pts)[0]
        pts = pts[dist < margin]
    return pts


def match_height(frame, io, table, model_points, resting, held, reference_color):
    """Compare image support for a table-fixed and palm-carried target in the same exposure."""
    _, pts, rgb, _ = foreground(frame, io, table)
    ref0 = model_points @ resting[:3, :3].T + resting[:3, 3]
    ref1 = model_points @ held[:3, :3].T + held[:3, 3]
    lo = np.minimum(ref0.min(0), ref1.min(0)) - 0.016
    hi = np.maximum(ref0.max(0), ref1.max(0)) + 0.016
    crop = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts, rgb = pts[crop], rgb[crop]
    if len(pts) < 18:
        return {"votes": "unknown", "points": len(pts)}
    color = rgb.astype(float) / 255
    chroma = color / (0.035 + color.sum(1, keepdims=True))
    compatible = (np.linalg.norm(chroma - reference_color[:3], axis=1) < 0.30) & (
        np.abs(color.mean(1) - reference_color[3]) < 0.27
    )
    pts = pts[compatible]
    if len(pts) < 18:
        return {"votes": "unknown", "points": len(pts)}
    if len(pts) > 2200:
        pts = pts[:: int(np.ceil(len(pts) / 2200))]
    d0 = cKDTree(ref0).query(pts)[0]
    d1 = cKDTree(ref1).query(pts)[0]
    eligible = np.minimum(d0, d1) < 0.010
    # Only pixels that distinguish the hypotheses vote. A vertical cylinder wall alone is ambiguous.
    wins0 = np.count_nonzero(eligible & (d0 + 0.003 < d1))
    wins1 = np.count_nonzero(eligible & (d1 + 0.003 < d0))
    vote = "unknown"
    if wins1 >= 10 and wins1 > 1.7 * wins0:
        vote = "carried"
    if wins0 >= 10 and wins0 > 1.7 * wins1:
        vote = "on_table"
    return {
        "votes": vote,
        "points": len(pts),
        "carried_pixels": int(wins1),
        "static_pixels": int(wins0),
    }
