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
    def __init__(self, library, mirror=False):
        self.library = library
        self.bundles = {
            name: load_mesh_bundle(row["prior"]) for name, row in library.items()
        }
        if mirror:
            from .tabletop_sides import reflected_bundle

            self.bundles = {
                name: reflected_bundle(b) for name, b in self.bundles.items()
            }
        self.features = {}
        for name, bundle in self.bundles.items():
            v, _, colors = bundle["visual"]
            self.features[name] = (
                dimensions(v, v[:, 2].min()),
                np.asarray(
                    library[name].get(
                        "color",
                        appearance(
                            (np.clip(colors, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
                        ),
                    )
                ),
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
        surface = RegisteredSurface(self.bundles[name], sample_faces=True)
        if previous is not None:
            surface.fit = previous.fit.copy()
        info = surface.register(points, table)
        pose = np.eye(4)
        pose[:3, :3] = info["pose_world_rotation"]
        pose[:3, 3] = info["pose_world_translation_m"]
        return surface, pose, info


def add_live_samples(surface, head, wrist):
    """Retain real camera samples for surface targets and for local handle cross sections."""
    surface.camera_samples = {}
    for name, cloud in (("head_depth", head), ("wrist_depth", wrist)):
        if not len(cloud):
            continue
        unique = np.unique(
            np.floor(cloud / 0.0025).astype(int), axis=0, return_index=True
        )[1]
        pts = cloud[unique]
        pts = pts[:: max(1, int(np.ceil(len(pts) / 1600)))]
        good = cKDTree(surface.world).query(pts)[0] < 0.013
        if good.sum() >= 8:
            surface.camera_samples[name] = pts[good]
    # Correct only locally observed parts. Unseen surfaces retain the registered shape prior.
    updated = surface.world.copy()
    for cloud in surface.camera_samples.values():
        dist, idx = cKDTree(cloud).query(surface.world, k=min(5, len(cloud)))
        local = dist[:, -1] < 0.012
        if local.any():
            residual = cloud[idx[local]].mean(1) - surface.world[local]
            updated[local] += np.clip(residual, -0.004, 0.004) * 0.35
    surface.handle_world = updated
    return surface


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
    d1, near1 = cKDTree(ref1).query(pts)
    eligible = np.minimum(d0, d1) < 0.010
    # Only pixels that distinguish the hypotheses vote. A vertical cylinder wall alone is ambiguous.
    wins0 = np.count_nonzero(eligible & (d0 + 0.003 < d1))
    wins1 = np.count_nonzero(eligible & (d1 + 0.003 < d0))
    vote = "unknown"
    if wins1 >= 10 and wins1 > 1.7 * wins0:
        vote = "carried"
    if wins0 >= 10 and wins0 > 1.7 * wins1:
        vote = "on_table"
    evidence = {}
    displacement = held[:3, 3] - resting[:3, 3]
    # One moving end of a long tool can resemble the raised template while
    # its other end still rests on the table. Require spatially separated
    # matches near the estimated bottom for a geometric height witness.
    if displacement[2] > 0.035:
        matches = (d1 < 0.005) & (d1 + 0.003 < d0)
        if matches.sum() >= 18:
            _, _, axes = np.linalg.svd(ref1 - ref1.mean(0), full_matrices=False)
            coord = ref1 @ axes[0]
            low, high = np.quantile(coord, [0.015, 0.985])
            sample = coord[near1[matches]]
            cover = np.quantile(sample, [0.06, 0.94])
            lower_band = ref1[near1[matches], 2] < ref1[:, 2].min() + max(
                0.005, 0.16 * np.ptp(ref1[:, 2])
            )
            if (
                cover[0] <= low + 0.26 * (high - low)
                and cover[1] >= high - 0.26 * (high - low)
                and lower_band.sum() >= 4
            ):
                dz = np.median((pts[matches, 2] - ref1[near1[matches], 2])[lower_band])
                evidence["geometry_clearance_m"] = round(
                    float(ref1[:, 2].min() - table + dz), 5
                )
                evidence["geometry_rise_m"] = round(float(displacement[2] + dz), 5)
    if 0.010 < displacement[2] < 0.042:
        # A compliant grasp can take up clearance before the object starts moving.
        # Estimate continuous visible object motion during the small verification lift.
        amounts = np.linspace(0, 1.20, 10)
        initial = cKDTree(ref0)
        scores = np.array(
            [initial.query(pts - alpha * displacement)[0] for alpha in amounts]
        )
        use = (scores.min(0) < 0.009) & (np.ptp(scores, axis=0) > 0.005)
        if use.sum() >= 24:
            loss = np.mean(np.minimum(scores[:, use], 0.012), axis=1)
            best = int(np.argmin(loss))
            evidence = {
                "visible_lift_m": round(float(amounts[best] * displacement[2]), 5),
                "height_support": int(use.sum()),
                "height_margin_m": round(float(loss[0] - loss[best]), 6),
            }
    return {
        **evidence,
        "votes": vote,
        "points": len(pts),
        "carried_pixels": int(wins1),
        "static_pixels": int(wins0),
    }
