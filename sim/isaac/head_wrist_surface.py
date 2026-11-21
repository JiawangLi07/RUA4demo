"""Register a metric shape prior to measured depth and plan a local opposed pinch."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .head_wrist_geometry import hand_shapes


class RegisteredSurface:
    """A support-aligned local mesh, with pose inferred from the exposed surface."""

    def __init__(self, bundle, *, sample_faces=False):
        vertices, faces, _ = bundle["visual"]
        points = np.vstack([vertices, vertices[faces].mean(axis=1)])
        # A coarse box or a shaft with two triangular end caps still has continuous
        # faces. Include area-weighted surface samples when vertex spacing is coarse.
        triangles = vertices[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1) / 2
        if sample_faces and (area.max() > 0.00005 or len(points) < 500):
            n = int(np.clip(area.sum() / 0.000006, 900, 6500))
            rng = np.random.default_rng(0)
            tri = triangles[rng.choice(len(triangles), n, p=area / area.sum())]
            u, v = rng.random((2, n))
            u = np.sqrt(u)
            points = np.vstack([points, ((1-u)[:, None] * tri[:, 0]
                                + (u*(1-v))[:, None] * tri[:, 1] + (u*v)[:, None] * tri[:, 2])])
        self.local = points[:: max(1, len(points) // 9000)]
        self.tree = cKDTree(self.local)
        self.parts = [v for v, _ in bundle["collisions"]]
        self.support = np.vstack(self.parts)
        self.fit = None
        self.detail_points = self.detail_normals = None
        if sample_faces:
            valid = area > 1e-12
            tri = triangles[valid]
            normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            # Equal-area sampling preserves the original visual surface, independent
            # of tessellation density and of the convex bodies used by dynamics.
            weights = area[valid]
            count = int(np.clip(weights.sum() / 0.000003, 1800, 18000))
            rng = np.random.default_rng(17)
            ids = rng.choice(len(tri), count, p=weights/weights.sum())
            u, v = rng.random((2, count))
            u = np.sqrt(u)
            self.detail_points = (1-u[:, None])*tri[ids, 0] + (u*(1-v))[:, None]*tri[ids, 1] + (u*v)[:, None]*tri[ids, 2]
            self.detail_normals = normal[ids]

    def register(self, observed, table):
        points = observed[:: max(1, len(observed) // 680)]
        zlo = self.support[:, 2].min()
        prior = self.fit

        def pose(params):
            # The rest face is a class prior. Small tilts are measured from this exposure.
            rotation = Rotation.from_euler("zyx", params[2:5]).as_matrix()
            shift = np.array([params[0], params[1], table - np.min(self.support @ rotation[2])])
            return rotation, shift

        def residual(params):
            rot, shift = pose(params)
            return self.tree.query((points - shift) @ rot)[0]

        if prior is None:
            center = (np.quantile(points[:, :2], 0.01, axis=0) + np.quantile(points[:, :2], 0.99, axis=0)) / 2
            starts = [np.array([*center, a, 0.0, 0.0]) for a in np.arange(0, 2 * np.pi, np.pi / 4)]
            coarse = sorted(starts, key=lambda p: np.mean(np.minimum(residual(p), 0.024)))[:4]
            bounds = ([*(center - 0.12), -2 * np.pi, -0.40, -0.40], [*(center + 0.12), 4 * np.pi, 0.40, 0.40])
        else:
            coarse = [prior]
            span = np.array([0.035, 0.035, 0.42, 0.22, 0.22])
            bounds = (prior - span, prior + span)
        best = None
        for seed in coarse:
            opt = least_squares(
                residual,
                seed,
                bounds=bounds,
                loss="soft_l1",
                f_scale=0.0023,
                max_nfev=46,
                x_scale=[0.025, 0.025, 0.28, 0.15, 0.15],
            )
            errors = residual(opt.x)
            score = float(np.mean(np.minimum(errors, 0.025)))
            if best is None or score < best[0]:
                best = (score, opt.x, errors)
        _, q, errors = best
        median = float(np.median(errors))
        p90 = float(np.quantile(errors, 0.9))
        if median > 0.006 or p90 > 0.016:
            raise ValueError(f"mesh-to-depth fit rejected: median {median:.4f} m, p90 {p90:.4f} m")
        self.fit = q
        rot, shift = pose(q)
        self.place(rot, shift, table)
        center = (self.world.min(0) + self.world.max(0)) / 2
        return {
            "center": center.tolist(),
            "radius_m": float(np.max(np.ptp(self.world[:, :2], axis=0)) / 2),
            "top_m": float(self.world[:, 2].max()),
            "median_residual_m": median,
            "p90_residual_m": p90,
            "pose_world_translation_m": shift.tolist(),
            "pose_world_rotation": rot.tolist(),
            "upright_local_bottom_m": float(zlo),
            "fit_samples": len(points),
        }

    def place(self, rot, shift, table):
        """Express this metric prior and its convex supports in a known table frame."""
        self.world = self.local @ rot.T + shift
        if self.detail_points is not None:
            self.detail_world = self.detail_points @ rot.T + shift
            self.detail_world_normals = self.detail_normals @ rot.T
        self.world_parts = [p @ rot.T + shift for p in self.parts]
        # Slab directions for conservative whole-object approach checks, including distinct colliders.
        angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        horizontal = np.c_[np.cos(angle), np.sin(angle)]
        dirs = [np.c_[horizontal, np.full(len(angle), s)] for s in (0.0, 0.65, -0.65, 1.65, -1.65)]
        self.directions = np.vstack([*dirs, [0.0, 0.0, 1.0]])
        self.directions /= np.linalg.norm(self.directions, axis=1, keepdims=True)
        self.bounds = []
        for part in self.world_parts:
            proj = part @ self.directions.T
            self.bounds.append((part.min(0), part.max(0), proj.min(0), proj.max(0)))
        self.table = table

    def clear(self, shapes, pose, margin):
        for v in shapes.values():
            p = v @ pose[:3, :3].T + pose[:3, 3]
            low, high = p.min(0), p.max(0)
            if low[2] < self.table + margin:
                return False
            projection = None
            for vlo, vhi, plo, phi in self.bounds:
                if np.any(high < vlo - margin) or np.any(low > vhi + margin):
                    continue
                if projection is None:
                    pp = p @ self.directions.T
                    projection = pp.min(0), pp.max(0)
                if not (np.any(projection[1] < plo - margin) or np.any(projection[0] > phi + margin)):
                    return False
        return True

    def bands(self):
        """Local widths along the minor horizontal direction, sampled at finger height."""
        points = self.world
        eig, axes = np.linalg.eigh(np.cov(points[:, :2].T))
        major = axes[:, -1]
        minor = np.array([-major[1], major[0]])
        t = points[:, :2] @ major
        tmin, tmax = np.quantile(t, [0.005, 0.995])
        length = tmax - tmin
        candidates = []
        for fraction in (0.50, 0.32, 0.68, 0.18, 0.82):
            level = tmin + fraction * length
            width = max(0.004, min(0.014, 0.14 * length))
            section = points[np.abs(t - level) < width]
            if len(section) < 16:
                continue
            zlo, zhi = np.quantile(section[:, 2], [0.012, 0.988])
            z = zlo + 0.54 * (zhi - zlo)
            mid = section[np.abs(section[:, 2] - z) < max(0.002, 0.24 * (zhi - zlo))]
            if len(mid) < 10:
                mid = section
            lo, hi = np.quantile(mid[:, :2] @ minor, [0.015, 0.985])
            span = hi - lo
            if span < 0.005:
                continue
            xy = major * level + minor * ((lo + hi) / 2)
            c = np.array([*xy, z])
            # Nearly circular footprints also admit the same contact line rotated 90 degrees.
            score = abs(span - 0.053) + 0.25 * abs(fraction - 0.5) * length
            candidates.append((score, c, minor, span, zhi, zlo))
            if eig[1] < 1.30 * max(eig[0], 1e-10) and fraction == 0.50:
                candidates.append((score + 0.002, c, major, tmax - tmin, zhi, zlo))
        return sorted(candidates, key=lambda row: row[0])


def unit(v):
    return v / np.linalg.norm(v)


def band_grasps(scene, surface, qopen, limits, postures, *, upright_tip=False, bands=None):
    """Synthesize a table-clearing pinch with real pad surfaces on the local thickness."""
    if bands is None:
        bands = upper_bands(surface) if upright_tip else surface.bands()
    if not bands:
        return []
    table = surface.table
    all_candidates = []
    fractions = np.linspace(0.0, 1.0, 17)
    for label, qlimit in postures.items():
        qc = np.clip(qlimit, limits[:, 0], limits[:, 1])
        q0 = qopen.copy()
        # Bring the three unused fingertips above the active opposing pads.
        q0[9:] = qc[9:]
        qs = [q0 + f * (qc - q0) for f in fractions]
        shapes = [hand_shapes(scene, q) for q in qs]
        local = shapes[9]
        a, b = (local["right_" + d + "_elastomer"].mean(0) for d in ("thumb", "index"))
        axis = unit(b - a)
        reach = (a + b) / 2
        down = unit(reach - axis * (reach @ axis))
        local_frame = np.stack([axis, down, np.cross(axis, down)], axis=1)
        for patch_score, center, n, width, top, bottom in bands[:4]:
            if width > 0.115:
                continue
            for flip in (1.0, -1.0):
                target_axis = np.r_[flip * n, 0]
                horizontal = np.cross([0.0, 0.0, 1.0], target_axis)
                if horizontal[0] < 0:
                    horizontal = -horizontal
                for tilt in (50, 68, 80, 92, 105, 32):
                    theta = np.radians(tilt)
                    pointing = horizontal * np.cos(theta) - np.array([0.0, 0.0, np.sin(theta)])
                    world_frame = np.stack([target_axis, pointing, np.cross(target_axis, pointing)], axis=1)
                    rot = world_frame @ local_frame.T
                    rows = []
                    for shape in shapes:
                        t = shape["right_thumb_elastomer"] @ rot.T
                        i = shape["right_index_elastomer"] @ rot.T
                        thumb = t[np.argmax(t @ target_axis)]
                        index = i[np.argmin(i @ target_axis)]
                        mid = (index + thumb) / 2
                        pose = np.eye(4)
                        pose[:3, :3] = rot
                        pose[:3, 3] = center - mid
                        low = min(np.min(v @ rot[2]) for v in shape.values()) + pose[2, 3]
                        pose[2, 3] += max(0.0, table + 0.003 - low)
                        levels = np.array([thumb[2], index[2]]) + pose[2, 3]
                        edge = float(min(np.min(top - levels), np.min(levels - bottom)))
                        rows.append(((index - thumb) @ target_axis, edge, pose))
                    for j, (gap, edge, grasp) in enumerate(rows):
                        squeeze = np.clip(width * 0.12, 0.0025, 0.009)
                        if (
                            edge < min(0.004, (top - bottom) * 0.12)
                            or not width - 2 * squeeze < gap < width - 0.0015
                        ):
                            continue
                        for i in reversed(range(j)):
                            pgap, _, pre = rows[i]
                            if pgap < width + 0.008 or not surface.clear(shapes[i], pre, 0.001):
                                continue
                            score = (
                                abs(gap - width + squeeze)
                                + patch_score * 0.35
                                + 0.00006 * abs(tilt - 65)
                                + 0.17 * max(0.0, 0.014 - edge)
                                + 0.0004 * (flip < 0)
                            )
                            all_candidates.append(
                                (
                                    score,
                                    pre,
                                    grasp,
                                    qs[i],
                                    qs[j],
                                    qc,
                                    {
                                        "posture": label,
                                        "tilt_deg": tilt,
                                        "width_m": float(width),
                                        "edge_margin_m": edge,
                                        "unloaded_gap_m": float(gap),
                                        "patch_world_m": center.tolist(),
                                    },
                                )
                            )
                            break
    return sorted(all_candidates, key=lambda c: c[0])


def upper_bands(surface):
    """Circular-tool contact sections measured at multiple heights below the tip."""
    cloud = surface.world
    bottom, top = np.quantile(cloud[:, 2], [0.006, 0.994])
    height = top - bottom
    out = []
    for f in (0.72, 0.82, 0.64, 0.54):
        z = bottom + f * height
        near = cloud[np.abs(cloud[:, 2] - z) < max(0.002, 0.065 * height)]
        if len(near) < 20:
            continue
        center = (np.quantile(near[:, :2], 0.01, axis=0) + np.quantile(near[:, :2], 0.99, axis=0)) / 2
        for a in (0.0, 0.65, 1.35, 2.0):
            axis = np.array([np.cos(a), np.sin(a)])
            span = np.ptp(np.quantile(near[:, :2] @ axis, [0.01, 0.99]))
            score = 0.2 * abs(span - 0.023) + 0.01 * abs(f - 0.75)
            out.append((score, np.r_[center, z], axis, span, min(top, z + 0.015), max(bottom, z - 0.015)))
    return sorted(out, key=lambda x: x[0])


def confirm_pad_surfaces(scene, surface, candidates):
    """Balance the unloaded pad overlap on the registered convex pieces."""
    from scipy.spatial import ConvexHull

    planes = [np.unique(np.round(ConvexHull(p).equations, 7), axis=0) for p in surface.world_parts]
    hand_cache = {}
    pad_cache = {}

    def shape(q):
        key = q.tobytes()
        if key not in hand_cache:
            hand_cache[key] = hand_shapes(scene, q)
        return hand_cache[key]

    def pad_points(q):
        key = q.tobytes()
        if key not in pad_cache:
            h = shape(q)
            points = []
            for digit in ("thumb", "index"):
                full = h["right_" + digit + "_elastomer"]
                extremes = np.argmax(full @ surface.directions.T, axis=0)
                ids = np.unique(np.r_[extremes, np.arange(0, len(full), max(1, len(full) // 30))])
                points.append(full[ids])
            pad_cache[key] = points
        return pad_cache[key]

    def gap(points):
        best = 0.03
        normal = None
        low, high = points.min(0), points.max(0)
        for (lo, hi, _, _), eq in zip(surface.bounds, planes):
            if np.linalg.norm(np.maximum(0.0, np.maximum(lo - high, low - hi))) > min(0.018, max(0.003, best)):
                continue
            support = points @ eq[:, :3].T + eq[:, 3]
            nearest = support.max(1)
            idx = nearest.argmin()
            if nearest[idx] < best:
                best = nearest[idx]
                normal = eq[np.argmax(support[idx]), :3]
        return best, normal

    checked = []
    for score, pre, grasp, qp, qg, ql, geom in candidates[:100]:
        pads = [p @ grasp[:3, :3].T + grasp[:3, 3] for p in pad_points(qg)]
        u = unit((pads[1].mean(0) - pads[0].mean(0)) * [1.0, 1.0, 0.0])
        tangent = np.cross([0.0, 0.0, 1.0], u)
        for across in (0.0, -0.004, 0.004, -0.008, 0.008):
            shift = across * tangent
            d0, _ = gap(pads[0] + shift)
            d1, _ = gap(pads[1] + shift)
            shift = shift + u * np.clip((d0 - d1) / 2.0, -0.007, 0.007)
            d0, n0 = gap(pads[0] + shift)
            d1, n1 = gap(pads[1] + shift)
            if max(d0, d1) > 0.0005 or min(d0, d1) < -0.014 or n0 @ n1 > -0.30:
                continue
            a = pre.copy()
            g = grasp.copy()
            a[:3, 3] += shift
            g[:3, 3] += shift
            if not surface.clear(shape(qp), a, 0.001):
                continue
            extra = dict(geom, pad_signed_gaps_m=[float(d0), float(d1)], normal_dot=float(n0 @ n1))
            cost = score + 0.5 * abs(d0 - d1) + 0.35 * (abs(d0 + 0.0035) + abs(d1 + 0.0035)) + 0.1 * abs(across)
            checked.append((cost, a, g, qp, qg, ql, extra))
    return sorted(checked, key=lambda x: x[0])
