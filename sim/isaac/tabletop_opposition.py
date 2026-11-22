"""Allow individual fingers to support a lower edge while keeping thumb opposition."""

import numpy as np
from scipy.spatial import ConvexHull

from .tabletop_control import NoPath

DIGITS = ("index", "middle", "ring", "pinky")
MAX_LOWER = 2


def long_horizontal_axis(points):
    """Horizontal major axis for objects long enough to pivot about an end contact."""
    values, vectors = np.linalg.eigh(np.cov(points[:, :2].T))
    if values[-1] > 3 * values[0] and values[-1] > 0.022**2:
        return np.r_[vectors[:, -1], 0.0]
    return None


def native_opposition(poses, data, links, present):
    # Use the projection in the gravity plane. A second fingertip on top does
    # not count as the opposing contact for the thumb.
    outward = -data["target_normal"] @ poses[:, :3, :3].transpose(0, 2, 1)
    outward /= np.maximum(1e-8, np.linalg.norm(outward, axis=2, keepdims=True))
    horizontal = outward[:, :, :2]
    horizontal_length = np.linalg.norm(horizontal, axis=2)
    unit = horizontal / np.maximum(1e-8, horizontal_length[:, :, None])
    thumb = present & np.char.startswith(links.astype(str), "right_thumb_")
    others = present & ~thumb
    thumb_xy = (unit * thumb[:, :, None]).sum(1)
    dot = (unit * thumb_xy[:, None]).sum(2)
    available = others & (horizontal_length >= 0.42)
    most_opposed = np.where(available, dot, 1.0).min(1)
    lower_face = present & (outward[:, :, 2] < -0.40)
    lower_count = np.sum(
        [
            np.any(
                lower_face
                & np.char.startswith(links.astype(str), "right_" + digit + "_"),
                axis=1,
            )
            for digit in DIGITS
        ],
        axis=0,
    )
    valid = (
        np.any(thumb & (horizontal_length >= 0.42), axis=1)
        & (most_opposed <= -0.45)
        & (lower_count <= MAX_LOWER)
    )
    return valid, most_opposed, lower_count


class LowerEdge:
    """Lower surface in the D455-registered frame, with each finger kept separate."""

    def __init__(self, surface, table):
        self.table = table
        self.spacing = 0.001
        self.parts = []
        vertices = np.vstack(surface.world_parts)
        self.top = vertices[:, 2].max()
        self.lo, self.hi = vertices[:, :2].min(0), vertices[:, :2].max(0)
        angles = np.arange(24) * np.pi / 12
        self.directions = np.array([np.cos(angles), np.sin(angles)])
        self.centroid = vertices[:, :2].mean(0)
        for component in surface.world_parts:
            hull = ConvexHull(component)
            eq = hull.equations
            base = component[:, :2].min(0) - 2 * self.spacing
            size = (
                np.ceil((component[:, :2].max(0) - base) / self.spacing).astype(int) + 3
            )
            xy = np.indices(size).reshape(2, -1).T * self.spacing + base
            z = np.full(len(xy), -np.inf)
            down, up, parallel = (
                eq[:, 2] < -1e-7,
                eq[:, 2] > 1e-7,
                abs(eq[:, 2]) <= 1e-7,
            )
            for j in range(0, len(xy), 256):
                sl = slice(j, j + 256)
                intercept = -(xy[sl] @ eq[:, :2].T + eq[:, 3])
                lower = (intercept[:, down] / eq[down, 2]).max(1)
                upper = (intercept[:, up] / eq[up, 2]).min(1)
                valid = lower <= upper
                if parallel.any():
                    valid &= (intercept[:, parallel] >= 0).all(1)
                z[sl] = np.where(valid, lower, -np.inf)
            outline = ((component[:, :2] - self.centroid) @ self.directions).max(0)
            self.parts.append((base, z.reshape(size), outline))
        self.carry_reference = None
        self.used = set()

    def under(self, world):
        # A pad low on the outside wall can remain in forceful side opposition.
        low = world[
            (world[:, 2] <= self.table + 0.53 * (self.top - self.table))
            & (world[:, :2] >= self.lo).all(1)
            & (world[:, :2] <= self.hi).all(1)
        ]
        if not len(low):
            return False
        projected = (low[:, :2] - self.centroid) @ self.directions
        exterior = np.full(len(low), np.inf)
        inside = np.zeros(len(low), dtype=bool)
        for base, grid, outline in self.parts:
            inset = (outline - projected).min(1)
            near = np.flatnonzero(inset >= -self.spacing)
            if not len(near):
                continue
            indices = np.rint((low[near, :2] - base) / self.spacing).astype(int)
            good = ((indices >= 0) & (indices < grid.shape)).all(1)
            near, indices = near[good], indices[good]
            height = grid[indices[:, 0], indices[:, 1]]
            occupied = np.isfinite(height)
            ids = near[occupied]
            exterior[ids] = np.minimum(exterior[ids], height[occupied])
            inside[ids] |= inset[ids] >= self.spacing
        return bool(np.any(inside & (low[:, 2] <= exterior + 0.0008)))

    def fingers(self, ctrl, palm, hand, enforce_table=True):
        groups = ctrl.link_shapes(hand)
        found = set()
        for name, vertices in groups.items():
            local = vertices[:: max(1, len(vertices) // 120)]
            world = local @ palm[:3, :3].T + palm[:3, 3]
            # The motor target may compress a soft pad on contact, but a hand
            # placement through the table is not an admissible approach.
            if (
                enforce_table
                and not getattr(ctrl, "allow_hand_table_contact", False)
                and world[:, 2].min() < self.table - 0.004
            ):
                raise NoPath("hand target below tabletop compliance margin")
            part = name.split("_")[1]
            if part in DIGITS and part not in found and self.under(world):
                found.add(part)
        return found

    def check(self, ctrl, palm, hand):
        found = self.fingers(ctrl, palm, hand)
        if self.carry_reference is not None:
            # Also check relative to the carried mesh, using the known grasp
            # frame. The static hypothesis remains active if the object slips.
            found |= self.fingers(ctrl, self.carry_reference, hand, enforce_table=False)
        if len(found | self.used) > MAX_LOWER:
            raise NoPath("three or more non-thumb fingers sweep under the same object")
        return found

    def sweep(self, ctrl, joints, start_hand, end_hand):
        joints = np.asarray(joints)
        arc = np.r_[0.0, np.cumsum(np.max(abs(np.diff(joints, axis=0)), axis=1))]
        found = set()
        for i in range(len(joints) - 1):
            count = max(
                3,
                1
                + int(
                    np.ceil(
                        (arc[i + 1] - arc[i] + max(abs(end_hand - start_hand))) / 0.065
                    )
                ),
            )
            for f in np.linspace(0.0, 1.0, count):
                pose = ctrl.ik.fk(joints[i] + f * (joints[i + 1] - joints[i]))
                t = (
                    (arc[i] + f * (arc[i + 1] - arc[i])) / arc[-1]
                    if arc[-1] > 1e-8
                    else f
                )
                found |= self.check(
                    ctrl, pose, start_hand + t * (end_hand - start_hand)
                )
                if len(found | self.used) > MAX_LOWER:
                    raise NoPath("group of non-thumb fingers would scoop the underside")
        return found


def lateral_load_capacity(surface, xyz, poses, data, active, eligible):
    """Best upward force at zero resultant torque with a bounded squeeze per finger."""
    from scipy.optimize import linprog
    from .tabletop_handle import volume_center

    center = volume_center(surface.world_parts)
    normals = -data["target_normal"] @ poses[:, :3, :3].transpose(0, 2, 1)
    normals /= np.maximum(1e-8, np.linalg.norm(normals, axis=2, keepdims=True))
    output = np.zeros(len(eligible))
    quality = np.zeros(len(eligible))
    angles = np.arange(8) * np.pi / 4
    for i in np.flatnonzero(eligible):
        n = normals[i, active[i]]
        p = xyz[i, active[i]]
        axis = np.zeros_like(n)
        axis[:, 2] = 1.0
        axis[np.abs(n[:, 2]) > 0.98] = [1.0, 0.0, 0.0]
        a = np.cross(n, axis)
        a /= np.maximum(1e-8, np.linalg.norm(a, axis=1, keepdims=True))
        b = np.cross(n, a)
        cone = -n[:, None, :] + 0.48 * (
            np.cos(angles)[None, :, None] * a[:, None]
            + np.sin(angles)[None, :, None] * b[:, None]
        )
        r = max(0.009, np.linalg.norm(p - center, axis=1).mean())
        torque = np.cross((p - center)[:, None] / r, cone)
        load = cone.reshape(-1, 3)
        eq = np.vstack([load[:, :2].T, torque.reshape(-1, 3).T])
        nf = len(n)
        bounds = np.kron(np.eye(nf), np.ones(8))
        # Allow a small dimensionless torque residual for soft finger patches.
        lim = np.r_[np.full(2, 0.008), np.full(3, 0.03)]
        lp = linprog(
            -load[:, 2],
            A_ub=np.vstack([bounds, eq, -eq]),
            b_ub=np.r_[np.ones(nf), lim, lim],
            bounds=(0, None),
            method="highs",
        )
        if lp.success:
            output[i] = max(0.0, -float(lp.fun))
        quality[i] = np.mean(n[:, 2])
    return output, quality
