"""Fit the cork and plan two pad trajectories around a common wrist pose."""

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .pinch_contact import PINCH_DIGITS
from .station import NoPlan


PAD_CLOSURE_DEPTH_M = .004
INDEX_MCP_ABDUCTION = 6


@dataclass(frozen=True)
class Cork:
    center: np.ndarray
    radius: float
    residual_m: float
    samples: np.ndarray


def fit_cork(points, up, table):
    # A bounded sphere is a local model of the rounded cork, including its unseen side.
    points = np.asarray(points)
    points = points[points @ up > table]
    if len(points) < 20:
        raise NoPlan("Insufficient cork points above the table")
    origin = np.median(points, axis=0)
    local = points - origin
    initial = -np.clip(origin @ up - table, 0., .013) * .4 * up
    fit = least_squares(lambda v: np.linalg.norm(local - v[:3], axis=1) - v[3],
                        np.r_[initial, .013], bounds=([-.019] * 3 + [.010],
                                                      [.019] * 3 + [.018]),
                        loss="soft_l1", f_scale=.0015)
    center, radius = origin + fit.x[:3], float(fit.x[3])
    error = float(np.quantile(np.abs(fit.fun), .90))
    if center @ up <= table or error > .006:
        raise NoPlan("Cork depth does not support an opposing-contact estimate")
    return Cork(center, radius, error, points)


@dataclass(frozen=True)
class PadPlan:
    wrist: np.ndarray
    contacts: np.ndarray
    closing_points: np.ndarray
    finger_path: np.ndarray

    def report(self, cork):
        return {
            "cork_center_m": cork.center.tolist(),
            "cork_radius_m": cork.radius,
            "cork_fit_p90_m": cork.residual_m,
            "contact_source": "fitted_cork_surface",
            "wrist_xyz_xyzw": self.wrist.tolist(),
            "contacts": {
                digit: {"point_mount_m": point.tolist(),
                        "closing_point_mount_m": closing.tolist(),
                        "nearest_observed_sample_m": float(np.linalg.norm(cork.samples - point, axis=1).min())}
                for digit, point, closing in zip(PINCH_DIGITS, self.contacts, self.closing_points)
            },
            "precontact_rad": self.finger_path[0].tolist(),
            "closing_waypoints": len(self.finger_path) - 1,
        }


class PadKinematics:
    def __init__(self, geometry, arm, fingers, joint_limits):
        self.geometry = geometry
        self.arm = arm
        self.fingers = fingers
        self.model, self.data = geometry.model, geometry.data
        self.pads = [self.model.geom(f"right_{digit}_elastomer_visual").id for digit in PINCH_DIGITS]
        self.bounds = np.asarray(joint_limits).copy()
        # The Sharpa SDK requires ordered MCP abduction. The middle finger stays parked.
        self.bounds[INDEX_MCP_ABDUCTION, 1] = min(self.bounds[INDEX_MCP_ABDUCTION, 1], fingers[10])
        self.indices = [sorted(j for j in geometry.finger_dependencies[i]
                               if self.bounds[j, 0] < self.bounds[j, 1]) for i in self.pads]
        geometry.set_pose(arm, fingers)
        tcp = self.data.site("right_tcp")
        self.position = tcp.xpos.copy()
        self.rotation = tcp.xmat.reshape(3, 3).copy()

    def set_fingers(self, q):
        self.data.qpos[self.geometry.hand] = q
        mj.mj_kinematics(self.model, self.data)

    def gap(self, q, fromto=None):
        self.set_fingers(q)
        return mj.mj_geomDistance(self.model, self.data, *self.pads, .5, fromto)

    def closed_pose(self, span, weights):
        indices = np.concatenate(self.indices)
        lo, hi = self.bounds[indices].T
        start = np.clip(self.fingers[indices], lo, hi)
        q = self.fingers.copy()
        # Prefer flexion to moving the index MCP abduction toward its travel limit.
        weights = weights * np.where(indices == INDEX_MCP_ABDUCTION, 5., 1.)

        def residual(x):
            q[indices] = x
            return np.r_[(self.gap(q) - span) / .00025, weights * (x - start)]

        fitted = least_squares(residual, start, bounds=(lo, hi), diff_step=1e-4, max_nfev=45)
        q[indices] = fitted.x
        pair = np.zeros(6)
        if abs(self.gap(q, pair) - span) > .0005:
            raise NoPlan("Requested cork span is outside the reachable two-pad aperture")
        return q.copy(), pair.reshape(2, 3)

    def finger_path(self, closed, pair):
        """Backsolve both digits along straight lines to get a compatible preshape."""
        positions = self.data.geom_xpos[self.pads].copy()
        rotations = self.data.geom_xmat[self.pads].reshape(2, 3, 3).copy()
        anchors = np.einsum("ni,nij->nj", pair - positions, rotations)
        axis = pair[1] - pair[0]
        axis /= np.linalg.norm(axis)
        # In wrist coordinates place each pad just outside its fitted contact.
        # Seed from the solved pinch rather than using the starting hand as the preshape.
        travel = PAD_CLOSURE_DEPTH_M + .001
        displacement = np.empty_like(pair)
        for i, (pad, indices) in enumerate(zip(self.pads, self.indices)):
            normal = (2 * i - 1) * axis
            q = closed.copy()
            seed = q[indices].copy()

            def open_residual(x):
                q[indices] = x
                self.set_fingers(q)
                delta = self.pad_point(pad, anchors[i]) - pair[i]
                axial = delta @ normal
                return np.r_[(axial - travel) / .0002, (delta - axial * normal) / .002, x - seed]

            opening = least_squares(open_residual, seed, bounds=self.bounds[indices].T)
            open_residual(opening.x)
            displacement[i] = self.pad_point(pad, anchors[i]) - pair[i]
            if abs(displacement[i] @ normal - travel) > .0005:
                raise NoPlan(f"{PINCH_DIGITS[i]} cannot reach its precontact opening")
        steps = int(np.ceil(np.linalg.norm(displacement, axis=1).max() / .0015))
        path = [closed]
        for fraction in np.linspace(0, 1, steps + 1)[1:]:
            q = path[-1].copy()
            for i, (pad, indices) in enumerate(zip(self.pads, self.indices)):
                target = pair[i] + fraction * displacement[i]

                def residual(x):
                    q[indices] = x
                    self.set_fingers(q)
                    point = self.pad_point(pad, anchors[i])
                    return np.r_[(point - target) / .0002, x - seed]

                seed = q[indices].copy()
                solved = least_squares(residual, seed, bounds=self.bounds[indices].T, max_nfev=25)
                if np.linalg.norm(solved.fun[:3]) > 2.5:
                    raise NoPlan(f"{PINCH_DIGITS[i]} cannot follow the interpolated pad line")
                q[indices] = solved.x
            path.append(q)
        return np.array(path[::-1])

    def pad_point(self, pad, anchor):
        return self.data.geom_xpos[pad] + self.data.geom_xmat[pad].reshape(3, 3) @ anchor

    def table_gap(self, q, rotation, translation):
        self.set_fingers(q)
        normal = rotation.T @ self.geometry.normal
        return min((mesh @ (self.data.geom_xmat[i].reshape(3, 3).T @ normal)).min()
                   + self.data.geom_xpos[i] @ normal
                   for i, mesh in self.geometry.meshes.items()) + (
                       translation @ self.geometry.normal - self.geometry.height)


def grasp_candidates(cork, pads, up, guard, *, expanded=False):
    """Yield reachable pad shapes paired with common wrist goals, nearest motion first."""
    if expanded:
        offsets = (0., .005, .008, .011, .014)
        headings = rolls = range(-165, 181, 15)
    else:
        offsets = (0., .005, .008)
        headings = (0, -30, 30, -60, 60, -90, 90, -135, 135, 180)
        rolls = (0, -15, 15, -30, 30, -60, 60)
    candidates = []
    for offset in offsets:
        if np.hypot(offset, PAD_CLOSURE_DEPTH_M) >= cork.radius:
            continue
        center = cork.center + offset * up
        radius = np.sqrt(cork.radius ** 2 - offset ** 2)
        # Allow measured closure to continue inside the fitted surface until tactile stops it.
        span = 2 * (radius - PAD_CLOSURE_DEPTH_M)
        for thumb_weight, index_weight in ((1., 1.), (1., 3.), (3., 1.)):
            guard()
            weights = np.repeat([thumb_weight, index_weight], [len(ids) for ids in pads.indices])
            try:
                closed, pair = pads.closed_pose(span, weights)
                finger_path = pads.finger_path(closed, pair)
            except NoPlan:
                continue
            midpoint = pair.mean(axis=0)
            axis = (pair[1] - pair[0]) / np.linalg.norm(pair[1] - pair[0])
            horizontal = axis - (axis @ up) * up
            if np.linalg.norm(horizontal) < 1e-6:
                horizontal = pads.rotation[:, 1] - (pads.rotation[:, 1] @ up) * up
            if np.linalg.norm(horizontal) < 1e-6:
                horizontal = pads.rotation[:, 2] - (pads.rotation[:, 2] @ up) * up
            horizontal /= np.linalg.norm(horizontal)
            for yaw in headings:
                direction = Rotation.from_rotvec(np.deg2rad(yaw) * up).apply(horizontal)
                alignment = Rotation.align_vectors([direction], [axis])[0]
                contacts = center + np.array([-radius, radius])[:, None] * direction
                closing_points = center + np.array([-.5, .5])[:, None] * span * direction
                support_error = np.linalg.norm(cork.samples - contacts[:, None], axis=2).min(axis=1).sum()
                for roll in rolls:
                    motion = Rotation.from_rotvec(np.deg2rad(roll) * direction) * alignment
                    rotation = motion.as_matrix()
                    translation = center - rotation @ midpoint
                    if any(pads.table_gap(q, rotation, translation) <= 0 for q in
                           (finger_path[0], finger_path[-1])):
                        continue
                    wrist = np.r_[rotation @ pads.position + translation,
                                  (motion * Rotation.from_matrix(pads.rotation)).as_quat()]
                    cost = (np.linalg.norm(wrist[:3] - pads.position)
                            + .07 * motion.magnitude()
                            + np.linalg.norm(finger_path[0] - pads.fingers) * .02
                            + .5 * support_error)
                    candidates.append((cost, PadPlan(wrist, contacts, closing_points, finger_path)))
    candidates.sort(key=lambda item: item[0])
    return [plan for _, plan in candidates]
