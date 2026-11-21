"""Plan tabletop grasps from calibrated RGB-D, robot CAD and commanded trajectories."""

import json
from itertools import islice, pairwise

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .head_wrist_geometry import hand_shapes
from .head_wrist_trial import ARM, PALM, save
from .tabletop_catalog import geometry_family
from .tabletop_vision import Models, match_height, proposals, wrist_points


class NoPath(ValueError):
    pass


def shifted(pose, offset):
    result = pose.copy()
    result[:3, 3] += offset
    return result


def swept_clear(stations, free):
    """Check intervening arm states and their synchronized hand interpolation fractions."""
    stations = np.asarray(stations)
    lengths = np.max(np.abs(np.diff(stations, axis=0)), axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    for i, (a, b) in enumerate(pairwise(stations)):
        count = max(1, int(np.ceil(lengths[i] / 0.04)))
        for t in np.arange(1, count + 1) / count:
            f = (arc[i] + t * lengths[i]) / arc[-1] if arc[-1] else t
            if not free((1 - t) * a + t * b, f):
                return False
    return True


class ArmIK:
    """Bounded seven-axis IK with the geometric Jacobian from the station URDF."""

    def __init__(self, tree):
        self.chain = tree._chain(PALM)
        self.bounds = tree.joint_limits(ARM).T
        if [j.name for j in self.chain if j.kind != "fixed"] != ARM:
            raise ValueError("unexpected right arm chain")

    def fk(self, q):
        t = np.eye(4)
        index = 0
        for joint in self.chain:
            t = t @ joint.origin
            if joint.kind != "fixed":
                t[:3, :3] = (
                    t[:3, :3] @ Rotation.from_rotvec(joint.axis * q[index]).as_matrix()
                )
                index += 1
        return t

    def solve(self, pose, initial):
        start = np.clip(initial, self.bounds[0] + 1e-7, self.bounds[1] - 1e-7)
        last, jac = None, None

        def evaluate(q):
            nonlocal last, jac
            t = np.eye(4)
            axes, origins = [], []
            for j in self.chain:
                t = t @ j.origin
                if j.kind != "fixed":
                    axes.append(t[:3, :3] @ j.axis)
                    origins.append(t[:3, 3].copy())
                    t[:3, :3] = (
                        t[:3, :3]
                        @ Rotation.from_rotvec(j.axis * q[len(axes) - 1]).as_matrix()
                    )
            a = np.asarray(axes)
            rot = Rotation.from_matrix(t[:3, :3] @ pose[:3, :3].T).as_rotvec()
            # Inverse left Jacobian of the SO(3) logarithm.
            x, y, z = rot
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            angle = np.linalg.norm(rot)
            c = (
                (1 - 0.5 * angle / np.tan(0.5 * angle)) / angle**2
                if angle > 1e-4
                else 1 / 12
            )
            inv = np.eye(3) - 0.5 * skew + c * skew @ skew
            jac = np.vstack([np.cross(a, t[:3, 3] - origins).T, 0.20 * inv @ a.T])
            last = q.copy()
            return np.r_[t[:3, 3] - pose[:3, 3], 0.20 * rot]

        def derivative(q):
            if last is None or not np.array_equal(q, last):
                evaluate(q)
            return jac

        result = least_squares(
            evaluate,
            start,
            jac=derivative,
            bounds=self.bounds,
            max_nfev=48,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-7,
        )
        r = result.fun
        if np.linalg.norm(r[:3]) > 0.0035 or np.linalg.norm(r[3:]) > 0.009:
            raise NoPath("arm IK")
        return result.x


class Controller:
    def __init__(
        self,
        io,
        library,
        out,
        template_comparison=False,
        shared=None,
        lift_height=0.05,
        lightning=None,
    ):
        self.io, self.library, self.out = io, library, out
        self.side = getattr(io, "side", "right")
        self.models = Models(library, mirror=self.side == "left")
        self.lightning = lightning
        self.shared = shared
        if not 0.03 <= lift_height <= 0.20:
            raise ValueError("expected a lift goal between 0.03 and 0.20 metres")
        self.lift_height = lift_height
        # Allow for a few millimetres of object settling during continuous closure.
        self.lift_travel = lift_height + (0.017 if lightning is not None else 0.009)
        from .tabletop_meshfit import HandContacts

        self.hand_contacts = HandContacts(io)
        self.handle_library = {}
        self.wrap_cache = {}
        self.template_comparison = template_comparison
        self.tracks = {}
        self.lift_views = {}
        self.ik = ArmIK(io.tree)
        self.events, self.results = [], {}
        self.name, self.attempt = "", 0
        home = io.commanded_joints()
        self.arm = home[io.arm]
        self.hand = home[io.hand]
        self.reference_arm = self.arm.copy()
        self.mid_arm = np.array([1.71, 1.15, -1.85, -1.60, 0.43, 0.13, -0.34])
        self.bank = None
        self.hints = {}
        self.table = None
        self.cloud = np.empty((0, 3))
        self.cache = {}
        self.open_hand = self.hand.copy()
        self.seen = {}
        self.camera_roles = getattr(io, "camera_roles", ("head", "wrist"))
        self.eye_from_palm = (
            None if io.palm_from_wrist is None else np.linalg.inv(io.palm_from_wrist)
        )

    def event(self, phase, **info):
        row = dict(
            time=round(self.io.time(), 5),
            name=self.name,
            attempt=self.attempt,
            phase=phase,
            hand=self.side,
            **info,
        )
        self.events.append(row)
        if self.out is None:
            return
        with (self.out / "events.jsonl").open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            "TABLE", self.name, self.attempt, phase, info.get("reason", ""), flush=True
        )

    def snapshot(self, role, tag):
        frame = self.io.camera(role)
        prefix = f"{self.side}_{len(self.events):04d}_{self.name or 'survey'}_{self.attempt}_{tag}_{role}"
        # Store raw samples from each decision exposure; RGB, depth and FK share a timestamp.
        np.savez_compressed(self.out / "observations" / (prefix + ".npz"), **frame)
        return frame

    def survey(self, wrist=False):
        frame = self.snapshot("wrist" if wrist else "head", "survey")
        self.table, pieces = proposals(frame, self.io, self.table)
        if not wrist:
            self.cloud = (
                np.vstack([x["xyz"][::4] for x in pieces])
                if pieces
                else np.empty((0, 3))
            )
            self.head_pieces = pieces
            self.seen = self.models.identify(pieces)
        else:
            # Merge two camera views by footprint proximity in world coordinates, then reassign globally.
            combined = [dict(p) for p in self.head_pieces]
            for item in pieces:
                if len(item["xyz"]) < 28:
                    continue
                tree = cKDTree(item["xyz"][::3, :2])
                overlap = [
                    np.mean(tree.query(p["xyz"][::4, :2])[0] < 0.013) for p in combined
                ]
                j = int(np.argmax(overlap)) if overlap else -1
                if j >= 0 and overlap[j] > 0.18:
                    p = combined[j]
                    from .tabletop_vision import appearance, dimensions

                    for field in ("xyz", "rgb"):
                        p[field] = np.concatenate([p[field], item[field]])
                    p.update(
                        dims=dimensions(p["xyz"], self.table),
                        color=appearance(p["rgb"]),
                        center=np.median(p["xyz"], 0),
                    )
                else:
                    combined.append(item)
            self.seen = self.models.identify(combined)
        self.event(
            "survey_wrist" if wrist else "survey_head",
            objects={
                n: {"xyz": p["center"].tolist(), "cost": round(p["match_cost"], 3)}
                for n, p in self.seen.items()
            },
        )
        return self.seen

    def look_poses(self, point, distance=0.20):
        current = self.io.fk(PALM) @ self.io.palm_from_wrist
        for way in (
            [-0.46, -0.36, 0.82],
            [-0.73, 0.12, 0.68],
            [-0.15, -0.68, 0.72],
            [0, 0, 1],
        ):
            eye = point + distance * np.asarray(way) / np.linalg.norm(way)
            z = (point - eye) / np.linalg.norm(point - eye)
            for roll in (0, 90, -90, 180):
                x = current[:3, 0] - z * (z @ current[:3, 0])
                if np.linalg.norm(x) < 0.01:
                    x = np.cross(z, [0, 1, 0])
                x /= np.linalg.norm(x)
                rot = (
                    np.column_stack([x, np.cross(z, x), z])
                    @ Rotation.from_euler("z", roll, degrees=True).as_matrix()
                )
                optical = np.eye(4)
                optical[:3, :3], optical[:3, 3] = rot, eye
                yield optical @ self.eye_from_palm

    def visible(self, pose, pts):
        optical = pose @ self.io.palm_from_wrist
        cam = (pts - optical[:3, 3]) @ optical[:3, :3]
        z = cam[:, 2]
        k = self.io.wrist_K
        uv = (
            cam[:, :2] * np.array([k[0, 0], k[1, 1]]) / np.maximum(z[:, None], 1e-4)
            + k[:2, 2]
        )
        return np.mean(
            (z > 0.03)
            & (z < 0.65)
            & (uv[:, 0] > 14)
            & (uv[:, 0] < 466)
            & (uv[:, 1] > 14)
            & (uv[:, 1] < 626)
        )

    def link_shapes(self, hand):
        return hand_shapes(self.io.model, hand)

    @staticmethod
    def surface_vertices(pieces):
        return np.concatenate([v[:: max(1, len(v) // 100)] for v in pieces.values()])

    def shape(self, hand):
        key = np.round(hand, 4).tobytes()
        if key not in self.cache:
            pieces = hand_shapes(self.io.model, hand)
            self.cache[key] = self.surface_vertices(pieces)
        return self.cache[key]

    def clear(self, pose, hand, others=None):
        return self.vertices_clear(pose, self.shape(hand), others)

    def vertices_clear(self, pose, local_vertices, others):
        if not self.approach_orientation_clear(pose) or not self.fixture_clear(pose):
            return False
        vertices = local_vertices @ pose[:3, :3].T + pose[:3, 3]
        if vertices[:, 2].min() < self.table - 0.0015:
            return False
        if others is not None and len(others.data):
            distance = others.query(vertices, distance_upper_bound=0.005)[0]
            if np.count_nonzero(np.isfinite(distance)) >= 3:
                return False
        return True

    def fixture_clear(self, pose):
        return all(
            (vertices @ pose[2, :3]).min() + pose[2, 3] >= self.table + 0.003
            for vertices in getattr(self.io.model, "fixture_palm", {}).values()
        )

    def approach_orientation_clear(self, palm):
        if self.lightning is None:
            return True
        # The parked hands have fingers upright and the radial axis horizontal.
        # Allow a small tracking tolerance from park and enforce the same half-space
        # along Cartesian and fallback joint-space paths, including local retries.
        return bool(palm[2, 1] >= -0.055 and palm[2, 2] >= -0.20)

    def command(self, joints, hand, duration):
        """Interpolate the planned joint path by arc length and keep the final hand target."""
        io = self.io
        start = self.hand.copy()
        hand = np.clip(hand, *io.model.limits["right"].T)
        joints = np.asarray(joints)
        if self.lightning is not None and not swept_clear(
            joints, lambda arm, _: self.approach_orientation_clear(self.ik.fk(arm))
        ):
            raise NoPath("approach would roll the thumb side down")
        if (
            self.table is not None
            and getattr(io.model, "fixture_palm", {})
            and not swept_clear(
                joints, lambda arm, _: self.fixture_clear(self.ik.fk(arm))
            )
        ):
            part = (
                "tool flange"
                if set(io.model.fixture_palm) == {"tool_flange"}
                else "camera fixture"
            )
            raise NoPath(f"{part} would intersect the table")
        arc = np.r_[0.0, np.cumsum(np.max(np.abs(np.diff(joints, axis=0)), axis=1))]
        ticks = max(2, round(duration / io.dt))
        for i in range(1, ticks + 1):
            f = min(1.0, i / (0.84 * ticks))
            f = f * f * (3 - 2 * f)
            distance = f * arc[-1]
            k = min(np.searchsorted(arc, distance, side="right") - 1, len(arc) - 2)
            blend = (distance - arc[k]) / max(arc[k + 1] - arc[k], 1e-12)
            arm = joints[k] + blend * (joints[k + 1] - joints[k])
            io.step(arm, start + f * (hand - start))
        self.arm, self.hand = joints[-1].copy(), hand.copy()

    def move(
        self,
        goal,
        hand,
        duration,
        others=None,
        transit=False,
        protected=None,
        object_margin=0.0025,
    ):
        start = self.io.fk(PALM)
        initial = self.arm.copy()
        hand = np.clip(hand, *self.io.model.limits["right"].T)
        initial_hand = self.hand.copy()
        arm_obstacles = others
        if protected is not None and others is not None:
            arm_obstacles = cKDTree(np.vstack([others.data, protected.world[::5]]))
        opened_shapes = {}
        arm_links = {
            k: v
            for k, v in self.io.bounds.items()
            if k.endswith("_R") and k.startswith(("Link", "TCP_Link"))
        }

        def clear_state(q, geometry, vertices):
            palm = self.ik.fk(q)
            if not self.vertices_clear(palm, vertices, others):
                return False
            if protected is not None and not protected.clear(
                geometry, palm, object_margin
            ):
                return False
            if arm_obstacles is not None and len(arm_obstacles.data):
                state = dict(zip(ARM, q))
                for link, bounds in arm_links.items():
                    transform = self.io.tree.pose(link, state)
                    center = transform[:3, :3] @ bounds.mean(0) + transform[:3, 3]
                    radius = np.linalg.norm(bounds[1] - bounds[0]) * 0.5 + 0.006
                    ids = arm_obstacles.query_ball_point(center, radius)
                    if ids:
                        local = (
                            arm_obstacles.data[ids] - transform[:3, 3]
                        ) @ transform[:3, :3]
                        if np.any(
                            np.all(
                                (local >= bounds[0] - 0.004)
                                & (local <= bounds[1] + 0.004),
                                axis=1,
                            )
                        ):
                            return False
            return True

        def free(q, f):
            f = round(8 * f) / 8
            h = (1 - f) * initial_hand + f * hand
            if protected is not None and f not in opened_shapes:
                opened_shapes[f] = self.link_shapes(h)
            return clear_state(q, opened_shapes.get(f), self.shape(h))

        delta = np.linalg.norm(start[:3, 3] - goal[:3, 3])
        angle = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).magnitude()
        anchors = [start]
        if transit and delta > 0.13:
            # Move above the observed objects along this particular transfer corridor.
            corridor = self.cloud
            if len(corridor):
                d = goal[:2, 3] - start[:2, 3]
                f = np.clip(
                    (corridor[:, :2] - start[:2, 3]) @ d / max(d @ d, 1e-8), 0, 1
                )
                corridor = corridor[
                    np.linalg.norm(
                        corridor[:, :2] - start[:2, 3] - f[:, None] * d, axis=1
                    )
                    < 0.085
                ]
            top = np.quantile(corridor[:, 2], 0.97) if len(corridor) else self.table
            h = max(start[2, 3], goal[2, 3], top + 0.085)
            anchors.extend(
                [
                    shifted(start, [0, 0, h - start[2, 3]]),
                    shifted(goal, [0, 0, h - goal[2, 3]]),
                ]
            )
        anchors.append(goal)

        def path(stations):
            joints = [initial]
            for seg, (a, b) in enumerate(pairwise(stations)):
                r = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))
                count = max(
                    1,
                    int(np.ceil(np.linalg.norm(a[:3, 3] - b[:3, 3]) / 0.065)),
                    int(
                        np.ceil(
                            Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
                            / 0.42
                        )
                    ),
                )
                for f in np.linspace(0, 1, count + 1)[1:]:
                    p = np.eye(4)
                    p[:3, :3] = r(f).as_matrix()
                    p[:3, 3] = (1 - f) * a[:3, 3] + f * b[:3, 3]
                    hf = round(4 * (seg + f) / (len(stations) - 1)) / 4
                    intermediate = (1 - hf) * initial_hand + hf * hand
                    if transit and not self.clear(p, intermediate, others):
                        raise NoPath("observed obstacle or table")
                    q = self.ik.solve(p, joints[-1])
                    if np.max(np.abs(q - joints[-1])) > 1.35:
                        raise NoPath("joint path discontinuity")
                    joints.append(q)
            return joints

        try:
            joints = path(anchors)
            if transit and not swept_clear(joints, free):
                raise NoPath("object in observation or approach path")
        except NoPath:
            if not transit:
                raise
            # Rotating from the upright park pose can cross a Cartesian singularity.
            # Inspect complete interpolated joint paths against the current observed obstacles.
            joints = None
            # Sample in joint space around the high transfer corridor, including alternate elbow branches.
            hint = self.hints.get(goal.round(5).tobytes())
            endpoints = [] if hint is None else [hint]
            for seed in (initial, self.mid_arm, self.reference_arm):
                try:
                    q = self.ik.solve(goal, seed)
                    if not any(np.linalg.norm(q - v) < 0.12 for v in endpoints):
                        endpoints.append(q)
                except NoPath:
                    continue
            lifted = []
            for anchor in (start, goal):
                qlist = []
                for dz in (0.11, 0.18):
                    high = shifted(anchor, [0, 0, dz])
                    for seed in (
                        (initial, self.mid_arm)
                        if anchor is start
                        else (self.mid_arm, *endpoints[:1])
                    ):
                        try:
                            q = self.ik.solve(high, seed)
                        except NoPath:
                            continue
                        if self.clear(
                            self.ik.fk(q),
                            initial_hand if anchor is start else hand,
                            others,
                        ):
                            qlist.append(q)
                            break
                lifted.append(qlist)
            for endpoint in endpoints:
                routes = [
                    [initial, a, b, endpoint] for a in lifted[0] for b in lifted[1]
                ]
                routes.extend(
                    [
                        [initial, q, endpoint]
                        for q in (*lifted[0], *lifted[1], self.mid_arm)
                    ]
                )
                routes.insert(0, [initial, endpoint])
                for stations in routes:
                    samples = np.vstack(
                        [
                            np.linspace(
                                a, b, max(4, int(np.max(np.abs(a - b)) / 0.13) + 2)
                            )
                            for a, b in pairwise(stations)
                        ]
                    )
                    if swept_clear(samples, free):
                        joints = samples
                        break
                if joints is not None:
                    break
            if joints is None and endpoints:
                for endpoint in endpoints[:2]:
                    joints = self.bank.route(
                        initial,
                        endpoint,
                        lambda q: free(q, 1),
                    )
                    if joints is not None and swept_clear(joints, free):
                        break
                    joints = None
            if joints is None:
                raise NoPath("no clear arm path")
        length = np.sum(np.max(np.abs(np.diff(joints, axis=0)), axis=1))
        # A smoothstep peaks at 1.5 times mean speed during the moving 84% of
        # the command. Bound the nominal velocity before entering observed clutter.
        duration = max(duration, length / (0.95 if transit else 2.4))
        self.command(joints, hand, max(duration, delta / 0.48, angle / 3.1))

    def hold(self, seconds):
        for _ in range(round(seconds / self.io.dt)):
            self.io.step(self.arm, self.hand)

    def leave_low_pose(self):
        # Clear the recently handled object before addressing a different object.
        # Preserve the local wrist pose on retry. Choose the smallest sufficient upward motion.
        start = self.io.fk(PALM)
        low = (self.shape(self.hand) @ start[2, :3]).min() + start[2, 3]
        if low >= self.table + 0.12:
            return
        if len(self.cloud):
            nearby = self.cloud[
                np.linalg.norm(self.cloud[:, :2] - start[:2, 3], axis=1) < 0.13
            ]
        else:
            nearby = []
        needed = (
            (np.quantile(np.asarray(nearby)[:, 2], 0.95) if len(nearby) else self.table)
            + 0.026
            - low
        )
        for scale in (1.0, 0.70, 0.44):
            dz = scale * np.clip(needed, 0.085, 0.17)
            try:
                self.move(shifted(start, [-0.012, 0, dz]), self.open_hand, 0.46)
                self.event("clear_previous_target", rise_m=round(float(dz), 3))
                return
            except NoPath:
                continue

    def inspect(self, goal, hand, others, protected=None):
        self.move(goal, hand, 0.53, others, True, protected)

    def lift_ik(self, palm, pre_joints):
        grasp = self.ik.solve(palm, pre_joints)
        probe = self.ik.solve(shifted(palm, [0, 0, 0.022]), grasp)
        self.ik.solve(shifted(palm, [0, 0, self.lift_travel]), probe)

    def coarse_surface(self, name, observed):
        from .head_wrist_surface import RegisteredSurface

        try:
            surface, pose, fit = self.models.fit(name, observed["xyz"], self.table)
            self.event(
                "registered_head",
                points=len(observed["xyz"]),
                median_m=fit["median_residual_m"],
                p90_m=fit["p90_residual_m"],
                center=fit["center"],
            )
            return surface, pose, True
        except ValueError:
            # The coarse pose only selects a camera ROI. A wrench/closure cannot use this estimate alone.
            surface = RegisteredSurface(self.models.bundles[name], sample_faces=True)
            obs, local = observed["xyz"], surface.local
            _, a = np.linalg.eigh(np.cov(local[:, :2].T))
            _, b = np.linalg.eigh(np.cov(obs[:, :2].T))
            angle = np.arctan2(b[1, -1], b[0, -1]) - np.arctan2(a[1, -1], a[0, -1])
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
            rotated = local @ pose[:3, :3].T
            pose[:2, 3] = np.median(obs[:, :2], 0) - np.median(rotated[:, :2], 0)
            pose[2, 3] = self.table - np.min(surface.support @ pose[2, :3])
            surface.world = rotated + pose[:3, 3]
            surface.table = self.table
            self.event(
                "coarse_head_roi", center=observed["center"].tolist(), points=len(obs)
            )
            return surface, pose, False

    def options(self, family, surface, pose, seeds, others):
        point = (surface.world.min(0) + surface.world.max(0)) / 2
        for s in seeds:
            qo, qt, qc = [np.array(s[k]) for k in ("q_open", "q_touch", "q_tight")]
            pad = hand_shapes(self.io.model, qt)
            digits = (
                ("index", "middle", "ring")
                if family in ("wrap", "handled")
                else ("index",)
            )
            fingers = np.mean(
                [pad["right_" + n + "_elastomer"].mean(0) for n in digits], 0
            )
            mid = (pad["right_thumb_elastomer"].mean(0) + fingers) / 2
            g = np.array(s["grasp"])
            grip = pose[:3, :3] @ (g[:3, :3] @ mid + g[:3, 3]) + pose[:3, 3]
            # Keep the same local patch on tools; upright cylinders admit more approach azimuths.
            cylindrical = family in ("wrap", "compact")
            pivot = point.copy() if cylindrical else grip
            angles = np.arange(0, 360, 30 if cylindrical else 45)
            templates = {
                k: pose @ np.array(s[k]) for k in ("pre", "grasp", "tight", "view")
            }

            def at(deg, k, templates=templates, pivot=pivot):
                rot = Rotation.from_euler("z", deg, degrees=True).as_matrix()
                p = templates[k].copy()
                p[:3, :3] = rot @ p[:3, :3]
                p[:3, 3] = pivot + rot @ (p[:3, 3] - pivot)
                return p

            for angle in sorted(
                angles,
                key=lambda a: np.linalg.norm(at(a, "pre")[:2, 3] - [0.39, -0.27]),
            ):
                poses = {k: at(angle, k) for k in templates}
                floor = min(
                    (self.shape(h) @ poses[k][2, :3]).min() + poses[k][2, 3]
                    for k, h in (("pre", qo), ("grasp", qt), ("tight", qc))
                )
                dz = max(0.0, self.table + 0.001 - floor)
                if dz > 0.032:
                    continue
                for p in poses.values():
                    p[2, 3] += dz
                if not self.clear(poses["pre"], qo, others) or not self.clear(
                    poses["grasp"], qt, others
                ):
                    continue
                for seed in (self.arm, self.mid_arm):
                    try:
                        q = self.ik.solve(poses["pre"], seed)
                        self.ik.solve(poses["grasp"], q)
                    except NoPath:
                        continue
                    yield s, poses, int(angle)
                    break

    def geometry(self, name, observed, choice):
        surface, pose, accurate = self.coarse_surface(name, observed)
        target = observed["xyz"]
        cloud = self.cloud
        if len(cloud):
            # Exclude target observations, including its sparse side wall, from neighbor checking.
            d1 = cKDTree(target[::2]).query(cloud)[0]
            d2 = cKDTree(surface.world[::3]).query(cloud)[0]
            cloud = cloud[(d1 > 0.017) & (d2 > 0.020)]
        if self.lightning is not None and hasattr(self.io, "bounds"):
            parked = []
            for link in self.io.bounds:
                if link.startswith("left_") and link in self.io.model.hand_support:
                    v = self.io.model.hand_support[link]
                    t = self.io.fk(link)
                    parked.append(v[:: max(1, len(v) // 60)] @ t[:3, :3].T + t[:3, 3])
                elif link.endswith("_L") and link.startswith(("Link", "TCP_Link")):
                    bounds = self.io.bounds[link]
                    # Sample the six faces of the parked arm link's conservative box.
                    unit = np.linspace(0.0, 1.0, 4)
                    grid = np.array(np.meshgrid(unit, unit, unit)).reshape(3, -1).T
                    grid = grid[np.any((grid == 0) | (grid == 1), axis=1)]
                    t = self.io.fk(link)
                    parked.append(
                        (bounds[0] + grid * (bounds[1] - bounds[0])) @ t[:3, :3].T
                        + t[:3, 3]
                    )
            if parked:
                cloud = np.vstack([cloud, *parked])
        others = cKDTree(cloud)
        if "wrist" not in self.camera_roles:
            if not accurate:
                raise NoPath("head mesh registration unconfirmed")
            from .tabletop_vision import add_live_samples
            from .tabletop_views import ViewBank

            add_live_samples(surface, target, [])
            if self.bank is None:
                # Only robot FK and the joint-space roadmap are used in this mode.
                self.bank = ViewBank(
                    self.ik, np.eye(4), self.mid_arm, self.reference_arm
                )
            if self.lightning is not None:
                family, seeds = "lightning", []
            else:
                family, seeds = (
                    self.shared.query(surface, pose)
                    if self.shared
                    else (geometry_family(surface), self.library[name]["seeds"])
                )
            self.event(
                "head_only_grasp", points=len(target), mesh_to_world=pose.tolist()
            )
            seed, poses = self.fit_registered(
                name, family, surface, pose, seeds, choice["prior_postures"], others
            )
            return surface, pose, poses, seed, others
        # Keep the target itself in the observation obstacle set. It is removed
        # from the hand's contact-allowed set only after the pregrasp is reached.
        observations = cKDTree(np.vstack([cloud, target[::3], surface.world[::5]]))
        protected = surface if accurate else None
        point = (surface.world.max(0) + surface.world.min(0)) / 2
        views = []
        # For the shared library the head estimate only guides the approaching wrist.
        # The final arm/hand IK is computed after that exposure.
        if self.lightning is not None:
            views.extend(self.lightning.head_views(self, name, pose, point))
            hand = self.open_hand
            self.event(
                "lightning_head_pregrasp",
                observed=accurate,
                visible_candidates=len(views),
            )
        else:
            family, available = (
                self.shared.query(surface, pose)
                if self.shared
                else (geometry_family(surface), self.library[name]["seeds"])
            )
            seeds = sorted(
                available, key=lambda s: s["posture"] in choice["prior_postures"]
            )
            first = (
                None
                if self.shared and family not in ("handled", "slender")
                else next(self.options(family, surface, pose, seeds[:5], others), None)
            )
            if self.shared and first is None and family not in ("handled", "slender"):
                head_goal = pose @ np.array(seeds[0]["view"])
                views.append((head_goal, np.array(seeds[0]["q_open"])))
            if first is not None:
                s, poses, _ = first
                views.append((poses["view"], np.array(s["q_open"])))
            else:
                s = seeds[0]
            # The camera can observe an item even when the preliminary hand approach is unreachable.
            hand = np.minimum(np.array(s["q_open"]), np.full(22, 0.44))
        # A reachable optical pose with an explicit arm solution often avoids a singularity.
        if self.bank is None:
            from .tabletop_views import ViewBank

            self.bank = ViewBank(
                self.ik, self.io.palm_from_wrist, self.mid_arm, self.reference_arm
            )
        for v, q in self.bank.looks(point, self.arm):
            self.hints[v.round(5).tobytes()] = q
            views.append((v, self.open_hand))
        views.extend(
            (v, hand)
            for distance in (0.22, 0.30, 0.15)
            for v in self.look_poses(point, distance)
        )
        if self.lightning is not None and self.attempt > 1 and len(views) > 3:
            offset = min(2 * (self.attempt - 1), len(views) - 1)
            views = views[offset:] + views[:offset]
        examined = 0
        for v, h in views:
            if self.visible(v, surface.world[::9]) < 0.29 or not self.clear(
                v, h, observations
            ):
                continue
            if protected is not None and not protected.clear(
                self.link_shapes(h), v, 0.0025
            ):
                continue
            try:
                self.inspect(v, h, observations, protected)
            except NoPath:
                continue
            fr = self.snapshot("wrist", "refine")
            near = wrist_points(
                fr, self.io, surface, self.table, 0.017 if accurate else 0.031
            )
            examined += 1
            if len(near) >= 28:
                coarse = target[:: max(1, len(target) // 310)]
                fine = near[:: max(1, len(near) // 640)]
                data = np.vstack([coarse, fine])
                try:
                    refined, reg, details = self.models.fit(
                        name, data, self.table, surface if accurate else None
                    )
                except ValueError:
                    try:
                        refined, reg, details = self.models.fit(
                            name, fine, self.table, surface if accurate else None
                        )
                    except ValueError:
                        if examined >= 5:
                            break
                        continue
                from .tabletop_vision import add_live_samples

                add_live_samples(refined, target, near)
                if self.lightning is not None:
                    family, seeds = "lightning", []
                else:
                    family, available = (
                        self.shared.query(refined, reg)
                        if self.shared
                        else (geometry_family(refined), self.library[name]["seeds"])
                    )
                    seeds = sorted(
                        available,
                        key=lambda s: s["posture"] in choice["prior_postures"],
                    )
                    self.event(
                        "mesh_contact_family",
                        family=family,
                        types_considered=(36 if self.shared else len(seeds)),
                        shared_catalog=bool(self.shared),
                    )
                self.event(
                    "registered_wrist",
                    points=len(near),
                    median_m=details["median_residual_m"],
                    mesh_to_world=reg.tolist(),
                    center=details["center"],
                )
                s, poses = self.fit_registered(
                    name, family, refined, reg, seeds, choice["prior_postures"], others
                )
                return refined, reg, poses, s, others
            if examined >= 5:
                break
        raise NoPath(
            "wrist target unconfirmed"
            if examined
            else "no clear arm path to wrist view"
        )

    def fit_registered(self, name, family, refined, reg, seeds, prior_postures, others):
        if self.lightning is not None:
            return self.lightning.select(
                self, name, refined, reg, others, prior_postures
            )
        best = None
        # After a failed contact or when a transferred hand shape is unreachable,
        # refit the catalog's thumb/finger pads to the new measured contact band.
        if family == "handled" and not self.template_comparison:
            from .tabletop_handle import plan

            cards = {s["posture"]: s["q_touch"] for s in seeds}
            if not self.shared:
                cards.update(self.library[name].get("handle_cards", {}))
            best = plan(self, refined, prior_postures, others, cards)
        if best is None and not self.template_comparison:
            from .tabletop_grips import patch_candidates
            from .tabletop_meshfit import refine

            coarse_options = list(
                islice(self.options(family, refined, reg, seeds[:4], others), 3)
            )
            anatomical = patch_candidates(self, family, refined, seeds[:6], others)
            # Include contact-band initializations with diverse palm directions.
            sampled = []
            for _, card, p in anatomical:
                if any(
                    other[0]["posture"] == card["posture"]
                    and np.linalg.norm(p["grasp"][:3, 3] - other[1]["grasp"][:3, 3])
                    < 0.022
                    for other in sampled
                ):
                    continue
                sampled.append((card, p, 0))
                if len(sampled) == 5:
                    break
            if family == "wrap":
                from .tabletop_grips import wrap_candidates

                sampled = [
                    *wrap_candidates(self, refined, seeds, prior_postures, others),
                    *sampled,
                ]
            best = refine(self, family, refined, [*sampled, *coarse_options], others)
        if self.template_comparison:
            best = next(self.options(family, refined, reg, seeds[:5], others), None)
        if self.template_comparison and (best is None or prior_postures):
            from .tabletop_grips import patch_candidates

            for gaps, card, targets in patch_candidates(
                self, family, refined, seeds[:6], others
            )[:24]:
                for initial in (self.arm, self.mid_arm):
                    try:
                        reached = self.ik.solve(targets["pre"], initial)
                        self.ik.solve(targets["grasp"], reached)
                    except NoPath:
                        continue
                    best = (card, targets, 0)
                    self.event(
                        "aligned_finger_pads",
                        predicted_gap_cost_m=round(float(gaps), 4),
                    )
                    break
                else:
                    continue
                break
        if best is None:
            raise NoPath("no clear reachable posture after wrist registration")
        s, poses, angle = best
        self.event("selected", posture=s["posture"], yaw_deg=angle, origin=s["origin"])
        return s, poses

    def reacquire_head(self, reference):
        """Match a fresh D455 component near a recently released or occluded target."""
        if reference is None:
            return None
        candidates = []
        for p in self.head_pieces:
            delta = np.linalg.norm(p["center"][:2] - reference["center"][:2])
            color = np.linalg.norm(p["color"][:3] - reference["color"][:3])
            size = np.linalg.norm(
                np.log(
                    np.maximum(p["dims"], 0.001) / np.maximum(reference["dims"], 0.001)
                )
            )
            if delta < 0.067 and color < 0.38 and size < 1.15:
                candidates.append((delta / 0.055 + color + 0.25 * size, p))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.16:
            return None
        score, p = candidates[0]
        self.event(
            "reacquired_head_roi",
            points=len(p["xyz"]),
            offset_m=float(np.linalg.norm(p["center"][:2] - reference["center"][:2])),
        )
        return dict(p, match_cost=score)

    def begin_lift(self, surface):
        from .tabletop_tracking import initialize

        self.tracks = {
            role: initialize(self.snapshot(role, "before_lift"), self.io, surface.world)
            for role in self.camera_roles
        }

    def vote(self, surface, pose, palm0, color, tag):
        held = self.io.fk(PALM) @ np.linalg.inv(palm0) @ pose
        votes = {}
        for role in self.camera_roles:
            frame = self.snapshot(role, tag)
            votes[role] = match_height(
                frame, self.io, self.table, surface.local, pose, held, color
            )
            if self.tracks:
                from .tabletop_tracking import measure

                delta = self.io.fk(PALM) @ np.linalg.inv(palm0)
                votes[role].update(
                    measure(self.tracks.get(role), frame, self.io, surface.world, delta)
                )
        carried = sum(x.get("carried_pixels", 0) for x in votes.values())
        static = sum(x.get("static_pixels", 0) for x in votes.values())
        vote = "unknown"
        if carried >= 12 and carried > 1.7 * static:
            vote = "carried"
        if static >= 12 and static > 1.7 * carried:
            vote = "on_table"
        if tag.startswith("test_observation") and not self.template_comparison:
            measured = [
                v
                for v in votes.values()
                if v.get("height_support", 0) >= 24
                and v.get("height_margin_m", 0) >= 0.0009
            ]
            if (
                measured
                and all(v["visible_lift_m"] >= 0.007 for v in measured)
                and not any(
                    v.get("static_pixels", 0) > 2 * max(v.get("carried_pixels", 0), 1)
                    and v.get("visible_lift_m", 0) < 0.005
                    for v in votes.values()
                )
            ):
                vote = "carried"
        tracks = [v for v in votes.values() if v.get("texture_inliers", 0) >= 5]
        if tracks:
            good = (
                (0.007, 0.0025)
                if tag.startswith("test_observation")
                else (self.lift_height, self.lift_height)
            )
            if all(
                v["tracked_rise_m"] >= good[0] and v["tracked_bottom_rise_m"] >= good[1]
                for v in tracks
            ):
                vote = "carried"
            elif all(
                abs(v["tracked_rise_m"]) < 0.004
                and abs(v["tracked_bottom_rise_m"]) < 0.004
                for v in tracks
            ):
                vote = "on_table"
            elif not tag.startswith("test_observation") and any(
                min(v["tracked_bottom_rise_m"], v["tracked_rise_m"]) < self.lift_height
                for v in tracks
            ):
                vote = "unknown"
        elif not tag.startswith("test_observation") and vote == "carried":
            heights = [
                min(v["geometry_clearance_m"], v["geometry_rise_m"])
                for v in votes.values()
                if "geometry_clearance_m" in v
            ]
            if not heights or min(heights) < self.lift_height:
                vote = "unknown"
        self.lift_views = votes
        self.event(tag, vote=vote, views=votes)
        return vote

    def trial(self, observed, result):
        name = self.name
        surface, pose, poses, seed, others = self.geometry(name, observed, result)
        result["prior_postures"].append(seed["posture"])
        pre, grasp, tight = [poses[k] for k in ("pre", "grasp", "tight")]
        qo, qt, qc = [np.array(seed[k]) for k in ("q_open", "q_touch", "q_tight")]
        if not self.clear(pre, qo, others) or not self.clear(grasp, qt, others):
            raise NoPath("wrist refinement invalidated clearance")
        self.event("approach", posture=seed["posture"])
        self.move(pre, qo, 0.58, others, True, surface, object_margin=0.0005)
        if seed["origin"] == "lightning_grasp":
            try:
                seated = self.ik.solve(shifted(grasp, -0.009 * grasp[:3, 0]), self.arm)
                self.lift_ik(grasp, seated)
            except NoPath:
                delta = pre[:3, 3] - grasp[:3, 3]
                delta = 0.035 * delta / max(np.linalg.norm(delta), 1e-6)
                try:
                    self.move(
                        shifted(self.io.fk(PALM), delta + [0, 0, 0.012]),
                        qo,
                        0.35,
                        others,
                    )
                except NoPath:
                    pass
                raise NoPath("executed pregrasp has insufficient lift reach")
        first = self.io.fk(PALM)
        close_started = False
        try:
            self.event(
                "close",
                posture=seed["posture"],
                visual_center=surface.world.mean(0).tolist(),
            )
            close_started = True
            sites = []
            if seed["origin"] == "lightning_grasp":
                # Seat the open hand before applying the native joint solution.
                seat = shifted(grasp, -0.009 * grasp[:3, 0])
                self.closing(seat, qo, 0.48)
                self.closing(grasp, qt, 0.65)
                self.closing(tight, qc, 0.42)
                self.hold(0.20)
            elif seed["origin"] == "wrist_handle_cross_sections":
                from .tabletop_handle import seat, surround_handle

                surround_handle(self, surface, grasp, qt)
                surface, pose, sites = seat(self, surface, pose, self.hand)
            elif seed["origin"] == "wrist_mesh_joint_fit":
                from .tabletop_handle import reobserve
                from .tabletop_meshfit import close_from_mesh

                # First surround the measured sides with an open hand. Moving a closed
                # thumb into an unsupported object can push it away from the fingers.
                self.closing(poses["open_seat"], qo, 0.38)
                self.closing(grasp, qt, 0.50)
                new_surface, new_pose = reobserve(
                    self, surface, pose, tag="mesh_seated"
                )
                movement = new_pose @ np.linalg.inv(pose)
                for site in seed["fit_sites"]:
                    site["goal"] = movement[:3, :3] @ site["goal"] + movement[:3, 3]
                    site["normal"] = movement[:3, :3] @ site["normal"]
                surface, pose = new_surface, new_pose
                sites = seed["fit_sites"]
                close_from_mesh(self, sites)
            else:
                self.closing(grasp, qt, 0.72)
                self.closing(tight, qc, 0.46)
                self.hold(0.22)
            palm0 = self.io.fk(PALM)
            first = palm0
            if not self.template_comparison:
                self.begin_lift(surface)
            carry = self.hand
            if sites:
                from .tabletop_meshfit import carry_shape

                carry = carry_shape(
                    self.hand_contacts,
                    palm0,
                    self.hand,
                    sites,
                    self.io.model.limits["right"],
                    self.table,
                )
                self.event(
                    "planned_carry",
                    q_supported=self.hand.tolist(),
                    q_carry=carry.tolist(),
                )
            self.event("test_lift")
            self.move(shifted(palm0, [0, 0, 0.022]), carry, 0.44)
            self.hold(0.16)
            verdict = self.vote(
                surface, pose, palm0, observed["color"], "test_observation"
            )
            if verdict == "unknown":
                self.hold(0.14)
                verdict = self.vote(
                    surface, pose, palm0, observed["color"], "test_observation_repeat"
                )
            if verdict != "carried":
                return verdict
            self.move(shifted(palm0, [0, 0, self.lift_travel]), self.hand, 0.65)
            if not self.template_comparison:
                # A compliant object can tilt or slip a little even as it follows the hand.
                # Use the observed motion of its bottom before starting the hold clock.
                for _ in range(2):
                    self.hold(0.12)
                    self.vote(
                        surface,
                        pose,
                        palm0,
                        observed["color"],
                        "measure_lift_clearance",
                    )
                    estimates = [
                        min(v["tracked_rise_m"], v["tracked_bottom_rise_m"])
                        for v in self.lift_views.values()
                        if v.get("texture_inliers", 0) >= 5
                    ]
                    if (
                        not estimates
                        or min(estimates) < 0.45 * self.lift_height
                        or min(estimates) >= self.lift_height + 0.005
                    ):
                        break
                    rise = min(0.024, self.lift_height + 0.010 - min(estimates))
                    current = self.io.fk(PALM)
                    self.event(
                        "correct_lift_clearance",
                        measured_rise_m=round(min(estimates), 4),
                        correction_m=round(rise, 4),
                    )
                    self.move(shifted(current, [0, 0, rise]), self.hand, 0.38)
            self.event("hold_start")
            verdicts = []
            for i in range(4):
                self.hold(0.51)
                verdicts.append(
                    self.vote(surface, pose, palm0, observed["color"], f"hold_{i + 1}")
                )
                if verdicts[-1] == "on_table":
                    break
            self.event("hold_end")
            return (
                "vision_pass"
                if len(verdicts) == 4
                and verdicts.count("carried") >= 3
                and verdicts[-1] == "carried"
                else "unconfirmed"
            )
        finally:
            if close_started:
                self.tracks = {}
                self.event("lower_and_release")
                current = self.io.fk(PALM)
                if current[2, 3] > first[2, 3] + 0.011:
                    try:
                        self.move(
                            shifted(current, [0, 0, first[2, 3] - current[2, 3]]),
                            self.hand,
                            0.57,
                        )
                    except NoPath:
                        pass
                self.command([self.arm, self.arm], (qo + self.open_hand) / 2, 0.35)
                back = pre[:3, 3] - grasp[:3, 3]
                if np.linalg.norm(back) < 0.005:
                    back = np.array([-0.3, -0.2, 0.8])
                back = back / np.linalg.norm(back) * 0.042
                back[2] = max(back[2], 0.016)
                try:
                    self.move(shifted(self.io.fk(PALM), back), qo, 0.40)
                except NoPath:
                    pass
                self.event("local_withdraw")

    def closing(self, goal, hand, seconds):
        start = self.arm.copy()
        target = self.ik.solve(goal, start)
        self.command([start, target], hand, seconds)

    def execute(self, targets=None, finish=True):
        (self.out / "observations").mkdir(exist_ok=True)
        self.survey()
        anchors = {name: dict(obs) for name, obs in self.seen.items()}
        pending = list(self.library if targets is None else dict.fromkeys(targets))
        if set(pending) - set(self.library):
            raise ValueError("target absent from the known object library")
        while pending:
            if self.results:
                self.leave_low_pose()
            head = self.survey()
            feedback = dict(head)
            for name, ref in anchors.items():
                if (
                    name in head
                    and np.linalg.norm(head[name]["center"][:2] - ref["center"][:2])
                    < 0.047
                ):
                    anchors[name] = head[name]
                    continue
                # Use the last observed location to direct a fresh wrist exposure during occlusion.
                if "wrist" in self.camera_roles:
                    feedback[name] = ref
            for name, item in head.items():
                if name not in anchors:
                    anchors[name] = item
            palm = self.io.fk(PALM)[:3, 3]

            def priority(n, feedback=feedback, palm=palm):
                if n not in feedback:
                    return 20.0
                obs = feedback[n]
                return obs["match_cost"] + 1.5 * np.linalg.norm(
                    obs["center"][:2] - palm[:2]
                )

            self.name = min(pending, key=priority)
            pending.remove(self.name)
            self.attempt = 0
            result = {"hand": self.side, "prior_postures": [], "outcomes": []}
            self.results[self.name] = result
            attempts = 3 if self.lightning is not None else 2
            for index in range(1, attempts + 1):
                self.attempt = index
                self.event("attempt_start")
                observed = feedback.get(self.name)
                if "wrist" in self.camera_roles and (
                    observed is None or observed["match_cost"] > 1.65
                ):
                    self.event("wrist_search")
                    # Registered table survey positions are independent of the randomized object poses.
                    for xy in ((0.80, -0.26), (0.76, 0.04)):
                        p = np.array([*xy, self.table + 0.035])
                        for v in self.look_poses(p, 0.24):
                            if not self.clear(v, self.hand):
                                continue
                            try:
                                self.inspect(v, self.hand, None)
                            except NoPath:
                                continue
                            self.survey(True)
                            break
                        if (
                            self.name in self.seen
                            and self.seen[self.name]["match_cost"] < 1.65
                        ):
                            observed = self.seen[self.name]
                            break
                if observed is None or observed["match_cost"] > 1.65:
                    reason = "unobserved"
                else:
                    try:
                        reason = self.trial(observed, result)
                    except (NoPath, ValueError) as error:
                        reason = str(error)
                result["outcomes"].append(reason)
                self.event("attempt_end", reason=reason)
                file = self.out / "decisions.json"
                previous = json.loads(file.read_bytes()) if file.exists() else {}
                save(file, dict(previous, **self.results))
                if reason == "vision_pass" or (
                    index == 1
                    and not result["prior_postures"]
                    and self.lightning is None
                ):
                    break
                if self.lightning is not None and (
                    reason.startswith("no Lightning Grasp bank")
                    or "zero collision-filtered" in reason
                    or (index >= 2 and not result["prior_postures"])
                ):
                    break
                if index < attempts:
                    fresh = self.survey()
                    if "wrist" not in self.camera_roles:
                        previous = feedback.pop(self.name, None)
                        local = self.reacquire_head(previous)
                        if local is not None:
                            feedback[self.name] = local
                    if self.name in fresh and (
                        observed is None
                        or np.linalg.norm(
                            fresh[self.name]["center"][:2] - observed["center"][:2]
                        )
                        < 0.055
                    ):
                        feedback[self.name] = fresh[self.name]
            self.event("target_end", reason=result["outcomes"][-1])
        if finish:
            self.event("episode_end")
        return self.results


def run(
    io,
    library,
    out,
    targets=None,
    template_comparison=False,
    shared=None,
    lift_height=0.05,
):
    Controller(io, library, out, template_comparison, shared, lift_height).execute(
        targets
    )
