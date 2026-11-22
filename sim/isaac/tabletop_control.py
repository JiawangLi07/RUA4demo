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


def wrap_entry_frames(tight):
    return shifted(tight, -0.015 * tight[:3, 0]), shifted(tight, -0.003 * tight[:3, 0])


def seating_frames(start, end):
    translation = np.linalg.norm(end[:3, 3] - start[:3, 3])
    angle = Rotation.from_matrix(start[:3, :3].T @ end[:3, :3]).magnitude()
    count = max(1, int(np.ceil(translation / 0.009)), int(np.ceil(angle / 0.10)))
    interpolation = Slerp(
        [0.0, 1.0], Rotation.from_matrix([start[:3, :3], end[:3, :3]])
    )
    for f in np.linspace(0, 1, count + 1)[1:]:
        frame = np.eye(4)
        frame[:3, :3] = interpolation(f).as_matrix()
        frame[:3, 3] = start[:3, 3] + f * (end[:3, 3] - start[:3, 3])
        yield float(f), frame


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
            error = NoPath("arm IK")
            error.residual = (
                float(np.linalg.norm(r[:3])),
                float(np.linalg.norm(r[3:]) / 0.20),
            )
            error.bound_joints = np.flatnonzero(
                np.minimum(result.x - self.bounds[0], self.bounds[1] - result.x) < 0.002
            ).tolist()
            raise error
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
        adaptive=False,
        hybrid_grasps=False,
        allow_hand_table_contact=False,
    ):
        self.io, self.library, self.out = io, library, out
        self.side = getattr(io, "side", "right")
        self.models = Models(library, mirror=self.side == "left", rest_faces=adaptive)
        self.lightning = lightning
        self.adaptive = adaptive
        self.allow_hand_table_contact = allow_hand_table_contact
        self.hybrid_grasps = hybrid_grasps
        self.shared = shared
        if not 0.03 <= lift_height <= 0.20:
            raise ValueError("expected a lift goal between 0.03 and 0.20 metres")
        self.seat_trim = None
        self.lift_height = lift_height
        # Allow for a few millimetres of object settling during continuous closure.
        self.lift_travel = lift_height + (0.017 if lightning is not None else 0.009)
        from .tabletop_meshfit import HandContacts

        self.hand_contacts = HandContacts(io)
        self.handle_library = {}
        self.wrap_cache = {}
        self.template_comparison = template_comparison
        self.tracks = {}
        self.last_measured_height = False
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
        self.table_front_x = None
        self.cloud = np.empty((0, 3))
        self.cache = {}
        self.open_hand = self.hand.copy()
        self.seen = {}
        self.camera_roles = getattr(io, "camera_roles", ("head", "wrist"))
        self.sampling_seed = getattr(io, "paired_scene_seed", None)
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
        frame = self.snapshot("wrist" if wrist else self.camera_roles[0], "survey")
        if not wrist:
            self.survey_frame = frame
        self.table, pieces = proposals(frame, self.io, self.table)
        if not wrist:
            if self.adaptive and self.table_front_x is None:
                from .tabletop_vision import world_image

                xyz, valid = world_image(frame)
                observed = xyz[
                    valid
                    & (abs(xyz[:, :, 2] - self.table) < 0.0025)
                    & (xyz[:, :, 0] > 0.35)
                    & (abs(xyz[:, :, 1]) < 0.55)
                ]
                if len(observed) >= 100:
                    self.table_front_x = (
                        float(np.quantile(observed[:, 0], 0.005)) - 0.030
                    )
            self.cloud = (
                np.vstack([x["xyz"][::4] for x in pieces])
                if pieces
                else np.empty((0, 3))
            )
            self.head_pieces = pieces
            self.seen = self.models.identify(pieces)
            if self.sampling_seed is not None:
                folder = self.out / "point_clouds"
                folder.mkdir(exist_ok=True)
                for name, part in self.seen.items():
                    path = folder / f"{name}.npz"
                    if not path.exists():
                        np.savez_compressed(
                            path,
                            xyz=part["xyz"],
                            rgb=part["rgb"],
                            uv=part["uv"],
                            sensor=frame["sensor"],
                            time=frame["time"],
                            K=frame["K"],
                            pose=frame["pose"],
                        )
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
            "survey_wrist" if wrist else "survey_" + self.camera_roles[0],
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
        if (
            not getattr(self, "allow_hand_table_contact", False)
            and vertices[:, 2].min() < self.table - 0.0015
        ):
            return False
        if others is not None and len(others.data):
            distance = others.query(vertices, distance_upper_bound=0.005)[0]
            if np.count_nonzero(np.isfinite(distance)) >= 3:
                return False
        return True

    def arm_above_table(self, joints):
        if self.table_front_x is None:
            return True
        angles = dict(zip(ARM, joints))
        for name, vertices in self.io.model.forearm_support.items():
            if not name.endswith("_R"):
                continue
            frame = self.io.tree.pose(name, angles)
            # Check the part of each forearm link over the observed tabletop.
            z = vertices @ frame[2, :3] + frame[2, 3]
            low = vertices[z < self.table + 0.004]
            if (
                len(low)
                and (low @ frame[0, :3] + frame[0, 3] >= self.table_front_x).any()
            ):
                return False
        return True

    def fixture_clear(self, pose):
        return all(
            (vertices @ pose[2, :3]).min() + pose[2, 3] >= self.table + 0.003
            for vertices in getattr(self.io.model, "fixture_palm", {}).values()
        )

    def approach_orientation_clear(self, palm):
        if getattr(self, "adaptive", False):
            # An oblique approach to a flat object may lower the knuckles while
            # its thumb remains lateral or uppermost. Bound radial inversion alone.
            return bool(palm[2, 1] >= -0.30)
        if self.lightning is None:
            return True
        # The parked hands have fingers upright and the radial axis horizontal.
        # Allow a small tracking tolerance from park and enforce the same half-space
        # along Cartesian and fallback joint-space paths, including local retries.
        return bool(palm[2, 1] >= -0.055 and palm[2, 2] >= -0.20)

    def command(self, joints, hand, duration, observe=None):
        """Interpolate the joint path, or stop at the issued command on a visual correction."""
        io = self.io
        start = self.hand.copy()
        hand = np.clip(hand, *io.model.limits["right"].T)
        joints = np.asarray(joints)
        if self.adaptive and not swept_clear(
            joints, lambda arm, _: self.arm_above_table(arm)
        ):
            raise NoPath("forearm would intersect the observed tabletop")
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
        guard = getattr(self, "lower_guard", None)
        lower = guard.sweep(self, joints, start, hand) if guard is not None else set()
        arc = np.r_[0.0, np.cumsum(np.max(np.abs(np.diff(joints, axis=0)), axis=1))]
        # The cubic has peak slope 1.5 and ends at 84% of the interval.
        # Leave tracking headroom below the arm drive's approximately 3.14 rad/s limit.
        duration = max(duration, 1.5 * arc[-1] / (0.84 * 2.25))
        ticks = max(2, round(duration / io.dt))
        sample_at = {round(ticks * t) for t in (0.29, 0.44, 0.65)} if observe else set()
        for i in range(1, ticks + 1):
            f = min(1.0, i / (0.84 * ticks))
            f = f * f * (3 - 2 * f)
            distance = f * arc[-1]
            k = min(np.searchsorted(arc, distance, side="right") - 1, len(arc) - 2)
            blend = (distance - arc[k]) / max(arc[k + 1] - arc[k], 1e-12)
            arm = joints[k] + blend * (joints[k + 1] - joints[k])
            drive = start + f * (hand - start)
            io.step(arm, drive)
            if i in sample_at and observe(i / ticks):
                self.arm, self.hand = arm.copy(), drive.copy()
                if guard is not None:
                    prefix = np.vstack((joints[: k + 1], arm))
                    guard.used.update(guard.sweep(self, prefix, start, drive))
                return
        self.arm, self.hand = joints[-1].copy(), hand.copy()
        if guard is not None:
            guard.used.update(lower)

    def move(
        self,
        goal,
        hand,
        duration,
        others=None,
        transit=False,
        protected=None,
        object_margin=0.0025,
        endpoint_check=None,
        plan_only=False,
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
            if self.adaptive and not self.arm_above_table(q):
                return False
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
            if endpoint_check is not None:
                endpoint_check(joints[-1])
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
            if endpoint_check is not None:
                valid = []
                for q in endpoints:
                    try:
                        endpoint_check(q)
                        valid.append(q)
                    except NoPath:
                        continue
                endpoints = valid
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
        if plan_only:
            return joints
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
        top = self.ik.solve(shifted(palm, [0, 0, self.lift_travel]), probe)
        if getattr(self, "adaptive", False) and not all(
            self.arm_above_table(q) for q in (pre_joints, grasp, probe, top)
        ):
            raise NoPath("forearm clearance along closing and lifting branch")

    def lightning_lift_ik(self, palm, pre_joints):
        seat = self.ik.solve(shifted(palm, -0.009 * palm[:3, 0]), pre_joints)
        self.lift_ik(palm, seat)

    def check_closure_path(self, origin, arm, grasp, tight):
        goals = []
        if origin == "seated_wrap":
            seat, near = wrap_entry_frames(tight)
            goals.extend((("seat", seat), ("opposition", near), ("tighten", tight)))
        elif origin == "power_wrap":
            goals.extend(
                (
                    ("native_entry", grasp),
                    ("seat", shifted(tight, -0.012 * tight[:3, 0])),
                    ("opposition", shifted(tight, -0.003 * tight[:3, 0])),
                    ("tighten", tight),
                )
            )
        else:
            if origin in ("head_wall_envelope", "lightning_grasp"):
                standoff = 0.012 if origin == "head_wall_envelope" else 0.009
                goals.append(("seat", shifted(grasp, -standoff * grasp[:3, 0])))
            goals.extend((("grasp", grasp), ("tighten", tight)))
        carry_pose = tight
        if origin in ("head_wall_envelope", "power_wrap", "seated_wrap"):
            carry_pose = shifted(tight, 0.006 * tight[:3, 0])
            goals.append(("preload", carry_pose))
        goals.extend(
            (step, shifted(carry_pose, [0, 0, dz]))
            for step, dz in (("probe", 0.022), ("lift", self.lift_travel))
        )
        path = [arm]
        for stage, goal in goals:
            try:
                path.append(self.ik.solve(goal, path[-1]))
            except NoPath as error:
                error.stage = stage + "_ik"
                raise

        def free(q, _):
            palm = self.ik.fk(q)
            if not self.arm_above_table(q):
                raise NoPath(stage + "_sweep_forearm")
            if not self.fixture_clear(palm):
                raise NoPath(stage + "_sweep_flange")
            if not self.approach_orientation_clear(palm):
                raise NoPath(stage + "_sweep_orientation")
            return True

        for (stage, _), (start, end) in zip(goals, pairwise(path)):
            swept_clear([start, end], free)

    def coarse_surface(self, name, observed):
        from .head_wrist_surface import RegisteredSurface

        try:
            surface, pose, fit = self.models.fit(
                name,
                observed["xyz"],
                self.table,
                depth_frame=self.survey_frame
                if self.allow_hand_table_contact
                else None,
            )
            self.event(
                "registered_" + self.camera_roles[0],
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
                f"coarse_{self.camera_roles[0]}_roi", center=observed["center"].tolist(), points=len(obs)
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
        surface.allow_hand_table_contact = self.allow_hand_table_contact
        if self.allow_hand_table_contact and self.lightning is not None:
            size = np.ptp(surface.world, axis=0)
            horizontal = np.linalg.norm(size[:2])
            # The far tip of a long tool can hang below its initial pose if it
            # rotates a few degrees on the pads. Reserve travel from the observed geometry.
            tip_allowance = (
                min(0.012, 0.5 * horizontal * np.sin(np.deg2rad(8)))
                if horizontal > 2 * size[2]
                else 0.0
            )
            self.lift_travel = self.lift_height + 0.017 + tip_allowance
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
                raise NoPath(f"{self.camera_roles[0]} mesh registration unconfirmed")
            from .tabletop_views import ViewBank
            from .tabletop_vision import add_live_samples

            add_live_samples(surface, target, [], self.camera_roles[0])
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
                self.camera_roles[0] + "_only_grasp", points=len(target), mesh_to_world=pose.tolist()
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
        if self.lightning is not None and not self.hybrid_grasps:
            return self.lightning.select(
                self, name, refined, reg, others, prior_postures
            )
        if self.adaptive:
            from .tabletop_adaptive import select

            return select(self, name, refined, reg, others, prior_postures)
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

    def reacquire_head(self, reference, *, after_release=False):
        """Match a fresh primary-camera component near a recently released or occluded target."""
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
            if not after_release:
                return None
            # A tipped cylinder can move half its own length and exchange its
            # horizontal and vertical extents. Verify a fresh depth component
            # against the target mesh before relaxing the old location prior.
            span = max(reference["dims"])
            for p in self.head_pieces:
                delta = np.linalg.norm(p["center"][:2] - reference["center"][:2])
                color = np.linalg.norm(p["color"] - reference["color"])
                old = np.maximum(np.sort(reference["dims"]), 0.002)
                new = np.maximum(np.sort(p["dims"]), 0.002)
                if (
                    delta > min(0.17, 0.06 + 0.52 * span)
                    or color > 0.28
                    or np.linalg.norm(np.log(new / old)) > 0.95
                ):
                    continue
                try:
                    _, _, fit = self.models.fit(self.name, p["xyz"], self.table)
                except ValueError:
                    continue
                if (
                    fit["median_residual_m"] <= 0.0025
                    and fit["p90_residual_m"] <= 0.007
                ):
                    candidates.append(
                        (0.35 * delta / 0.10 + color + 80.0 * fit["p90_residual_m"], p)
                    )
            if not candidates:
                return None
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.16:
            return None
        score, p = candidates[0]
        self.event(
            f"reacquired_{self.camera_roles[0]}_roi",
            points=len(p["xyz"]),
            offset_m=float(np.linalg.norm(p["center"][:2] - reference["center"][:2])),
        )
        return dict(p, match_cost=score)

    def begin_lift(self, surface):
        from .tabletop_tracking import initialize

        self.last_measured_height = False
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
            pivot = (
                tag.startswith("test_observation")
                and self.adaptive
                and any(
                    v["texture_inliers"] >= 6
                    and v.get("texture_span_m", 0) >= 0.008
                    and v["tracked_bottom_rise_m"] <= 0.0035
                    and v["tracked_rise_m"] - v["tracked_bottom_rise_m"] >= 0.006
                    and v["tracked_rise_m"] >= 0.006
                    for v in tracks
                )
            )
            if pivot:
                vote = "pivot_on_table"
            elif all(
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
                follows_last_height = (
                    self.adaptive
                    and not heights
                    and self.last_measured_height
                    and static == 0
                    and carried >= 24
                )
                if not follows_last_height:
                    vote = "unknown"
        if not tag.startswith("test_observation") and not self.template_comparison:
            # A later frame can lose its texture while still strongly matching
            # the same raised mesh. Establish height metrically before using it.
            if vote != "carried":
                self.last_measured_height = False
            elif tracks or any("geometry_clearance_m" in v for v in votes.values()):
                self.last_measured_height = True
        self.lift_views = votes
        self.event(tag, vote=vote, views=votes)
        return vote

    def approach(self, seed, poses, surface, others):
        pre, grasp, tight = [poses[k] for k in ("pre", "grasp", "tight")]
        qo, qt = [np.asarray(seed[k]) for k in ("q_open", "q_touch")]
        entry, entry_hand = grasp, qt
        if seed.get("closure_mode") == "seated_wrap":
            entry = wrap_entry_frames(tight)[0]
            entry_hand = seed["seated_q"]
        if not self.clear(pre, qo, others) or not self.clear(entry, entry_hand, others):
            raise NoPath("registered target invalidated clearance")
        endpoint_check = None
        if self.adaptive:

            def endpoint_check(arm):
                self.check_closure_path(
                    seed.get("closure_mode") or seed["origin"],
                    arm,
                    grasp,
                    tight,
                )
        elif seed["origin"] == "lightning_grasp":

            def endpoint_check(arm):
                self.lightning_lift_ik(grasp, arm)

        self.lower_guard = seed.get("lower_guard")
        self.event("approach", posture=seed["posture"])
        self.move(
            pre,
            qo,
            0.58,
            others,
            True,
            surface,
            object_margin=0.0005,
            endpoint_check=endpoint_check,
        )
        if not self.adaptive and seed["origin"] == "lightning_grasp":
            try:
                self.lightning_lift_ik(grasp, self.arm)
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

    def trial(self, observed, result):
        self.seat_trim = None
        name = self.name
        surface, pose, poses, seed, others = self.geometry(name, observed, result)
        for plan_no in range(4):
            result["prior_postures"].append(seed["posture"])
            start = self.io.time()
            try:
                self.approach(seed, poses, surface, others)
                break
            except NoPath as exc:
                self.lower_guard = None
                if not self.adaptive or self.io.time() != start or plan_no == 3:
                    raise
                self.event(
                    "pregrasp_route_rejected", posture=seed["posture"], reason=str(exc)
                )
                seed, poses = self.fit_registered(
                    name, None, surface, pose, (), result["prior_postures"], others
                )
        pre, grasp, tight = [poses[k] for k in ("pre", "grasp", "tight")]
        qo, qt, qc = [np.asarray(seed[k]) for k in ("q_open", "q_touch", "q_tight")]
        first = self.io.fk(PALM)
        close_started = False
        source = {"origin": seed["origin"], **seed.get("lightning_candidate", {})}
        native_registration = pose.copy()
        try:
            verified = (
                self.adaptive
                and "wrist" not in self.camera_roles
                and seed["origin"] == "lightning_grasp"
                and not seed.get("power_wrap")
            )
            if verified:
                from .tabletop_servo import seat_before_close

                surface, pose, shift = seat_before_close(self, surface, pose, grasp, qo)
                grasp, tight = shifted(grasp, shift), shifted(tight, shift)
                if getattr(self, "lower_guard", None) is not None:
                    from .tabletop_opposition import LowerEdge

                    old = self.lower_guard
                    self.lower_guard = LowerEdge(surface, self.table)
                    self.lower_guard.used.update(old.used)
            self.event(
                "close",
                posture=seed["posture"],
                visual_center=surface.world.mean(0).tolist(),
                **source,
            )
            close_started = True
            sites = []
            power_wrap = (
                seed.get("power_wrap") or seed["origin"] == "head_wall_envelope"
            )
            if seed["origin"] == "lightning_grasp" and not power_wrap:
                # Seat the open hand before applying the native joint solution.
                if not verified:
                    seat = shifted(grasp, -0.009 * grasp[:3, 0])
                    self.closing(seat, qo, 0.48)
                from .tabletop_servo import (
                    follow_during_closure,
                    move_native_sites,
                    opposed_touch,
                )

                sites = seed.get("native_sites", [])
                geometry = seed["lightning_candidate"]
                capacity = geometry.get("estimated_upward_capacity")
                risky = capacity is not None and (
                    capacity < 0.5 or geometry["contact_lever_xy_m"] > 0.020
                )
                if sites:
                    move_native_sites(sites, native_registration, pose)
                if verified and sites and risky:
                    surface, pose, correction = opposed_touch(
                        self, surface, pose, sites, grasp, qt
                    )
                    tight = shifted(tight, correction)
                elif verified and sites:
                    surface, pose, grasp, tight = follow_during_closure(
                        self, surface, pose, sites, grasp, tight, qt, others
                    )
                else:
                    self.closing(grasp, qt, 0.65)
                self.closing(tight, qc, 0.42)
                if sites:
                    from .tabletop_handle import reobserve
                    from .tabletop_meshfit import close_from_mesh

                    observed_surface, measured = reobserve(
                        self, surface, pose, tag="native_seated"
                    )
                    move_native_sites(sites, pose, measured)
                    surface, pose = observed_surface, measured
                    close_from_mesh(self, sites, seated=True)
                self.hold(0.20)
            elif power_wrap:
                from .paths import RIGHT_JOINTS

                seated_q = seed.get("seated_q")
                if seated_q is not None:
                    # Keep all fingers outside the object until the palm reaches
                    # the near wall, then oppose the thumb across that wall.
                    tips = np.array(
                        [n.endswith(("_PIP", "_DIP")) for n in RIGHT_JOINTS]
                    )
                    hooked = qo.copy()
                    hooked[tips] += 0.52 * (seated_q[tips] - qo[tips])
                    seat, near = wrap_entry_frames(tight)
                    self.closing(seat, hooked, 0.68)
                    opposed = hooked + 0.44 * (seated_q - hooked)
                    opposed[:5] = qo[:5] + 0.76 * (seated_q[:5] - qo[:5])
                    self.closing(near, opposed, 0.63)
                    # The native anchor solution at this wrist supplies the
                    # direction of the final finger travel. Follow under load
                    # with a bounded motor target on the measured cross section.
                    knots = seed.get("seating_knots", [])
                    projected = knots[-1][1] if knots else qt
                    first_touch = seated_q + 0.16 * np.clip(
                        projected - seated_q, -0.45, 0.45
                    )
                    self.closing(tight, first_touch, 0.59)
                    self.event(
                        "native_targets_seated", posture=seed["posture"], **source
                    )
                    self.closing(tight, qc, 0.36)
                    self.hold(0.18)
                else:
                    hooked = qo.copy()
                    tips = np.array(
                        [n.endswith(("_PIP", "_DIP")) for n in RIGHT_JOINTS]
                    )
                    hooked[tips] += 0.52 * (qt[tips] - qo[tips])
                    seat = shifted(grasp, -0.012 * grasp[:3, 0])
                    self.closing(seat, hooked, 0.68)
                    opposed = hooked + 0.44 * (qt - hooked)
                    opposed[:5] = qo[:5] + 0.76 * (qt[:5] - qo[:5])
                    self.closing(grasp, opposed, 0.68)
                    self.closing(tight, qt, 0.65)
                    self.closing(tight, qc, 0.35)
                    self.hold(0.18)
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
            self.event("closure_completed", posture=seed["posture"], **source)
            palm0 = self.io.fk(PALM)
            first = palm0
            if not self.template_comparison:
                self.begin_lift(surface)
            carry = self.hand.copy()
            if power_wrap:
                # Compensate finger compliance on the wall while seating the
                # palm. Keep this motor target through the probe and final lift.
                upper = self.io.model.limits["right"][:, 1]
                for j, joint in enumerate(RIGHT_JOINTS):
                    if joint.endswith(("MCP_FE", "_PIP", "_DIP", "_IP")):
                        preload = 0.22 if "_ring_" in joint else 0.13
                        if "_thumb_" in joint:
                            preload = 0.07
                        carry[j] = min(upper[j], carry[j] + preload)
                from .tabletop_adaptive import draw_pads_to_palm

                squeeze = 0.006
                advance = np.zeros(3)
                digit_offsets = None
                if seed.get("seated_q") is not None:
                    from .tabletop_handle import reobserve
                    from .tabletop_lightning import central_palm_samples

                    new_surface, new_pose = reobserve(
                        self, surface, pose, tag="seated_wrap_head"
                    )
                    # A consistent head-registered lateral shift locates the
                    # object after seating. Avoid following a large outlier.
                    measured = new_pose[:3, 3] - pose[:3, 3]
                    if np.linalg.norm(measured[:2]) <= 0.018:
                        advance[:2] = 0.70 * measured[:2]
                        surface, pose = new_surface, new_pose
                    patch = central_palm_samples(self.io.model.hand_support[PALM])
                    rel = np.linalg.solve(pose, palm0)
                    gaps = surface.tree.query(patch @ rel[:3, :3].T + rel[:3, 3])[0]
                    squeeze = float(
                        np.clip(0.005 + np.quantile(gaps, 0.15), 0.006, 0.012)
                    )
                    digit_offsets = {
                        digit: np.array([-0.005, 0.0, -0.003])
                        for digit, explicit in zip(
                            ("thumb", "index", "middle", "ring", "pinky"),
                            seed["explicit_digits"],
                        )
                        if not explicit
                    }
                    self.event(
                        "wrap_head_pressure",
                        head_shift_m=measured.tolist(),
                        palm_seat_m=squeeze,
                        unconstrained_digits=list(digit_offsets),
                    )
                carry = draw_pads_to_palm(
                    self,
                    carry,
                    0.011 if digit_offsets is not None else 0.009,
                    digit_offsets,
                )
                default = 0.006 * palm0[:3, 0]
                requested = advance + squeeze * palm0[:3, 0]
                for fraction in (1.0, 0.5, 0.0):
                    dest = shifted(palm0, default + fraction * (requested - default))
                    if not self.clear(dest, carry, others):
                        continue
                    try:
                        self.lift_ik(dest, self.arm)
                        self.closing(dest, carry, 0.56)
                    except NoPath:
                        continue
                    break
                else:
                    self.closing(palm0, carry, 0.52)
                palm0 = self.io.fk(PALM)
                first = palm0
            if getattr(self, "lower_guard", None) is not None:
                self.lower_guard.carry_reference = palm0.copy()
                self.event(
                    "opposed_carry",
                    lower_non_thumb_digits=sorted(self.lower_guard.used),
                )
            elif sites:
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
            first_vote = verdict
            if verdict in ("unknown", "pivot_on_table"):
                self.hold(0.14)
                verdict = self.vote(
                    surface, pose, palm0, observed["color"], "test_observation_repeat"
                )
                if first_vote == verdict == "pivot_on_table":
                    return verdict
            if seed.get("seated_q") is not None and verdict == "carried":
                free = {
                    d: np.array([-0.004, 0.0, -0.003])
                    for d, has in zip(
                        ("thumb", "index", "middle", "ring", "pinky"),
                        seed["explicit_digits"],
                    )
                    if not has
                }
                if free:
                    from .tabletop_adaptive import draw_pads_to_palm

                    held = self.io.fk(PALM)
                    supported = draw_pads_to_palm(self, self.hand, 0.0, free)
                    if self.clear(held, supported, others):
                        try:
                            self.closing(held, supported, 0.30)
                            self.event(
                                "wrap_probe_pressure",
                                palm_advance_m=0.0,
                                free_digits=list(free),
                            )
                        except NoPath:
                            pass
            if verdict != "carried":
                # When the closing hand masks the object, static/raised mesh
                # hypotheses can be visually indistinguishable at 22 mm. Allow
                # the bounded verification motion unless texture stayed fixed.
                stationary = any(
                    v.get("texture_inliers", 0) >= 5
                    and abs(v["tracked_rise_m"]) < 0.004
                    and abs(v["tracked_bottom_rise_m"]) < 0.004
                    for v in self.lift_views.values()
                )
                if not self.adaptive or stationary:
                    return verdict
                self.event("continue_visual_probe", initial_vote=verdict)
            travel = self.lift_travel
            if self.allow_hand_table_contact and sites and verdict == "carried":
                measured = []
                for view in self.lift_views.values():
                    if view.get("texture_inliers", 0) >= 5:
                        measured.append(
                            min(view["tracked_rise_m"], view["tracked_bottom_rise_m"])
                        )
                    if (
                        view.get("height_support", 0) >= 24
                        and view.get("height_margin_m", 0) >= 0.0009
                    ):
                        measured.append(view["visible_lift_m"])
                if measured and min(measured) >= 0.005:
                    wrist_rise = self.io.fk(PALM)[2, 3] - palm0[2, 3]
                    takeup = float(np.clip(wrist_rise - min(measured), 0, 0.016))
                    if takeup >= 0.002:
                        suggested = shifted(palm0, [0, 0, travel + takeup])
                        try:
                            checked = self.ik.solve(suggested, self.arm)
                        except NoPath:
                            checked = None
                        if checked is not None and self.arm_above_table(checked):
                            travel += takeup
                            self.event(
                                "compensate_probe_slip",
                                visual_rise_m=min(measured),
                                commanded_wrist_rise_m=float(wrist_rise),
                                extra_travel_m=takeup,
                            )
            top = shifted(palm0, [0, 0, travel])
            if power_wrap:
                from .tabletop_adaptive import thumb_up_carry

                upright = thumb_up_carry(
                    self, top, qt if seed.get("seated_q") is None else seed["seated_q"]
                )
                try:
                    if not self.clear(upright, self.hand, others):
                        raise NoPath("no clearance for a thumb-up carry")
                    self.move(upright, self.hand, 0.65)
                    self.event(
                        "thumb_up_carry",
                        rotation_deg=round(
                            float(
                                np.rad2deg(
                                    Rotation.from_matrix(
                                        upright[:3, :3] @ top[:3, :3].T
                                    ).magnitude()
                                )
                            ),
                            2,
                        ),
                    )
                except NoPath:
                    self.move(top, self.hand, 0.65)
            else:
                self.move(top, self.hand, 0.65)
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
                if verdicts[-1] == "on_table" and (
                    not self.adaptive
                    or any(
                        v.get("texture_inliers", 0) >= 5
                        and abs(v["tracked_rise_m"]) < 0.004
                        for v in self.lift_views.values()
                    )
                ):
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
            self.seat_trim = None
            if getattr(self, "lower_guard", None) is not None:
                self.event(
                    "opposition_guard_end",
                    lower_non_thumb_digits=sorted(self.lower_guard.used),
                )
                self.lower_guard = None
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
                if self.adaptive:
                    # Clear the target before planning the next guarded approach.
                    # Releasing inside its silhouette otherwise blocks every retry.
                    start = self.io.fk(PALM)
                    for offset in (
                        1.7 * back + [0, 0, 0.018],
                        1.3 * back + [0, 0, 0.045],
                        back,
                    ):
                        dest = shifted(start, offset)
                        if not surface.clear(self.link_shapes(qo), dest, 0.002):
                            continue
                        try:
                            self.move(dest, qo, 0.48, others, transit=True)
                        except NoPath:
                            continue
                        self.event("local_withdraw", translation_m=offset.tolist())
                        break
                else:
                    try:
                        self.move(shifted(self.io.fk(PALM), back), qo, 0.40)
                    except NoPath:
                        pass
                    self.event("local_withdraw")

    def closing(self, goal, hand, seconds, observe=None):
        start = self.arm.copy()
        target = self.ik.solve(goal, start)
        if self.seat_trim is not None:
            target = np.clip(target + self.seat_trim, *self.ik.bounds)
        self.command([start, target], hand, seconds, observe=observe)

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
                if (
                    self.adaptive
                    and name not in feedback
                    and "wrist" not in self.camera_roles
                ):
                    current = self.reacquire_head(ref)
                    if current is not None:
                        feedback[name] = current
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
            file = self.out / "decisions.json"
            previous_decisions = json.loads(file.read_bytes()) if file.exists() else {}
            earlier = previous_decisions.get(self.name, {})
            offset = len(earlier.get("outcomes", ()))
            result = {
                "hand": self.side,
                "prior_postures": [],
                "outcomes": list(earlier.get("outcomes", ())),
                "attempt_hands": list(
                    earlier.get("attempt_hands", [earlier.get("hand")] * offset)
                ),
            }

            self.results[self.name] = result
            last_seen = feedback.get(self.name)
            attempts = 3 if self.lightning is not None else 2
            for index in range(1, attempts + 1):
                self.attempt = index + offset
                result["attempt_hands"].append(self.side)
                self.event("attempt_start")
                observed = feedback.get(self.name)
                if observed is not None:
                    last_seen = observed
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
                save(file, dict(previous, **{self.name: result}))
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
                        feedback.pop(self.name, None)
                        released = self.adaptive and any(
                            e["name"] == self.name and e["phase"] == "closure_completed"
                            for e in self.events
                        )
                        local = self.reacquire_head(last_seen, after_release=released)
                        if (
                            local is None
                            and self.name not in fresh
                            and released
                            and self.allow_hand_table_contact
                            and last_seen is not None
                        ):
                            palm = self.io.fk(PALM)
                            eye = self.io.camera(self.camera_roles[0])["pose"][:3, 3]
                            sight = np.asarray(last_seen["center"])[:2] - eye[:2]
                            lateral = np.array([-sight[1], sight[0]])
                            lateral /= max(1e-8, np.linalg.norm(lateral))
                            if lateral @ (palm[:2, 3] - last_seen["center"][:2]) < 0:
                                lateral *= -1
                            for sign in (1, -1):
                                goal = shifted(
                                    palm, np.r_[sign * 0.065 * lateral, 0.018]
                                )
                                try:
                                    self.move(
                                        goal,
                                        self.hand,
                                        0.48,
                                        cKDTree(self.cloud),
                                        transit=True,
                                    )
                                except NoPath:
                                    continue
                                self.event(f"clear_{self.camera_roles[0]}_view")
                                fresh = self.survey()
                                local = self.reacquire_head(
                                    last_seen, after_release=True
                                )
                                break
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
