"""Plan a continuous approach through an object-specific, camera-visible open grasp."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from .head_wrist_geometry import hand_shapes, pad_targets
from .head_wrist_surface import band_grasps, confirm_pad_surfaces
from .head_wrist_trial import FINGERS, HeadWristTrial, Rejected, cylinder


@dataclass
class Grasp:
    pre: np.ndarray
    close: np.ndarray
    opened: np.ndarray
    closed: np.ndarray
    limit: np.ndarray
    details: dict


def family(points):
    """Classify the contact region from horizontal cross sections of the registered mesh."""
    bottom, top = np.quantile(points[:, 2], [0.008, 0.992])
    height = top - bottom
    widths = []
    for fraction in (0.25, 0.45, 0.65):
        z = bottom + fraction * height
        section = points[np.abs(points[:, 2] - z) < max(0.002, 0.07 * height), :2]
        if len(section) < 8:
            section = points[:, :2]
        _, axes = np.linalg.eigh(np.cov(section.T))
        widths.append(np.sort(np.ptp(np.quantile(section @ axes, [0.025, 0.975], axis=0), axis=0)))
    widths = np.asarray(widths)
    minor, major = np.median(widths, axis=0)
    if major > 1.65 * minor:
        return "across_handle"
    if height < 0.52 * minor:
        return "over_edge"
    if height > 1.1 * minor:
        if np.min(widths[:, 0]) >= 0.78 * np.max(widths[:, 0]):
            return "side_wall"
        return "narrow_tip"
    return "round_wrap"


def rotation_error(a, b):
    return float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def interpolate(a, b, f):
    t = np.eye(4)
    t[:3, :3] = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(f).as_matrix()
    t[:3, 3] = a[:3, 3] * (1 - f) + b[:3, 3] * f
    return t


class WristVisibility:
    """Project the target and the preshaped hand into the actual offset D405 imager."""

    def __init__(self, palm_camera, K, shape, table_bounds=None):
        self.palm_camera, self.K = palm_camera, K
        self.table_bounds = table_bounds
        self.scale = 8
        self.width, self.height = 480, 640
        w, h = self.width // self.scale, self.height // self.scale
        zbuffer = np.full((h, w), np.inf)
        for vertices in shape.values():
            camera = (vertices - palm_camera[:3, 3]) @ palm_camera[:3, :3]
            camera = camera[camera[:, 2] > 0.012]
            if len(camera) < 4:
                continue
            pix = (camera[:, :2] / camera[:, 2:3] * K.diagonal()[:2] + K[:2, 2]) / self.scale
            if np.any(pix.max(0) < 0) or np.any(pix.min(0) >= [w, h]):
                continue
            hull = ConvexHull(pix).vertices
            mask = Image.new("1", (w, h))
            ImageDraw.Draw(mask).polygon([tuple(x) for x in pix[hull]], fill=1)
            inside = np.asarray(mask)
            zbuffer[inside] = np.minimum(zbuffer[inside], np.min(camera[:, 2]))
        self.depth = zbuffer

    def fraction(self, palm, points):
        world = palm @ self.palm_camera
        cam = (points - world[:3, 3]) @ world[:3, :3]
        z = cam[:, 2]
        uv = cam[:, :2] / np.maximum(z[:, None], 1e-4) * self.K.diagonal()[:2] + self.K[:2, 2]
        inside = (
            (z > 0.032)
            & (z < 0.40)
            & (uv[:, 0] >= 6)
            & (uv[:, 0] < self.width - 6)
            & (uv[:, 1] >= 6)
            & (uv[:, 1] < self.height - 6)
        )
        if self.table_bounds is not None and world[2, 3] < self.table_bounds[1, 2] + 0.002:
            # Rays from a wrist imager near the table edge can meet the tabletop
            # even when the virtual hand and all image pixels appear clear.
            ray = points - world[:3, 3]
            top = self.table_bounds[1, 2] + 0.002
            alpha = (top - world[2, 3]) / np.maximum(ray[:, 2], 1e-8)
            crossing = world[:2, 3] + alpha[:, None] * ray[:, :2]
            blocked = (
                (alpha >= 0)
                & (alpha <= 1)
                & np.all(crossing >= self.table_bounds[0, :2] - 0.003, axis=1)
                & np.all(crossing <= self.table_bounds[1, :2] + 0.003, axis=1)
            )
            inside &= ~blocked
        if not inside.any():
            return 0.0
        u, v = (uv[inside] / self.scale).astype(int).T
        return float(np.sum(z[inside] < self.depth[v, u] + 0.002) / len(points))


def path_clear(surface, a, b, shapes):
    for f, shape in zip(np.linspace(0, 1, len(shapes)), shapes):
        if not surface.clear(shape, interpolate(a, b, f), 0.001):
            return False
    return True


def view_options(scene, surface, grasp, home, palm_camera, K, qhome):
    fractions = np.linspace(0, 1, 9)
    closing = [hand_shapes(scene, (1 - f) * qhome + f * grasp.opened) for f in fractions]
    fixed = [closing[-1]] * len(fractions)
    vision = WristVisibility(palm_camera, K, closing[-1], getattr(scene, "table_bounds", None))
    clouds = surface.world[:: max(1, len(surface.world) // 170)]
    center = (surface.world.min(0) + surface.world.max(0)) * 0.5
    dest = grasp.pre
    base_length = np.linalg.norm(home[:3, 3] - dest[:3, 3])
    options = []

    def offer(pose):
        to_pre = np.linalg.norm(pose[:3, 3] - dest[:3, 3])
        from_home = np.linalg.norm(pose[:3, 3] - home[:3, 3])
        if not 0.024 <= to_pre <= 0.18 or from_home < 0.025 or from_home > 0.32:
            return
        if dest[2, 3] - pose[2, 3] > 0.012:
            return
        extra = from_home + to_pre - base_length
        if extra > 0.133:
            return
        visible = vision.fraction(pose, clouds)
        if visible < 0.21:
            return
        turn = rotation_error(pose, dest)
        score = extra + 0.34 * to_pre + 0.030 * turn - 0.071 * min(visible, 0.9)
        options.append((score, pose, visible, extra, turn))

    for blend in (1.0, 0.86, 0.72, 0.58, 0.44, 0.32):
        base = interpolate(home, dest, blend)
        for progress in (0.46, 0.60, 0.72, 0.82):
            pose = base.copy()
            pose[:3, 3] = home[:3, 3] * (1 - progress) + dest[:3, 3] * progress
            offer(pose)
        rot = base[:3, :3]
        for depth in (0.095, 0.13, 0.165, 0.20, 0.24, 0.28):
            for u in (0.28, 0.48, 0.68, 0.84):
                for v in (0.27, 0.44, 0.60, 0.76):
                    ray = np.array([(u * 480 - K[0, 2]) / K[0, 0], (v * 640 - K[1, 2]) / K[1, 1], 1.0]) * depth
                    pose = base.copy()
                    pose[:3, 3] = center - rot @ (palm_camera[:3, 3] + palm_camera[:3, :3] @ ray)
                    offer(pose)
    attempts = 0
    for _, pose, visible, extra, turn in sorted(options, key=lambda row: row[0]):
        if not path_clear(surface, home, pose, closing) or not path_clear(surface, pose, dest, fixed):
            continue
        yield (
            pose,
            dict(
                predicted_visible_fraction=visible,
                path_excess_m=float(extra),
                remaining_rotation_deg=float(np.rad2deg(turn)),
            ),
        )
        attempts += 1
        if attempts >= 6:
            break


class DirectGrasp(HeadWristTrial):
    def side_options(self, role, base, opened, bounds):
        with np.load(self.observation_dir / (role + "_mesh.npz")) as data:
            points = data["vertices_world_m"]
        cyl = cylinder(points, self.table_z, self.head_cylinder if role == "wrist" else None)
        if role == "head":
            self.head_cylinder = cyl
        world = self.surface.world
        height = cyl["top_m"] - self.table_z
        wall = world[(world[:, 2] > self.table_z + 0.23 * height) & (world[:, 2] < self.table_z + 0.60 * height)]
        if len(wall) < 8:
            raise Rejected("insufficient mesh support at the side grasp height")
        core = np.quantile(wall[:, :2], [0.008, 0.992], axis=0)
        center = np.array(cyl["center"])
        center[:2] = core.mean(0) + np.clip(center[:2] - core.mean(0), -0.003, 0.003) * 0.65
        radius = (core[1] - core[0]).mean() / 2
        limit = np.clip(base["q_squeeze"], *bounds.T)
        poses = pad_targets(self.scene, center, radius, cyl["top_m"], self.table_z, base, opened, limit)
        result, cached = [], {}
        for pre, grasp, qo, qc, details in poses:
            key = qo.tobytes()
            if key not in cached:
                cached[key] = hand_shapes(self.scene, qo)
            away = pre[:2, 3] - center[:2]
            away = away / np.linalg.norm(away)
            # Allow room for the full phalanges and the irregular neck or nozzle.
            # Closing then spans this measured small lateral standoff.
            for standoff in (0.0, 0.004, 0.008, 0.014, 0.021, 0.030, 0.042):
                candidate = pre.copy()
                candidate[:2, 3] += standoff * away
                if self.surface.clear(cached[key], candidate, 0.0015):
                    info = dict(details, posture="cylindrical_side", side_standoff_m=standoff)
                    result.append(Grasp(candidate, grasp, qo, qc, limit, info))
                    break
        return result

    def options(self, role, coarse=None):
        s = self.scene
        mode = self.family
        base = self.cfg["posture"]
        bounds = s.limits["right"]
        opened = np.clip(np.asarray(base["q_pre"]) * 0.45, *bounds.T)
        if mode == "side_wall":
            candidates = self.side_options(role, base, opened, bounds)
        else:
            names = {
                "over_edge": ("1_Large_Diameter", "22_Parallel_Extension", "24_Tip_Pinch"),
                "across_handle": ("5_Light_Tool", "14_Tripod", "1_Large_Diameter", "24_Tip_Pinch"),
                "narrow_tip": ("14_Tripod", "24_Tip_Pinch", "2_Small_Diameter"),
                "round_wrap": ("14_Tripod", "11_Power_Sphere", "26_Sphere_4_Finger"),
            }[mode]
            library = self.cfg.get("pinch_postures", {})
            postures = {
                n: library[n] for n in names if n in library and n not in getattr(self, "used_postures", ())
            }
            order = self.cfg.get("gesture_order", ())
            index = getattr(self, "attempt_number", 1) - 1
            if index < len(order) and order[index] in postures:
                postures = {order[index]: postures[order[index]]}
            if not postures:
                raise Rejected(f"no configured Sharpa finger posture for {mode}")
            rows = band_grasps(
                s,
                self.surface,
                opened,
                bounds,
                postures,
                upright_tip=mode == "narrow_tip",
                bands=self.contact_bands(),
            )
            if self.cfg.get("check_pad_surfaces"):
                rows = confirm_pad_surfaces(s, self.surface, rows)
            candidates = [Grasp(p, g, qo, qc, lim, details) for _, p, g, qo, qc, lim, details in rows]
        if coarse is not None:

            def change(c):
                return (
                    np.linalg.norm(c.close[:3, 3] - coarse.close[:3, 3])
                    + 0.07 * rotation_error(c.close, coarse.close)
                    + 0.02 * np.max(np.abs(c.opened - coarse.opened))
                    + 0.02 * (c.details["posture"] != coarse.details["posture"])
                )

            return sorted(candidates, key=change)
        representatives = []
        for c in candidates:
            if any(
                rotation_error(c.pre, other.pre) < 0.12
                and np.linalg.norm(c.pre[:3, 3] - other.pre[:3, 3]) < 0.025
                and c.details["posture"] == other.details["posture"]
                for other in representatives
            ):
                continue
            representatives.append(c)
            if len(representatives) >= 18:
                break
        return representatives

    def contact_bands(self):
        return None

    def select(self, home, hand):
        s = self.scene
        camera = np.linalg.solve(home, self.wrist_optical())
        K = s.wrist_K.copy()
        K[:2, 2] -= 0.5
        seeds = self.options("head")
        for j, grasp in enumerate(seeds):
            for view, metadata in view_options(s, self.surface, grasp, home, camera, K, hand):
                try:
                    joints, _ = self.joint_path(home, view)
                    self.following_path(view, grasp.pre, joints[-1])
                except Rejected:
                    continue
                return grasp, view, dict(metadata, grasp_candidate=j + 1)
        raise Rejected(f"no direct visible and reachable approach for {self.family} ({len(seeds)} candidate grips)")

    def following_path(self, start, target, arm):
        # Supply the end of the first leg as the second leg's IK seed.
        actual = self.actual()
        actual[self.arm_ids] = arm
        return self.joint_path(start, target, initial=actual)

    def execute(self):
        s = self.scene
        if not self.cfg.get("use_wrist"):
            raise Rejected("direct pregrasp requires a fresh wrist observation")
        if not s.settle()["verification"]["passed"]:
            raise Rejected("initial contact-free settling failed")
        self.event("settled")
        self.attempt_grasp()

    def attempt_grasp(self):
        s = self.scene
        self.head = self.observe("head")
        self.family = family(self.surface.world)
        if self.cfg.get("side_lying_taper"):
            self.family = "across_handle"
        self.event("select_object_grasp", family=self.family)
        home = self.palm()
        homeq = self.actual()[self.hand_ids]
        self.digit_indices = {
            d: [i for i, n in enumerate(s.hand_names["right"]) if n.startswith("right_" + d + "_")] for d in FINGERS
        }
        coarse, view, details = self.select(home, homeq)
        self.selected_grasp, self.selected_view = coarse, view
        self.event(
            "object_specific_pregrasp",
            family=self.family,
            posture=coarse.details["posture"],
            palm=view.tolist(),
            q_pre=coarse.opened.tolist(),
            head_grasp=coarse.close.tolist(),
            **details,
        )
        self.move("move_to_pregrasp", view, coarse.opened, 1.40)
        if np.max(np.abs(self.actual()[self.hand_ids] - coarse.opened)) > 0.15:
            raise Rejected("pregrasp finger tracking")
        self.event(
            "pregrasp_arrived",
            measured_palm=self.palm().tolist(),
            measured_fingers=self.actual()[self.hand_ids].tolist(),
        )
        self.command(s.target[0, self.arm_ids].cpu().numpy(), coarse.opened, view)
        self.observe("wrist")
        current = self.palm()
        for grip in self.options("wrist", coarse)[:60]:
            correction = np.linalg.norm(grip.close[:3, 3] - coarse.close[:3, 3])
            angle = rotation_error(grip.close, coarse.close)
            if correction > 0.038 or angle > np.deg2rad(22):
                continue
            shapes = [hand_shapes(s, (1 - f) * coarse.opened + f * grip.opened) for f in np.linspace(0, 1, 9)]
            if not path_clear(self.surface, current, grip.pre, shapes):
                continue
            try:
                self.joint_path(current, grip.pre)
            except Rejected:
                continue
            break
        else:
            raise Rejected("fresh wrist mesh has no collision-free local correction")
        self.event(
            "refined_grasp",
            family=self.family,
            **grip.details,
            correction_m=float(correction),
            correction_deg=float(np.rad2deg(angle)),
            close_world_palm=grip.close.tolist(),
        )
        self.selected_grasp = grip
        self.move("approach_contact_band", grip.pre, grip.opened, 0.78)
        self.close_and_lift(grip.close, grip.closed, grip.limit, *s.limits["right"].T)
