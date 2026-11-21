"""Metric head/wrist observations, arm interpolation, contact feedback and lift verification."""

from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .cloud_geometry import instance_mask
from .scene_fixture import pose_matrix

ARM = [f"Joint{i}_R" for i in range(1, 8)]
PALM = "right_hand_C_MC"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class Rejected(Exception):
    """A measured stage failed its criterion."""


def save(path, data):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def depth_surface(depth, mask, K, camera):
    """Triangulate neighboring valid pixels of the observed object in world metres."""
    valid = mask & np.isfinite(depth) & (depth > 0.025) & (depth < 2.0)
    y, x = np.nonzero(valid)
    z = depth[y, x]
    if len(z) < 80:
        raise Rejected(f"too few valid depth pixels: {len(z)}")
    index = np.full(depth.shape, -1, dtype=int)
    index[y, x] = np.arange(len(z))
    vertices = np.column_stack(((x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z))
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    faces = np.concatenate([np.stack(triple, axis=-1).reshape(-1, 3) for triple in ((a, c, b), (b, c, d))])
    faces = faces[(faces >= 0).all(axis=1)]
    faces = faces[np.ptp(z[faces], axis=1) < 0.012]
    if len(faces) < 70:
        raise Rejected("insufficient connected measured surface")
    return vertices @ camera[:3, :3].T + camera[:3, 3], faces


def cylinder(surface, table_z, previous=None):
    """Fit the horizontal cross-section on the visible mid-height side wall."""
    top = float(np.quantile(surface[:, 2], 0.995)) if previous is None else previous["top_m"]
    height = top - table_z
    wall = surface[(surface[:, 2] > table_z + 0.18 * height) & (surface[:, 2] < top - 0.18 * height), :2]
    if len(wall) < 40 or height < 0.03:
        raise Rejected(f"insufficient visible side wall: {len(wall)} pixels, height {height:.3f} m")
    wall = wall[:: max(1, len(wall) // 3000)]
    xy = (wall.max(0) + wall.min(0)) / 2 if previous is None else previous["center"][:2]
    radius = 0.5 * np.ptp(wall, axis=0).max() if previous is None else previous["radius_m"]

    def error(params):
        return np.linalg.norm(wall - params[:2], axis=1) - params[2]

    estimate = least_squares(
        error,
        [*xy, np.clip(radius, 0.006, 0.089)],
        bounds=([-np.inf, -np.inf, 0.005], [np.inf, np.inf, 0.09]),
        loss="soft_l1",
        f_scale=0.001,
    )
    residual = float(np.median(np.abs(error(estimate.x))))
    if residual > 0.005 or (previous is not None and np.linalg.norm(estimate.x[:2] - xy) > 0.025):
        raise Rejected(f"local surface registration rejected, median residual {residual:.5f} m")
    return {
        "center": [*estimate.x[:2], (top + table_z) / 2],
        "radius_m": estimate.x[2],
        "top_m": top,
        "median_residual_m": residual,
        "wall_samples": len(wall),
    }


class HeadWristTrial:
    def __init__(self, scene):
        self.scene = scene
        self.out = scene.out
        self.observation_dir = self.out
        self.approach_origin = None
        self.cfg = scene.config["head_wrist_validation"]
        self.dt = scene.config["physics_dt"]
        self.arm_ids = [scene.arm_ids[name] for name in ARM]
        self.hand_ids = scene.ids["right"]
        self.events, self.samples = [], []
        self.stage = "initialization"
        self.tracked = []
        self.object_name = scene.config["assignment"]["right"]
        self.held = set()
        self.streak = dict.fromkeys(FINGERS, 0)
        self.pad_names = {f: "right_" + f + "_elastomer" for f in FINGERS}
        self.outcome = {}
        self.surface = None
        if self.cfg.get("shape_model") == "local_band" or self.cfg.get("direct_pregrasp"):
            from .head_wrist_surface import RegisteredSurface
            from .scene_meshes import load_mesh_bundle

            prior = self.cfg.get("metric_shape_prior")
            if prior is None and self.cfg.get("direct_pregrasp"):
                prior = next(
                    (o.get("mesh") for o in scene.config.get("added_objects", []) if o["name"] == self.object_name),
                    None,
                )
            if prior is None:
                raise ValueError("mesh registration requires a metric shape prior for the assigned object")
            self.surface = RegisteredSurface(load_mesh_bundle(prior))

    def event(self, stage, **details):
        self.stage = stage
        self.events.append({"stage": stage, "sim_time": self.scene.time, **details})
        save(self.out / "events.json", self.events)
        print("HEAD_WRIST", self.events[-1], flush=True)

    def palm(self):
        s = self.scene
        i = s.palm_ids["right"]
        return pose_matrix(
            s.robot.data.body_pos_w[0, i].cpu().numpy(), s.robot.data.body_quat_w[0, i].cpu().numpy()
        )

    def actual(self):
        return self.scene.robot.data.joint_pos[0].cpu().numpy().copy()

    def wrist_optical(self):
        s = self.scene
        joints = dict(zip(s.robot.joint_names, self.actual()))
        palm_from_tcp = np.linalg.solve(s.tree.pose(PALM, joints), s.tree.pose("TCP_Link_R", joints))
        return self.palm() @ palm_from_tcp @ s.tcp_from_wrist_optical

    def read_exposure(self, role):
        s = self.scene
        sensor = s.camera if role == "head" else s.wrist_camera
        if role == "wrist":
            expected = self.wrist_optical()
            quat = np.roll(Rotation.from_matrix(expected[:3, :3]).as_quat(), 1)
            sensor.set_world_poses(
                expected[None, :3, 3].astype("float32"), quat[None].astype("float32"), convention="ros"
            )
            s.world.render()
        s.world.render()
        sensor.update(self.dt, force_recompute=True)
        data = sensor.data
        depth = data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        ids = data.output["instance_id_segmentation_fast"][0, ..., 0].cpu().numpy()
        labels = data.info[0]["instance_id_segmentation_fast"]["idToLabels"]
        K = s.wrist_K.copy() if role == "wrist" else data.intrinsic_matrices[0].cpu().numpy().astype(float)
        K[:2, 2] -= 0.5
        pose = pose_matrix(data.pos_w[0].cpu().numpy(), data.quat_w_ros[0].cpu().numpy())
        if role == "wrist" and not np.allclose(expected, pose, atol=1e-4):
            raise Rejected("wrist render pose does not agree with measured hand-eye pose")
        rgb = data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        return rgb, depth, ids, labels, K, pose

    def observe(self, role):
        from .cloud_geometry import depth_to_world_points, fit_table_plane

        s = self.scene
        rgb, depth, ids, labels, K, pose = self.read_exposure(role)
        Image.fromarray(rgb).save(self.observation_dir / (role + ".jpg"))
        try:
            mask = instance_mask(ids, labels, s.config["object_prims"][self.object_name])
        except ValueError as exc:
            raise Rejected(f"{role}: target absent from native segmentation") from exc
        np.savez_compressed(
            self.observation_dir / (role + "_frame.npz"),
            depth_z_m=depth,
            target_mask=mask,
            K=K,
            T_world_optical=pose,
            exposure_sim_seconds=s.time,
            joints_rad=self.actual(),
        )
        if role == "head":
            table_mask = instance_mask(ids, labels, s.config["table_prim"])
            table = depth_to_world_points(depth, table_mask, K, pose, max_depth_m=2.0)
            normal, _ = fit_table_plane(table)
            if normal[2] < 0.999:
                raise Rejected("this upright-can probe requires a horizontal table")
            self.table_z = -normal[3] / normal[2]
            self.head_camera = pose
        vertices, faces = depth_surface(depth, mask, K, pose)
        np.savez_compressed(self.observation_dir / (role + "_mesh.npz"), vertices_world_m=vertices, faces=faces)
        if self.surface is None:
            result = cylinder(vertices, self.table_z, None if role == "head" else self.head)
        else:
            self.event(role + "_exposure", vertices=len(vertices), triangles=len(faces))
            try:
                result = self.surface.register(vertices, self.table_z)
            except ValueError as exc:
                raise Rejected(f"{role}: {exc}") from exc
        self.event(
            role + "_mesh",
            camera_world_xyz=pose[:3, 3].tolist(),
            vertices=len(vertices),
            triangles=len(faces),
            **result,
        )
        return result

    def command(self, arm, hand, goal, *, allow_contact=False):
        s = self.scene
        s.target[0, self.arm_ids] = s.target.new_tensor(arm)
        s.target[0, self.hand_ids] = s.target.new_tensor(hand)
        s.wrist_targets["right"] = goal
        row = s._step()
        r = row["right"]
        pads = row["pad_contacts"]
        self.samples.append(
            {
                "time": s.time,
                "stage": self.stage,
                "pads": pads,
                "hand_contacts": r["contact_count"],
                "clearance_m": r["clearance_m"],
                "rise_m": r["rise_m"],
                "table_penetration_m": row["hand_table_penetration_m"],
                "palm_error_m": r["wrist_position_error_m"],
                "palm_error_rad": r["wrist_rotation_error_rad"],
            }
        )
        displacement = r["object_translation_m"]
        if "link_normal_N" in row:
            pose = r["object_pose_wxyz"]
            self.samples[-1].update(
                link_normal_N=row["link_normal_N"],
                object_pose_wxyz=pose,
                attempt=getattr(self, "attempt_number", 1),
            )
            if row.get("palm_contact_local_m") is not None:
                self.samples[-1]["palm_contact_local_m"] = row["palm_contact_local_m"]
            if self.cfg.get("multi_finger_handle"):
                self.samples[-1]["contact_surfaces"] = row["link_contact_surfaces"]
            if self.approach_origin is not None:
                displacement = np.linalg.norm(np.asarray(pose[:3]) - self.approach_origin)
        if s.step_number % 8 == 0:
            self.tracked.append(
                {
                    "time": s.time,
                    "palm": self.palm().tolist(),
                    "goal": goal.tolist(),
                    "actual": self.actual().tolist(),
                    "target": s.target[0].cpu().tolist(),
                }
            )
        if not row["finite"] or row["hand_table_penetration_m"] > 0.001:
            raise Rejected(f"{self.stage}: invalid state/table penetration {row['hand_table_penetration_m']:.4f} m")
        if not allow_contact and (r["contact_count"] or displacement > 0.005):
            raise Rejected(f"{self.stage}: premature contact or displacement {displacement:.4f} m")
        return row

    def joint_path(self, start, goal, *, initial=None):
        actual = self.actual() if initial is None else initial
        s = self.scene
        names = s.robot.joint_names
        limits = s.tree.joint_limits(names)
        seed = dict(zip(names, np.clip(actual, limits[:, 0], limits[:, 1])))
        rotations = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], goal[:3, :3]]))
        span = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).magnitude()
        count = max(2, int(np.ceil(max(np.linalg.norm(goal[:3, 3] - start[:3, 3]) / 0.012, span / 0.10))))
        joints = [actual[self.arm_ids]]
        for f in np.linspace(0, 1, count + 1)[1:]:
            target = np.eye(4)
            target[:3, :3] = rotations(f).as_matrix()
            target[:3, 3] = start[:3, 3] * (1 - f) + goal[:3, 3] * f
            solution = s.tree.solve_pose(PALM, target, seed, ARM)
            if not solution.converged:
                raise Rejected(
                    f"{self.stage}: IK at {f:.2f}, {solution.position_error_m:.4f} m, {solution.rotation_error_rad:.3f} rad"
                )
            seed = solution.joint_positions
            joints.append([seed[name] for name in ARM])
        arr = np.asarray(joints)
        if np.max(np.abs(np.diff(arr, axis=0))) > 0.48:
            raise Rejected(f"{self.stage}: discontinuous IK")
        return arr, rotations

    def move(self, stage, goal, hand_goal, seconds, *, closure=False, allow_contact=False):
        start = self.palm()
        self.event(stage, start=start.tolist(), goal=goal.tolist())
        joints, rotations = self.joint_path(start, goal)
        hand_start = self.scene.target[0, self.hand_ids].cpu().numpy().copy()
        hold = hand_start.copy()
        ticks = max(2, int(np.ceil(seconds / self.dt)))
        for i in range(1, ticks + 1):
            f = min(1.0, i / (ticks * 0.80))
            # Smooth position/orientation and preshaping begin together.
            f = f * f * (3 - 2 * f)
            x = f * (len(joints) - 1)
            j = min(int(x), len(joints) - 2)
            arm = joints[j] + (x - j) * (joints[j + 1] - joints[j])
            hand = hand_start + (hand_goal - hand_start) * f
            for digit in self.held:
                indices = self.digit_indices[digit]
                hand[indices] = hold[indices] if closure else hand[indices]
            pose = np.eye(4)
            pose[:3, 3] = start[:3, 3] * (1 - f) + goal[:3, 3] * f
            pose[:3, :3] = rotations(f).as_matrix()
            row = self.command(arm, hand, pose, allow_contact=allow_contact or closure)
            if closure:
                for digit, pad in self.pad_names.items():
                    hits = row["pad_contacts"].get(pad, 0)
                    self.streak[digit] = self.streak[digit] + 1 if hits else 0
                    if self.streak[digit] >= 3 and digit not in self.held:
                        indices = self.digit_indices[digit]
                        hold[indices] = hand[indices]
                        self.held.add(digit)
        measured = row["right"]
        if measured["wrist_position_error_m"] > 0.005 or measured["wrist_rotation_error_rad"] > 0.08:
            raise Rejected(
                f"{stage}: wrist tracking {measured['wrist_position_error_m']:.4f} m, {measured['wrist_rotation_error_rad']:.3f} rad"
            )
        return row

    def bilateral(self, pads):
        return bool(pads.get(self.pad_names["thumb"]) and any(pads.get(self.pad_names[d]) for d in FINGERS[1:]))

    def reacquire(self, grasp, qlimit):
        """Keep loaded fingers steady and increment the missing side of the planned pinch."""
        self.event("reacquire_contact")
        s = self.scene
        arm = s.target[0, self.arm_ids].cpu().numpy().copy()
        hand = s.target[0, self.hand_ids].cpu().numpy().copy()
        hand_limit = hand + 0.6 * (qlimit - hand)
        stable = 0
        pads = self.samples[-1]["pads"]
        for _ in range(round(0.9 / self.dt)):
            for digit in ("thumb", "index"):
                if not pads.get(self.pad_names[digit]):
                    ids = self.digit_indices[digit]
                    step = 0.6 * self.dt
                    hand[ids] += np.clip(hand_limit[ids] - hand[ids], -step, step)
            row = self.command(arm, hand, grasp, allow_contact=True)
            pads = row["pad_contacts"]
            stable = stable + 1 if self.bilateral(pads) else 0
            if stable >= round(0.10 / self.dt):
                return
        raise Rejected("contact reacquisition did not sustain opposing finger pads")

    def close_and_lift(self, grasp, qcontact, qclose, low, hi):
        s = self.scene
        self.move("feedback_closure", grasp, qcontact, 1.6, closure=True)
        window = self.samples[-round(0.10 / self.dt) :]
        if not all(self.bilateral(row["pads"]) for row in window):
            self.reacquire(grasp, qclose)
        loaded = [digit for digit, pad in self.pad_names.items() if self.samples[-1]["pads"].get(pad)]
        self.event("opposition_confirmed", digits=loaded, stopped_on_contact=sorted(self.held))
        current = s.target[0, self.hand_ids].cpu().numpy().copy()
        tight = np.clip(current + 0.25 * (qclose - current), low, hi)
        self.move("tighten", grasp, tight, 0.4, allow_contact=True)
        if not all(self.bilateral(row["pads"]) for row in self.samples[-round(0.10 / self.dt) :]):
            raise Rejected("opposition lost during tightening")

        self.lift_and_hold(grasp, tight, lambda row: self.bilateral(row["pad_contacts"]))

    def regulate_loaded_fingers(self, hand):
        if not self.cfg.get("regulate_power_lift"):
            return
        nominal = self.loaded_hand_reference
        links = self.samples[-1]["link_normal_N"]
        palm = links.get(PALM, 0.0)
        share = self.cfg.get("share_palm_load") and palm < 1.2 and links.get("right_thumb_MC", 0) > 5
        names = self.scene.hand_names["right"]
        for digit, ids in self.digit_indices.items():
            force = sum(v for k, v in links.items() if k.startswith("right_" + digit + "_"))
            proximal = sum(links.get(f"right_{digit}_{k}", 0.0) for k in ("PP", "MP"))
            for j in ids:
                name = names[j]
                if digit == "thumb":
                    if share and name.endswith("CMC_FE"):
                        # Open the base to make room for the broad palm while the thumb pad opposes.
                        hand[j] = max(nominal[j] - 0.40, hand[j] - 0.65 * self.dt)
                    elif share and name.endswith("MCP_FE") and links.get("right_thumb_elastomer", 0) < 2.5:
                        hand[j] += 0.42 * self.dt
                    elif force < 1.8 and name.endswith(("CMC_FE", "MCP_FE", "_IP")):
                        hand[j] += 0.42 * self.dt
                elif name.endswith("MCP_FE") and proximal < 0.65 and force < 5.5:
                    hand[j] += 0.54 * self.dt
                elif name.endswith(("PIP", "DIP")) and (force < 1.9 or (palm < 0.70 and force < 4.8)):
                    hand[j] += 0.55 * self.dt
        lower, upper = self.scene.limits["right"].T
        hand[:] = np.clip(hand, lower, np.minimum(upper, nominal + 0.40))

    def move_with_preload(self, stage, goal, hand, duration, *, allow_contact):
        scene = self.scene
        measured = self.palm()
        state = self.actual()
        state[self.arm_ids] = scene.target[0, self.arm_ids].cpu().numpy()
        start = scene.tree.pose(PALM, dict(zip(scene.robot.joint_names, state)))
        end = start.copy()
        end[:3, 3] += goal[:3, 3] - measured[:3, 3]
        rotate = stage == "lift" and self.cfg.get("cradle_lift_deg", 0)
        if rotate:
            end[:3, :3] = goal[:3, :3] @ measured[:3, :3].T @ start[:3, :3]
        joints, rotations = self.joint_path(start, end, initial=state)
        self.event(stage, goal=goal.tolist(), preload_offset_m=(start[:3, 3] - measured[:3, 3]).tolist())
        ticks = round(duration / self.dt)
        for i in range(1, ticks + 1):
            f = min(1.0, i / (0.8 * ticks))
            f = f * f * (3 - 2 * f)
            v = f * (len(joints) - 1)
            k = min(int(v), len(joints) - 2)
            q = joints[k] + (v - k) * (joints[k + 1] - joints[k])
            target = measured.copy()
            target[:3, 3] += f * (goal[:3, 3] - measured[:3, 3])
            if rotate:
                target[:3, :3] = rotations(f).as_matrix() @ start[:3, :3].T @ measured[:3, :3]
            self.regulate_loaded_fingers(hand)
            row = self.command(q, hand, target, allow_contact=allow_contact)
        return row

    def lift_and_hold(self, grasp, tight, supported, *, maintain_preload=False, verify_hold=None):
        s = self.scene

        def relative_pose():
            obj = s.objects[self.object_name].data.root_state_w[0].cpu().numpy()
            return np.linalg.inv(self.palm()) @ pose_matrix(obj[:3], obj[3:7])

        relative = relative_pose()
        move = self.move_with_preload if maintain_preload else self.move
        active = maintain_preload and self.cfg.get("regulate_power_lift")
        if active:
            self.loaded_hand_reference = tight.copy()
            arm = s.target[0, self.arm_ids].cpu().numpy().copy()
            self.event("preload_before_lift")
            for _ in range(round(0.25 / self.dt)):
                self.regulate_loaded_fingers(tight)
                self.command(arm, tight, grasp, allow_contact=True)
        short = grasp.copy()
        short[2, 3] += 0.02
        row = move("test_lift", short, tight, 0.5, allow_contact=True)
        if active and row["right"]["clearance_m"] >= 0.005 and row["right"]["contact_count"] and not supported(row):
            self.event("stabilize_test_lift")
            arm = s.target[0, self.arm_ids].cpu().numpy().copy()
            stable = 0
            for _ in range(round(0.60 / self.dt)):
                self.regulate_loaded_fingers(tight)
                row = self.command(arm, tight, short, allow_contact=True)
                stable = stable + 1 if supported(row) else 0
                if stable >= round(0.10 / self.dt) or row["right"]["clearance_m"] < 0.005:
                    break
        if maintain_preload:
            self.event(
                "test_lift_check",
                clearance_m=row["right"]["clearance_m"],
                rise_m=row["right"]["rise_m"],
                links_N=row["link_normal_N"],
                supported=bool(supported(row)),
            )
        if row["right"]["clearance_m"] < 0.005 or not supported(row):
            raise Rejected("object failed 2 cm test lift")
        lift = grasp.copy()
        lift[2, 3] += 0.10
        cradle_angle = self.cfg.get("cradle_lift_deg", 0) if maintain_preload else 0
        if cradle_angle:
            measured = self.palm()
            axis = np.cross(measured[:3, 0], [0, 0, 1])
            points = [point for point, _ in row["link_contact_surfaces"].values()]
            if points and np.linalg.norm(axis) > 1e-6:
                # Turn the loaded palm upward about the observed contact patch.
                rotation = Rotation.from_rotvec(np.deg2rad(cradle_angle) * axis / np.linalg.norm(axis)).as_matrix()
                offset = np.median(points, axis=0) - measured[:3, 3]
                lift[:3, :3] = rotation @ measured[:3, :3]
                lift[:3, 3] += offset - rotation @ offset
                self.event("cradle_loaded_object", angle_deg=cradle_angle)
        move("lift", lift, tight, 0.9, allow_contact=True)
        self.event("hold")
        arm = s.target[0, self.arm_ids].cpu().numpy().copy()
        worst_translation = worst_rotation = 0.0
        for _ in range(round(2 / self.dt)):
            if active:
                self.regulate_loaded_fingers(tight)
            row = self.command(arm, tight, lift, allow_contact=True)
            delta = np.linalg.inv(relative) @ relative_pose()
            worst_translation = max(worst_translation, float(np.linalg.norm(delta[:3, 3])))
            worst_rotation = max(worst_rotation, float(Rotation.from_matrix(delta[:3, :3]).magnitude()))
            if (
                not supported(row)
                or row["right"]["rise_m"] < 0.09
                or row["right"]["clearance_m"] < 0.005
                or worst_translation > 0.03
                or worst_rotation > 0.5
            ):
                raise Rejected("height, clearance, slip, or opposing contacts failed during hold")
        if verify_hold is not None:
            verify_hold(self.samples[-round(2 / self.dt) :])
        self.outcome.update(hold_seconds=2.0, maximum_slip_m=worst_translation, maximum_slip_rad=worst_rotation)
        self.event("success")

    def finish(self, passed, reason):
        s = self.scene
        s.capture()
        save(
            self.out / "result.json",
            {
                "passed": passed,
                "reason": reason,
                "last_stage": self.stage,
                "sim_seconds": s.time,
                "wall_seconds": time.monotonic() - s.started,
                "frames": s.frame_number,
                "use_wrist": self.cfg["use_wrist"],
                "events": self.events,
                **self.outcome,
            },
        )
        save(self.out / "physics.json", self.samples)
        save(self.out / "tracking.json", self.tracked)
        save(self.out / "frames.json", s.frame_times)
        print("HEAD_WRIST_RESULT", passed, reason, flush=True)


def run_trial(scene):
    settings = scene.config["head_wrist_validation"]
    if settings.get("max_grasp_attempts") or settings.get("multi_finger_handle"):
        from .retry_grasp import RetryGrasp

        trial = RetryGrasp(scene)
    else:
        from .direct_pregrasp import DirectGrasp

        trial = DirectGrasp(scene)
    try:
        trial.execute()
    except Rejected as exc:
        trial.finish(False, str(exc))
    else:
        trial.finish(True, "10 cm lift with continuous measured hold for 2 s")
