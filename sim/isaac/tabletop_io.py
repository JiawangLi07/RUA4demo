"""Separate the native sensor/command interface from the independent physics recorder."""

from types import SimpleNamespace

import numpy as np

from .head_wrist_trial import ARM
from .scene_fixture import pose_matrix


class PhysicsRecording:
    def __init__(self, scene, out):
        self.scene, self.out = scene, out
        self.names = list(scene.objects)
        self.pending = []
        self.chunks = []
        self.table = float(scene.table_bounds[1, 2])
        self.sides = ("right", "left")
        self.links = {
            side: [
                i for i, n in enumerate(scene.contact_names) if n.startswith(side + "_")
            ]
            for side in self.sides
        }
        self.digits = {
            side: [
                [
                    j
                    for j, i in enumerate(self.links[side])
                    if scene.contact_names[i].startswith(side + "_" + d + "_")
                    and scene.contact_names[i].endswith(
                        ("_PP", "_MP", "_DP", "_elastomer")
                    )
                ]
                for d in ("thumb", "index", "middle", "ring", "pinky")
            ]
            for side in self.sides
        }
        self.tcp_ids = [
            scene.robot.body_names.index("TCP_Link_" + s) for s in ("R", "L")
        ]

        pad_names = [
            side + "_" + d + "_elastomer"
            for side in self.sides
            for d in ("thumb", "index", "middle", "ring", "pinky")
        ]
        self.pad_ids = [scene.robot.body_names.index(n) for n in pad_names]
        self.pad_centers = scene.target.new_tensor(
            np.stack([scene.hand_support[n].mean(0) for n in pad_names])
        )

    def record(self):
        import torch

        s = self.scene
        forces = s.contact_view.get_contact_force_matrix(s.config["physics_dt"]).norm(
            dim=-1
        )
        pieces = [o.data.root_state_w[0, :7] for o in s.objects.values()]
        for side in self.sides:
            i = s.palm_ids[side]
            loads = forces[self.links[side]]
            pieces.extend(
                [
                    s.robot.data.body_pos_w[0, i],
                    s.robot.data.body_quat_w[0, i],
                    loads.sum(dim=0),
                    *(loads[ids].sum(dim=0) for ids in self.digits[side]),
                    loads.reshape(-1),
                ]
            )
        for i in self.tcp_ids:
            pieces.extend(
                [s.robot.data.body_pos_w[0, i], s.robot.data.body_quat_w[0, i]]
            )
        q = s.robot.data.body_quat_w[0, self.pad_ids]
        xyz = s.robot.data.body_pos_w[0, self.pad_ids]
        cross = torch.cross(q[:, 1:], self.pad_centers, dim=1)
        offset = self.pad_centers + 2 * (
            q[:, 0:1] * cross + torch.cross(q[:, 1:], cross, dim=1)
        )
        pieces.append(xyz[:, 2] + offset[:, 2])
        if s.table_contact_view is not None:
            table_load = (
                s.table_contact_view.get_contact_force_matrix(s.config["physics_dt"])
                .norm(dim=-1)
                .sum(dim=1)
            )
            for side in self.sides:
                pieces.append(table_load[self.links[side]])
        pieces.append(s.robot.data.joint_pos[0])
        self.pending.append(torch.cat(pieces))
        if len(self.pending) == 240:
            self.flush()

    def flush(self):
        import torch

        if self.pending:
            self.chunks.append(torch.stack(self.pending).cpu().numpy())
            self.pending.clear()

    def save(self):
        self.flush()
        rows = np.concatenate(self.chunks)
        n = len(self.names)
        end = 7 * n
        fields = {}
        for side in self.sides:
            start = end
            end = start + 7
            fields[side + "_palm_wxyz"] = rows[:, start:end]
            fields[side + "_hand_load_N"] = rows[:, end : end + n]
            fields[side + "_digit_load_N"] = rows[:, end + n : end + 6 * n].reshape(
                -1, 5, n
            )
            start = end + 6 * n
            end = start + len(self.links[side]) * n
            fields[side + "_link_names"] = [
                self.scene.contact_names[i] for i in self.links[side]
            ]
            fields[side + "_link_load_N"] = rows[:, start:end].reshape(
                -1, len(self.links[side]), n
            )
        for side in self.sides:
            fields[side + "_tcp_wxyz"] = rows[:, end : end + 7]
            end += 7
        for side in self.sides:
            fields[side + "_pad_center_z_m"] = rows[:, end : end + 5]
            end += 5
        if self.scene.table_contact_view is not None:
            for side in self.sides:
                size = len(self.links[side])
                fields[side + "_link_table_load_N"] = rows[:, end : end + size]
                end += size
        if self.scene.fixture_path is not None:
            from .scene_wrist_mount import mount_parts

            for side in self.scene.fixture_sides:
                fields[side + "_fixture_tcp_vertices"] = np.vstack(
                    list(mount_parts(self.scene.fixture_path, side).values())
                )
        for side, vertices in self.scene.bare_tcp_support.items():
            fields[side + "_flange_tcp_vertices"] = vertices
        np.savez_compressed(
            self.out / "physics_audit.npz",
            object_names=self.names,
            pose_wxyz=rows[:, : 7 * n].reshape(-1, n, 7),
            **fields,
            palm_wxyz=fields["right_palm_wxyz"],
            hand_load_N=fields["right_hand_load_N"],
            digit_load_N=fields["right_digit_load_N"],
            digit_names=["thumb", "index", "middle", "ring", "pinky"],
            q_measured=rows[:, end:],
            dt=self.scene.config["physics_dt"],
            table_z=self.table,
        )


def connect(scene, out, side="right", audit=None):
    audit = audit if audit is not None else PhysicsRecording(scene, out)
    tree = scene.tree
    names = list(scene.robot.joint_names)
    dt = scene.config["physics_dt"]
    suffix = "R" if side == "right" else "L"
    tcp = "TCP_Link_" + suffix
    palm = side + "_hand_C_MC"
    arm = [names.index(n.replace("_R", "_" + suffix)) for n in ARM]
    hand = list(scene.ids[side])
    left_primary = scene.config.get("camera_mode") == "left-observer"
    head, head_pose, head_k = scene.camera, None, None
    if not left_primary:
        head_pose = pose_matrix(
            head.data.pos_w[0].cpu().numpy(), head.data.quat_w_ros[0].cpu().numpy()
        )
        head_k = head.data.intrinsic_matrices[0].cpu().numpy().astype(float)
        head_k[:2, 2] -= 0.5
    wrist_sensor, eye, wrist_intrinsics = scene.wrist_devices.get(
        side, (None, None, None)
    )
    roles = ("head",) if wrist_sensor is None else ("head", "wrist")
    if left_primary:
        roles = ("left_observer",)
    wrist_k = None if wrist_intrinsics is None else wrist_intrinsics.copy()
    if wrist_k is not None:
        wrist_k[:2, 2] -= 0.5
    if not hasattr(scene, "sensor_calls"):
        scene.sensor_calls = {"head": 0, "wrist": 0}
        if scene.config.get("wrist_hardware") == "left-only":
            scene.sensor_calls["left_observer"] = 0

    if not hasattr(scene, "commanded_targets"):
        scene.commanded_targets = scene.target[0].cpu().numpy().astype(float, copy=True)
    command = scene.commanded_targets

    def commanded_joints():
        return command.copy()

    arm_joints = [i for i, n in enumerate(names) if n.startswith("Joint")]
    use_feedback = scene.config.get("arm_joint_feedback", False)

    def joint_feedback():
        # Joint-state telemetry available from the arm drives. Fingers retain their issued values.
        angles = command.copy()
        if use_feedback:
            angles[arm_joints] = (
                scene.robot.data.joint_pos[0, arm_joints].detach().cpu().numpy()
            )
        return angles

    def fk(link, angles=None):
        return tree.pose(
            link, dict(zip(names, joint_feedback() if angles is None else angles))
        )

    palm_eye = None if eye is None else np.linalg.solve(fk(palm), fk(tcp)) @ eye

    def camera(role):
        if role not in roles:
            raise RuntimeError(f"camera {role} is disabled in this run")
        scene.sensor_calls[role] += 1
        angles = joint_feedback()
        if role == "head":
            sensor, pose, k = head, head_pose, head_k
        elif role in ("wrist", "left_observer"):
            camera_side = "left" if role == "left_observer" else side
            sensor, hand_eye, intrinsics = scene.wrist_devices[camera_side]
            k = intrinsics.copy()
            k[:2, 2] -= 0.5
            pose = fk("TCP_Link_" + camera_side[0].upper(), angles) @ hand_eye
            scene.sync_wrist_camera(camera_side)
            scene.world.render()
        else:
            raise ValueError(role)
        scene.world.render()
        sensor.update(dt, force_recompute=True)
        data = sensor.data.output
        return {
            "rgb": data["rgb"][0, ..., :3].cpu().numpy().copy(),
            "depth": data["distance_to_image_plane"][0, ..., 0].cpu().numpy().copy(),
            "sensor": role,
            "K": k.copy(),
            "pose": pose.copy(),
            "q": angles,
            "q_command": commanded_joints(),
            "joint_source": "arm_feedback_finger_command"
            if use_feedback
            else "command",
            "time": scene.time,
        }

    def step(arm_q, hand_q):
        command[arm], command[hand] = arm_q, hand_q
        scene.target[0, arm] = scene.target.new_tensor(arm_q)
        scene.target[0, hand] = scene.target.new_tensor(hand_q)
        scene.advance_dynamics()
        audit.record()
        if scene.step_number % round(1 / dt / scene.config["images"]["fps"]) == 0:
            scene.capture()

    # These are robot CAD vertices. No object collider, mask, transform, or contact is exposed.
    bounds = {n: np.array([v.min(0), v.max(0)]) for n, v in scene.hand_support.items()}
    bounds.update(scene.arm_mask_bounds)
    from .scene_wrist_mount import palm_parts

    model = SimpleNamespace(
        fixture_palm=palm_parts(
            scene.fixture_path if side in scene.fixture_sides else None, np.linalg.solve(fk(palm), fk(tcp)), side
        ),
        tree=tree,
        robot=SimpleNamespace(joint_names=names),
        hand_names=scene.hand_names,
        hand_bounds=scene.hand_bounds,
        hand_support=scene.hand_support,
        forearm_support=scene.forearm_support,
        limits=scene.limits,
    )
    if side in scene.bare_tcp_support:
        t = np.linalg.solve(fk(palm), fk(tcp))
        model.fixture_palm["tool_flange"] = (
            scene.bare_tcp_support[side] @ t[:3, :3].T + t[:3, 3]
        )
    physical = SimpleNamespace(
        side=side,
        paired_scene_seed=scene.config["tabletop_seed"] if scene.config.get("wrist_hardware") == "left-only" else None,
        camera_roles=roles,
        commanded_joints=commanded_joints,
        joint_feedback=joint_feedback,
        step=step,
        fk=fk,
        camera=camera,
        time=lambda: scene.time,
        names=names,
        arm=arm,
        hand=hand,
        tree=tree,
        dt=dt,
        bounds=bounds,
        model=model,
        palm_from_wrist=palm_eye,
        wrist_K=wrist_k,
    )
    if side == "left":
        from .tabletop_sides import canonical_left

        return canonical_left(physical), audit
    return physical, audit
