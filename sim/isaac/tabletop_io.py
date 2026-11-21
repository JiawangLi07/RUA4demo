"""Separate the native sensor/command interface from the independent physics recorder."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .head_wrist_trial import ARM, PALM
from .scene_fixture import pose_matrix


class PhysicsRecording:
    def __init__(self, scene, out):
        self.scene, self.out = scene, out
        self.names = list(scene.objects)
        self.pending = []
        self.chunks = []
        self.table = float(scene.table_bounds[1, 2])
        self.right = [
            i for i, name in enumerate(scene.contact_names) if name.startswith("right_")
        ]
        self.digit_links = [
            [
                j
                for j, i in enumerate(self.right)
                if scene.contact_names[i].startswith("right_" + d + "_")
                and scene.contact_names[i].endswith(("_PP", "_MP", "_DP", "_elastomer"))
            ]
            for d in ("thumb", "index", "middle", "ring", "pinky")
        ]

    def record(self):
        import torch

        s = self.scene
        i = s.palm_ids["right"]
        forces = s.contact_view.get_contact_force_matrix(s.config["physics_dt"])[
            self.right
        ]
        # Keep one independent sample per physics tick. A batched host copy avoids
        # synchronizing the simulator once per body and finger on every tick.
        magnitudes = forces.norm(dim=-1)
        sample = torch.cat(
            [
                *(o.data.root_state_w[0, :7] for o in s.objects.values()),
                s.robot.data.body_pos_w[0, i],
                s.robot.data.body_quat_w[0, i],
                magnitudes.sum(dim=0),
                *(magnitudes[ids].sum(dim=0) for ids in self.digit_links),
                s.robot.data.joint_pos[0],
            ]
        )
        self.pending.append(sample)
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
        np.savez_compressed(
            self.out / "physics_audit.npz",
            object_names=self.names,
            pose_wxyz=rows[:, :end].reshape(-1, n, 7),
            palm_wxyz=rows[:, end : end + 7],
            hand_load_N=rows[:, end + 7 : end + 7 + n],
            digit_load_N=rows[:, end + 7 + n : end + 7 + 6 * n].reshape(-1, 5, n),
            digit_names=["thumb", "index", "middle", "ring", "pinky"],
            q_measured=rows[:, end + 7 + 6 * n :],
            dt=self.scene.config["physics_dt"],
            table_z=self.table,
        )


def connect(scene, out):
    audit = PhysicsRecording(scene, out)
    tree = scene.tree
    names = list(scene.robot.joint_names)
    dt = scene.config["physics_dt"]
    arm = [names.index(n) for n in ARM]
    hand = list(scene.ids["right"])
    head = scene.camera
    # Fixed head extrinsics and hand-eye calibration are available at the station.
    head_pose = pose_matrix(
        head.data.pos_w[0].cpu().numpy(), head.data.quat_w_ros[0].cpu().numpy()
    )
    eye = scene.tcp_from_wrist_optical.copy()
    head_k = head.data.intrinsic_matrices[0].cpu().numpy().astype(float)
    wrist_k = scene.wrist_K.copy()
    for k in (head_k, wrist_k):
        k[:2, 2] -= 0.5

    def q():
        return scene.robot.data.joint_pos[0].cpu().numpy().astype(float, copy=True)

    def fk(link, angles=None):
        return tree.pose(link, dict(zip(names, q() if angles is None else angles)))

    def camera(role):
        angles = q()
        if role == "head":
            sensor, pose, k = head, head_pose, head_k
        elif role == "wrist":
            sensor, pose, k = (
                scene.wrist_camera,
                fk("TCP_Link_R", angles) @ eye,
                wrist_k,
            )
            quat = np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1).astype(
                "float32"
            )
            sensor.set_world_poses(
                pose[None, :3, 3].astype("float32"), quat[None], convention="ros"
            )
            scene.world.render()
        else:
            raise ValueError(role)
        scene.world.render()
        sensor.update(dt, force_recompute=True)
        data = sensor.data.output
        return {
            "rgb": data["rgb"][0, ..., :3].cpu().numpy().copy(),
            "depth": data["distance_to_image_plane"][0, ..., 0].cpu().numpy().copy(),
            "K": k.copy(),
            "pose": pose.copy(),
            "q": angles,
            "time": scene.time,
        }

    def step(arm_q, hand_q):
        scene.target[0, arm] = scene.target.new_tensor(arm_q)
        scene.target[0, hand] = scene.target.new_tensor(hand_q)
        scene.advance_dynamics()
        audit.record()
        if scene.step_number % round(1 / dt / scene.config["images"]["fps"]) == 0:
            scene.capture()

    # These are robot CAD vertices. No object collider, mask, transform, or contact is exposed.
    bounds = {n: np.array([v.min(0), v.max(0)]) for n, v in scene.hand_support.items()}
    bounds.update(scene.arm_mask_bounds)
    model = SimpleNamespace(
        tree=tree,
        robot=SimpleNamespace(joint_names=names),
        hand_names=scene.hand_names,
        hand_bounds=scene.hand_bounds,
        hand_support=scene.hand_support,
        limits=scene.limits,
    )
    return SimpleNamespace(
        q=q,
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
        palm_from_wrist=np.linalg.solve(fk(PALM), fk("TCP_Link_R")) @ eye,
        wrist_K=wrist_k,
    ), audit
