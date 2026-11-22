"""Request a bounded native solve from the live head registration, once per target and hand."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tabletop_control import NoPath
from .tabletop_lightning import LightningBank


def placements_from_prior(data, surface, reg):
    poses = data["object_pose"].astype(float)
    indices = np.arange(len(poses))
    groups = [indices]
    # Pull the wrist toward the head-observed side by the measured registration
    # residual. The native solver projects the contact sites separately below.
    live = surface.camera_samples.get("head_depth", np.empty((0, 3)))
    delta = np.zeros(3)
    if len(live):
        cloud = (live - reg[:3, 3]) @ reg[:3, :3]
        d, j = surface.tree.query(cloud)
        seen = d < 0.006
        if seen.any():
            delta = np.clip(
                np.median(cloud[seen] - surface.tree.data[j[seen]], axis=0),
                -0.0015,
                0.0015,
            )
    shifted = poses.copy()
    shifted[:, :3, 3] -= shifted[:, :3, :3] @ delta
    batches = [shifted]
    # Slide an off-centre grasp along a long horizontal object before solving its contacts again.
    # Keep both the original and translated placements only where every active site follows the mesh.
    from .tabletop_handle import volume_center
    from .tabletop_opposition import long_horizontal_axis

    world = surface.local @ reg[:3, :3].T + reg[:3, 3]
    world_axis = long_horizontal_axis(world)
    if world_axis is not None:
        axis = world_axis @ reg[:3, :3]
        com = (volume_center(surface.world_parts) - reg[:3, 3]) @ reg[:3, :3]
        obj_sites = (data["target_pos"] - shifted[:, None, :3, 3]) @ shifted[:, :3, :3]
        present = data["contact_link_id"] >= 0
        mean = (obj_sites * present[:, :, None]).sum(1) / np.maximum(
            1, present.sum(1, keepdims=True)
        )
        offset = np.clip((com - mean) @ axis, -0.045, 0.045)
        for fraction in (1.0, 0.55):
            move = fraction * offset[:, None] * axis
            gaps = surface.tree.query(obj_sites + move[:, None])[0]
            keep = indices[
                (np.abs(fraction * offset) >= 0.006)
                & (np.max(np.where(present, gaps, 0), axis=1) < 0.006)
            ]
            if len(keep):
                trial = shifted[keep].copy()
                trial[:, :3, 3] -= (trial[:, :3, :3] @ move[keep, :, None])[:, :, 0]
                batches.append(trial)
                groups.append(keep)
    if len(poses) < 24 or reg[2, 2] < -0.5:
        center = np.quantile(surface.local, [0.02, 0.98], axis=0).mean(axis=0)
        _, axes = np.linalg.eigh(np.cov(surface.local.T))
        # Check roll about the long axis, then alternative wrist bearings in the
        # tabletop plane. Projected contacts must still be near this object's skin.
        rotations = [
            (axes[:, -1], (-90, 90, 180, -45, 45)),
            (reg[2, :3], (90, -90, 180)),
        ]
        if reg[2, 2] < -0.5:
            rotations = [
                (np.array([1.0, 0.0, 0.0]), (180,)),
                (np.array([0.0, 1.0, 0.0]), (180,)),
            ] + rotations
        for axis, angles in rotations:
            for angle in angles:
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_rotvec(
                    np.deg2rad(angle) * axis
                ).as_matrix()
                transform[:3, 3] = center - transform[:3, :3] @ center
                trial = shifted @ np.linalg.inv(transform)
                obj_points = (data["target_pos"] - trial[:, None, :3, 3]) @ trial[
                    :, :3, :3
                ]
                gaps = surface.tree.query(obj_points)[0]
                ok = (
                    np.max(np.where(data["contact_link_id"] >= 0, gaps, 0), axis=1)
                    < 0.008
                )
                if ok.any():
                    keep = indices[ok]
                    if len(keep) > 12:
                        palms = reg @ np.linalg.inv(trial[keep])
                        merit = (
                            palms[:, 2, 1]
                            + 0.3 * palms[:, 2, 2]
                            - 10 * gaps[keep].min(1)
                        )
                        keep = keep[np.argsort(-merit)[:12]]
                    batches.append(trial[keep])
                    groups.append(keep)
    parents = np.concatenate(groups)
    cache = {
        k: data[k][parents]
        for k in (
            "q",
            "contact_link_id",
            "target_pos",
            "target_normal",
            "source",
            "source_id",
        )
    }
    cache.update(object_pose=np.concatenate(batches), link_names=data["link_names"])
    return cache, parents


def section_warm_starts(ctrl, surface, reg, others, link_names, count):
    """Use measured sections and arm FK to initialize remaining native contact queries."""
    from .tabletop_adaptive import DIGITS, atlas_grasps, nearest_arm, opposing_grasps
    from .tabletop_meshfit import ContactSurface, unit

    family, cards = ctrl.shared.query(surface, reg)
    skin = ContactSurface(surface)
    sets = (
        opposing_grasps(ctrl, family, surface, cards, others),
        atlas_grasps(ctrl, family, surface, cards),
    )
    used = []
    for proposals in sets:
        for _, card, poses in sorted(proposals, key=lambda entry: entry[0])[:36]:
            palm, pre = poses["grasp"], poses["pre"]
            if any(
                np.linalg.norm(palm[:3, 3] - p[:3, 3]) < 0.008
                and np.trace(p[:3, :3].T @ palm[:3, :3]) > 2.92
                for p in used
            ):
                continue
            q = np.asarray(card["q_touch"])
            qo = np.asarray(card["q_open"])
            if not ctrl.clear(palm, q, others) or not ctrl.clear(pre, qo, others):
                continue
            if not surface.clear(ctrl.link_shapes(qo), pre, 0.0005):
                continue
            links = ["right_" + d + "_elastomer" for d in DIGITS]
            channels = [ctrl.hand_contacts.link(k, palm, q) for k in links]
            clouds = [
                ctrl.io.model.hand_support[k] @ tf[:3, :3].T + tf[:3, 3]
                for k, (tf, _) in zip(links, channels)
            ]
            side = unit(
                np.mean([c.mean(0) for c in clouds[1:3]], axis=0) - clouds[0].mean(0)
            )
            if clouds[0][:, 2].mean() + 0.014 < np.mean(
                [c[:, 2].mean() for c in clouds[1:]]
            ):
                continue
            contacts = []
            for digit, (link, cloud, (tf, axes)) in enumerate(
                zip(links, clouds, channels)
            ):
                facing = (-side if digit == 0 else side) + 0.50 * unit(
                    cloud.mean(0) - skin.center
                )
                goal, normal, _ = skin.assign(cloud.mean(0), unit(facing))
                dist = np.linalg.norm(cloud - goal, axis=1)
                anchor = cloud[np.argmin(dist)]
                contacts.append(
                    (
                        float(dist.min()),
                        link,
                        axes,
                        tf,
                        (anchor - tf[:3, 3]) @ tf[:3, :3],
                        goal,
                        normal,
                    )
                )
            best = sorted(contacts[1:], key=lambda entry: entry[0])
            if contacts[0][0] > 0.023 or best[0][0] > 0.023:
                continue
            sites = [
                contacts[0],
                *[s for s in best if s[0] <= min(0.021, best[0][0] + 0.010)],
            ]
            if len(sites) < 2 or (family == "handled" and len(sites) < 3):
                continue
            if np.mean([s[6] @ sites[0][6] for s in sites[1:]]) > -0.20:
                continue

            def endpoint(arm, goal=palm):
                ctrl.check_closure_path("lightning_grasp", arm, goal, goal)

            solved = None
            for arm in nearest_arm(ctrl, pre)[1]:
                try:
                    branch = ctrl.ik.solve(pre, arm)
                    endpoint(branch)
                    ctrl.hints[pre.round(5).tobytes()] = branch
                    route = ctrl.move(
                        pre,
                        qo,
                        0.58,
                        others,
                        True,
                        surface,
                        object_margin=0.0005,
                        endpoint_check=endpoint,
                        plan_only=True,
                    )
                    solved = route[-1]
                    break
                except NoPath:
                    continue
            if solved is None:
                continue
            used.append(palm)
            warm = {
                "q": q,
                "q_mask": np.zeros(len(q), bool),
                "contact_link_id": np.full(5, -1, dtype=np.int32),
            }
            for field in (
                "target_pos",
                "target_normal",
                "contact_pos",
                "contact_normal",
            ):
                warm[field] = np.zeros((5, 3), dtype=np.float32)
            for k, (_, link, axes, tf, anchor, goal, normal) in enumerate(sites):
                warm["q_mask"][[joint for joint, _, _ in axes]] = True
                warm["contact_link_id"][k] = list(link_names).index(link)
                warm["contact_pos"][k] = anchor
                warm["contact_normal"][k] = -normal @ tf[:3, :3]
                warm["target_pos"][k] = (goal - palm[:3, 3]) @ palm[:3, :3]
                warm["target_normal"][k] = -normal @ palm[:3, :3]
            yield (
                np.linalg.solve(palm, reg),
                np.linalg.solve(palm, pre),
                qo,
                solved,
                warm,
            )
            if len(used) >= count:
                return


class LiveSceneBank(LightningBank):
    def __init__(self, path, side, directory, limit):
        super().__init__(path, side)
        self.directory = Path(directory) / side
        self.limit = limit
        self.requested = set()
        self.response = {}

    def select(self, ctrl, name, surface, reg, others, previous):
        folder = self.directory / name
        if name not in self.requested:
            cfg = ctrl.library[name]["prior"]
            data, provenance = self.read(
                name, cfg["sha256"], cfg["lightning_input_sha256"][self.side]
            )
            folder.mkdir(parents=True, exist_ok=False)
            chart, parents = placements_from_prior(data, surface, reg)
            sampler = LightningBank(self.path, self.side)
            sampler.cache[name] = chart, provenance
            selected = sampler.select(
                ctrl, name, surface, reg, others, (), placement_limit=self.limit
            )
            fields = (
                "q",
                "q_mask",
                "contact_link_id",
                "contact_pos",
                "contact_normal",
                "target_pos",
                "target_normal",
            )
            proposals = []
            with np.load(self.path / f"{name}.{self.side}.npz") as original:
                for _, row, pre, opened, arm in selected:
                    parent = int(parents[row])
                    o = chart["object_pose"][row]
                    palm = reg @ np.linalg.inv(o)
                    warm = {key: original[key][parent] for key in fields}
                    proposals.append(
                        (o, np.linalg.solve(palm, pre), opened, arm, warm, parent)
                    )
            if len(proposals) < self.limit and ctrl.shared is not None:
                for section in section_warm_starts(
                    ctrl,
                    surface,
                    reg,
                    others,
                    data["link_names"],
                    self.limit - len(proposals),
                ):
                    proposals.append((*section, -1))
            self.requested.add(name)
            self.response[name] = LightningBank(folder, self.side)
            receipt = {
                "target": name,
                "side": self.side,
                "limit": self.limit,
                "prior_rows": len(data["q"]),
                "placement_hypotheses": len(parents),
                "issued": len(proposals),
                "section_starts": len(proposals) - len(selected),
                "parent_bank_sha256": provenance["bank_sha256"],
                "prior_sha256": cfg["sha256"],
                "input_sha256": provenance["input_sha256"],
                "hand_sha256": provenance["hand_sha256"],
                "upstream": provenance["upstream"],
            }
            (folder / "placements.json").write_text(json.dumps(receipt, indent=2))
            ctrl.event("scene_native_request", **receipt)
            if proposals:
                obj, pre_rel, opened, arms, warm, source = zip(*proposals)
                head = surface.camera_samples.get("head_depth", np.empty((0, 3)))
                with (folder / "request.part").open("wb") as f:
                    np.savez_compressed(
                        f,
                        receipt=json.dumps(receipt),
                        parent=source,
                        object_pose=np.asarray(obj, np.float32),
                        arm_pre=arms,
                        pre_in_palm=pre_rel,
                        q_open=opened,
                        **{"warm_" + k: np.stack([w[k] for w in warm]) for k in fields},
                        mesh_to_world=reg,
                        head_local=(head - reg[:3, 3]) @ reg[:3, :3],
                        commanded_arm=ctrl.arm,
                        table_m=ctrl.table,
                    )
                (folder / "request.part").replace(folder / "request.npz")
                deadline = time.monotonic() + 600
                while not (folder / "result.json").is_file():
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "timed out waiting for the scene's native Lightning solver"
                        )
                    time.sleep(0.25)
                report = json.loads((folder / "result.json").read_bytes())
                sha = hashlib.sha256((folder / "request.npz").read_bytes()).hexdigest()
                if (
                    "error" in report
                    or report["request_sha256"] != sha
                    or report["issued"] != len(proposals)
                    or not 0 <= report["candidates"] <= len(proposals)
                ):
                    raise RuntimeError(f"invalid scene solver result: {report}")
                if report["candidates"]:
                    bank = folder / f"{name}.{self.side}.npz"
                    if (
                        hashlib.sha256(bank.read_bytes()).hexdigest()
                        != report["bank_sha256"]
                    ):
                        raise ValueError("scene result bank checksum mismatch")
                    with np.load(bank) as response:
                        if len(response["q"]) != report["candidates"]:
                            raise ValueError(
                                "scene solver returned an inconsistent row count"
                            )
                ctrl.event("scene_native_result", **report)
        if not (folder / f"{name}.{self.side}.npz").is_file():
            raise NoPath(
                "no native contact solution for this scene's reachable placements"
            )
        return self.response[name].select(ctrl, name, surface, reg, others, previous)
