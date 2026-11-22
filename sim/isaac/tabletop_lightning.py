"""Place native Lightning Grasp solutions using the RGB-D registration and station constraints."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from .paths import RIGHT_JOINTS
from .tabletop_control import NoPath, shifted


def central_palm_samples(vertices):
    """Sample the front of the metacarpal plate, away from the flange and knuckles."""
    hull = ConvexHull(vertices)
    front = vertices[hull.simplices[hull.equations[:, 0] > 0.65]]
    samples = []
    for a in range(9):
        for b in range(9 - a):
            samples.extend(
                (a * front[:, 0] + b * front[:, 1] + (8 - a - b) * front[:, 2]) / 8
            )
    points = np.unique(np.round(samples, 5), axis=0)
    lo, hi = vertices.min(0), vertices.max(0)
    rel = (points - lo) / np.maximum(hi - lo, 1e-6)
    patch = points[
        (rel[:, 1] > 0.20)
        & (rel[:, 1] < 0.80)
        & (rel[:, 2] > 0.30)
        & (rel[:, 2] < 0.82)
    ]
    if not len(patch):
        raise ValueError("no working surface on Sharpa palm CAD")
    return patch[:: max(1, len(patch) // 120)]


class LightningBank:
    def __init__(self, path, side):
        self.path, self.side = Path(path), side
        self.cache = {}

    def read(self, name, digest, mesh_hash=None):
        if name not in self.cache:
            file = self.path / f"{name}.{self.side}.npz"
            if not file.is_file():
                raise NoPath("no Lightning Grasp bank for registered object and hand")
            with np.load(file, allow_pickle=False) as z:
                repo = json.loads(str(z["receipt"]))
                if (
                    repo["source_frame"] != "object_to_palm"
                    or repo["prior_sha256"] != digest
                    or list(z["joint_names"]) != RIGHT_JOINTS
                ):
                    raise ValueError(
                        "Lightning Grasp bank frame, metric mesh or joint order mismatch"
                    )
                if not len(z["q"]):
                    raise NoPath(
                        "Lightning Grasp returned zero collision-filtered grasps"
                    )
                data = {
                    key: z[key]
                    for key in (
                        "q",
                        "object_pose",
                        "contact_link_id",
                        "target_pos",
                        "target_normal",
                        "link_names",
                    )
                }
                data["source"] = (
                    z["candidate_source"]
                    if "candidate_source" in z
                    else np.full(len(z["q"]), "official")
                )
                data["source_id"] = (
                    z["source_id"] if "source_id" in z else np.arange(len(z["q"]))
                )
                if not all(
                    np.isfinite(data[k]).all()
                    for k in ("q", "object_pose", "target_pos", "target_normal")
                ):
                    raise ValueError("nonfinite Lightning Grasp output")
                repo["bank_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
                self.cache[name] = data, repo
        data, receipt = self.cache[name]
        if receipt["prior_sha256"] != digest or (
            mesh_hash and receipt["input_sha256"] != mesh_hash
        ):
            raise ValueError(
                "Lightning Grasp input SHA256 does not match the registered mesh"
            )
        return data, receipt

    def head_views(self, ctrl, name, reg, center):
        data, _ = self.read(
            name,
            ctrl.library[name]["prior"]["sha256"],
            ctrl.library[name]["prior"]
            .get("lightning_input_sha256", {})
            .get(self.side),
        )
        palms = reg @ np.linalg.inv(data["object_pose"])
        # Favor a visible wrist view close to an official pregrasp before moving
        # toward the object. The new wrist registration selects the final grasp.
        candidates = []
        for i, palm in enumerate(palms):
            for d in (0.11, 0.17):
                pre = shifted(palm, -d * palm[:3, 0] + [0, 0, 0.035])
                eye = pre @ ctrl.io.palm_from_wrist
                local = eye[:3, :3].T @ (center - eye[:3, 3])
                if local[2] < 0.065 or local[2] > 0.38 or not ctrl.fixture_clear(pre):
                    continue
                off = np.linalg.norm(local[:2]) / local[2]
                if off < 0.40:
                    candidates.append((off + 0.6 * abs(local[2] - 0.19), i, pre))
        return [
            (p, ctrl.open_hand)
            for _, _, p in sorted(candidates, key=lambda x: x[0])[:5]
        ]

    def select(self, ctrl, name, surface, reg, others, previous):
        data, receipt = self.read(
            name,
            ctrl.library[name]["prior"]["sha256"],
            ctrl.library[name]["prior"]
            .get("lightning_input_sha256", {})
            .get(self.side),
        )
        poses = reg @ np.linalg.inv(data["object_pose"])
        ids = data["contact_link_id"]
        present = ids >= 0
        links = data["link_names"][ids.clip(min=0)]
        thumb = (links == "right_thumb_elastomer") & present
        fingers = present & ~thumb
        xyz = (
            np.einsum("nij,nkj->nki", poses[:, :3, :3], data["target_pos"])
            + poses[:, None, :3, 3]
        )
        thumb_z = (xyz[:, :, 2] * thumb).sum(1)
        mean_z = (xyz[:, :, 2] * fingers).sum(1) / np.maximum(1, fingers.sum(1))
        thumb_dz = np.where(thumb.any(1), thumb_z - mean_z, -0.060)
        if "pad_centers" not in data:
            pads = [
                "right_" + d + "_elastomer"
                for d in ("thumb", "index", "middle", "ring", "pinky")
            ]
            centers = []
            for q in data["q"]:
                row = []
                for link in pads:
                    frame, _ = ctrl.hand_contacts.link(link, np.eye(4), q)
                    row.append(
                        frame[:3, :3] @ ctrl.io.model.hand_support[link].mean(0)
                        + frame[:3, 3]
                    )
                centers.append(row)
            data["pad_centers"] = np.asarray(centers)
        pad_z = np.einsum("ni,nki->nk", poses[:, 2, :3], data["pad_centers"])
        pad_dz = pad_z[:, 0] - pad_z[:, 1:].mean(1)
        if "palm_gaps" not in data:
            patch = central_palm_samples(ctrl.io.model.hand_support["right_hand_C_MC"])
            obj_from_palm = np.linalg.inv(data["object_pose"])
            samples = (
                np.einsum("nij,pj->npi", obj_from_palm[:, :3, :3], patch)
                + obj_from_palm[:, None, :3, 3]
            )
            # A patch fraction rejects candidates grazing the edge with a single vertex.
            gaps = surface.tree.query(samples)[0]
            data["palm_gaps"] = np.quantile(gaps, 0.15, axis=1)
        palm_gap = data["palm_gaps"]
        digits = np.stack(
            [
                (
                    np.char.startswith(links.astype(str), "right_" + d + "_") & present
                ).any(1)
                for d in ("thumb", "index", "middle", "ring", "pinky")
            ],
            axis=1,
        )
        # A tall body with room for the four-finger span can accept a power grasp.
        span = np.ptp(surface.local, axis=0)
        power_body = span[2] >= 0.090 and np.min(span[:2]) >= 0.042
        park_rot = ctrl.ik.fk(ctrl.reference_arm)[:3, :3]
        trace = np.einsum("nij,ij->n", poses[:, :3, :3], park_rot)
        turn = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        # Contact normals on opposite sides and a palm close to the surface carry
        # more weight than the number of candidate fingertip contacts alone.
        normal = data["target_normal"]
        tn = (normal * thumb[:, :, None]).sum(1)
        opposing = (-np.einsum("ni,nki->nk", tn, normal) * fingers).sum(1) / np.maximum(
            1, fingers.sum(1)
        )
        hand_z = poses[:, 2, 0]
        center = reg[:3, 3]
        centroid = (xyz * present[:, :, None]).sum(1) / present.sum(1)[:, None]
        local_center = data["object_pose"][:, :3, 3]
        phalanges = np.char.endswith(links.astype(str), "_MP") & present
        score = (
            0.12 * phalanges.sum(1)
            + 2.0 * thumb.any(1)
            + 0.7 * fingers.sum(1)
            + opposing
            - 1.2 * turn
            - 65 * palm_gap
            - 22 * np.maximum(0, -thumb_dz + 0.003)
            - 45 * np.maximum(0, -pad_dz + 0.004)
            - 0.45 * abs(hand_z)
            - 12 * np.linalg.norm(centroid - center, axis=1)
            - 7 * np.maximum(0, local_center[:, 0] - 0.062)
        )
        # Prune clearly infeasible bracket orientations before expensive arm IK.
        fixture = getattr(ctrl.io.model, "fixture_palm", {})
        eligible = np.ones(len(poses), dtype=bool)
        if fixture:
            verts = np.vstack(list(fixture.values()))
            for lo in range(0, len(poses), 1024):
                subset = poses[lo : lo + 1024]
                low = (subset[:, 2, :3] @ verts.T).min(1) + subset[:, 2, 3]
                eligible[lo : lo + len(subset)] = low >= ctrl.table + 0.003
        counts = {
            "bank": len(poses),
            "mount_clear": int(eligible.sum()),
            "hand_clear": 0,
            "open_approach": 0,
            "arm_reachable": 0,
        }
        # The right and mirrored-left controller both use a canonical right palm.
        # +Y is radial (pinky towards index), +Z points from wrist to knuckles.
        adaptive = getattr(ctrl, "adaptive", False) and not power_body
        if adaptive:
            # Keep opposed contacts and prefer the thumb above the other pads in
            # the score. A short low object may require knuckles directed down.
            eligible &= (poses[:, 2, 1] >= -0.25) & (pad_dz >= -0.014) & thumb.any(1)
        else:
            eligible &= (
                (poses[:, 2, 1] >= 0.25)
                & (poses[:, 2, 2] >= -0.15)
                & (pad_dz >= 0.004)
                & thumb.any(1)
            )
        counts["thumb_up"] = int(eligible.sum())
        supported = digits.copy()
        if power_body:
            # Upstream assigns unrequested fingers and collision-checks them.
            # Credit an unassigned digit only if its pad or inner phalanx actually
            # approaches this object's measured surface at the returned native q.
            inv = np.linalg.inv(data["object_pose"])
            for i in np.flatnonzero(eligible & (palm_gap <= 0.012)):
                for j, digit in enumerate(("index", "middle", "ring", "pinky"), 1):
                    if digits[i, j]:
                        continue
                    for suffix in ("elastomer", "MP"):
                        link = f"right_{digit}_{suffix}"
                        t, _ = ctrl.hand_contacts.link(link, inv[i], data["q"][i])
                        vertices = ctrl.io.model.hand_support[link]
                        points = vertices @ t[:3, :3].T + t[:3, 3]
                        if np.quantile(surface.tree.query(points)[0], 0.05) <= 0.006:
                            supported[i, j] = True
                            break
        eligible &= digits.sum(1) >= (2 if adaptive else 3)
        # Prefer a full wrap whenever the arm can execute it. Narrow surfaces
        # and obstructed approaches can still support an upright opposed grip.
        tier = np.ones(len(poses), dtype=int)
        if power_body:
            close_palm = palm_gap <= 0.012
            tier[close_palm & (supported.sum(1) >= 4)] = 2
            tier[close_palm & supported.all(1)] = 3
        counts["nominal_full_wrap"] = int(np.count_nonzero(eligible & (tier == 3)))
        counts["contact_geometry"] = int(eligible.sum())
        excluded = {int(s.split("-")[-1]) for s in previous if s.startswith("LG-")}

        def distinct(i):
            for j in excluded:
                if i == j or (
                    np.linalg.norm(poses[i, :3, 3] - poses[j, :3, 3]) < 0.018
                    and np.trace(poses[i, :3, :3].T @ poses[j, :3, :3]) > 2.55
                ):
                    return False
            return True

        lo, hi = ctrl.io.model.limits["right"].T
        flex = np.array(
            [n.endswith(("_FE", "_IP", "_PIP", "_DIP")) for n in RIGHT_JOINTS]
        )
        choices = np.flatnonzero(eligible)
        order = choices[np.lexsort((-score[choices], -tier[choices]))]
        reached = []
        approach_checks = np.zeros(4, dtype=int)
        for i in order:
            if reached and tier[i] < tier[reached[0][1]]:
                break
            if approach_checks[tier[i]] >= 110 or not distinct(i):
                continue
            palm, q = poses[i], data["q"][i].astype(float)
            if (
                np.any(q < lo - 0.002)
                or np.any(q > hi + 0.002)
                or not ctrl.clear(palm, q, others)
                or (fixture and not surface.clear(fixture, palm, 0.001))
            ):
                continue
            counts["hand_clear"] += 1
            open_q = q.copy()
            open_q[flex] = ctrl.open_hand[flex] + 0.23 * (
                q[flex] - ctrl.open_hand[flex]
            )
            geometry = ctrl.link_shapes(open_q)
            toward_out = -palm[:3, 0]
            pre_options = [
                shifted(palm, d * toward_out - s * palm[:3, 2] + [0, 0, h])
                for d, s, h in (
                    (0.063, 0.018, 0.012),
                    (0.080, 0.025, 0.027),
                    (0.045, 0.040, 0.028),
                )
            ]
            pre_options = [
                p
                for p in pre_options
                if ctrl.clear(p, open_q, others)
                and surface.clear(geometry, p, 0.0005)
                and (
                    not fixture
                    or all(
                        surface.clear(
                            fixture, shifted(palm, t * (p[:3, 3] - palm[:3, 3])), 0.001
                        )
                        for t in (0.25, 0.5, 0.75)
                    )
                )
            ]
            if not pre_options:
                continue
            counts["open_approach"] += 1
            approach_checks[tier[i]] += 1
            for pre in pre_options:
                solved = None
                ik_seeds = [ctrl.arm, ctrl.mid_arm]
                if ctrl.bank is not None:
                    # Use native arm FK nearest to this pose, including other elbow branches.
                    atlas = ctrl.bank.palm
                    dot = np.einsum("nij,ij->n", atlas[:, :3, :3], pre[:3, :3])
                    cost = np.linalg.norm(
                        atlas[:, :3, 3] - pre[:3, 3], axis=1
                    ) + 0.10 * (3 - dot)
                    ik_seeds.extend(ctrl.bank.q[np.argsort(cost)[:4]])
                for seed in ik_seeds:
                    try:
                        solved = ctrl.ik.solve(pre, seed)
                        if getattr(ctrl, "adaptive", False):
                            ctrl.check_closure_path(
                                "lightning_grasp", solved, palm, palm
                            )
                        else:
                            ctrl.lightning_lift_ik(palm, solved)
                        break
                    except NoPath:
                        solved = None
                if solved is not None:
                    counts["arm_reachable"] += 1
                    effort = np.linalg.norm(solved - ctrl.arm)
                    reached.append((score[i] - 0.14 * effort, i, pre, open_q, solved))
                    break
            if len(reached) >= 6:
                break
        ctrl.event("lightning_filter", **counts)
        if not reached:
            raise NoPath("Lightning Grasp has no reachable collision-clear candidate")
        _, i, pre, open_q, solved = max(reached, key=lambda item: item[0])
        palm, q = poses[i], data["q"][i].astype(float)
        tight = q.copy()
        tight[flex] = np.clip(q[flex] + 0.15, lo[flex], hi[flex])
        posture = f"LG-{i}"
        ctrl.hints[pre.round(5).tobytes()] = solved
        # The native pose is the initial contact target. Use a separate, bounded
        # seating command along the measured palm normal to take up surface clearance.
        seat_depth = (
            min(float(palm_gap[i]) + 0.0015, 0.005)
            if power_body and palm_gap[i] <= 0.012
            else 0.0
        )
        seated = shifted(palm, seat_depth * palm[:3, 0])
        if not ctrl.clear(seated, tight, others):
            seated, seat_depth = palm, 0.0
        identity = {
            "bank_sha256": receipt["bank_sha256"],
            "bank_row": int(i),
            "candidate_source": str(data["source"][i]),
            "source_id": int(data["source_id"][i]),
        }
        ctrl.event(
            "selected",
            origin="lightning_grasp",
            posture=posture,
            upstream=receipt["upstream"],
            **identity,
            native_contact_links=links[i, present[i]].tolist(),
            physical_contact_links=[
                str(link).replace("right_", ctrl.side + "_")
                for link in links[i, present[i]]
            ],
            thumb_above_mean_fingers_m=float(thumb_dz[i]) if thumb[i].any() else None,
            predicted_thumb_pad_above_four_m=float(pad_dz[i]),
            radial_axis_world_z=float(palm[2, 1]),
            wrist_to_knuckles_world_z=float(palm[2, 2]),
            turn_from_park_deg=float(np.rad2deg(turn[i])),
            central_palm_gap_m=float(palm_gap[i]),
            palm_seat_m=seat_depth,
            contact_tier=("multi_digit", "palm_and_three", "palm_and_four")[
                tier[i] - 1
            ],
            supported_finger_geometry=supported[i].tolist(),
            rgbd_object_pose=reg.tolist(),
            palm_grasp=palm.tolist(),
            native_q=q.tolist(),
            hand_model="sharpa_22",
            bank_side=self.side,
            bank_input_sha256=receipt["input_sha256"],
        )
        return dict(
            posture=posture,
            origin="lightning_grasp",
            lightning_candidate=identity,
            q_open=open_q,
            q_touch=q,
            q_tight=tight,
        ), dict(pre=pre, grasp=palm, tight=seated)
