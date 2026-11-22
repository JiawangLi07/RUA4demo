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


def requires_full_wrap(points):
    """Long bodies with two broad transverse dimensions can span the four fingers."""
    centered = points - np.mean(points, axis=0)
    _, basis = np.linalg.eigh(centered.T @ centered)
    lower, upper = np.quantile(centered @ basis, [0.005, 0.995], axis=0)
    narrow, wide, length = np.sort(upper - lower)
    return bool(narrow >= 0.042 and length >= max(0.090, 2 * wide))


def fit_pressure_pose(
    ctrl, surface, object_from_palm, native, section_guess, constrained
):
    """Solve compliant motor targets at the seated wrist using the registered mesh."""
    guess = section_guess.astype(float)
    drive = guess.copy()
    lower, upper = ctrl.io.model.limits["right"].T
    center = np.median(surface.local, axis=0)
    for digit, active in zip(
        ("thumb", "index", "middle", "ring", "pinky"), constrained
    ):
        columns = np.array(
            [
                i
                for i, n in enumerate(RIGHT_JOINTS)
                if n.startswith("right_" + digit + "_")
            ]
        )
        jacobians = []
        residuals = []
        for suffix, weight, compression in (
            ("elastomer", 1.0, 0.003),
            ("PP" if digit == "thumb" else "MP", 0.65, 0.002),
        ):
            link = f"right_{digit}_{suffix}"
            frame, axes = ctrl.hand_contacts.link(link, object_from_palm, guess)
            cloud = ctrl.io.model.hand_support[link] @ frame[:3, :3].T + frame[:3, 3]
            gap, nearest = surface.tree.query(cloud)
            band = gap <= np.quantile(gap, 0.08) + 0.0004
            point = cloud[band].mean(0)
            target = surface.tree.data[nearest[band]].mean(0)
            inward = center - target
            inward[2] = 0
            inward /= max(1e-8, np.linalg.norm(inward))
            J = np.zeros((3, len(RIGHT_JOINTS)))
            for idx, axis, origin in axes:
                J[:, idx] = np.cross(axis, point - origin)
            jacobians.append(weight * J[:, columns])
            residuals.append(weight * (target + compression * inward - point))
        jacobians.append(0.012 * np.eye(len(columns)))
        # Use the native solution to break ties between motor configurations
        # that seat the same two patches on the scanned wall.
        residuals.append(
            0.012 * np.clip(native[columns] - guess[columns], -0.18, 0.18)
            if active
            else np.zeros(len(columns))
        )
        delta = np.linalg.lstsq(
            np.vstack(jacobians), np.concatenate(residuals), rcond=None
        )[0]
        drive[columns] = np.clip(
            guess[columns] + np.clip(delta, -0.14, 0.14), lower[columns], upper[columns]
        )
    return drive


def seat_contact_knots(ctrl, palm, seated, native, pressure, links, anchors, goals):
    """Follow the original native contact sites as the palm advances around the object."""
    from .tabletop_control import seating_frames

    targets = goals @ palm[:3, :3].T + palm[:3, 3]
    low, high = ctrl.io.model.limits["right"].T
    q = native.copy()
    knots = []
    for fraction, frame in seating_frames(palm, seated):
        # Release the kinematic overconstraint from the compliant motor target:
        # keep contacted sites stationary in the measured object frame.
        bias = native + 0.55 * fraction * (pressure - native)
        for _ in range(5):
            J = []
            residual = []
            for name, anchor, target in zip(links, anchors, targets):
                link, axes = ctrl.hand_contacts.link(str(name), frame, q)
                point = link[:3, :3] @ anchor + link[:3, 3]
                jac = np.zeros((3, len(q)))
                for col, axis, origin in axes:
                    jac[:, col] = np.cross(axis, point - origin)
                J.append(jac)
                residual.append(target - point)
            J.append(0.008 * np.eye(len(q)))
            residual.append(0.008 * (bias - q))
            update = np.linalg.lstsq(
                np.vstack(J), np.concatenate(residual), rcond=None
            )[0]
            q = np.clip(q + np.clip(update, -0.18, 0.18), low, high)
        knots.append((frame, q.copy()))
    return knots


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
                if "source_domain_sha256" in z:
                    data["source_domain_sha256"] = z["source_domain_sha256"]
                data["source_id"] = (
                    z["source_id"] if "source_id" in z else np.arange(len(z["q"]))
                )
                # Native, transferred and scene-refitted banks use the same link-local anchors.
                if "contact_pos" in z and "contact_normal" in z:
                    data["contact_pos"] = z["contact_pos"]
                    data["contact_normal"] = z["contact_normal"]
                if repo.get("method") == "live_scene_contact_refit":
                    for key in (
                        "arm_pre",
                        "pre_in_palm",
                        "q_open",
                    ):
                        data[key] = z[key]
                if repo.get("method") in (
                    "reachable_placement_native_five_link_contact_ik",
                    "arm_chart_native_contact_field",
                    "arm_chart_native_contact_refinement",
                ):
                    for key in (
                        "pre_in_palm",
                        "tight_in_palm",
                        "arm_pre",
                        "q_open",
                        "cost",
                        "warm_delta_rad",
                        "standoff_m",
                    ):
                        data[key] = z[key]
                    if "seated_path" in z:
                        data["seated_path"] = z["seated_path"]
                    if "contact_pos" in z:
                        data["contact_pos"] = z["contact_pos"]
                    if "q_wrap_guide" in z:
                        data["q_wrap_guide"] = z["q_wrap_guide"]
                    if not np.all(data["source"] == "reach_conditioned"):
                        raise ValueError("unexpected source in arm-conditioned bank")
                if not all(
                    np.isfinite(data[k]).all()
                    for k in (
                        "q",
                        "object_pose",
                        "target_pos",
                        "target_normal",
                        "contact_pos",
                        "contact_normal",
                    )
                    if k in data
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

    def select(
        self, ctrl, name, surface, reg, others, previous, *, placement_limit=None
    ):
        data, receipt = self.read(
            name,
            ctrl.library[name]["prior"]["sha256"],
            ctrl.library[name]["prior"]
            .get("lightning_input_sha256", {})
            .get(self.side),
        )
        guided = "arm_pre" in data
        conditioned = "tight_in_palm" in data
        uniform = False
        if conditioned:
            low, high = np.quantile(surface.local[:, 2], [0.01, 0.99])
            widths = []
            for f in (0.32, 0.49, 0.72, 0.82):
                section = surface.local[
                    np.abs(surface.local[:, 2] - (low + f * (high - low))) < 0.006, :2
                ]
                if len(section) >= 10:
                    widths.append(
                        np.mean(
                            np.diff(np.quantile(section, [0.01, 0.99], axis=0), axis=0)
                        )
                    )
            uniform = len(widths) == 4 and min(widths) > 0.87 * max(widths)
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
        require_wrap = not getattr(ctrl, "hybrid_grasps", False) and requires_full_wrap(
            surface.local
        )
        power_body = require_wrap or (span[2] >= 0.090 and np.min(span[:2]) >= 0.042)
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
        elif conditioned:
            # A cylinder can sit between the pads with the thumb near their
            # mean height. Rank pad height within anatomically radial-up palms.
            eligible &= (
                (poses[:, 2, 1] >= 0.60)
                & (poses[:, 2, 2] >= -0.50)
                & (thumb.any(1) | ("q_wrap_guide" in data))
            )
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
        # A native finger solution and a nearby palm patch are both required for
        # an enveloping grasp. Other shapes can use an opposed multi-digit grip.
        tier = np.ones(len(poses), dtype=int)
        if power_body:
            close_palm = palm_gap <= 0.012
            tier[close_palm & (supported.sum(1) >= 4)] = 2
            tier[close_palm & supported.all(1)] = 3
        if require_wrap and conditioned:
            patch = central_palm_samples(ctrl.io.model.hand_support["right_hand_C_MC"])
            seat_obj = np.linalg.inv(data["object_pose"]) @ data["tight_in_palm"]
            points = (
                patch @ seat_obj[:, :3, :3].transpose(0, 2, 1)
                + seat_obj[:, None, :3, 3]
            )
            seated_gaps = np.quantile(surface.tree.query(points)[0], 0.15, axis=1)
            pad_motion = np.zeros((len(poses), 5, 3))
            if "q_wrap_guide" in data:
                # Re-evaluate every finger on the registered surface at the final
                # commanded seat, including fingers left free by the native solver.
                if "seated_q" not in data:
                    data["seated_q"] = data["q_wrap_guide"].copy()
                for i in np.flatnonzero(eligible & (seated_gaps <= 0.005)):
                    data["seated_q"][i] = fit_pressure_pose(
                        ctrl,
                        surface,
                        seat_obj[i],
                        data["q"][i],
                        data["q_wrap_guide"][i],
                        digits[i],
                    )
                    for j, digit in enumerate(
                        ("thumb", "index", "middle", "ring", "pinky")
                    ):
                        near = False
                        for suffix in ("elastomer", "PP" if digit == "thumb" else "MP"):
                            link = f"right_{digit}_{suffix}"
                            frame, _ = ctrl.hand_contacts.link(
                                link, seat_obj[i], data["seated_q"][i]
                            )
                            cloud = (
                                ctrl.io.model.hand_support[link] @ frame[:3, :3].T
                                + frame[:3, 3]
                            )
                            if suffix == "elastomer":
                                initial = np.linalg.solve(
                                    data["object_pose"][i],
                                    np.r_[data["pad_centers"][i, j], 1.0],
                                )[:3]
                                pad_motion[i, j] = cloud.mean(0) - initial
                            if np.quantile(surface.tree.query(cloud)[0], 0.05) <= 0.006:
                                near = True
                                break
                        supported[i, j] = near
                tier[(seated_gaps <= 0.005) & supported.all(1)] = 3
        counts["nominal_full_wrap"] = int(np.count_nonzero(eligible & (tier == 3)))
        counts["full_wrap_required"] = require_wrap
        if require_wrap:
            if conditioned:
                # The native start may stand away from the wall. Check the
                # central patch again at the actual commanded seated transform.
                eligible &= (tier == 3) & supported.all(1) & (seated_gaps <= 0.005)
                score = (
                    -1100 * data["cost"]
                    - 0.25 * data["warm_delta_rad"]
                    - 42 * data["standoff_m"]
                    - 65 * np.linalg.norm(pad_motion[:, 0, :2], axis=1)
                    - 95 * np.linalg.norm(pad_motion[:, :, :2].mean(axis=1), axis=1)
                    - 20 * np.linalg.norm(pad_motion[:, :, :2], axis=2).mean(axis=1)
                    + 8 * pad_dz
                    + 0.43 * digits.sum(1)
                    + 5.0 * uniform * digits.all(1)
                )
            else:
                eligible &= (tier == 3) & (palm_gap <= 0.005)
        excluded = {int(s.split("-")[-1]) for s in previous if s.startswith("LG-")}

        def distinct(i):
            for j in excluded:
                if (
                    conditioned
                    and i != j
                    and np.linalg.norm(data["q"][i] - data["q"][j]) > 0.65
                ):
                    continue
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

        def seating(i):
            tight = data.get("seated_q", data["q"])[i].astype(float)
            tight[flex] = np.clip(
                tight[flex] + (0.055 if conditioned else 0.15), lo[flex], hi[flex]
            )
            if conditioned:
                goal = poses[i] @ data["tight_in_palm"][i]
                return goal, tight, float(np.linalg.norm(goal[:3, 3] - poses[i, :3, 3]))
            depth = (
                min(float(palm_gap[i]) + 0.0015, 0.005)
                if power_body and palm_gap[i] <= 0.012
                else 0.0
            )
            return shifted(poses[i], depth * poses[i, :3, 0]), tight, depth

        direct_seat = conditioned and require_wrap and "seated_q" in data
        if direct_seat and "seated_path" in data:
            eligible &= data["seated_path"]

        counts["contact_geometry"] = int(eligible.sum())
        closure_mode = "lightning_grasp"
        if conditioned:
            closure_mode = "seated_wrap" if direct_seat else "power_wrap"

        from .tabletop_opposition import LowerEdge, native_opposition

        guard = LowerEdge(surface, ctrl.table) if np.any(eligible) else None
        opposed, lateral_dot, lower_count = native_opposition(
            poses, data, links, present
        )
        eligible &= opposed
        capacity = lever = None
        if np.any(eligible) and (
            (not conditioned and not power_body) or receipt.get("continuous_lower_edge")
        ):
            from .tabletop_handle import volume_center
            from .tabletop_opposition import lateral_load_capacity, long_horizontal_axis

            capacity, _ = lateral_load_capacity(
                surface, xyz, poses, data, present, eligible
            )
            lever = np.linalg.norm(
                centroid[:, :2] - volume_center(surface.world_parts)[:2], axis=1
            )
            if receipt.get("continuous_lower_edge"):
                eligible &= capacity >= 0.14
            # A row with normal opposition can still pivot about one end of a tool.
            # Score how much upward load its friction cones can balance about the mesh's centre.
            if long_horizontal_axis(surface.local @ reg[:3, :3].T) is not None:
                score += 10.0 * np.minimum(capacity, 1.25) - 65.0 * lever
            else:
                score -= 16.0 * np.maximum(0.55 - capacity, 0.0) + 85.0 * np.maximum(
                    lever - 0.020, 0.0
                )
        counts["thumb_opposed_with_limited_lower_support"] = int(eligible.sum())
        choices = np.flatnonzero(eligible)
        if getattr(ctrl, "allow_hand_table_contact", False):
            # A light brush is admissible. A palm placement well through the rigid
            # desk will deflect the wrist and miss the intended mesh contacts.
            for row in choices:
                lowest = (ctrl.shape(data["q"][row]) @ poses[row, 2, :3]).min() + poses[
                    row, 2, 3
                ]
                # Include the open approach: extended fingers can hit the desk
                # before a valid final contact shape is reached.
                preview = data["q"][row].copy()
                preview[flex] = ctrl.open_hand[flex] + 0.4 * (
                    preview[flex] - ctrl.open_hand[flex]
                )
                open_low = (ctrl.shape(preview) @ poses[row, 2, :3]).min() + poses[
                    row, 2, 3
                ]
                score[row] -= 1600 * max(0.0, ctrl.table - lowest - 0.001)
                score[row] -= 900 * max(0.0, ctrl.table - open_low - 0.001)
        order = choices[np.lexsort((-score[choices], -tier[choices]))]
        reached = []
        reach_trace = {}
        approach_checks = np.zeros(4, dtype=int)
        for i in order:
            if reached and tier[i] < tier[reached[0][1]]:
                break
            if approach_checks[tier[i]] >= 110 or not distinct(i):
                continue
            palm, q = poses[i], data["q"][i].astype(float)
            if placement_limit is not None and any(
                np.linalg.norm(palm[:3, 3] - poses[j, :3, 3]) < 0.006
                and np.trace(palm[:3, :3].T @ poses[j, :3, :3]) > 2.96
                and np.linalg.norm(q - data["q"][j]) < 0.40
                for _, j, *_ in reached
            ):
                continue
            if (
                np.any(q < lo - 0.002)
                or np.any(q > hi + 0.002)
                or (not direct_seat and not ctrl.clear(palm, q, others))
                or (fixture and not surface.clear(fixture, palm, 0.001))
            ):
                continue
            closure_goal = palm
            if require_wrap:
                closure_goal, tight_q, _ = seating(i)
                if not ctrl.clear(closure_goal, tight_q, others):
                    continue
            counts["hand_clear"] += 1
            open_q = q.copy()
            open_q[flex] = ctrl.open_hand[flex] + 0.23 * (
                q[flex] - ctrl.open_hand[flex]
            )
            if guided:
                open_q = data["q_open"][i]
            elif not require_wrap and getattr(ctrl, "allow_hand_table_contact", False):
                # Keep a finger opening around the object while avoiding the
                # extended-finger posture driving into the support plane.
                for fraction in (0.23, 0.42, 0.60):
                    open_q[flex] = ctrl.open_hand[flex] + fraction * (
                        q[flex] - ctrl.open_hand[flex]
                    )
                    if (ctrl.shape(open_q) @ palm[2, :3]).min() + palm[
                        2, 3
                    ] >= ctrl.table - 0.002:
                        break
            try:
                _, tight_q, _ = seating(i)
                lower = set()
                for start, end in ((open_q, q), (q, tight_q)):
                    for fraction in (0.0, 0.33, 0.66, 1.0):
                        lower |= guard.check(
                            ctrl, palm, start + fraction * (end - start)
                        )
                if len(lower) > 2:
                    continue
            except NoPath:
                continue
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
            if guided:
                pre_options = [palm @ data["pre_in_palm"][i]]
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
            failures = {}
            reach_trace[int(i)] = {"open_poses": len(pre_options), "branches": failures}
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
                if guided:
                    ik_seeds.insert(0, data["arm_pre"][i])
                for seed in ik_seeds:
                    stage = "pregrasp_ik"
                    try:
                        solved = ctrl.ik.solve(pre, seed)
                        stage = "closure_path"
                        if require_wrap or getattr(ctrl, "adaptive", False):
                            ctrl.check_closure_path(
                                closure_mode,
                                solved,
                                palm,
                                closure_goal,
                            )
                        else:
                            ctrl.lightning_lift_ik(palm, solved)
                        break
                    except NoPath as error:
                        reason = getattr(
                            error,
                            "stage",
                            stage if str(error) == "arm IK" else str(error),
                        )
                        entry = failures.setdefault(reason, {"count": 0})
                        entry["count"] += 1
                        if hasattr(error, "residual"):
                            pos, angle = error.residual
                            if (
                                "closest_position_m" not in entry
                                or pos < entry["closest_position_m"]
                            ):
                                entry.update(
                                    closest_position_m=pos,
                                    rotation_at_closest_rad=angle,
                                    bound_joint_numbers=[
                                        j + 1 for j in error.bound_joints
                                    ],
                                )
                        solved = None
                if solved is not None:
                    if placement_limit is not None:
                        ctrl.hints[pre.round(5).tobytes()] = solved
                        target, _, _ = seating(i)
                        try:
                            route = ctrl.move(
                                pre,
                                open_q,
                                0.58,
                                others,
                                True,
                                surface,
                                object_margin=0.0005,
                                plan_only=True,
                                endpoint_check=lambda arm, grasp=palm, goal=target: (
                                    ctrl.check_closure_path(
                                        closure_mode, arm, grasp, goal
                                    )
                                ),
                            )
                            solved = route[-1]
                        except NoPath:
                            failures.setdefault("transit_path", {"count": 0})[
                                "count"
                            ] += 1
                            continue
                    counts["arm_reachable"] += 1
                    effort = np.linalg.norm(solved - ctrl.arm)
                    reached.append((score[i] - 0.14 * effort, i, pre, open_q, solved))
                    break
            if len(reached) >= (placement_limit or 6):
                break
        if placement_limit is not None:
            ctrl.event(
                "scene_placement_filter", **counts, reach_by_bank_row=reach_trace
            )
            return sorted(reached, key=lambda row: row[0], reverse=True)
        ctrl.event("lightning_filter", **counts, reach_by_bank_row=reach_trace)
        if not reached:
            if require_wrap:
                raise NoPath(
                    "Lightning Grasp has no executable four-finger and central-palm wrap"
                )
            raise NoPath("Lightning Grasp has no reachable collision-clear candidate")
        _, i, pre, open_q, solved = max(reached, key=lambda item: item[0])
        palm, q = poses[i], data["q"][i].astype(float)
        posture = f"LG-{i}"
        ctrl.hints[pre.round(5).tobytes()] = solved
        # The native pose is the initial contact target. Use a separate, bounded
        # seating command along the measured palm normal to take up surface clearance.
        seated, tight, seat_depth = seating(i)
        if not require_wrap and not ctrl.clear(seated, tight, others):
            seated, seat_depth = palm, 0.0
        identity = {
            "bank_sha256": receipt["bank_sha256"],
            "bank_row": int(i),
            "candidate_source": str(data["source"][i]),
            "source_id": int(data["source_id"][i]),
            "estimated_upward_capacity": float(capacity[i])
            if capacity is not None
            else None,
            "contact_lever_xy_m": float(lever[i]) if lever is not None else None,
        }
        if "source_domain_sha256" in data:
            identity["domain_sha256"] = str(data["source_domain_sha256"][i])
        else:
            identity["domain_sha256"] = receipt.get("request_sha256")
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
            explicitly_constrained_fingers=digits[i].tolist(),
            radial_axis_world_z=float(palm[2, 1]),
            wrist_to_knuckles_world_z=float(palm[2, 2]),
            turn_from_park_deg=float(np.rad2deg(turn[i])),
            central_palm_gap_m=float(palm_gap[i]),
            seated_central_palm_gap_m=float(seated_gaps[i])
            if require_wrap and conditioned
            else None,
            full_wrap_required=require_wrap,
            palm_seat_m=seat_depth,
            native_five_on_uniform_shaft=bool(uniform and digits[i].all()),
            closure_mode=closure_mode,
            pad_travel_to_seat_m=np.linalg.norm(pad_motion[i], axis=1).tolist()
            if require_wrap and conditioned
            else None,
            contact_tier=("multi_digit", "palm_and_three", "palm_and_four")[
                tier[i] - 1
            ],
            supported_finger_geometry=supported[i].tolist(),
            rgbd_object_pose=reg.tolist(),
            palm_grasp=palm.tolist(),
            native_q=q.tolist(),
            seated_q=data["seated_q"][i].tolist() if "seated_q" in data else None,
            approach_policy="thumb_opposed_limited_lower_edge",
            opposed_horizontal_dot=float(lateral_dot[i]),
            native_lower_non_thumb_digits=int(lower_count[i]),
            hand_model="sharpa_22",
            bank_side=self.side,
            bank_input_sha256=receipt["input_sha256"],
        )
        knots = []
        native_sites = []
        if not conditioned and "contact_pos" in data and "contact_normal" in data:
            for j in np.flatnonzero(present[i]):
                native_sites.append(
                    dict(
                        link=str(links[i, j]),
                        anchor=data["contact_pos"][i, j],
                        hand_normal=data["contact_normal"][i, j],
                        goal=palm[:3, :3] @ data["target_pos"][i, j] + palm[:3, 3],
                        normal=-palm[:3, :3] @ data["target_normal"][i, j],
                        weight=1.2 if thumb[i, j] else 1.0,
                    )
                )
        if conditioned and "seated_q" in data and "contact_pos" in data:
            knots = seat_contact_knots(
                ctrl,
                palm,
                seated,
                q,
                data["seated_q"][i],
                links[i, present[i]],
                data["contact_pos"][i, present[i]],
                data["target_pos"][i, present[i]],
            )
        return dict(
            lower_guard=guard,
            native_sites=native_sites,
            seating_knots=knots,
            explicit_digits=digits[i].tolist(),
            closure_mode=closure_mode,
            posture=posture,
            origin="lightning_grasp",
            lightning_candidate=identity,
            power_wrap=conditioned,
            q_open=open_q,
            q_touch=q,
            q_tight=tight,
            seated_q=data["seated_q"][i] if "seated_q" in data else None,
        ), dict(pre=pre, grasp=palm, tight=seated)
