"""Fit a Sharpa palm pose and independent finger angles to a wrist-registered metric surface."""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .head_wrist_trial import FINGERS, PALM


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def cross_matrix(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def gravity_residual(contacts, center):
    """Best upward unit wrench obtainable with unilateral geometric friction cones."""
    locations = np.array([c["goal"] for c in contacts])
    radius = max(0.012, np.linalg.norm(locations - center, axis=1).mean())
    columns = []
    for c in contacts:
        n = c["normal"]
        ref = np.eye(3)[np.argmin(np.abs(n))]
        a = unit(np.cross(n, ref))
        b = np.cross(n, a)
        for angle in np.arange(8) * (np.pi / 4):
            force = -n + 0.55 * (np.cos(angle) * a + np.sin(angle) * b)
            columns.append(np.r_[force, np.cross(c["goal"] - center, force) / radius])
    wrenches = np.array(columns).T
    target = np.array([0, 0, 1, 0, 0, 0], dtype=float)
    fit = lsq_linear(wrenches, target, bounds=(0, 1.5), method="bvls", tol=1e-6)
    return float(np.linalg.norm(wrenches @ fit.x - target))


class ContactSurface:
    """Exposed faces of the registered convex parts, in the camera-calibrated robot frame."""

    def __init__(self, surface):
        pts, norms = [], []
        self.planes = []
        for part in surface.world_parts:
            hull = ConvexHull(part)
            tri = part[hull.simplices]
            normal = hull.equations[:, :3]
            self.planes.append(hull.equations)
            # Centroids, biased barycentric samples and edge midpoints cover long flat facets.
            for w in (
                (1 / 3, 1 / 3, 1 / 3),
                (0.72, 0.14, 0.14),
                (0.14, 0.72, 0.14),
                (0.14, 0.14, 0.72),
                (0.49, 0.49, 0.02),
                (0.49, 0.02, 0.49),
                (0.02, 0.49, 0.49),
            ):
                pts.append(np.einsum("nkj,k->nj", tri, w))
                norms.append(normal)
        pts, norms = np.vstack(pts), np.vstack(norms)
        keep = np.ones(len(pts), bool)
        for eq in self.planes:
            # Discard facets covered by another convex part.
            for i in range(0, len(pts), 1024):
                dis = (pts[i : i + 1024] @ eq[:, :3].T + eq[:, 3]).max(1)
                keep[i : i + 1024] &= dis >= -0.0008
        self.pts, self.normals = pts[keep], norms[keep]
        self.source = np.full(len(self.pts), "registered_prior", dtype="U20")
        refxyz, refn = self.pts.copy(), self.normals.copy()
        from scipy.spatial import cKDTree

        template = cKDTree(refxyz)
        for label, cloud in getattr(surface, "camera_samples", {}).items():
            if len(cloud) < 8:
                continue
            d, indices = cKDTree(cloud).query(cloud, k=min(14, len(cloud)))
            near = cloud[indices] - cloud[indices].mean(1, keepdims=True)
            values, basis = np.linalg.eigh(np.einsum("nki,nkj->nij", near, near))
            normals = basis[:, :, 0].copy()
            prior_normal = refn[template.query(cloud)[1]]
            flip = np.sum(normals * prior_normal, axis=1) < 0
            normals[flip] *= -1
            good = (d[:, -1] < 0.018) & (
                values[:, 0] < 0.25 * np.maximum(values[:, 1], 1e-9)
            )
            if not good.any():
                continue
            self.pts = np.vstack([self.pts, cloud[good]])
            self.normals = np.vstack([self.normals, normals[good]])
            self.source = np.r_[self.source, np.full(good.sum(), label, dtype="U20")]
        self.center = (surface.world.min(0) + surface.world.max(0)) / 2
        self.height = np.ptp(surface.world[:, 2])
        self.table = surface.table

    def assign(self, point, direction, avoid_bottom=True):
        distance = np.linalg.norm(self.pts - point, axis=1)
        cosine = self.normals @ direction
        cost = distance + 0.028 * (1 - cosine)
        cost += 0.0030 * (self.source == "registered_prior") + 0.001 * (
            self.source == "head_depth"
        )
        cost += 0.09 * np.maximum(0.16 - cosine, 0)
        if avoid_bottom:
            cost += 0.05 * (self.pts[:, 2] < self.table + 0.0025) + 0.035 * (
                self.normals[:, 2] < -0.5
            )
        i = np.argmin(cost)
        # Project onto the same tangent face for continuous targets within a sample cell.
        offset = point - self.pts[i]
        tangent = offset - self.normals[i] * (self.normals[i] @ offset)
        goal = self.pts[i] + tangent * min(
            1.0, 0.0025 / max(np.linalg.norm(tangent), 1e-9)
        )
        return goal, self.normals[i], str(self.source[i])


class HandContacts:
    def __init__(self, io):
        self.io = io
        self.names = io.model.hand_names["right"]
        tree = io.tree
        self.base = len(tree._chain(PALM))
        self.chains = {
            n: tree._chain(n)[self.base :]
            for n in io.model.hand_support
            if n.startswith("right_")
        }
        self.columns = {n: i for i, n in enumerate(self.names)}

    def link(self, name, palm, q):
        t = palm.copy()
        axes = []
        for j in self.chains[name]:
            t = t @ j.origin
            if j.kind != "fixed":
                i = self.columns[j.name]
                axes.append((i, t[:3, :3] @ j.axis, t[:3, 3].copy()))
                t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(q[i] * j.axis).as_matrix()
        return t, axes

    def anchors(self, roles, palm, q, surf):
        contacts = []
        for link in roles:
            t, _ = self.link(link, palm, q)
            v = self.io.model.hand_support[link]
            mid = v.mean(0)
            center = t[:3, :3] @ mid + t[:3, 3]
            toward = unit(surf.center - center)
            if link == PALM:
                # Central metacarpal surface, clear of the wrist flange and the knuckles.
                zlo, zhi = np.quantile(v[:, 2], [0.35, 0.83])
                face = v[(v[:, 2] >= zlo) & (v[:, 2] <= zhi)]
                if len(face) < 3:
                    face = v
                ids = np.argsort(face[:, 0])[-min(12, len(face)) :]
                anchor = face[ids].mean(0)
                normal = np.array([1.0, 0, 0])
            else:
                direction = t[:3, :3].T @ toward
                score = v @ direction
                anchor = np.mean(v[np.argsort(score)[-min(8, len(v)) :]], 0)
                normal = unit(anchor - mid)
            point = t[:3, :3] @ anchor + t[:3, 3]
            contacts.append(
                {
                    "link": link,
                    "anchor": anchor,
                    "hand_normal": normal,
                    "initial": point,
                }
            )
        thumb = next(c["initial"] for c in contacts if "thumb" in c["link"])
        opp = np.mean(
            [
                c["initial"]
                for c in contacts
                if ("index" in c["link"] or "middle" in c["link"])
            ],
            0,
        )
        axis = unit((opp - thumb) * [1.0, 1.0, 0.30])
        for c in contacts:
            link = c["link"]
            d = unit(c["initial"] - surf.center)
            if "thumb" in link:
                d = unit(0.28 * d - 0.85 * axis)
            elif link.endswith("elastomer"):
                d = unit(0.4 * d + 0.70 * axis)
            elif link == PALM:
                d = -(palm[:3, :3] @ c["hand_normal"])
            target = c["initial"].copy()
            if PALM not in roles and link.endswith("elastomer"):
                # A side contact below the equator can support a rounded item on lift.
                # Choose its lateral side before optimizing FK, independently of pad height.
                target[2] = np.clip(
                    target[2],
                    surf.table + 0.29 * surf.height,
                    surf.table + 0.51 * surf.height,
                )
                side = -axis if "thumb" in link else axis
                projection = (surf.pts - surf.center) @ side
                target += side * (
                    np.quantile(projection, 0.96) - (target - surf.center) @ side
                )
                d = unit(0.18 * d + 0.86 * side - np.array([0, 0, 0.10]))
            c["goal"], c["normal"], c["source"] = surf.assign(target, d)
            c["weight"] = (
                1.45
                if link == PALM or "thumb" in link
                else (1.0 if link.endswith("elastomer") else 0.68)
            )
        return contacts

    def point_jac(self, contact, palm, q):
        t, axes = self.link(contact["link"], palm, q)
        p = t[:3, :3] @ contact["anchor"] + t[:3, 3]
        normal = t[:3, :3] @ contact["hand_normal"]
        j = np.zeros((3, 28))
        jn = np.zeros((3, 28))
        j[:, :3] = np.eye(3)
        j[:, 3:6] = -cross_matrix(p - palm[:3, 3])
        jn[:, 3:6] = -cross_matrix(normal)
        for i, axis, origin in axes:
            j[:, 6 + i] = np.cross(axis, p - origin)
            jn[:, 6 + i] = np.cross(axis, normal)
        return p, normal, j, jn


def roles_for(family, card, hand):
    wrap = family == "wrap"
    wanted = card.get("intended_contacts", [])
    digits = [d for d in FINGERS if any(f"_{d}_" in n for n in wanted)]
    if not digits:
        digits = ["thumb", "index", "middle"] if not wrap else list(FINGERS)
    if "thumb" not in digits:
        digits.insert(0, "thumb")
    if not any(d in digits for d in ("index", "middle")):
        digits.append("index")
    if family in ("slender", "small") and len(digits) > 3:
        digits = digits[:3]
    links = ["right_" + d + "_elastomer" for d in digits]
    if wrap:
        links.extend(
            [
                PALM,
                *["right_" + d + "_PP" for d in ("index", "middle", "ring")],
                "right_middle_MP",
                "right_ring_MP",
            ]
        )
    return [link for link in dict.fromkeys(links) if link in hand.chains]


def open_around_surface(hand, contacts, palm, initial, lo, hi):
    """Set the unloaded aperture from opposed surface patches in the final palm frame."""
    opened = initial.copy()
    for _ in range(8):
        jac, error = [], []
        for site in contacts:
            if not site["link"].endswith("elastomer"):
                continue
            point, _, j, _ = hand.point_jac(site, palm, opened)
            jac.append(j[:, 6:])
            error.append(site["goal"] + 0.009 * site["normal"] - point)
        j = np.vstack(jac)
        damping = 0.000035
        delta = np.linalg.solve(
            j.T @ j + damping * np.eye(len(opened)),
            j.T @ np.concatenate(error) + damping * (initial - opened),
        )
        opened = np.clip(opened + np.clip(delta, -0.15, 0.15), lo, hi)
    return opened


def refine(controller, family, surface, candidates, others):
    hand = controller.hand_contacts
    surf = ContactSurface(surface)
    lo, hi = controller.io.model.limits["right"].T
    solutions = []
    for seed, poses, angle in candidates:
        q0 = np.clip(np.array(seed["q_touch"]), lo, hi)
        palm0 = poses["grasp"]
        palm = palm0.copy()
        q = q0.copy()
        contacts = hand.anchors(roles_for(family, seed, hand), palm, q, surf)
        if len(contacts) < 2:
            continue
        for _ in range(11):
            rows = []
            errors = []
            for c in contacts:
                pt, n, j, jn = hand.point_jac(c, palm, q)
                w = c["weight"]
                rows.extend((w * j, w * 0.007 * jn))
                errors.extend(
                    (
                        w * (c["goal"] - 0.002 * c["normal"] - pt),
                        w * 0.007 * (-c["normal"] - n),
                    )
                )
                if pt[2] < surf.table + 0.002:
                    rows.append(2 * j[2:3])
                    errors.append(np.array([2 * (surf.table + 0.002 - pt[2])]))
            j = np.vstack(rows)
            err = np.concatenate(errors)
            reg = np.r_[np.full(3, 0.13), np.full(3, 0.013), np.full(22, 0.0065)]
            prior = np.r_[
                palm0[:3, 3] - palm[:3, 3],
                Rotation.from_matrix(palm0[:3, :3] @ palm[:3, :3].T).as_rotvec(),
                q0 - q,
            ]
            delta = np.linalg.lstsq(
                np.vstack([j, np.diag(reg)]), np.r_[err, reg * prior], rcond=1e-6
            )[0]
            palm[:3, 3] = palm0[:3, 3] + np.clip(
                palm[:3, 3] + np.clip(delta[:3], -0.009, 0.009) - palm0[:3, 3],
                -0.045,
                0.045,
            )
            palm[:3, :3] = (
                Rotation.from_rotvec(np.clip(delta[3:6], -0.11, 0.11)).as_matrix()
                @ palm[:3, :3]
            )
            q = np.clip(q + np.clip(delta[6:], -0.18, 0.18), lo, hi)
            if (
                np.linalg.norm(delta[:3]) < 0.00025
                and np.max(np.abs(delta[6:])) < 0.009
            ):
                break
        # Enforce the whole hand's table clearance after contact optimization.
        low = (controller.shape(q) @ palm[2, :3]).min() + palm[2, 3]
        correction = max(0.0, surf.table + 0.001 - low)
        if correction > 0.014:
            continue
        palm[2, 3] += correction
        if not controller.clear(palm, q, others):
            continue
        gaps = {
            c["link"]: float(np.linalg.norm(hand.point_jac(c, palm, q)[0] - c["goal"]))
            for c in contacts
        }
        thumb = gaps.get("right_thumb_elastomer", 0.08)
        opponents = sorted(
            g for k, g in gaps.items() if k.endswith("elastomer") and "thumb" not in k
        )
        if not opponents:
            continue
        wrap = PALM in gaps
        quality = (
            1.4 * thumb
            + np.mean(opponents[:2])
            + np.mean(list(gaps.values()))
            + 0.8 * gaps.get(PALM, 0)
        )
        if thumb > 0.024 or opponents[0] > 0.024 or (wrap and gaps[PALM] > 0.031):
            continue
        thumb_normal = next(c["normal"] for c in contacts if "thumb" in c["link"])
        opposition = min(
            thumb_normal @ c["normal"] for c in contacts if "thumb" not in c["link"]
        )
        if opposition > -0.15:
            continue
        residual = gravity_residual(contacts, surf.center)
        quality += 0.035 * residual + 0.010 * (1 + opposition)
        opened = q.copy()
        nonthumb = np.array(
            [
                i
                for i, n in enumerate(hand.names)
                if n.endswith(("_FE", "PIP", "DIP", "_IP"))
            ]
        )
        opened[nonthumb] -= np.array(
            [0.19 if "thumb" in hand.names[i] else 0.29 for i in nonthumb]
        )
        opened = np.clip(opened, lo, hi)
        opened = open_around_surface(hand, contacts, palm, opened, lo, hi)
        offset = poses["pre"][:3, 3] - poses["grasp"][:3, 3]
        if np.linalg.norm(offset) < 0.012:
            offset = np.array([-0.22, -0.1, 1.0])
        offset = unit(offset)
        for direction in (
            offset,
            unit(0.6 * offset + np.array([0, 0, 0.6])),
            np.array([0.0, 0.0, 1.0]),
        ):
            for distance in (0.032, 0.049, 0.065):
                pre = palm.copy()
                pre[:3, 3] += distance * direction
                if controller.clear(pre, opened, others) and surface.clear(
                    controller.link_shapes(opened), pre, 0.0005
                ):
                    break
            else:
                continue
            break
        else:
            continue
        gap_pose = pre.copy()
        open_shape = controller.link_shapes(opened)
        for remaining in (0, 0.005, 0.010, 0.017, 0.026):
            trial = palm.copy()
            trial[:3, 3] += remaining * direction
            if controller.clear(trial, opened, others) and surface.clear(
                open_shape, trial, 0.0003
            ):
                gap_pose = trial
                break
        quality += 0.22 * np.linalg.norm(gap_pose[:3, 3] - palm[:3, 3])
        for initial in (controller.arm, controller.mid_arm):
            try:
                qa = controller.ik.solve(pre, initial)
                controller.ik.solve(palm, qa)
            except ValueError:
                continue
            break
        else:
            continue
        tighter = q.copy()
        tighter[nonthumb] += 0.16
        tighter = np.clip(tighter, lo, hi)
        card = dict(
            seed,
            q_open=opened,
            q_touch=q,
            q_tight=tighter,
            origin="wrist_mesh_joint_fit",
            fit_sites=contacts,
            seed_origin=seed["origin"],
            wrench_residual=round(residual, 4),
            opposite_normal_dot=round(float(opposition), 4),
            fit_gaps=gaps,
            seed_palm_mm=round(
                float(1000 * np.linalg.norm(palm[:3, 3] - palm0[:3, 3])), 2
            ),
            seed_q_rms=round(float(np.sqrt(np.mean((q - q0) ** 2))), 4),
        )
        solutions.append(
            (
                quality,
                card,
                {
                    "pre": pre,
                    "open_seat": gap_pose,
                    "grasp": palm,
                    "tight": palm.copy(),
                },
                angle,
            )
        )
    if not solutions:
        return None
    _, card, poses, angle = min(solutions, key=lambda r: r[0])
    controller.event(
        "mesh_fitted_palm_and_joints",
        posture=card["posture"],
        seed_origin=card["seed_origin"],
        palm_seed_offset_mm=card["seed_palm_mm"],
        joint_seed_rms_rad=card["seed_q_rms"],
        wrench_residual=card["wrench_residual"],
        opposing_normal_dot=card["opposite_normal_dot"],
        gap_m={k: round(v, 4) for k, v in card["fit_gaps"].items()},
        palm=poses["grasp"].tolist(),
        q_target=card["q_touch"].tolist(),
        targets={
            c["link"]: {
                "point_m": c["goal"].tolist(),
                "normal": c["normal"].tolist(),
                "source": c["source"],
            }
            for c in card["fit_sites"]
        },
    )
    return card, poses, angle


def close_from_mesh(controller, seed):
    io = controller.io
    hand = controller.hand_contacts
    contacts = seed["fit_sites"]
    lo, hi = io.model.limits["right"].T
    groups = {
        d: np.array(
            [i for i, n in enumerate(hand.names) if n.startswith("right_" + d + "_")]
        )
        for d in FINGERS
    }
    active = [d for d in FINGERS if any("_" + d + "_" in c["link"] for c in contacts)]
    retained = {d: 0 for d in active}
    held = set()
    last = {}
    controller.event("continuous_mesh_closure")
    for iteration in range(29):
        positions = io.q()
        palm = io.fk(PALM, positions)
        actual = positions[io.hand]
        command = controller.hand.copy()
        nxt = command.copy()
        for digit in active:
            ids = groups[digit]
            js = []
            errors = []
            gaps = []
            for c in contacts:
                if "_" + digit + "_" not in c["link"]:
                    continue
                p, _, j, _ = hand.point_jac(c, palm, actual)
                gap = float((p - c["goal"]) @ c["normal"])
                gaps.append(gap)
                squeeze = min(0.008, 0.002 + iteration * 0.00040)
                js.append(c["weight"] * j[:, 6 + ids])
                errors.append(c["weight"] * (c["goal"] - squeeze * c["normal"] - p))
            flex = [
                i for i in ids if hand.names[i].endswith(("_FE", "PIP", "DIP", "_IP"))
            ]
            lag = float(np.max(np.abs(command[flex] - actual[flex])))
            retained[digit] = (
                retained[digit] + 1
                if lag > 0.055 and min(abs(np.array(gaps))) < 0.010
                else 0
            )
            if retained[digit] >= 3:
                held.add(digit)
            elif lag < 0.025 or min(abs(np.array(gaps))) > 0.013:
                held.discard(digit)
            last[digit] = [round(lag, 4), round(float(min(gaps, key=abs)), 4)]
            if digit in held:
                continue
            j = np.vstack(js)
            error = np.concatenate(errors)
            dq = np.linalg.solve(j.T @ j + 0.000044 * np.eye(len(ids)), j.T @ error)
            nxt[ids] += np.clip(actual[ids] + dq - command[ids], -0.068, 0.068)
        nxt = np.clip(nxt, lo, hi)
        for k in range(1, 11):
            io.step(controller.arm, command + (nxt - command) * k / 10)
        controller.hand = nxt
        if iteration >= 11 and "thumb" in held and len(held) >= min(3, len(active)):
            break
    controller.event(
        "mesh_closure_readback",
        iterations=iteration + 1,
        blocked_near_surface=sorted(held),
        finger_lag_and_gap=last,
    )
