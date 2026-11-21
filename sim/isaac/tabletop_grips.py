"""Align catalog finger pairs with a contact band measured in the wrist reconstruction."""

import numpy as np
from scipy.spatial import cKDTree

from .head_wrist_geometry import hand_shapes


def basis(axis, direction):
    u = axis / np.linalg.norm(axis)
    v = direction - u * (u @ direction)
    v /= np.linalg.norm(v)
    return np.column_stack([u, v, np.cross(u, v)])


def patch_candidates(controller, name, surface, seeds, others):
    from .handle_grasp import handle_sections
    from .head_wrist_trial import Rejected

    world = surface.world
    center = (world.max(0) + world.min(0)) / 2
    _, axes = np.linalg.eigh(np.cov(world[:, :2].T))
    across = axes[:, 0]
    if name == "driver":
        try:
            origin, along, across, sections = handle_sections(world)
        except Rejected:
            return []
        row = sections[len(sections) // 2]
        center = np.r_[origin + along * row[0] + across * row[1], row[2]]
    elif name in (
        "shuttle",
        "marker",
        "metal",
        "wire",
        "redtool",
        "white",
        "cushion",
        "box",
    ):
        sections = surface.bands()
        if sections:
            # Small contact patches on thin edges can be reached without straddling the full width.
            _, center, across, _, _, _ = min(
                sections[:6],
                key=lambda r: r[0] + 0.14 * np.linalg.norm(r[1][:2] - [0.42, -0.30]),
            )
    ref = cKDTree(world)
    proposals = []
    for s in seeds[:3]:
        qo, qt, _qc = [np.array(s[k]) for k in ("q_open", "q_touch", "q_tight")]
        support = hand_shapes(controller.io.model, qt)
        thumb = support["right_thumb_elastomer"]
        digits = (
            ("index", "middle", "ring")
            if name in ("driver", "bottle", "tube", "can", "spray")
            else ("index", "middle")
        )
        clouds = [support[f"right_{d}_elastomer"] for d in digits]
        finger = np.mean([v.mean(0) for v in clouds], 0)
        t = thumb.mean(0)
        mid = (finger + t) * 0.5
        src = basis(finger - t, mid)
        axes2 = [across, -across]
        if name in ("ball", "bottle", "tube", "can", "spray"):
            axes2 = [
                np.array([np.cos(a), np.sin(a)])
                for a in np.arange(0, 2 * np.pi, np.pi / 4)
            ]
        for ax in axes2:
            towards = np.array([-ax[1], ax[0]])
            if towards @ (center[:2] - [0.32, -0.28]) < 0:
                towards = -towards
            for tilt in (0.43, 0.80, 1.20):
                rot = (
                    basis(np.r_[ax, 0], np.r_[towards * np.cos(tilt), -np.sin(tilt)])
                    @ src.T
                )
                grasp = np.eye(4)
                grasp[:3, :3] = rot
                grasp[:3, 3] = center - rot @ mid
                # Match the contact-facing pad faces, whose widths vary with the catalog joints.
                chain = (finger - t) / np.linalg.norm(finger - t)
                tip = t + chain * (np.quantile(thumb @ chain, 0.92) - t @ chain)
                far = finger + chain * (
                    np.mean([np.quantile(v @ chain, 0.08) for v in clouds])
                    - finger @ chain
                )
                grasp[:3, 3] = center - rot @ ((tip + far) / 2)
                low = (controller.shape(qt) @ rot[2]).min() + grasp[2, 3]
                grasp[2, 3] += max(0.0, controller.table + 0.001 - low)
                tight = grasp.copy()
                pre = grasp.copy()
                pre[:3, 3] -= rot @ src[:, 1] * 0.038
                low = (controller.shape(qo) @ rot[2]).min() + pre[2, 3]
                pre[2, 3] += max(0.005, controller.table + 0.003 - low)
                if not controller.clear(pre, qo, others) or not controller.clear(
                    grasp, qt, others
                ):
                    continue
                gaps = [
                    ref.query(v[:: max(1, len(v) // 90)] @ rot.T + grasp[:3, 3])[
                        0
                    ].min()
                    for v in (thumb, *clouds)
                ]
                score = (
                    gaps[0]
                    + np.mean(sorted(gaps[1:])[:2])
                    + 0.1
                    * max(0.0, np.linalg.norm(grasp[:2, 3] - [0.43, -0.28]) - 0.26)
                )
                proposals.append(
                    (score, s, {"pre": pre, "grasp": grasp, "tight": tight})
                )
    return sorted(proposals, key=lambda x: x[0])
