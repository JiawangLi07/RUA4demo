"""Prepare object-to-palm placements from registered head depth and arm kinematics."""

import json
from pathlib import Path

import numpy as np

from .tabletop_adaptive import nearest_arm, wall_grasps
from .tabletop_control import NoPath
from .tabletop_lightning import requires_full_wrap


def checked_placements(ctrl, name, surface, reg, others, out, limit=24):
    if not requires_full_wrap(surface.local):
        raise NoPath(
            "reachable wall placement requires a long object with room for four fingers"
        )
    ranked = []
    for cost, card, targets in wall_grasps(ctrl, surface):
        distance, seeds = nearest_arm(ctrl, targets["pre"])
        ranked.append((cost + 0.22 * distance, card, targets, seeds))
    rows, visited = [], set()
    for cost, card, targets, seeds in sorted(ranked, key=lambda row: row[0]):
        palm, pre, tight = (targets[k] for k in ("grasp", "pre", "tight"))
        key = palm.round(5).tobytes()
        if key in visited:
            continue
        opened, warm = card["q_open"], card["q_touch"]
        if not ctrl.clear(palm, warm, others) or not ctrl.clear(
            tight, card["q_tight"], others
        ):
            continue
        if not ctrl.clear(pre, opened, others) or not surface.clear(
            ctrl.link_shapes(opened), pre, 0.0005
        ):
            continue
        for seed in (seeds[0], ctrl.arm, *seeds[1:]):
            try:
                arm = ctrl.ik.solve(pre, seed)
                ctrl.check_closure_path("head_wall_envelope", arm, palm, tight)
            except NoPath:
                continue
            break
        else:
            continue
        visited.add(key)
        rows.append(
            (
                np.linalg.solve(palm, reg),
                np.linalg.solve(palm, pre),
                np.linalg.solve(palm, tight),
                warm,
                opened,
                arm,
                cost,
            )
        )
        if len(rows) >= limit:
            break
    if not rows:
        raise NoPath("no arm-feasible placements before native finger synthesis")
    keys = (
        "object_pose",
        "pre_in_palm",
        "tight_in_palm",
        "q_warm",
        "q_open",
        "arm_pre",
        "cost",
    )
    data = dict(zip(keys, map(np.asarray, zip(*rows))))
    prior = ctrl.library[name]["prior"]
    receipt = {
        "target": name,
        "side": ctrl.side,
        "input_sha256": prior["lightning_input_sha256"][ctrl.side],
        "prior_sha256": prior["sha256"],
        "source_frame": "object_to_palm",
        "inputs": "head RGB-D, registered mesh, issued joints, station CAD",
        "placements": len(rows),
    }
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{name}.{ctrl.side}.npz"
    np.savez_compressed(
        file,
        **data,
        mesh_to_world=reg,
        arm_start=ctrl.arm,
        table_m=ctrl.table,
        table_edge_m=ctrl.table_front_x,
        receipt=json.dumps(receipt),
    )
    ctrl.event(
        "reach_placements", count=len(rows), considered=len(ranked), file=file.name
    )
    return {}, {}


class PlacementExport:
    def __init__(self, directory, base=None, priors=None):
        self.directory, self.base, self.priors = directory, base, priors

    def select(self, ctrl, name, surface, reg, others, previous):
        chart_dir = self.base
        if chart_dir is None:
            chart_dir = Path(self.directory) / "arm-chart"
            checked_placements(ctrl, name, surface, reg, others, chart_dir)
        return checked_domain(
            ctrl, name, surface, reg, others, self.directory, chart_dir, self.priors
        )


def checked_domain(
    ctrl, name, surface, reg, others, out, base_dir, priors_dir, limit=160
):
    """Check actual shifted palm poses and whole transit before GPU finger synthesis."""
    import hashlib

    from scipy.spatial.transform import Rotation
    from scipy.stats import qmc

    f = f"{name}.{ctrl.side}.npz"
    with np.load(Path(base_dir) / f, allow_pickle=False) as z:
        base = dict(z)
    meta = json.loads(str(base["receipt"]))
    if (
        meta["input_sha256"]
        != ctrl.library[name]["prior"]["lightning_input_sha256"][ctrl.side]
    ):
        raise ValueError("arm chart belongs to a different mesh")
    native = None
    if priors_dir is not None and (Path(priors_dir) / f).is_file():
        prior_path = Path(priors_dir) / f
        with np.load(prior_path, allow_pickle=False) as z:
            prior_meta = json.loads(str(z["receipt"]))
            if (
                prior_meta.get("request_sha256")
                == hashlib.sha256((Path(base_dir) / f).read_bytes()).hexdigest()
            ):
                native = dict(z)
                meta["parent_bank_sha256"] = hashlib.sha256(
                    prior_path.read_bytes()
                ).hexdigest()
    poses = reg @ np.linalg.inv(base["object_pose"])
    transits = {}

    def check_transit(j):
        if j not in transits:
            pre = poses[j] @ base["pre_in_palm"][j]
            opened = base["q_open"][j]
            ctrl.hints[pre.round(5).tobytes()] = base["arm_pre"][j]
            try:
                path = ctrl.move(
                    pre,
                    opened,
                    0.58,
                    others,
                    True,
                    surface,
                    object_margin=0.0005,
                    endpoint_check=lambda arm: ctrl.check_closure_path(
                        "power_wrap", arm, poses[j], poses[j] @ base["tight_in_palm"][j]
                    ),
                    plan_only=True,
                )
                transits[j] = path[-1]
            except NoPath:
                transits[j] = None
        return transits[j]

    def proposals():
        if native is not None:
            for i, (pose, parent) in enumerate(
                zip(native["object_pose"], native["source_id"])
            ):
                yield int(parent), reg @ np.linalg.inv(pose), i
        # Explore a bounded patch in translation and rotation around each
        # observed wall; the arm, fixture and table are checked at each sample.
        points = qmc.Sobol(6, scramble=True, seed=137).random_base2(7)
        for u in points:
            for j, p in enumerate(poses):
                delta = np.r_[-0.020, -0.017, -0.034] + u[:3] * [0.019, 0.034, 0.041]
                rotation = Rotation.from_euler(
                    "xyz", (2 * u[3:] - 1) * [12, 14, 10], degrees=True
                ).as_matrix()
                pose = p.copy()
                pose[:3, :3] = p[:3, :3] @ rotation
                # Pivot the small change near the enclosed mesh rather than at the flange.
                pivot = np.array([0.056, 0, 0.076])
                pose[:3, 3] += p[:3, :3] @ (delta + pivot - rotation @ pivot)
                yield j, pose, -1

    rows = []
    sampled = 0
    keys = (
        "object_pose",
        "pre_in_palm",
        "tight_in_palm",
        "q_wrap_guide",
        "q_open",
        "arm_pre",
        "cost",
        "parent_chart_row",
        "parent_native_row",
        "seated_path",
    )
    for j, palm, source in proposals():
        sampled += 1
        end = check_transit(j)
        if end is None:
            continue
        pre, tight = [poses[j] @ base[k][j] for k in ("pre_in_palm", "tight_in_palm")]
        if not ctrl.fixture_clear(palm) or not ctrl.approach_orientation_clear(palm):
            continue
        try:
            ctrl.check_closure_path("power_wrap", end, palm, tight)
            ctrl.check_closure_path("seated_wrap", end, palm, tight)
        except NoPath:
            continue
        rows.append(
            (
                np.linalg.solve(palm, reg),
                np.linalg.solve(palm, pre),
                np.linalg.solve(palm, tight),
                base["q_warm"][j],
                base["q_open"][j],
                end,
                base["cost"][j],
                j,
                source,
                True,
            )
        )
        if len(rows) >= limit:
            break
    if not rows:
        raise NoPath("no continuous arm motion from park through lift")
    data = dict(zip(keys, map(np.asarray, zip(*rows))))
    meta.update(
        placements=len(rows),
        tested=sampled,
        charts=len(poses),
        mesh_source="registered head RGB-D",
        certification="park transit, pregrasp, seat, grasp, preload, probe and lift; swept forearm and flange",
    )
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(out) / f,
        **data,
        mesh_to_world=reg,
        arm_start=ctrl.arm,
        receipt=json.dumps(meta),
    )
    ctrl.event("reach_placements", count=len(rows), considered=sampled, file=f)
    return {}, {}
