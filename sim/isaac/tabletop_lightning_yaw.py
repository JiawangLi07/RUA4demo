"""Re-solve native grasp seeds at mesh-verified rotations around the table normal."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .tabletop_lightning import central_palm_samples
from .tabletop_lightning_generate import prepare


def yaw_candidates(mesh, seeds, palm_patch, step_deg):
    if not len(seeds["object_pose"]):
        return
    vertices, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    tree = cKDTree(vertices)
    cloud = vertices[:: max(1, len(vertices) // 6000)]
    names = seeds["link_names"][seeds["contact_link_id"].clip(min=0)]
    mask = seeds["contact_link_id"] >= 0
    encoded = seeds["object_pose"]
    up = np.linalg.inv(encoded)[:, 2]
    rows = np.flatnonzero(
        (up[:, 1] > 0.18)
        & (up[:, 2] > -0.21)
        & ((names == "right_thumb_elastomer") & mask).any(1)
    )
    for degree in range(step_deg, 360, step_deg):
        yaw = Rotation.from_euler("z", degree, degrees=True).as_matrix()
        deviation = np.maximum(tree.query(cloud @ yaw.T)[0], tree.query(cloud @ yaw)[0])
        if (
            np.quantile(deviation, 0.90) > 0.0045
            or np.quantile(deviation, 0.98) > 0.008
        ):
            continue
        for i in rows:
            rotation = encoded[i, :3, :3] @ yaw
            translation = encoded[i, :3, 3]
            n = int(mask[i].sum())
            points = (seeds["target_pos"][i, :n] - translation) @ rotation
            # v1 project() returns the negative object outer normal to the IK.
            expected_normals = -seeds["target_normal"][i, :n] @ rotation
            dist, index = tree.query(points, k=12)
            match = (normals[index] * expected_normals[:, None]).sum(2)
            cost = dist + 0.005 * (1 - match)
            choice = np.argmin(cost, axis=1)
            jj = np.arange(n)
            if dist[jj, choice].max() > 0.004 or match[jj, choice].min() < 0.72:
                continue
            points, outward = vertices[index[jj, choice]], normals[index[jj, choice]]
            patch = (palm_patch - translation) @ rotation
            gap, near = tree.query(patch)
            k = int(np.argmin(gap))
            palm_matched = bool(
                gap[k] < 0.005
                and np.quantile(gap, 0.15) < 0.009
                and (normals[near[k]] @ rotation.T)[0] < -0.45
            )
            pose = encoded[i].copy()
            pose[:3, :3] = rotation
            yield (
                i,
                degree,
                pose,
                points,
                outward,
                vertices[near[k]],
                normals[near[k]],
                palm_matched,
            )


def solve(api, torch, ctx, mesh, seeds, proposals):
    from lygra.batch_grasp_optimizer import compute_wrench_score_vectorized
    from lygra.pipeline.module.collision import batch_filter_collision

    _, tree, projection, _, hand_mesh, pairs, _, _, _, _, buffer = ctx
    points, _ = mesh.sample_point_and_normal(4096)

    def gpu(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device="cuda")

    check_points = gpu(points)
    results, source_rows, angles = {}, [], []
    for width in range(2, 6):
        group = [p for p in proposals if len(p[3]) == width]
        for off in range(0, len(group), 96):
            chunk = group[off : off + 96]
            ids, degrees, poses, pts, nrm, ppts, pnrm, palm_ok = map(
                np.asarray, zip(*chunk)
            )
            points, normal = gpu(pts), gpu(nrm)
            wrench = compute_wrench_score_vectorized(points, normal)[0]
            with_palm = compute_wrench_score_vectorized(
                torch.cat((points, gpu(ppts)[:, None]), 1),
                torch.cat((normal, gpu(pnrm)[:, None]), 1),
            )[0]
            valid = (
                (
                    (wrench < 0.15)
                    | ((with_palm < 0.15) & torch.as_tensor(palm_ok, device="cuda"))
                )
                .cpu()
                .numpy()
            )
            if not valid.any():
                continue
            ids, degrees, poses = ids[valid], degrees[valid], poses[valid]
            obj = gpu(poses)
            link_ids = torch.as_tensor(
                seeds["contact_link_id"][ids, :width], device="cuda", dtype=torch.int32
            )
            result = api.batch_contact_adjustment(
                tree=tree,
                mesh=projection,
                q_init=gpu(seeds["q"][ids]),
                q_mask=torch.as_tensor(seeds["q_mask"][ids], device="cuda"),
                contact_ids=link_ids,
                contact_link_ids=link_ids,
                contact_pos_in_linkf=gpu(seeds["contact_pos"][ids, :width]),
                contact_normal_in_linkf=gpu(seeds["contact_normal"][ids, :width]),
                target_contact_pos=gpu(pts[valid]) @ obj[:, :3, :3].transpose(1, 2)
                + obj[:, None, :3, 3],
                target_contact_normal=-gpu(nrm[valid]) @ obj[:, :3, :3].transpose(1, 2),
                object_pose=obj,
                gpu_memory_pool=buffer,
                n_iter=5,
            )
            if result is None or not len(result["q"]):
                continue
            clear = batch_filter_collision(
                tree, result["q"], result["object_pose"], check_points, pairs, hand_mesh
            )
            src = {
                p.tobytes(): (int(i), int(deg))
                for p, i, deg in zip(poses, ids, degrees)
            }
            for pose in result["object_pose"][clear].cpu().numpy():
                i, deg = src[pose.tobytes()]
                source_rows.append(i)
                angles.append(deg)
            if not clear.any():
                continue
            for key, val in result.items():
                if not torch.is_tensor(val):
                    continue
                value = val[clear].cpu().numpy()
                if key.startswith(("contact_", "target_")):
                    value = np.pad(
                        value,
                        [(0, 0), (0, 5 - width)]
                        + ([(0, 0)] if value.ndim == 3 else []),
                        constant_values=-1 if key == "contact_link_id" else 0,
                    )
                results.setdefault(key, []).append(value)
    return {k: np.concatenate(a) for k, a in results.items()}, source_rows, angles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("upstream", "hand", "banks", "out"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--meshes", type=Path, nargs="+", required=True)
    p.add_argument("--step-deg", type=int, default=15)
    args = p.parse_args()
    if not 3 <= args.step_deg <= 90:
        p.error("expected a step between 3 and 90 degrees")
    args.field_samples, args.outer, args.inner, args.power_contact = 1000, 96, 1, True
    sys.path.insert(0, str(args.upstream.resolve()))
    import demo as api
    import torch
    import trimesh

    np.random.seed(164)
    torch.manual_seed(164)
    hand_hash = hashlib.sha256(args.hand.read_bytes()).hexdigest()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    ctx = prepare(api, torch, args)
    cad = trimesh.load(args.hand.parent / "right_hand_C_MC.stl", process=True)
    patch = central_palm_samples(np.asarray(cad.vertices))
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.meshes:
        bank = args.banks / (path.stem + ".npz")
        with np.load(bank, allow_pickle=False) as z:
            seeds = dict(z)
        base = json.loads(str(seeds["receipt"]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != base["input_sha256"]:
            raise ValueError("source mesh and native seed bank differ")
        if hand_hash != base["hand_sha256"] or revision != base["upstream"]:
            raise ValueError(
                "hand or Lightning Grasp revision differs from the seed bank"
            )
        start = time.perf_counter()
        mesh = api.MeshObject(str(path))
        proposals = list(yaw_candidates(mesh.mesh, seeds, patch, args.step_deg))
        with torch.no_grad():
            data, rows, yaw = solve(api, torch, ctx, mesh, seeds, proposals)
        data.setdefault("q", np.empty((0, 22), np.float32))
        data.setdefault("object_pose", np.empty((0, 4, 4), np.float32))
        receipt = {
            k: base[k]
            for k in (
                "upstream",
                "input_sha256",
                "hand_sha256",
                "prior_sha256",
                "source_frame",
            )
        }
        receipt.update(
            candidates=len(rows),
            method="mesh_checked_yaw_then_native_contact_adjustment",
            parent_bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
            proposals=len(proposals),
            step_deg=args.step_deg,
            solve_s=time.perf_counter() - start,
        )
        np.savez_compressed(
            args.out / (path.stem + ".npz"),
            **data,
            link_names=seeds["link_names"],
            joint_names=seeds["joint_names"],
            parent_row=rows,
            proposal_yaw_deg=yaw,
            receipt=json.dumps(receipt),
        )
        (args.out / (path.stem + ".json")).write_text(json.dumps(receipt, indent=2))
        print(
            "Saved",
            path.stem,
            len(proposals),
            len(rows),
            round(receipt["solve_s"], 2),
            flush=True,
        )


if __name__ == "__main__":
    main()
