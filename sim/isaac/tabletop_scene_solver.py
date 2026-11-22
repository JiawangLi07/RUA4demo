"""Solve a small set of reachable placements on a registered object using native Lightning Grasp."""

import argparse
import fcntl
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tabletop_lightning_generate import native_collision_mask, prepare

MAX_SURFACE_GAP_M = 0.009
MIN_NORMAL_DOT = 0.35
NORMAL_COST_M = 0.005


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_surface_patches(tree, normals, sites, outward):
    """Match each contact to a nearby patch with the requested outward normal."""
    distances, indices = tree.query(sites, k=min(24, tree.n))
    dots = (normals[indices] * outward[:, None]).sum(-1)
    choice = np.argmin(distances + NORMAL_COST_M * (1 - dots), axis=1)
    rows = np.arange(len(sites))
    matched = indices[rows, choice]
    gaps = distances[rows, choice]
    alignment = dots[rows, choice]
    recovered = 0
    # On a dense mesh the nearest 24 vertices may all lie on the back of a
    # thin wall. Search the full physical tolerance before rejecting a site.
    for i in np.flatnonzero((gaps > MAX_SURFACE_GAP_M) | (alignment < MIN_NORMAL_DOT)):
        near = np.asarray(tree.query_ball_point(sites[i], MAX_SURFACE_GAP_M), dtype=int)
        patch_dot = normals[near] @ outward[i]
        compatible = patch_dot >= MIN_NORMAL_DOT
        if not compatible.any():
            continue
        near, patch_dot = near[compatible], patch_dot[compatible]
        gap = np.linalg.norm(tree.data[near] - sites[i], axis=1)
        k = np.argmin(gap + NORMAL_COST_M * (1 - patch_dot))
        matched[i], gaps[i], alignment[i] = near[k], gap[k], patch_dot[k]
        recovered += 1
    return matched, gaps, alignment, recovered


def solve_request(folder, args, ctx, api, torch, collision_ik=None):
    import trimesh
    from lygra.pipeline.module.collision import batch_filter_collision

    begin = time.monotonic()
    with np.load(folder / "request.npz") as z:
        req = dict(z)
    rec = json.loads(str(req["receipt"]))
    stem = f"{rec['target']}.{rec['side']}"
    mesh_file, bank = args.meshes / (stem + ".ply"), args.priors / (stem + ".npz")
    sha = digest(folder / "request.npz")
    if not (1 <= len(req["parent"]) == rec["issued"] <= rec["limit"] <= 4):
        raise ValueError("invalid native proposal budget")
    if (
        digest(bank) != rec["parent_bank_sha256"]
        or digest(mesh_file) != rec["input_sha256"]
    ):
        raise ValueError("native prior or input mesh checksum mismatch")
    with np.load(bank) as loaded:
        seed = dict(loaded)
    parent = json.loads(str(seed["receipt"]))
    if (
        rec["upstream"] != args.revision
        or digest(args.hand) != rec["hand_sha256"]
        or any(
            parent[k] != rec[k]
            for k in ("upstream", "input_sha256", "prior_sha256", "hand_sha256")
        )
        or not np.array_equal(seed["link_names"], ctx[1].get_all_link_names())
    ):
        raise ValueError("native solver provenance mismatch")
    draw_seed = rec.get("sampling_seed", int(sha[:8], 16))
    np.random.seed(draw_seed)
    torch.manual_seed(draw_seed)
    mesh = trimesh.load(mesh_file, force="mesh", process=False)
    verts, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    if len(verts) < 3 or not len(mesh.faces):
        raise ValueError("contact fitting requires a triangle mesh")
    tree = cKDTree(verts)
    observed = req["observed_local"] if "observed_local" in req else req["head_local"]
    depth = cKDTree(observed) if len(observed) >= 3 else None
    points = trimesh.sample.sample_surface(mesh, 6144)[0]

    def gpu(value):
        return torch.as_tensor(
            np.ascontiguousarray(value), dtype=torch.float32, device="cuda"
        )

    samples = gpu(points)
    out = {}
    kept, rows = [], []
    warm = {k[5:]: v for k, v in req.items() if k.startswith("warm_")} or seed
    for j, parent_id in enumerate(req["parent"]):
        idx = j if "warm_q" in req else int(parent_id)
        pose = req["object_pose"][j]
        active = np.flatnonzero(warm["contact_link_id"][idx] >= 0)
        # The original link-local sites initialize the native joint iteration.
        # Their new object-surface targets follow the proposed wrist and the head scan.
        pred = (warm["target_pos"][idx, active] - pose[:3, 3]) @ pose[:3, :3]
        outward = -warm["target_normal"][idx, active] @ pose[:3, :3]
        near, gaps, alignment, recovered = match_surface_patches(
            tree, normals, pred, outward
        )
        row = {
            "parent": int(parent_id),
            "surface_gap_m": float(gaps.max()),
            "min_normal_dot": float(alignment.min()),
            "radius_recovered_sites": recovered,
        }
        rows.append(row)
        if gaps.max() > MAX_SURFACE_GAP_M or alignment.min() < MIN_NORMAL_DOT:
            row["stage"] = "surface_projection"
            continue
        # Contact sites from the parent already lie on its triangles. Retain
        # subvertex accuracy for near-identity transfers, and reproject rotated
        # sites onto the tangent plane of the matching patch of the dense mesh.
        n = normals[near].copy()
        target = pred - ((pred - verts[near]) * n).sum(1)[:, None] * n
        if depth is not None:
            dp, ip = depth.query(target, k=3)
            at = dp[:, -1] < 0.008
            offset = ((observed[ip].mean(1) - target) * n).sum(1)
            target[at] += n[at] * (0.35 * np.clip(offset[at], -0.003, 0.003))[:, None]
            row[rec.get("observation_source", "head") + "_corrected_sites"] = int(at.sum())
        n = n * 0.25 + 0.75 * outward
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        link = torch.as_tensor(
            warm["contact_link_id"][idx, active][None], device="cuda", dtype=torch.int32
        )
        result = api.batch_contact_adjustment(
            tree=ctx[1],
            mesh=ctx[2],
            q_init=gpu(warm["q"][idx : idx + 1]),
            q_mask=torch.as_tensor(warm["q_mask"][idx : idx + 1], device="cuda"),
            contact_ids=link,
            contact_link_ids=link,
            contact_pos_in_linkf=gpu(warm["contact_pos"][idx, active][None]),
            contact_normal_in_linkf=gpu(warm["contact_normal"][idx, active][None]),
            target_contact_pos=gpu((target @ pose[:3, :3].T + pose[:3, 3])[None]),
            target_contact_normal=gpu((-n @ pose[:3, :3].T)[None]),
            object_pose=gpu(pose[None]),
            gpu_memory_pool=ctx[-1],
            n_iter=5,
        )
        if result is None or not len(result["q"]):
            row["stage"] = "native_contact_ik"
            continue
        # The native contact IK zeros independent, unrequested chains. Reuse
        # their arm-checked initialization, then collision-check the entire hand.
        result["q"] = torch.where(
            result["q_mask"].bool(), result["q"], gpu(warm["q"][idx : idx + 1])
        )
        flags = batch_filter_collision(
            ctx[1], result["q"], result["object_pose"], samples, ctx[5], ctx[4]
        )
        if (
            not native_collision_mask(flags, 1).item()
            and not result["q_mask"].all().item()
        ):
            result["q"] = torch.where(
                result["q_mask"].bool(), result["q"], gpu(req["q_open"][j : j + 1])
            )
            flags = batch_filter_collision(
                ctx[1], result["q"], result["object_pose"], samples, ctx[5], ctx[4]
            )
            row["free_joints_from_pregrasp"] = True
        if collision_ik is not None and not native_collision_mask(flags, 1).item():
            world_from_palm = req["mesh_to_world"] @ np.linalg.inv(pose)
            repaired, row["collision_refit"] = collision_ik.refine(
                result, samples, world_from_palm, float(req["table_m"])
            )
            if repaired:
                flags = batch_filter_collision(
                    ctx[1], result["q"], result["object_pose"], samples, ctx[5], ctx[4]
                )
        if not native_collision_mask(flags, 1).item():
            row["stage"] = "native_collision"
            continue
        q = result["q"][0].detach().cpu().numpy()
        row.update(
            stage="accepted",
            max_joint_change_rad=float(np.max(np.abs(q - warm["q"][idx]))),
        )
        for key in (
            "q",
            "q_mask",
            "object_pose",
            "target_pos",
            "target_normal",
            "contact_pos",
            "contact_normal",
            "contact_link_id",
        ):
            v = result[key][0].detach().cpu().numpy()
            if key.startswith(("target_", "contact_")):
                v = np.pad(
                    v,
                    [(0, 5 - len(v))] + ([(0, 0)] if v.ndim == 2 else []),
                    constant_values=-1 if key == "contact_link_id" else 0,
                )
            out.setdefault(key, []).append(v)
        kept.append(j)
    result = {
        "request_sha256": sha,
        "candidates": len(kept),
        "issued": len(rows),
        "rows": rows,
        "gpu": torch.cuda.get_device_name(),
        "upstream": args.revision,
        "collision_aware_ik": collision_ik is not None,
        "seconds": time.monotonic() - begin,
    }
    if kept:
        receipt = dict(
            rec,
            method="live_scene_contact_refit",
            collision_aware_ik=collision_ik is not None,
            source_frame="object_to_palm",
            request_sha256=sha,
        )
        file = folder / (stem + ".npz")
        np.savez_compressed(
            file,
            **out,
            **{k: req[k][kept] for k in ("pre_in_palm", "arm_pre", "q_open")},
            candidate_source=np.where(
                req["parent"][kept] < 0,
                "live_section_contact_refit",
                "live_scene_contact_refit",
            ),
            source_id=np.where(
                req["parent"][kept] >= 0, req["parent"][kept], -1 - np.asarray(kept)
            ),
            joint_names=seed["joint_names"],
            link_names=seed["link_names"],
            receipt=json.dumps(receipt),
        )
        result["bank_sha256"] = digest(file)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("upstream", "hand", "meshes", "priors", "queue"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--collision-aware-ik",
        action="store_true",
        help="Refit collision-rejected candidates using the contact and separation objectives of Mana algorithm 1.",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    import demo as api
    import torch

    args.revision = subprocess.check_output(
        ["git", "-C", args.upstream, "rev-parse", "HEAD"], text=True
    ).strip()
    args.power_contact, args.field_samples, args.outer, args.inner = True, 1000, 6, 2
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    ctx = prepare(api, torch, args)
    collision_ik = None
    if args.collision_aware_ik:
        from .tabletop_collision_ik import CollisionAwareIK

        collision_ik = CollisionAwareIK(ctx, torch)
    print("ready for scene requests", flush=True)
    while True:
        pending = sorted(
            p.parent
            for p in args.queue.rglob("request.npz")
            if not (p.parent / "result.json").exists()
        )
        for folder in pending:
            with (folder / "request.npz").open("rb") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if (folder / "result.json").exists():
                    continue
                try:
                    result = solve_request(folder, args, ctx, api, torch, collision_ik)
                except Exception as exc:  # noqa: BLE001 - report worker errors to the waiting episode.
                    import traceback

                    traceback.print_exc()
                    result = {
                        "error": str(exc),
                        "request_sha256": digest(folder / "request.npz"),
                    }
                tmp = folder / "result.partial"
                tmp.write_text(json.dumps(result, indent=2))
                tmp.replace(folder / "result.json")
                print(
                    folder,
                    result.get("candidates"),
                    result.get("error", ""),
                    flush=True,
                )
        if args.once:
            break
        time.sleep(0.25)


if __name__ == "__main__":
    main()
