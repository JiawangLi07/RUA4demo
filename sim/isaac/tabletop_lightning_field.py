"""Run Lightning Grasp's CUDA contact-field solver only at certified station arm placements."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .paths import RIGHT_JOINTS
from .tabletop_lightning import central_palm_samples
from .tabletop_lightning_generate import batch, native_collision_mask, prepare


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pad_contacts(values):
    for k, v in values.items():
        if k.startswith(("target_", "contact_")):
            values[k] = np.pad(
                v,
                [(0, 0), (0, 5 - v.shape[1])] + ([(0, 0)] if v.ndim == 3 else []),
                constant_values=-1 if k == "contact_link_id" else 0,
            )
    return values


def neighbor_contacts(poses, seeds, vertices, normals, tree, patch):
    """Transfer native link-local seeds between similar mesh cross sections."""

    def shaft(transforms):
        up = transforms[:, :3, 2]
        pos = transforms[:, :3, 3]
        return up, pos - (pos * up).sum(1)[:, None] * up

    prior_axis, prior_center = shaft(seeds["object_pose"])
    axes, centers = shaft(poses)
    ids = seeds["contact_link_id"]
    thumb = (
        (seeds["link_names"][ids.clip(min=0)] == "right_thumb_elastomer") & (ids >= 0)
    ).any(1)
    for j, (pose, axis, center) in enumerate(zip(poses, axes, centers)):
        r, t = pose[:3, :3], pose[:3, 3]
        gap, near = tree.query((patch - t) @ r)
        ii = int(gap.argmin())
        palm_ok = (
            gap[ii] < 0.005
            and np.quantile(gap, 0.15) < 0.009
            and (normals[near[ii]] @ r.T)[0] < -0.4
        )
        distance = np.linalg.norm(
            prior_center - center, axis=1
        ) + 0.06 * np.linalg.norm(prior_axis - axis, axis=1)
        distance[~thumb] = np.inf
        for row in np.argsort(distance)[:20]:
            if distance[row] > 0.040:
                break
            width = (ids[row] >= 0).sum()
            if width < 3:
                continue
            pred = (seeds["target_pos"][row, :width] - t) @ r
            outward = -seeds["target_normal"][row, :width] @ r
            dist, nbr = tree.query(pred, k=32)
            cos = (normals[nbr] * outward[:, None]).sum(2)
            best = (dist + 0.006 * (1 - cos)).argmin(1)
            ix = np.arange(width)
            if dist[ix, best].max() > 0.012 or cos[ix, best].min() < 0.55:
                continue
            m = nbr[ix, best]
            yield (
                row,
                j,
                pose,
                vertices[m],
                normals[m],
                vertices[near[ii]],
                normals[near[ii]],
                palm_ok,
            )


def solve_domain(api, torch, ctx, obj, chart, patch, args, prior, neighbors):
    from lygra.pipeline.module.collision import (
        batch_filter_collision,
        batch_object_hand_collision_check,
    )
    from lygra.utils.transform_utils import batch_object_transform

    _, tree, proj, _, bodies, pairs, *_, pool = ctx
    row_ids = np.flatnonzero(
        chart.get("seated_path", np.ones(len(chart["object_pose"]), bool))
    )
    if not len(row_ids):
        raise ValueError("no arm placement reaches the seated wrap and lift")
    chart = {
        key: chart[key][row_ids]
        for key in (
            "object_pose",
            "arm_pre",
            "q_open",
            "q_wrap_guide",
            "pre_in_palm",
            "tight_in_palm",
            "cost",
            "parent_native_row",
        )
    }
    poses = chart["object_pose"].astype(np.float32)
    rng = np.random.default_rng(args.seed)

    def gpu(x):
        return torch.as_tensor(
            np.ascontiguousarray(x), dtype=torch.float32, device="cuda"
        )

    verts = np.asarray(obj.mesh.vertices)
    normals = np.asarray(obj.mesh.vertex_normals)
    index = cKDTree(verts)
    inv = np.linalg.inv(poses)
    locals_ = patch @ inv[:, :3, :3].transpose(0, 2, 1) + inv[:, None, :3, 3]
    gaps, near = index.query(locals_)
    point = gaps.argmin(1)
    palm_norm = -np.einsum(
        "nij,nj->ni", poses[:, :3, :3], normals[near[np.arange(len(poses)), point]]
    )
    palm_site = patch[point]
    lookup = {p.tobytes(): i for i, p in enumerate(poses)}
    parts = {}
    logs = []

    def append(result, ids=None):
        result = pad_contacts(result)
        if ids is None:
            ids = np.array([lookup[p.tobytes()] for p in result["object_pose"]])
        else:
            ids = np.asarray(ids, dtype=int)
        for k in (
            "arm_pre",
            "q_open",
            "q_wrap_guide",
            "pre_in_palm",
            "tight_in_palm",
            "cost",
        ):
            result[k] = chart[k][ids]
        result["source_id"] = row_ids[ids]
        result["standoff_m"] = np.linalg.norm(result["tight_in_palm"][:, :3, 3], axis=1)
        result["warm_delta_rad"] = np.max(
            np.abs(result["q"] - chart["q_wrap_guide"][ids]), axis=1
        )
        for k, v in result.items():
            parts.setdefault(k, []).append(v)

    # A warm restart still requires native contact IK and both collision filters.
    if prior is not None:
        check = gpu(obj.sample_point_and_normal(4096)[0])
        for width in (3, 4, 5):
            r = np.flatnonzero(chart["parent_native_row"] >= 0)
            ix = chart["parent_native_row"][r]
            r = r[(prior["contact_link_id"][ix] >= 0).sum(1) == width]
            if not len(r):
                continue
            ix = chart["parent_native_row"][r]
            old = prior["object_pose"][ix]
            objgoals = (prior["target_pos"][ix, :width] - old[:, None, :3, 3]) @ old[
                :, :3, :3
            ]
            normals_obj = prior["target_normal"][ix, :width] @ old[:, :3, :3]
            o = gpu(poses[r])
            link = torch.as_tensor(
                prior["contact_link_id"][ix, :width], device="cuda", dtype=torch.int32
            )
            g = gpu(objgoals) @ o[:, :3, :3].transpose(1, 2) + o[:, None, :3, 3]
            n = gpu(normals_obj) @ o[:, :3, :3].transpose(1, 2)
            result = api.batch_contact_adjustment(
                tree=tree,
                mesh=proj,
                q_init=gpu(prior["q"][ix]),
                q_mask=torch.as_tensor(prior["q_mask"][ix], device="cuda"),
                contact_ids=link,
                contact_link_ids=link,
                contact_pos_in_linkf=gpu(prior["contact_pos"][ix, :width]),
                contact_normal_in_linkf=gpu(prior["contact_normal"][ix, :width]),
                target_contact_pos=g,
                target_contact_normal=n,
                object_pose=o,
                gpu_memory_pool=pool,
                n_iter=5,
            )
            if result is not None and len(result["q"]):
                mask = native_collision_mask(
                    batch_filter_collision(
                        tree, result["q"], result["object_pose"], check, pairs, bodies
                    ),
                    len(result["q"]),
                )
                if mask.any():
                    append({k: v[mask].cpu().numpy() for k, v in result.items()})
    if neighbors is not None:
        from .tabletop_lightning_yaw import solve as native_refine

        proposals = list(
            neighbor_contacts(poses, neighbors, verts, normals, index, patch)
        )
        result, _, parent_rows = native_refine(
            api, torch, ctx, obj, neighbors, proposals
        )
        if result:
            append(result, parent_rows)
        logs.append(
            {
                "native_neighbor_proposals": len(proposals),
                "collision_filtered": len(result.get("q", [])),
            }
        )
        print("native neighbors", logs[-1], flush=True)

    def at_certified_poses(
        api, torch, robot, field, tree, static, pts, nrm, obj, settings
    ):
        rows = rng.integers(0, len(poses), size=settings.outer)
        o = gpu(poses[rows])
        flags = batch_object_hand_collision_check(
            tree, batch_object_transform(o, pts)["pos"], mesh=static
        )
        valid = native_collision_mask(flags, len(rows))
        r = rows[valid.cpu().numpy()]
        if not len(r):
            return o[:0], {}
        return o[valid], {
            "extra_contact_pos": gpu(palm_site[r, None]),
            "extra_contact_normal": gpu(palm_norm[r, None]),
            "extra_contact_mask": gpu((gaps[r, point[r]] < 0.005)[:, None]),
        }

    for iteration in range(args.batches):
        n = (5, 4, 4, 3)[iteration % 4]
        result, log = batch(
            api, torch, ctx, obj, args, n, placement_sampler=at_certified_poses
        )
        logs.append(log)
        if result:
            append(result)
        if (iteration + 1) % 12 == 0:
            total = sum(len(v) for v in parts.get("q", []))
            print("native field", iteration + 1, total, flush=True)
            if total >= args.target_rows:
                break
    return {k: np.concatenate(v) for k, v in parts.items()}, logs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("upstream", "hand", "domain", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--priors", type=Path)
    p.add_argument(
        "--contact-priors",
        type=Path,
        help="optional native seeds for both canonical hand charts of the same object",
    )
    p.add_argument("--meshes", nargs="+", type=Path, required=True)
    p.add_argument("--batches", type=int, default=120)
    p.add_argument("--target-rows", type=int, default=48)
    p.add_argument("--seed", type=int, default=2209)
    args = p.parse_args()
    if args.batches < 0 or args.target_rows <= 0:
        p.error("expected nonnegative batches and a positive number of target rows")
    sys.path.insert(0, str(args.upstream.resolve()))
    import demo as api
    import torch
    import trimesh

    args.outer, args.inner, args.field_samples, args.points, args.power_contact = (
        192,
        40,
        14000,
        3072,
        True,
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ctx = prepare(api, torch, args)
    patch = central_palm_samples(
        trimesh.load(args.hand.parent / "right_hand_C_MC.stl").vertices
    )
    revision = subprocess.check_output(
        ["git", "-C", args.upstream, "rev-parse", "HEAD"], text=True
    ).strip()
    args.out.mkdir(exist_ok=True, parents=True)
    for mesh in args.meshes:
        start = time.perf_counter()
        file = args.domain / (mesh.stem + ".npz")
        with np.load(file) as z:
            chart = dict(z)
        rec = json.loads(str(chart["receipt"]))
        if rec["input_sha256"] != digest(mesh) or not rec.get("certification"):
            raise ValueError("expected matching metric mesh and arm-checked placements")
        prior = None
        if args.priors is not None and "parent_bank_sha256" in rec:
            parent = args.priors / file.name
            if digest(parent) != rec["parent_bank_sha256"]:
                raise ValueError("prior bank checksum mismatch")
            with np.load(parent) as z:
                prior = dict(z)
            pre = json.loads(str(prior["receipt"]))
            if pre["hand_sha256"] != digest(args.hand) or pre["upstream"] != revision:
                raise ValueError("prior hand or revision mismatch")
        neighbors = None
        if args.contact_priors is not None:
            arrays = []
            receipts = {}
            for side in ("right", "left"):
                path = args.contact_priors / (rec["target"] + "." + side + ".npz")
                if not path.is_file():
                    continue
                with np.load(path) as z:
                    if not len(z["q"]):
                        continue
                    provenance = json.loads(str(z["receipt"]))
                    if (
                        provenance["hand_sha256"] != digest(args.hand)
                        or provenance["upstream"] != revision
                        or provenance["prior_sha256"] != rec["prior_sha256"]
                    ):
                        raise ValueError(
                            "neighbor priors need the same hand, upstream and metric object"
                        )
                    if not np.array_equal(z["link_names"], ctx[1].get_all_link_names()):
                        raise ValueError("native link order differs")
                    arrays.append(
                        {
                            k: z[k]
                            for k in (
                                "object_pose",
                                "q",
                                "q_mask",
                                "contact_link_id",
                                "target_pos",
                                "target_normal",
                                "contact_pos",
                                "contact_normal",
                            )
                        }
                    )
                    receipts[side] = digest(path)
            if arrays:
                neighbors = {
                    k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]
                }
                neighbors["link_names"] = np.asarray(ctx[1].get_all_link_names())
            rec["contact_prior_sha256"] = receipts
        obj = api.MeshObject(str(mesh))
        rng = np.random.default_rng(args.seed)

        def sample(count, obj=obj, rng=rng):
            pts, indices = trimesh.sample.sample_surface(obj.mesh, count, seed=rng)
            return pts, obj.mesh.face_normals[indices]

        obj.sample_point_and_normal = sample
        with torch.no_grad():
            data, logs = solve_domain(
                api, torch, ctx, obj, chart, patch, args, prior, neighbors
            )
        data.setdefault("q", np.empty((0, 22), np.float32))
        data.setdefault("object_pose", np.empty((0, 4, 4), np.float32))
        rec.update(
            method="arm_chart_native_contact_field",
            upstream=revision,
            hand_sha256=digest(args.hand),
            request_sha256=digest(file),
            seed=args.seed,
            candidates=len(data["q"]),
            batches=logs,
            solve_s=time.perf_counter() - start,
        )
        np.savez_compressed(
            args.out / file.name,
            **data,
            joint_names=RIGHT_JOINTS,
            link_names=ctx[1].get_all_link_names(),
            candidate_source=np.full(len(data["q"]), "reach_conditioned"),
            receipt=json.dumps(rec),
        )
        (args.out / (mesh.stem + ".json")).write_text(json.dumps(rec, indent=2) + "\n")
        print("Saved", mesh.stem, rec["candidates"], flush=True)


if __name__ == "__main__":
    main()
