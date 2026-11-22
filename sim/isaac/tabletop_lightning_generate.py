"""Run Lightning Grasp's public contact-field, IK and collision kernels on the station hand."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .paths import RIGHT_JOINTS

PALM = "right_hand_C_MC"
DIGITS = ("thumb", "index", "middle", "ring", "pinky")


def native_collision_mask(flags, count):
    """v1 returns integer 0/1 flags, one flag per native pose."""
    if tuple(flags.shape) != (count,):
        raise ValueError("collision kernel returned a different number of rows")
    return flags.bool()


def prepare(api, torch, args):
    from lygra.contact_field import build_contact_field
    from lygra.robot.base import RobotInterface

    class Sharpa(RobotInterface):
        def get_active_joints(self):
            return RIGHT_JOINTS

        def get_base_link(self):
            return PALM

        def get_static_links(self):
            return [PALM]

        def get_white_list_pairs(self):
            path = args.hand.parent / "self_collision_exclusions.json"
            return json.loads(path.read_bytes())

        def get_mesh_scale(self):
            return 1.0

        def get_canonical_space(self):
            return np.array([0.024, -0.03, 0.045]), np.array([0.08, 0.03, 0.11])

        def get_contact_field_config(self):
            movable = {f"right_{d}_elastomer": {"disabled_normal": []} for d in DIGITS}
            if args.power_contact:
                for link, normal in json.loads(
                    (args.hand.parent / "inner_phalanx_normals.json").read_bytes()
                ).items():
                    movable[link] = {
                        "disabled_normal": [(-np.asarray(normal), np.pi * 0.46)]
                    }
            return dict(
                type="v1",
                movable_link=movable,
                static_link={
                    PALM: {"allowed_normal": [(np.array([[1.0, 0.0, 0.0]]), np.pi / 4)]}
                },
            )

    robot = Sharpa(str(args.hand.resolve()))
    tree = robot.get_kinematics_tree()
    config = robot.get_contact_field_config()
    projection = api.get_urdf_mesh_for_projection(robot.urdf_path, tree, config)
    static = api.get_urdf_mesh_decomposed(robot.urdf_path, tree, [PALM])
    geometry = api.get_urdf_mesh_decomposed(robot.urdf_path, tree)
    pairs = tree.get_self_collision_check_link_pairs(
        geometry["link_body_id"], whitelist_pairs=robot.get_white_list_pairs()
    )
    pairs = torch.as_tensor(pairs, device="cuda", dtype=torch.int32)
    field = build_contact_field(
        robot.urdf_path, config, patch_pos_dist_tol=0.01, patch_rot_dist_tol=np.pi / 5
    )
    field.generate_contact_field(tree, n_iter=args.field_samples)
    accel = field.generate_acceleration_structure(method="lbvhs2")
    dependency = api.get_link_dependency_matrix(
        field, tree.get_dependency_sets([PALM])
    ).cuda()
    parent = torch.tensor(
        [tree.get_link_id(n) for n in field.get_all_parent_link_names()], device="cuda"
    )
    buffer = api.IKGPUBufferPool(
        n_dof=tree.n_dof(),
        n_link=tree.n_link(),
        n_actuated_dof=tree.n_actuated_dof(),
        max_batch=args.outer * args.inner,
        retry=10,
    )
    return (
        robot,
        tree,
        projection,
        static,
        geometry,
        pairs,
        field,
        accel,
        dependency,
        parent,
        buffer,
    )


def side_placements(torch, field, tree, static, pts, normals, obj, args):
    """Seed the native contact search at measured side walls with the radial side up."""
    from lygra.pipeline.module.collision import batch_object_hand_collision_check

    def unit(x):
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-7)

    def basis(x, up, twist):
        y = unit(up - (x * up).sum(1, keepdim=True) * x)
        z = torch.linalg.cross(x, y)
        y = torch.cos(twist) * y + torch.sin(twist) * z
        return torch.stack((x, y, torch.linalg.cross(x, y)), dim=-1)

    low, high = obj.mesh.vertices[:, 2].min(), obj.mesh.vertices[:, 2].max()
    rel_z = (pts[:, 2] - low) / max(high - low, 1e-6)
    wall = torch.where((normals[:, 2].abs() < 0.45) & (rel_z > 0.23) & (rel_z < 0.75))[
        0
    ]
    if not len(wall):
        return pts.new_empty((0, 4, 4)), {}
    n = args.outer * 3
    ix = wall[torch.randint(len(wall), (n,), device=pts.device)]
    site = torch.as_tensor(
        field.sample_static_contact_geometry(n), device=pts.device, dtype=pts.dtype
    )
    hand = basis(unit(site[:, 3:]), pts.new_tensor([0, 1, 0]), pts.new_zeros((n, 1)))
    twist = (torch.rand((n, 1), device=pts.device) - 0.5) * 1.40
    body = basis(-unit(normals[ix]), pts.new_tensor([0, 0, 1]), twist)
    rot = hand @ body.transpose(1, 2)
    # A clearance of a few millimetres accommodates scan noise. The runtime
    # independently rechecks the palm patch and controls the final seating travel.
    stand_off = torch.rand((n, 1), device=pts.device) * 0.0045
    pos = (
        site[:, :3] + stand_off * unit(site[:, 3:]) - (rot @ pts[ix, :, None])[:, :, 0]
    )
    pose = torch.eye(4, device=pts.device, dtype=pts.dtype).repeat(n, 1, 1)
    pose[:, :3, :3], pose[:, :3, 3] = rot, pos
    inv = torch.linalg.inv(pose)
    mount = torch.as_tensor(args.fixture, device=pts.device, dtype=pts.dtype)
    clear = (inv[:, 2, :3] @ mount.T).amin(1) + inv[:, 2, 3] > low + 0.004
    clear &= (inv[:, 2, 1] >= 0.25) & (inv[:, 2, 2] >= -0.15)
    idx = torch.where(clear)[0]
    if len(idx):
        points = pts @ rot[idx].transpose(1, 2) + pos[idx, None]
        no_collision = batch_object_hand_collision_check(tree, points, mesh=static)
        idx = idx[native_collision_mask(no_collision, len(idx))][: args.outer]
    return pose[idx], {
        "extra_contact_pos": (site[:, :3] + stand_off * unit(site[:, 3:]))[idx, None],
        "extra_contact_normal": site[idx, None, 3:],
        "extra_contact_mask": torch.ones(
            (len(idx), 1), device=pts.device, dtype=pts.dtype
        ),
    }


def sample_placements(api, torch, robot, field, tree, static, pts, normals, obj, args):
    if args.side_placements:
        return side_placements(torch, field, tree, static, pts, normals, obj, args)
    proposal_args = api.get_object_pose_sampling_args("canonical", robot)
    if not args.tabletop:
        return api.sample_object_pose(
            args.outer, pts, normals, field, tree, static, proposal_args
        )
    from scipy.spatial.transform import Rotation

    fixture = torch.as_tensor(args.fixture, dtype=pts.dtype, device=pts.device)
    bottom = float(obj.mesh.vertices[:, 2].min())
    poses, conditions = [], {}
    # Changing coordinates before the upstream normal-alignment sampler randomizes
    # its otherwise fixed tangent direction. Undo that change before contact search.
    for _ in range(12):
        chart = torch.eye(4, device=pts.device, dtype=pts.dtype)
        chart[:3, :3] = torch.as_tensor(
            Rotation.random().as_matrix(), device=pts.device, dtype=pts.dtype
        )
        sample, cond = api.sample_object_pose(
            args.outer,
            pts @ chart[:3, :3].T,
            normals @ chart[:3, :3].T,
            field,
            tree,
            static,
            proposal_args,
            use_static_prob=1.0
            if args.upright_wrap
            else (0.8 if args.power_contact else 0.5),
        )
        sample = sample @ chart
        inverse = torch.linalg.inv(sample)
        floor = (inverse[:, 2, :3] @ fixture.T).amin(1) + inverse[:, 2, 3]
        good = floor > bottom + 0.004
        if args.upright_wrap:
            # In the canonical Sharpa frame +Y runs from the little finger to
            # the index, +Z runs from the wrist to the knuckles. The mirrored
            # left mesh uses the same canonical hand; table height is unchanged.
            good &= (inverse[:, 2, 1] >= 0.25) & (inverse[:, 2, 2] >= -0.15)
        if not good.any():
            continue
        poses.append(sample[good])
        for k, v in cond.items():
            conditions.setdefault(k, []).append(v[good])
        if sum(map(len, poses)) >= args.outer:
            break
    if not poses:
        return pts.new_empty((0, 4, 4)), {}
    return torch.cat(poses)[: args.outer], {
        k: torch.cat(v)[: args.outer] for k, v in conditions.items()
    }


def palm_condition_in_object_frame(poses, condition):
    # The placement sampler reports a point and outward normal on the palm.
    # Native contact search evaluates object-frame points. Its v1 project()
    # negates the object's outer normals, yielding directions into the object.
    # The outward palm normal already points into the object at contact.
    r = poses[:, :3, :3]
    return dict(
        condition,
        extra_contact_pos=(condition["extra_contact_pos"] - poses[:, None, :3, 3]) @ r,
        extra_contact_normal=condition["extra_contact_normal"] @ r,
    )


def batch(api, torch, ctx, obj, args, ncontact, placement_sampler=sample_placements):
    (
        robot,
        tree,
        projection,
        static,
        geometry,
        pairs,
        field,
        accel,
        dependency,
        parent,
        buffer,
    ) = ctx
    pts, normals = obj.sample_point_and_normal(args.points)
    all_pts, all_normals = [
        torch.as_tensor(x, device="cuda", dtype=torch.float32) for x in (pts, normals)
    ]
    valid = api.get_support_point_mask(all_pts, all_normals, [0.01])[0]
    if getattr(args, "room", None) is not None:
        from .tabletop_plane_sampling import placements

        n = all_pts.new_tensor(args.room["normal"])
        valid &= (all_pts @ n + args.room["offset"] > 0.004) & (all_normals @ n > -0.42)
        placement_sampler = placements
    pts, normals = all_pts[valid], all_normals[valid]
    poses, condition = placement_sampler(
        api, torch, robot, field, tree, static, pts, normals, obj, args
    )
    log = {"placements": len(poses), "requested_digits": ncontact}
    if not len(poses):
        return {}, log
    if "extra_contact_pos" in condition:
        condition = palm_condition_in_object_frame(poses, condition)
    hits = api.batch_object_all_contact_fields_interaction(pts, normals, poses, accel)
    hit = (hits >= 0).int()
    try:
        domain = api.sample_pose_and_contact_from_interaction(
            n_contact=ncontact,
            interaction_matrix=field.reduce_link_interaction(hit),
            dependency_matrix=dependency,
            object_points=pts,
            object_normals=normals,
            object_poses=poses,
            condition=condition,
        )
    except SystemExit as error:
        # Upstream v1 exits with -1 if no independent set of two links exists.
        if error.code != -1:
            raise
        return {}, dict(log, domains=0)
    except NameError as error:
        # The pinned v1 contact_collection also calls sys.exit on an empty
        # independent set without importing sys. Preserve other NameErrors.
        if error.name != "sys":
            raise
        return {}, dict(log, domains=0)
    pos, normal, idx, poses, link, condition, outer = domain
    log.update(domains=len(poses), actual_digits=link.shape[-1])
    if not len(poses):
        return {}, log
    goals, directions, point_idx, poses, links, outer_ids = api.search_contact_point(
        contact_domain_pos=pos,
        contact_domain_normal=normal,
        contact_domain_point_idx=idx,
        object_poses=poses,
        contact_ids=link,
        batch_size=args.inner,
        return_hand_frame=True,
        condition=condition,
        zo_lr=5 * (obj.get_area() / args.points) ** 0.5,
    )
    if not len(goals):
        return {}, dict(log, optimized_contacts=0)
    contact, local = field.sample_contact_ids(
        interaction_matrix=hit[outer],
        interaction_matrix_hand_point_idx=hits[outer],
        target_batch_outer_ids=outer_ids,
        target_contact_link_ids=links,
        target_contact_point_idx=point_idx,
    )
    anchors, axes = field.sample_contact_geometry(contact, local)
    result = api.batch_ik(
        tree=tree,
        contact_ids=contact,
        contact_parent_ids=parent,
        contact_pos_in_linkf=anchors.float(),
        contact_normal_in_linkf=axes.float(),
        target_contact_pos=goals.float(),
        target_contact_normal=directions.float(),
        object_pose=poses.float(),
        gpu_memory_pool=buffer,
    )
    log["ik"] = len(result["q"])
    if not log["ik"]:
        return {}, log
    result = api.batch_contact_adjustment(
        tree=tree,
        mesh=projection,
        q_init=result["q"],
        q_mask=result["q_mask"],
        contact_ids=contact,
        contact_link_ids=result["contact_link_id"],
        contact_pos_in_linkf=result["contact_pos"],
        contact_normal_in_linkf=result["contact_normal"],
        target_contact_pos=result["target_pos"],
        target_contact_normal=result["target_normal"],
        object_pose=result["object_pose"],
        n_iter=5,
        gpu_memory_pool=buffer,
        ret_mesh_buffer=True,
    )
    if result is None or not len(result["q"]):
        return {}, log
    collision_points = all_pts
    if getattr(args, "room", None) is not None:
        from .tabletop_plane_sampling import tabletop_points

        collision_points = torch.cat(
            (all_pts, tabletop_points(torch, all_pts, args.room)), dim=0
        )
    result = api.batch_assign_free_finger_and_filter(
        tree=tree,
        result=result,
        object_point=collision_points,
        self_collision_link_pairs=pairs,
        decomposed_mesh_data=geometry,
    )
    log["collision_filtered"] = len(result["q"])
    if not log["collision_filtered"]:
        return {}, log
    return {
        k: v.detach().cpu().numpy()
        for k, v in result.items()
        if isinstance(v, torch.Tensor)
    }, log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--hand", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meshes", type=Path, nargs="+", required=True)
    p.add_argument("--field-samples", type=int, default=10000)
    p.add_argument("--outer", type=int, default=192)
    p.add_argument("--inner", type=int, default=48)
    p.add_argument("--points", type=int, default=2048)
    p.add_argument("--batches", type=int, default=6)
    p.add_argument("--seed", type=int, default=36)
    p.add_argument(
        "--power-contact",
        action="store_true",
        help="include inner middle phalanges in the official contact field",
    )
    p.add_argument(
        "--tabletop",
        action="store_true",
        help="sample diverse contact tangents above the registered mesh support plane",
    )
    p.add_argument("--contacts", nargs="+", type=int, default=[5, 4])
    p.add_argument(
        "--upright-wrap",
        action="store_true",
        help="use static palm contact on every placement and sample thumb-side-up grasps",
    )
    p.add_argument(
        "--side-placements",
        action="store_true",
        help="seed the native solver from upright registered side-wall contacts",
    )
    p.add_argument(
        "--table-patch-prior",
        type=Path,
        help="optional NPZ of surface patches for plane-conditioned wrist proposals",
    )
    p.add_argument(
        "--registered-planes",
        type=Path,
        help="head-registered support planes indexed by mesh stem",
    )
    p.add_argument(
        "--placement-priors",
        type=Path,
        help="native object-to-palm poses for each mesh stem",
    )
    a = p.parse_args()
    if (a.registered_planes is None) != (a.placement_priors is None):
        p.error("--registered-planes and --placement-priors must be supplied together")
    if a.registered_planes and not a.tabletop:
        p.error("--registered-planes requires --tabletop")
    if a.table_patch_prior and not a.registered_planes:
        p.error("--table-patch-prior requires --registered-planes")
    a.patch_prior = None
    if a.table_patch_prior:
        with np.load(a.table_patch_prior) as patches:
            a.patch_prior = dict(patches)
    if a.side_placements and not a.upright_wrap:
        p.error("--side-placements needs --upright-wrap")
    if a.upright_wrap and not (a.tabletop and a.power_contact):
        p.error("--upright-wrap needs --tabletop and --power-contact")
    if a.tabletop:
        a.fixture = np.load(a.hand.parent / "palm_and_fixture.npy", allow_pickle=False)
    sys.path.insert(0, str(a.upstream.resolve()))
    import demo as api
    import torch

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    t = time.perf_counter()
    ctx = prepare(api, torch, a)
    torch.cuda.synchronize()
    prepare_s = time.perf_counter() - t
    print("Prepared native Sharpa field", round(prepare_s, 2), flush=True)
    sha = subprocess.check_output(
        ["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    a.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for path in a.meshes:
            if a.registered_planes:
                a.room = json.loads(a.registered_planes.read_bytes())[path.stem]
                with np.load(a.placement_priors / (path.stem + ".npz")) as prior:
                    a.parents = prior["object_pose"]
            obj = api.MeshObject(str(path))
            attempts = []
            arrays = {}
            start = time.perf_counter()
            for j in range(a.batches):
                data, counts = batch(
                    api, torch, ctx, obj, a, a.contacts[j % len(a.contacts)]
                )
                attempts.append(counts)
                print(path.stem, j, counts, flush=True)
                for key, v in data.items():
                    if key in (
                        "target_pos",
                        "target_normal",
                        "contact_pos",
                        "contact_normal",
                        "contact_link_id",
                    ):
                        extra = 5 - v.shape[1]
                        if extra < 0:
                            raise ValueError("unexpected contact count")
                        v = np.pad(
                            v,
                            [(0, 0), (0, extra)] + ([(0, 0)] if v.ndim == 3 else []),
                            constant_values=-1 if key == "contact_link_id" else 0,
                        )
                    arrays.setdefault(key, []).append(v)
            torch.cuda.synchronize()
            values = {k: np.concatenate(v) for k, v in arrays.items()}
            values.setdefault("q", np.empty((0, len(RIGHT_JOINTS)), np.float32))
            values.setdefault("object_pose", np.empty((0, 4, 4), np.float32))
            receipt = dict(
                upstream=sha,
                seed=a.seed,
                device=torch.cuda.get_device_name(),
                hand_sha256=hashlib.sha256(a.hand.read_bytes()).hexdigest(),
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                prior_sha256=json.loads(
                    path.with_name(path.name.split(".")[0] + ".json").read_text()
                )["prior_sha256"],
                field_samples=a.field_samples,
                field_s=prepare_s,
                solve_s=time.perf_counter() - start,
                source_frame="object_to_palm",
                outer=a.outer,
                inner=a.inner,
                points=a.points,
                tabletop=a.tabletop,
                power_contact=a.power_contact,
                static_contact_frame="object_inward",
                upright_wrap=a.upright_wrap,
                side_placements=a.side_placements,
                candidates=len(values["q"]),
                batches=attempts,
            )
            if a.registered_planes:
                receipt["head_registered_table"] = a.room
                if a.table_patch_prior:
                    receipt["table_patch_prior_sha256"] = hashlib.sha256(
                        a.table_patch_prior.read_bytes()
                    ).hexdigest()
                receipt["placement_parent_sha256"] = hashlib.sha256(
                    (a.placement_priors / (path.stem + ".npz")).read_bytes()
                ).hexdigest()
            provenance = a.hand.parent / "station_provenance.json"
            if provenance.exists():
                receipt["station"] = json.loads(provenance.read_bytes())
            np.savez_compressed(
                a.out / (path.stem + ".npz"),
                **values,
                link_names=ctx[1].get_all_link_names(),
                joint_names=RIGHT_JOINTS,
                receipt=json.dumps(receipt),
            )
            (a.out / (path.stem + ".json")).write_text(json.dumps(receipt, indent=2))
            print(
                "Saved",
                path.stem,
                len(values["q"]),
                round(receipt["solve_s"], 3),
                flush=True,
            )


if __name__ == "__main__":
    main()
