"""Collision-aware contact refit for a fixed, reachable Sharpa palm (Mana, Alg. 1)."""

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


class ConvexBody:
    def __init__(self, vertices, link):
        hull = ConvexHull(vertices)
        self.vertices = vertices[hull.vertices]
        self.planes = np.unique(hull.equations.round(9), axis=0)
        self.center = self.vertices.mean(axis=0)
        self.radius = np.linalg.norm(self.vertices - self.center, axis=1).max()
        self.link = link

    def penetration(self, local):
        """Center and mean outward separation for a coherent patch of inside points."""
        if not len(local):
            return None
        margin = local @ self.planes[:, :3].T + self.planes[:, 3]
        face = margin.argmax(axis=1)
        depths = -margin[np.arange(len(local)), face]
        inside = depths > 0
        if not inside.any():
            return None
        deepest = depths.argmax()
        normals = self.planes[face, :3]
        patch = (
            inside
            & (normals @ normals[deepest] > 0.7)
            & (depths >= 0.5 * depths[deepest])
        )
        direction = np.average(normals[patch], weights=depths[patch], axis=0)
        direction /= np.linalg.norm(direction)
        return local[patch].mean(axis=0), direction, depths[deepest]


def point_jacobian(jac, offset):
    """Velocity Jacobian at a point displaced from the link origin, in the palm frame."""
    return jac[:3] + np.cross(jac[3:].T, offset).T


def separate_bodies(a, b, vertices_a, vertices_b, poses):
    """Face-axis translation of A relative to B for overlapping convex bodies."""
    axes = np.vstack(
        [body.planes[:, :3] @ poses[body.link, :3, :3].T for body in (a, b)]
    )
    proj_a, proj_b = vertices_a @ axes.T, vertices_b @ axes.T
    plus = proj_b.max(axis=0) - proj_a.min(axis=0)
    minus = proj_a.max(axis=0) - proj_b.min(axis=0)
    depths = np.minimum(plus, minus)
    j = depths.argmin()
    if depths[j] <= 0:
        return None
    normal = axes[j] if plus[j] < minus[j] else -axes[j]
    x, y = vertices_a @ normal, vertices_b @ normal
    center = (
        vertices_a[x < x.min() + 0.0005].mean(axis=0)
        + vertices_b[y > y.max() - 0.0005].mean(axis=0)
    ) / 2
    return a.link, b.link, center, normal * (depths[j] + 0.0006)


def constraints(bodies, poses, obj, self_pairs, table_plane, intersect=None):
    """Return relative point displacements in palm coordinates, with residual weights."""
    rows = []
    world = {}
    bounds = {}
    for i, body in enumerate(bodies):
        if body is None:
            continue
        R, t = poses[body.link, :3, :3], poses[body.link, :3, 3]
        world[i] = body.vertices @ R.T + t
        bounds[i] = (world[i].min(axis=0), world[i].max(axis=0))
        if obj is not None:
            near = obj.query_ball_point(R @ body.center + t, body.radius)
            hit = body.penetration((obj.data[near] - t) @ R)
            if hit is not None:
                p, normal, depth = hit
                # Object samples lie inside the hand. Move the hand away from the samples.
                rows.append(
                    (body.link, None, R @ p + t, -(R @ normal) * (depth + 0.0006), 3.0)
                )
        normal, origin = table_plane
        heights = world[i] @ normal + origin
        deepest = heights.min()
        # Light contact with the support is allowed by the tabletop task.
        if deepest < -0.001:
            p = world[i][heights < deepest + 0.0005].mean(axis=0)
            rows.append((body.link, None, p, -deepest * normal, 0.35))
    for a, b in self_pairs:
        if a not in world or b not in world:
            continue
        if np.any(bounds[a][1] < bounds[b][0]) or np.any(bounds[b][1] < bounds[a][0]):
            continue
        if intersect is not None and not intersect(a, b):
            continue
        correction = separate_bodies(bodies[a], bodies[b], world[a], world[b], poses)
        if correction is not None:
            rows.append((*correction, 3.0))
    return rows


class CollisionAwareIK:
    def __init__(self, ctx, torch):
        self.tree, mesh, self.geometry, self.pairs = ctx[1], ctx[2], ctx[4], ctx[5]
        self.pair_ids = self.pairs.tolist()
        self.vertices = torch.as_tensor(
            self.geometry["v"], device="cuda", dtype=torch.float32
        )
        self.offsets = torch.as_tensor(
            self.geometry["vi"], device="cuda", dtype=torch.int32
        )
        self.projection = [
            torch.as_tensor(mesh[k], device="cuda", dtype=dtype)
            for k, dtype in (
                ("v", torch.float32),
                ("vi", torch.int32),
                ("f", torch.int32),
                ("n", torch.float32),
                ("fi", torch.int32),
            )
        ]
        self.bodies = []
        for i, link in enumerate(self.geometry["body_id_to_link_id"]):
            verts = self.geometry["v"][
                self.geometry["vi"][i] : self.geometry["vi"][i + 1]
            ]
            self.bodies.append(ConvexBody(verts, link) if len(verts) else None)
        self.limits = self.tree.get_actuated_joint_limit()

    def refine(self, result, samples, world_from_palm, table_m, max_steps=30):
        from lygra import gem
        from lygra.kinematics import batch_jacobian
        from lygra.pipeline.module.collision import (
            batch_hand_self_collision_check,
            batch_object_hand_collision_check,
        )
        from lygra.utils.transform_utils import batch_object_transform

        q = result["q"].clone()
        obj_tensor = batch_object_transform(result["object_pose"], samples)["pos"]
        obj = cKDTree(obj_tensor[0].cpu().numpy())
        normal = world_from_palm[2, :3]
        plane = (normal, world_from_palm[2, 3] - table_m)
        links = result["contact_link_id"]
        ids = links[0].tolist()
        targets, normals = (
            result[k][0].cpu().numpy() for k in ("target_pos", "target_normal")
        )
        report = {}
        for step in range(max_steps + 1):
            lin = batch_jacobian(self.tree, q, ret_fk=True)
            fk = lin["fk"]["link"]
            poses = fk[0].cpu().numpy()
            jac = lin["J"][0].reshape(len(poses), 6, -1).cpu().numpy()
            sites = gem.batch_solve_contact_with_normal(
                fk,
                links.int(),
                *self.projection,
                result["target_pos"],
                result["target_normal"],
                True,
                0.01,
            )
            local = sites[0].cpu().numpy()
            error, design = [], []
            for i, link in enumerate(ids):
                R, t = poses[link, :3, :3], poses[link, :3, 3]
                # Reproduce the native IK's point and normal residual.
                for length in (0.0, 0.01):
                    r = R @ (local[i, :3] + length * local[i, 3:])
                    error.append(targets[i] + length * normals[i] - (t + r))
                    design.append(point_jacobian(jac[link], r))
            site_error = np.reshape(error, (len(ids), 6))
            worst = float(np.sqrt((site_error**2).sum(axis=1).max() / 2))
            self_ok, _, _ = batch_hand_self_collision_check(
                self.tree, q, self.pairs, mesh=self.geometry
            )
            obj_ok = bool(
                batch_object_hand_collision_check(
                    self.tree, obj_tensor, q=q, mesh=self.geometry
                ).item()
            )
            self_collision = not bool(self_ok.item())
            body_fk = fk[:, self.geometry["body_id_to_link_id"]]

            def intersect(a, b, body_fk=body_fk):
                pair = self.pairs.new_tensor([[a, b]])
                return (
                    gem.batch_link_mesh_gjk(body_fk, self.vertices, self.offsets, pair)
                    .any()
                    .item()
                )

            free = obj_ok and not self_collision
            # Once clear, use the remaining tolerance to recover contact accuracy.
            # The native filters are checked again after every such step.
            pushes = (
                []
                if free
                else constraints(
                    self.bodies,
                    poses,
                    None if obj_ok else obj,
                    self.pair_ids if self_collision else (),
                    plane,
                    intersect,
                )
            )
            status = {
                "hand_object_collision": not obj_ok,
                "self_collision": self_collision,
                "correction_sites": len(pushes),
                "max_contact_error_m": worst,
            }
            if step == 0:
                report["initial"] = status
            report.update(iterations=step, final=status)
            if free:
                report["best_clear_contact_error_m"] = min(
                    worst, report.get("best_clear_contact_error_m", worst)
                )
            if free and worst < 0.004:
                result.update(
                    q=q, contact_pos=sites[:, :, :3], contact_normal=sites[:, :, 3:]
                )
                return True, report
            if step == max_steps or (not pushes and worst < 0.004):
                break
            for link, other, p, d, weight in pushes:
                J = point_jacobian(jac[link], p - poses[link, :3, 3])
                if other is not None:
                    J = J - point_jacobian(jac[other], p - poses[other, :3, 3])
                # Only the separating velocity is constrained; sliding along the patch is free.
                distance = np.linalg.norm(d)
                direction = d / distance
                design.append(weight * (direction @ J)[None])
                error.append(np.array([weight * distance]))
            A, b = np.vstack(design), np.concatenate(error)
            dq = np.linalg.solve(A.T @ A + np.eye(q.shape[1]) * 2e-5, A.T @ b)
            step_limit = 0.04 if free else 0.14
            dq *= min(0.8, step_limit / max(np.max(np.abs(dq)), 1e-12))
            values = np.clip(q[0].cpu().numpy() + dq, *self.limits)
            if np.max(np.abs(values - q[0].cpu().numpy())) < 1e-6:
                break
            q = q.new_tensor(values[None])
        return False, report
