"""Build a reproducible simultaneous tabletop scene. The layout is simulator-side only."""

import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .scene_meshes import load_mesh_bundle

PARK_ARMS = {
    "L": [
        -1.290963836,
        1.299714337,
        1.527006662,
        -1.882286094,
        -1.840669591,
        -0.088832653,
        -1.538807605,
    ],
    "R": [
        1.377736218,
        1.222414972,
        -1.691926096,
        -1.842133491,
        1.881570539,
        -0.155392425,
        1.545743866,
    ],
}


def separate(a, b, gap):
    for poly in (a, b):
        edges = np.roll(poly, 1, axis=0) - poly
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        x, y = a @ normals.T, b @ normals.T
        if np.any((x.max(0) + gap < y.min(0)) | (y.max(0) + gap < x.min(0))):
            return True
    return False


def layout(objects, seed, table_z=0.7165):
    rng = np.random.default_rng(seed)
    prepared = []
    for spec in objects:
        colliders = load_mesh_bundle(spec["mesh"])["collisions"]
        vertices = np.vstack([v for v, _ in colliders])
        rest = Rotation.from_quat(np.roll(spec["quaternion_wxyz"], -1))
        placed = rest.apply(vertices)
        footprint = placed[ConvexHull(placed[:, :2]).vertices, :2]
        prepared.append(
            (
                np.prod(np.ptp(footprint, axis=0)),
                spec,
                rest,
                footprint,
                placed[:, 2].min(),
            )
        )
    # Sample without replacement inside the measured tabletop and the head camera's field of view.
    low, high = np.array([0.505, -0.52]), np.array([0.905, 0.19])
    for _ in range(20):
        all_polys, items = [], []
        for _, spec, rest, footprint, zmin in sorted(prepared, key=lambda row: -row[0]):
            for _ in range(600):
                yaw = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi))
                poly = footprint @ yaw.as_matrix()[:2, :2].T
                xy = rng.uniform(low - poly.min(0), high - poly.max(0))
                poly = poly + xy
                if all(separate(poly, other, 0.012) for other in all_polys):
                    obj = copy.deepcopy(spec)
                    obj["position"] = [*xy, table_z - zmin + 0.001]
                    obj["quaternion_wxyz"] = np.roll((yaw * rest).as_quat(), 1).tolist()
                    all_polys.append(poly)
                    items.append(obj)
                    break
            else:
                break
        if len(items) == len(objects):
            return items
    raise ValueError(
        "could not pack this tabletop without intersecting initial footprints"
    )


def make_scene(base, objects, seed):
    result = copy.deepcopy(base)
    items = layout(objects, seed)
    result["added_objects"] = items
    result["object_prims"] = {o["name"]: o["prim_path"] for o in items}
    result["assignment"] = dict.fromkeys(("left", "right"), items[0]["name"])
    result["tabletop_seed"] = seed
    # In the station frame +X points toward the table, +Y toward the left arm.
    # The named postures put the palms at x=.24, y=+/-.31, z=.95 with fingers up.
    result["initial_joint_positions"] = {
        f"Joint{i}_{side}": angle
        for side, angles in PARK_ARMS.items()
        for i, angle in enumerate(angles, 1)
    }
    result["head_wrist_validation"] = {
        "camera_prim": "/World/Robot/TCP_Link_R/wrist_camera"
    }
    eye, target = np.array([1.66, -0.43, 1.47]), np.array([0.49, -0.02, 0.91])
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    rotation = Rotation.from_matrix(np.column_stack((x, np.cross(z, x), z)))
    result["images"]["observer"] = {
        "position_m": eye.tolist(),
        "rotation_wxyz_ros": np.roll(rotation.as_quat(), 1).tolist(),
        "focal_length_mm": 18.0,
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--aliases", type=Path, help="registry key to display name for offline scoring"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    objects = json.loads(args.objects.read_bytes())
    config = make_scene(
        json.loads(args.base.read_bytes()),
        list(objects.values()) if isinstance(objects, dict) else objects,
        args.seed,
    )
    if args.aliases:
        config["evaluation_aliases"] = json.loads(args.aliases.read_bytes())
    args.out.write_text(json.dumps(config, indent=2) + "\n")
