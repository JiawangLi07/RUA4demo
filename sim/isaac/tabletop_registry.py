"""Prepare an anonymous metric mesh registry for the RGB-D tabletop controller."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .scene_meshes import load_mesh_bundle
from .tabletop_catalog import SharedPostures


def import_mesh(path, scale, up):
    import trimesh

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("raw meshes require a positive scale in metres per mesh unit")
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("the file has no triangle surface")
    mesh.update_faces(mesh.nondegenerate_faces() & mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    vis = mesh.visual
    if vis.kind == "texture":
        mesh.visual = vis.to_color()
        vis = mesh.visual
    rgb = np.asarray(vis.face_colors[:, :3], float) / 255
    color = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    mesh.apply_translation(-mesh.bounds.mean(0))
    mesh.apply_scale(scale)
    axes = {
        "x": ("y", -90),
        "-x": ("y", 90),
        "y": ("x", 90),
        "-y": ("x", -90),
        "z": ("x", 0),
        "-z": ("x", 180),
    }
    axis, degrees = axes[up]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    mesh.apply_transform(transform)
    data = {
        "visual_vertices": np.asarray(mesh.vertices),
        "visual_faces": np.asarray(mesh.faces),
        "visual_colors": color,
    }
    # Weld duplicate STL vertices before separating connected components. The visual surface
    # keeps its original triangles and colours. Supply prepared NPZ parts for concavities.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    parts = [
        part
        for part in mesh.split(only_watertight=False)
        if len(part.vertices) >= 4
        and np.linalg.matrix_rank(part.vertices - part.vertices.mean(0)) == 3
    ]
    if not 1 <= len(parts) <= 128:
        raise ValueError(
            "expected 1–128 volumetric components; supply a prepared collision decomposition"
        )
    data["collision_count"] = np.array(len(parts))
    for i, part in enumerate(parts):
        hull = part.convex_hull
        data[f"collision_{i:03d}_vertices"] = np.asarray(hull.vertices)
        data[f"collision_{i:03d}_faces"] = np.asarray(hull.faces)
    return data


def prepare(meshes, catalog, out, scale, up, mass):
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("nominal simulation mass must be positive")
    SharedPostures(catalog)
    out = out.resolve()
    source = "metric_surfaces"
    folder = out / source
    folder.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("RUA_SIM_ASSETS")
    os.environ["RUA_SIM_ASSETS"] = str(out)
    items, actors = {}, []
    try:
        for mesh in meshes:
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz") as tmp:
                if mesh.suffix.lower() == ".npz":
                    shutil.copyfile(mesh, tmp.name)
                else:
                    if scale is None:
                        raise ValueError("pass --meters-per-unit for raw meshes")
                    np.savez_compressed(tmp, **import_mesh(mesh, scale, up))
                    tmp.flush()
                digest = hashlib.sha256(Path(tmp.name).read_bytes()).hexdigest()
                key = "part_" + digest[:16]
                spec = {"source": source, "file": Path(tmp.name).name, "sha256": digest}
                load_mesh_bundle(spec)
                spec["file"] = key + ".npz"
                if key in items:
                    raise ValueError("duplicate geometry in this registry")
                shutil.copyfile(tmp.name, folder / spec["file"])
            items[key] = {"prior": spec}
            actors.append(
                {
                    "name": key,
                    "prim_path": "/World/SkillObjects/" + key,
                    "kind": "mesh",
                    "mesh": spec,
                    "quaternion_wxyz": [1, 0, 0, 0],
                    "position": [0, 0, 0],
                    "mass_kg": mass,
                    "friction": {"static": 0.6, "dynamic": 0.6, "restitution": 0},
                }
            )
    finally:
        if previous is None:
            os.environ.pop("RUA_SIM_ASSETS", None)
        else:
            os.environ["RUA_SIM_ASSETS"] = previous
    hand_file = out / "shared_postures.json.gz"
    if catalog.resolve() != hand_file:
        shutil.copyfile(catalog, hand_file)
    manifest = {
        "schema": "rua.tabletop-geometries.v1",
        "shared_postures": hand_file.name,
        "objects": items,
    }
    (out / "registry.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "objects.json").write_text(json.dumps(actors, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mesh",
        action="append",
        type=Path,
        required=True,
        help="metric NPZ or raw GLB/OBJ/STL",
    )
    p.add_argument(
        "--catalog", type=Path, required=True, help="exported shared hand catalog"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--meters-per-unit", type=float, help="calibrated scale for all raw inputs"
    )
    p.add_argument(
        "--up-axis",
        choices=("x", "-x", "y", "-y", "z", "-z"),
        default="z",
        help="raw mesh axis pointing up in its nominal resting orientation",
    )
    p.add_argument(
        "--mass-kg",
        type=float,
        default=0.080,
        help="nominal simulation mass for each rigid body",
    )
    a = p.parse_args()
    prepare(a.mesh, a.catalog, a.out, a.meters_per_unit, a.up_axis, a.mass_kg)
