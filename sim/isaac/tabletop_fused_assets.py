"""Prepare metre-scale, table-aligned fused SAM3D scans for native physics and Lightning Grasp."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import trimesh


def hull_mesh(points):
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    mapping = np.empty(len(points), dtype=np.int64)
    mapping[hull.vertices] = np.arange(len(vertices))
    faces = mapping[hull.simplices]
    a, b, c = vertices[faces].transpose(1, 0, 2)
    reverse = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0
    faces[reverse] = faces[reverse, ::-1]
    return vertices.astype(np.float32), faces.astype(np.int32)


def prepare(path, record, out, mass=None):
    source = trimesh.load(path, force="mesh", process=False)
    raw = np.array(source.vertices, dtype=float)
    if len(raw) != record["vertices"] or len(source.faces) != record["faces"]:
        raise ValueError(
            f"mesh topology disagrees with evaluation record for {path.name}"
        )
    if not np.allclose(
        np.ptp(raw, axis=0) * 1000, record["extents_table_axes_mm"], atol=0.03, rtol=0
    ):
        raise ValueError(
            f"metric scale disagrees with evaluation record for {path.name}"
        )
    # The GLB is already local and metric. The reference camera transform in the
    # capture report applies to raw camera points, not to these GLB vertices.
    transform = np.eye(4)
    transform[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    transform[:3, 3] = -transform[:3, :3] @ ((raw.max(0) + raw.min(0)) / 2)
    v = raw @ transform[:3, :3].T + transform[:3, 3]
    f = np.asarray(source.faces).copy()
    if trimesh.Trimesh(v, f, process=False).volume < 0:
        f = f[:, ::-1]
    srgb = np.asarray(source.visual.vertex_colors)[:, :3].astype(float) / 255
    cell = max(0.0011, np.ptp(v, axis=0).max() / 90)
    _, idx = np.unique(np.floor(v / cell).astype(int), axis=0, return_inverse=True)
    n = np.bincount(idx)

    def mean(x):
        return np.column_stack([np.bincount(idx, weights=a) / n for a in x.T])

    vertices = mean(v)
    faces = idx[f]
    tri = vertices[faces]
    faces = faces[
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        > 1e-12
    ]
    _, selected = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(selected)]
    color = mean(srgb)[faces].mean(1)
    color = np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)
    data = dict(
        visual_vertices=vertices.astype(np.float32),
        visual_faces=faces.astype(np.int32),
        visual_colors=color.astype(np.float32),
    )
    # Intersect the surface with overlapping axial slabs for elongated parts.
    # Short round objects use one hull. Sharp concavities remain approximations.
    vv, axes = np.linalg.eigh(np.cov(v.T))
    along = v @ axes[:, -1]
    ratio = np.sqrt(vv[-1] / max(vv[-2], 1e-12))
    parts = int(np.clip(np.ceil(ratio * 2), 3, 10)) if ratio > 1.55 else 1
    edges = np.linspace(along.min(), along.max(), parts + 1)
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        points = v[(along >= lo - 0.0007) & (along <= hi + 0.0007)]
        data[f"collision_{i:03d}_vertices"], data[f"collision_{i:03d}_faces"] = (
            hull_mesh(points)
        )
    data["collision_count"] = np.array(parts)
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    key = "fused_" + raw_sha[:9]
    file = out / "fused_metric" / (key + ".npz")
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file, **data)
    prior = dict(
        source="fused_metric",
        file=file.name,
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
    )
    dense = out / "dense"
    dense.mkdir(exist_ok=True)
    metric_mesh = trimesh.Trimesh(v, f, process=False)
    for side in ("right", "left"):
        obj = metric_mesh.copy()
        if side == "left":
            obj.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
        obj.export(dense / f"{key}.{side}.ply")
    prior["lightning_input_sha256"] = {
        side: hashlib.sha256((dense / f"{key}.{side}.ply").read_bytes()).hexdigest()
        for side in ("right", "left")
    }
    if mass is None:
        mass = float(np.clip(abs(metric_mesh.volume) * 550, 0.007, 0.32))
        mass_source = "assumed 550 kg/m3 bounded to 7–320 g"
    else:
        mass_source = "previous simulation assumption for matching object"
    receipt = dict(
        prior_sha256=prior["sha256"],
        raw_sha256=raw_sha,
        source="fused_metric_glb",
        faces=len(f),
        coarse_faces=len(faces),
        raw_to_object=transform.tolist(),
        dimensions_m=np.ptp(v, axis=0).tolist(),
        hulls=parts,
        views_in_offline_fusion=record["views"],
        fused_head_depth_p50_mm=record["fused_check"]["head"]["depth_p50_mm"],
        mass_kg=mass,
        mass_source=mass_source,
    )
    (dense / (key + ".json")).write_text(json.dumps(receipt, indent=2))
    physical = dict(
        name=path.stem,
        kind="mesh",
        mesh=prior,
        mass_kg=mass,
        prim_path="/World/SkillObjects/" + path.stem,
        position=[0, 0, 0],
        quaternion_wxyz=[1, 0, 0, 0],
        friction={"static": 0.6, "dynamic": 0.6, "restitution": 0},
    )
    return key, prior, physical, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--mass-reference", type=Path, help="previous scene JSON with added_objects"
    )
    a = p.parse_args()
    evaluation = json.loads((a.input / "evaluation.json").read_bytes())
    rows = evaluation["objects_metrics"]
    if len({r["id"] for r in rows}) != evaluation["objects"]:
        raise ValueError("incomplete or nonunique fused scan manifest")
    masses = {}
    if a.mass_reference:
        masses = {
            Path(o["mesh"]["file"]).stem: o["mass_kg"]
            for o in json.loads(a.mass_reference.read_bytes())["added_objects"]
        }
    entries, objects, aliases, receipts = {}, {}, {}, {}
    for r in sorted(
        rows, key=lambda r: (r["id"] not in ("bottle", "driver", "tube"), r["id"])
    ):
        name = r["id"]
        if Path(name).name != name:
            raise ValueError("invalid scan ID")
        k, prior, physical, info = prepare(
            a.input / (name + ".glb"), r, a.out, masses.get(name)
        )
        entries[k] = {"prior": prior}
        objects[name] = physical
        aliases[k] = name
        receipts[name] = info
        print(name, k, info["faces"], flush=True)
    manifest = {
        "schema": "rua.tabletop-geometries.v1",
        "shared_postures": "shared_postures.json.gz",
        "objects": entries,
    }
    for name, data in (
        ("registry", manifest),
        ("objects", objects),
        ("aliases", aliases),
        ("geometry", receipts),
    ):
        (a.out / (name + ".json")).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
