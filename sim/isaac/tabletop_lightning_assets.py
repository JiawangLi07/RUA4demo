"""Export station hand geometry and a registered SAM3D surface in Lightning Grasp's metric frames."""

import argparse
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .paths import RIGHT_JOINTS, asset_path
from .scene_meshes import load_mesh_bundle

PALM = "right_hand_C_MC"


def export_hand(source, out, wrist_mount="removed"):
    from .tabletop_synthesize import station_kinematics

    out.mkdir(parents=True, exist_ok=True)
    io_model = station_kinematics(source, wrist_mount)
    support = io_model.model.hand_support
    tree = ET.parse(asset_path(source, "source", "robot_cad.urdf"))
    links = {link.get("name"): link for link in tree.findall("link")}
    selected = {PALM}
    joints = []
    for _ in range(len(RIGHT_JOINTS) + 3):
        for joint in tree.findall("joint"):
            parent, child = (joint.find(key).get("link") for key in ("parent", "child"))
            if parent in selected and child not in selected:
                selected.add(child)
                joints.append(joint)
    if {j.get("name") for j in joints if j.get("type") != "fixed"} != set(RIGHT_JOINTS):
        raise ValueError("subtree does not match the station's 22 actuators")
    robot = ET.Element("robot", name="sharpa_station_right")
    for name in links:
        if name not in selected:
            continue
        link = ET.SubElement(robot, "link", name=name)
        if name not in support:
            continue
        filename = name + ".stl"
        mesh = trimesh.convex.convex_hull(support[name])
        mesh.export(out / filename)
        for tag in ("visual", "collision"):
            shape = ET.SubElement(ET.SubElement(link, tag), "geometry")
            ET.SubElement(shape, "mesh", filename=filename)
    robot.extend(joints)
    ET.indent(robot)
    ET.ElementTree(robot).write(
        out / "sharpa.urdf", encoding="utf-8", xml_declaration=True
    )
    # Inner faces follow finger flexion: take the instantaneous velocity of the
    # pad under a positive interphalangeal flex at the neutral configuration.
    from .tabletop_meshfit import HandContacts

    contacts = HandContacts(io_model)
    normals = {}
    for d in ("index", "middle", "ring", "pinky"):
        name = f"right_{d}_MP"
        pad = f"right_{d}_elastomer"
        q = np.zeros(len(RIGHT_JOINTS))
        t, _ = contacts.link(name, np.eye(4), q)
        t0, _ = contacts.link(pad, np.eye(4), q)
        q[RIGHT_JOINTS.index(f"right_{d}_PIP")] = 0.005
        t1, _ = contacts.link(pad, np.eye(4), q)
        v = t[:3, :3].T @ (t1[:3, 3] - t0[:3, 3])
        normals[name] = (v / np.linalg.norm(v)).tolist()
    (out / "inner_phalanx_normals.json").write_text(json.dumps(normals, indent=2))
    fixture = io_model.model.fixture_palm
    if fixture:
        from scipy.spatial import ConvexHull

        points = np.vstack([support[PALM], *fixture.values()])
        np.save(out / "palm_and_fixture.npy", points[ConvexHull(points).vertices])
    from pxr import Usd

    stage = Usd.Stage.Open(str(asset_path(source, "source", "scene.usdc")))
    pairs = set()
    for name in selected:
        body = stage.GetPrimAtPath("/World/Robot/" + name)
        if body:
            rel = body.GetRelationship("physics:filteredPairs")
            if rel:
                for target in rel.GetTargets():
                    if target.name in selected:
                        pairs.add(tuple(sorted((name, target.name))))
    (out / "self_collision_exclusions.json").write_text(
        json.dumps(sorted(pairs), indent=2)
    )
    records = {
        key: hashlib.sha256(asset_path(source, "source", key).read_bytes()).hexdigest()
        for key in ("robot_cad.urdf", "scene.usdc")
    }
    records.update(
        station=source,
        wrist_hardware="removed" if wrist_mount == "removed" else "mounted",
        collision_geometry_sha256={
            n: hashlib.sha256((out / (n + ".stl")).read_bytes()).hexdigest()
            for n in sorted(selected.intersection(support))
        },
    )
    (out / "station_provenance.json").write_text(json.dumps(records, indent=2))
    return len(selected), len(selected.intersection(support))


def export_object(prior, out, raw=None):
    bundle = load_mesh_bundle(prior)
    coarse_v, coarse_f, _ = bundle["visual"]
    mesh = trimesh.Trimesh(coarse_v, coarse_f, process=False)
    record = dict(
        prior_sha256=prior["sha256"],
        coarse_faces=len(coarse_f),
        source="registered_visual",
    )
    if raw is not None:
        # Inverse of the original preprocessing: the mesh library uses SAM3D Y-up
        # rotated into station Z-up, centered at the original bounding-box midpoint.
        source = trimesh.load(
            io.BytesIO(raw), file_type="glb", force="mesh", process=False
        )
        hull = np.vstack([v for v, _ in bundle["collisions"]])
        v = np.asarray(source.vertices)[:, [0, 2, 1]] * [1.0, -1.0, 1.0]
        scale = np.ptp(hull, axis=0).max() / np.ptp(v, axis=0).max()
        v = (v - (v.min(0) + v.max(0)) / 2) * scale + (hull.min(0) + hull.max(0)) / 2
        residual = np.quantile(cKDTree(coarse_v).query(v[::4])[0], 0.99)
        hull_residual = np.quantile(cKDTree(v).query(hull)[0], 0.99)
        if (
            hull_residual > 0.0001
            or residual > 0.004
            or np.max(np.abs(np.ptp(v, axis=0) - np.ptp(hull, axis=0))) > 0.001
        ):
            raise ValueError(
                f"raw GLB disagrees with the registered prior, p99={residual:.5f} m"
            )
        mesh = trimesh.Trimesh(v, source.faces, process=False)
        record.update(
            source="dense_main_sam3d",
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            coarse_vertex_nn_p99_m=float(residual),
            hull_vertex_nn_p99_m=float(hull_residual),
            scale=float(scale),
        )
    if mesh.volume < 0:
        mesh.invert()
    out.parent.mkdir(parents=True, exist_ok=True)
    for side in ("right", "left"):
        copy = mesh.copy()
        if side == "left":
            copy.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
        path = out.with_name(out.name + "." + side + ".ply")
        copy.export(path)
    record["faces"] = len(mesh.faces)
    out.with_suffix(".json").write_text(json.dumps(record, indent=2))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--station", default="real2sim_0417")
    p.add_argument(
        "--wrist-mount",
        default="removed",
        help="wrist fixture bundle for reproducing mounted trials",
    )
    p.add_argument("--registry", type=Path)
    p.add_argument("--object")
    p.add_argument("--raw-zip", type=Path)
    p.add_argument("--raw-entry")
    a = p.parse_args()
    if not a.object:
        print(
            "hand links / collider bodies", export_hand(a.station, a.out, a.wrist_mount)
        )
        return
    prior = json.loads(a.registry.read_bytes())["objects"][a.object]["prior"]
    raw = None
    if a.raw_zip:
        with ZipFile(a.raw_zip) as z:
            raw = z.read(a.raw_entry)
    print(export_object(prior, a.out / a.object, raw))


if __name__ == "__main__":
    main()
