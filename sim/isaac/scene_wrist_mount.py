"""Extract the registered wrist fixture from the station CAD and attach it to both TCPs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from .paths import asset_path

REGISTRATION = Path(__file__).parent / "config" / "wrist_mount" / "registration.json"


def prepare(cad, output):
    import rhino3dm as rh

    reg = json.loads(REGISTRATION.read_bytes())
    if hashlib.sha256(cad.read_bytes()).hexdigest() != reg["input_sha256"]:
        raise ValueError("the mount registration belongs to a different Rhino assembly")
    model = rh.File3dm.Read(str(cad))
    if model.Settings.ModelUnitSystem != rh.UnitSystem.Millimeters:
        raise ValueError("expected millimetre CAD")
    transform = np.asarray(reg["right_tcp_from_cad_m"])
    data = {}
    for name in ("bracket", "housing"):
        body = next(
            o for o in model.Objects if str(o.Attributes.Id) == reg[name + "_guid"]
        )
        vertices, triangles = [], []
        for face in body.Geometry.Faces:
            mesh = face.GetMesh(rh.MeshType.Render)
            offset = len(vertices)
            vertices.extend((p.X, p.Y, p.Z) for p in mesh.Vertices)
            for a, b, c, d in mesh.Faces:
                triangles.append((a + offset, b + offset, c + offset))
                if c != d:
                    triangles.append((a + offset, c + offset, d + offset))
        v = 0.001 * np.asarray(vertices) @ transform[:3, :3].T + transform[:3, 3]
        f = np.asarray(triangles)
        data[name + "_vertices"] = v.astype(np.float32)
        data[name + "_faces"] = f.astype(np.int32)
        # Clip real triangles before convexifying each short section. A single hull
        # would fill the space between the mounting foot and camera head.
        axis = np.eye(3)[np.argmax(np.ptp(v, axis=0))]
        levels = np.linspace((v @ axis).min() - 1e-6, (v @ axis).max() + 1e-6, 7)
        edges = np.unique(
            np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
            axis=0,
        )
        a, b = v[edges].transpose(1, 0, 2)
        u, w = a @ axis, b @ axis
        for i, (lo, hi) in enumerate(zip(levels, levels[1:])):
            clipped = [v[((v @ axis) >= lo) & ((v @ axis) <= hi)]]
            for level in (lo, hi):
                cut = (u - level) * (w - level) < 0
                fraction = (level - u[cut]) / (w[cut] - u[cut])
                clipped.append(a[cut] + fraction[:, None] * (b[cut] - a[cut]))
            points = np.unique(np.round(np.vstack(clipped), 7), axis=0)
            if len(points) >= 4 and np.linalg.matrix_rank(points - points.mean(0)) == 3:
                hull = ConvexHull(points)
                data[f"hull_{name}_{i}"] = points[hull.vertices].astype(np.float32)
    data["cad_sha256"] = np.asarray(reg["input_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)


def resolve_mount(path=None):
    if path == "removed":
        return None
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    try:
        return asset_path("wrist_mount", "mount.npz")
    except FileNotFoundError:
        return None


def mount_parts(path, side):
    mirror = np.array([1.0, -1.0 if side == "left" else 1.0, 1.0])
    with np.load(path, allow_pickle=False) as z:
        if (
            str(z["cad_sha256"])
            != json.loads(REGISTRATION.read_bytes())["input_sha256"]
        ):
            raise ValueError("unknown wrist fixture registration")
        return {
            k: np.asarray(z[k], dtype=float) * mirror
            for k in z.files
            if k.startswith("hull_")
        }


def palm_parts(path, palm_from_tcp, side="right"):
    if path is None:
        return {}
    return {
        name: v @ palm_from_tcp[:3, :3].T + palm_from_tcp[:3, 3]
        for name, v in mount_parts(path, side).items()
    }


def remove_wrist_hardware(stage, robot):
    """Remove native camera prims and camera-specific geometry before PhysX starts."""
    from pxr import Usd, UsdGeom, UsdPhysics

    report = {}
    for side, suffix in (("right", "R"), ("left", "L")):
        tcp_path = f"{robot}/TCP_Link_{suffix}"
        tcp = stage.GetPrimAtPath(tcp_path)
        if not tcp:
            raise ValueError(f"missing wrist TCP {tcp_path}")
        paths = [str(x.GetPath()) for x in Usd.PrimRange(tcp) if x.IsA(UsdGeom.Camera)]
        paths.append(tcp_path + "/WristCameraFixture")
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if prim:
                stage.RemovePrim(path)
                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsActive():
                    prim.SetActive(False)
        active = list(Usd.PrimRange(tcp, Usd.TraverseInstanceProxies()))
        cameras = [str(x.GetPath()) for x in active if x.IsA(UsdGeom.Camera)]
        fixture = [x for x in active if "WristCameraFixture" in str(x.GetPath())]
        if cameras or fixture:
            raise RuntimeError(f"wrist hardware remains under {tcp_path}")
        report[side] = {
            "active_wrist_cameras": cameras,
            "fixture_prims": len(fixture),
            "fixture_colliders": sum(
                x.HasAPI(UsdPhysics.CollisionAPI) for x in fixture
            ),
            "retained_tcp_colliders": [
                str(x.GetPath()) for x in active if x.HasAPI(UsdPhysics.CollisionAPI)
            ],
        }
    return report


def install_mount(stage, robot, path):
    from pxr import UsdGeom, UsdPhysics

    with np.load(path, allow_pickle=False) as data:
        for side, suffix in (("right", "R"), ("left", "L")):
            root = robot + "/TCP_Link_" + suffix + "/WristCameraFixture"
            if not stage.GetPrimAtPath(robot + "/TCP_Link_" + suffix):
                raise ValueError("missing wrist TCP")

            def mesh(name, vertices, faces):
                usd = UsdGeom.Mesh.Define(stage, root + "/" + name)
                usd.CreatePointsAttr(vertices.astype(np.float32).tolist())
                usd.CreateFaceVertexCountsAttr([3] * len(faces))
                usd.CreateFaceVertexIndicesAttr(faces.astype(int).ravel().tolist())
                usd.CreateSubdivisionSchemeAttr("none")
                return usd

            for name in ("bracket", "housing"):
                vertices = data[name + "_vertices"].copy()
                faces = data[name + "_faces"]
                if side == "left":
                    vertices[:, 1] *= -1
                    faces = faces[:, ::-1]
                visual = mesh(name, vertices, faces)
                visual.CreateDisplayColorAttr(
                    [(0.21, 0.25, 0.28) if name == "bracket" else (0.11, 0.14, 0.16)]
                )
            for name, vertices in mount_parts(path, side).items():
                hull = ConvexHull(vertices)
                faces = hull.simplices.copy()
                a, b, c = vertices[faces].transpose(1, 0, 2)
                flip = (np.cross(b - a, c - a) * hull.equations[:, :3]).sum(1) < 0
                faces[flip] = faces[flip, ::-1]
                collider = mesh(name, vertices, faces)
                UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(
                    collider.GetPrim()
                ).CreateApproximationAttr("convexHull")
                collider.MakeInvisible()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.cad, args.out)
