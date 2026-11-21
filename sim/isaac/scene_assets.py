"""Read native collision geometry and relocate render assets for the local runtime."""

import importlib.util
from pathlib import Path, PurePosixPath


def collision_bounds(stage, path, cache):
    """Read enabled collider envelopes, including instance proxies and guide-purpose meshes."""
    import numpy as np
    from pxr import Usd, UsdPhysics

    if "guide" not in cache.GetIncludedPurposes():
        raise ValueError("collision bounds require guide-purpose geometry")
    body = stage.GetPrimAtPath(path)
    if not body:
        raise ValueError(f"missing native body: {path}")
    ranges = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            bounds = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
            if not bounds.IsEmpty():
                ranges.append(np.array([bounds.GetMin(), bounds.GetMax()]))
    if not ranges:
        return None
    return np.array(
        [np.min([entry[0] for entry in ranges], axis=0), np.max([entry[1] for entry in ranges], axis=0)]
    )


def collision_support(stage, path):
    """Convex support vertices preserve plane extrema without empty body-box corners."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull

    from .scene_fixture import box_corners, transform_points

    body = stage.GetPrimAtPath(path)
    if not body:
        raise ValueError(f"missing native body: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)
    seen, groups = set(), []
    for collider in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not collider.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:
            continue
        for geometry in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()):
            if not geometry.IsA(UsdGeom.Boundable) or geometry.GetPath() in seen:
                continue
            seen.add(geometry.GetPath())
            if geometry.IsA(UsdGeom.Mesh):
                vertices = np.asarray(UsdGeom.Mesh(geometry).GetPointsAttr().Get(), dtype=float)
            elif geometry.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(geometry).GetSizeAttr().Get()) / 2
                vertices = box_corners([[-half] * 3, [half] * 3])
            else:
                raise ValueError(f"unsupported collider geometry: {geometry.GetTypeName()}")
            relative = inverse @ np.asarray(cache.GetLocalToWorldTransform(geometry)).T
            groups.append(transform_points(relative, vertices))
    if not groups:
        raise ValueError(f"body has no supported collision geometry: {path}")
    vertices = np.unique(np.concatenate(groups), axis=0)
    if len(vertices) > 4 and np.linalg.matrix_rank(vertices - vertices.mean(axis=0)) == 3:
        vertices = vertices[ConvexHull(vertices).vertices]
    return vertices


def remap_render_assets(stage, source):
    """Resolve only texture/builtin shader locations; preserve all physics properties."""
    from pxr import Sdf

    package = importlib.util.find_spec("isaacsim")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("Isaac Sim runtime package is unavailable")
    runtime = Path(next(iter(package.submodule_search_locations)))
    targets = {
        "tube_texture.png": source / "tube_texture.png",
        "OmniPBR.mdl": runtime / "kit/mdl/core/Base/OmniPBR.mdl",
    }
    changes = []
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            if value is None or not value.path:
                continue
            name = PurePosixPath(value.path).name
            if name not in targets:
                raise ValueError(f"unregistered native render dependency: {attribute.GetPath()} {name}")
            replacement = targets[name]
            if not replacement.is_file():
                raise FileNotFoundError(replacement)
            # MDL builtins are module identifiers, not relocated filesystem assets.
            identifier = "OmniPBR.mdl" if name == "OmniPBR.mdl" else str(replacement)
            attribute.Set(Sdf.AssetPath(identifier))
            changes.append(
                {
                    "attribute": str(attribute.GetPath()),
                    "source": value.path,
                    "resolved": identifier,
                    "runtime_file": str(replacement),
                }
            )
    return changes
