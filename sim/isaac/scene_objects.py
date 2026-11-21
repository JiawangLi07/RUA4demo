"""Research meshes and initialization-only USD scene overlays.

Call ``scenario = install_objects(stage, scenario, config)`` after cloning/remapping
render assets and before registering RigidObjects or resetting the world. Poses are
world-space metres and unit wxyz quaternions. Only the anonymous session layer is
authored; no source layer is saved. This API cannot detect a running simulator:
the caller owns the before-reset lifecycle requirement. Sphere means a faceted
physical mesh, not an analytic sphere. Tessellation is fixed, not configurable.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Real

import numpy as np

from .scene_meshes import load_mesh_bundle


def _number(value, label, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"invalid {label}")
    return value


def _vector(value, length, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (length,):
        raise ValueError(f"{label} must have {length} components")
    return np.array([_number(item, label) for item in value], dtype=float)


def _required(mapping, key):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"missing required field: {key}")
    return mapping[key]


def primitive_mesh(spec):
    """Return centered float64 vertices and outward-wound int32 triangles.

    Cylinders point along local Z. Spheres have 8 latitude intervals and 16
    longitudes; cylinders have 32 sides. All meshes are closed convex surfaces,
    with exact axis-aligned requested extents and fewer than 256 vertices.
    """
    kind = _required(spec, "kind")
    if kind == "box":
        size = _vector(_required(spec, "size_m"), 3, "size_m")
        if np.any(size <= 0):
            raise ValueError("size_m must be positive")
        vertices = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=float,
            )
            * size
            / 2
        )
        faces = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    elif kind == "sphere":
        radius = _number(_required(spec, "radius_m"), "radius_m", positive=True)
        sides, intervals = 16, 8
        vertices = [[0.0, 0.0, radius]]
        for ring in range(1, intervals):
            latitude = np.pi * ring / intervals
            for side in range(sides):
                angle = 2 * np.pi * side / sides
                vertices.append(
                    [
                        radius * np.sin(latitude) * np.cos(angle),
                        radius * np.sin(latitude) * np.sin(angle),
                        radius * np.cos(latitude),
                    ]
                )
        bottom = len(vertices)
        vertices.append([0.0, 0.0, -radius])
        faces = []
        for side in range(sides):
            following = (side + 1) % sides
            faces.append([0, 1 + side, 1 + following])
            for ring in range(intervals - 2):
                a, b = 1 + ring * sides + side, 1 + ring * sides + following
                faces.extend([[a, a + sides, b], [b, a + sides, b + sides]])
            faces.append([bottom, bottom - sides + following, bottom - sides + side])
    elif kind == "cylinder":
        radius = _number(_required(spec, "radius_m"), "radius_m", positive=True)
        height = _number(_required(spec, "height_m"), "height_m", positive=True)
        sides = 32
        vertices = [
            [radius * np.cos(2 * np.pi * side / sides), radius * np.sin(2 * np.pi * side / sides), z]
            for z in (-height / 2, height / 2)
            for side in range(sides)
        ]
        faces = []
        for side in range(sides):
            following = (side + 1) % sides
            faces.extend([[side, following, sides + following], [side, sides + following, sides + side]])
        for side in range(1, sides - 1):
            faces.extend([[0, side + 1, side], [sides, sides + side, sides + side + 1]])
    else:
        raise ValueError("kind must be box, sphere or cylinder")
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)
    # USD points use float32. Reject dimensions that collapse or overflow there,
    # rather than silently installing a zero-volume or nonfinite collider.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        points = vertices.astype(np.float32).astype(float)
        triangles = points[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if (
        not np.isfinite(points).all()
        or not np.isfinite(normals).all()
        or np.any(np.linalg.norm(normals, axis=1) == 0)
    ):
        raise ValueError("geometry is not representable as a physical USD mesh")
    return vertices, faces


def _pose(spec):
    position = _vector(_required(spec, "position"), 3, "position")
    quaternion = _vector(_required(spec, "quaternion_wxyz"), 4, "quaternion_wxyz")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6, rtol=0):
        raise ValueError("quaternion_wxyz must be unit length")
    return position, quaternion


def install_objects(stage, scenario, config):
    """Validate the complete batch, then add, park or deactivate objects in memory.

    Returns a deep copy with added/parked entries and without removed sources;
    scenario/config are never mutated. Source geometry, materials, mass and scale
    are preserved when parking. All authored changes live in the anonymous USD
    session layer, including when the root stage is file-backed. No reset, physics
    stepping, retry, source save or success inference occurs here.
    """
    if not isinstance(config, dict) or not isinstance(scenario, dict):
        raise ValueError("config and scenario must be dictionaries")
    objects = _required(scenario, "objects")
    if not isinstance(objects, dict):
        raise ValueError("scenario.objects must be a dictionary")
    added = config.get("added_objects", [])
    parked = config.get("parked_source_objects", [])
    removed = config.get("removed_source_objects", [])
    if not all(isinstance(entries, list) for entries in (added, parked, removed)):
        raise ValueError("added_objects, parked_source_objects and removed_source_objects must be lists")
    prepared, names, paths = [], set(), set()
    for adding, entries in ((True, added), (False, parked)):
        for spec in entries:
            name, path = _required(spec, "name"), _required(spec, "prim_path")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("invalid or duplicate object name")
            if not isinstance(path, str) or path in paths:
                raise ValueError("invalid or duplicate prim_path")
            if adding and name in objects:
                raise ValueError(f"object name collision: {name}")
            if not adding and (name not in objects or not isinstance(objects[name], dict)):
                raise ValueError(f"missing source scenario object: {name}")
            if not adding and objects[name].get("prim_path", path) != path:
                raise ValueError(f"source scenario path mismatch: {name}")
            position, quaternion = _pose(spec)
            entry = dict(adding=adding, name=name, path=path, position=position, quaternion=quaternion)
            if adding:
                compound = spec.get("kind") == "mesh"
                entry["compound"] = compound
                if compound:
                    entry.update(load_mesh_bundle(_required(spec, "mesh")))
                else:
                    vertices, faces = primitive_mesh(spec)
                    color = _vector(_required(spec, "color"), 3, "color")
                    if np.any((color < 0) | (color > 1)):
                        raise ValueError("color components must be in [0, 1]")
                    entry["visual"] = (vertices, faces, color[None])
                    entry["collisions"] = [(vertices, faces)]
                entry["mass"] = _number(_required(spec, "mass_kg"), "mass_kg", positive=True)
                friction = _required(spec, "friction")
                entry["friction"] = {
                    key: _number(_required(friction, key), key) for key in ("static", "dynamic", "restitution")
                }
                if (
                    entry["friction"]["static"] < 0
                    or entry["friction"]["dynamic"] < 0
                    or not 0 <= entry["friction"]["restitution"] <= 1
                ):
                    raise ValueError("invalid friction or restitution")
                for value in (entry["mass"], *entry["friction"].values()):
                    if value > float(np.finfo(np.float32).max) or (
                        value > 0 and value < float(np.finfo(np.float32).tiny)
                    ):
                        raise ValueError("physical scalar is not representable in USD")
            names.add(name)
            paths.add(path)
            prepared.append(entry)
    for spec in removed:
        name, path = _required(spec, "name"), _required(spec, "prim_path")
        if not isinstance(name, str) or name in names or name not in objects:
            raise ValueError("removed source must have a unique existing scenario name")
        if not isinstance(path, str) or path in paths:
            raise ValueError("invalid or duplicate removal prim_path")
        names.add(name)
        paths.add(path)

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    session = stage.GetSessionLayer()
    if session is None or not session.anonymous or not session.permissionToEdit:
        raise ValueError("installation requires an editable anonymous session layer")
    protected = [
        *config.get("object_prims", {}).values(),
        *(config[key] for key in ("robot_prim", "table_prim", "camera_prim") if key in config),
    ]
    for spec in removed:
        path = Sdf.Path(spec["prim_path"])
        if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
            raise ValueError("removal requires an absolute non-root prim path")
        if any(
            path.HasPrefix(Sdf.Path(other)) or Sdf.Path(other).HasPrefix(path)
            for other in paths
            if other != spec["prim_path"]
        ):
            raise ValueError("object prim paths must not overlap")
        if any(path.HasPrefix(Sdf.Path(other)) or Sdf.Path(other).HasPrefix(path) for other in protected):
            raise ValueError("cannot remove a robot, table, camera or active grasp target")
        source_path = objects[spec["name"]].get("prim_path")
        if source_path is not None and not Sdf.Path(source_path).HasPrefix(path):
            raise ValueError("removed prim does not contain the named source object")
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive() or prim.IsInstanceProxy():
            raise ValueError(f"source removal requires an active editable prim: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for entry in prepared:
        path = Sdf.Path(entry["path"])
        if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
            raise ValueError("prim_path must be an absolute non-root prim path")
        if any(path.HasPrefix(Sdf.Path(other)) for other in paths if other != entry["path"]):
            raise ValueError("object prim paths must not overlap")
        prim = stage.GetPrimAtPath(path)
        if entry["adding"]:
            if prim:
                raise ValueError(f"prim path collision: {path}")
        elif (
            not prim
            or not prim.IsActive()
            or prim.IsInstanceProxy()
            or prim.IsInstance()
            or not prim.IsA(UsdGeom.Xform)
            or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            raise ValueError(f"source must be an editable rigid Xform: {path}")
        parent = path.GetParentPath()
        while parent != Sdf.Path.absoluteRootPath:
            ancestor = stage.GetPrimAtPath(parent)
            if ancestor and (
                not ancestor.IsActive()
                or ancestor.IsInstance()
                or ancestor.IsInstanceProxy()
                or ancestor.HasAPI(UsdPhysics.RigidBodyAPI)
                or ancestor.IsA(UsdGeom.Gprim)
            ):
                raise ValueError(f"invalid object ancestor: {parent}")
            parent = parent.GetParentPath()
        position, q = entry["position"], entry["quaternion"]
        rotation = Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:])))
        transform = Gf.Matrix4d(1.0)
        transform.SetRotate(rotation)
        if not entry["adding"]:
            xform = UsdGeom.Xformable(prim)
            if any(op.GetNumTimeSamples() for op in xform.GetOrderedXformOps()):
                raise ValueError("cannot park an animated source transform")
            before = cache.GetLocalToWorldTransform(prim)
            # Row-vector USD convention: preserve world scale/shear, replacing
            # only the rigid orientation and translation.
            old_rotation = Gf.Matrix4d(1.0)
            old_rotation.SetRotate(Gf.Transform(before).GetRotation())
            transform = before * old_rotation.GetInverse() * transform
        transform.SetTranslateOnly(Gf.Vec3d(*position))
        entry["transform"] = transform

    result = deepcopy(scenario)
    with Usd.EditContext(stage, session):
        for spec in removed:
            stage.GetPrimAtPath(spec["prim_path"]).SetActive(False)
            del result["objects"][spec["name"]]
        for entry in prepared:
            path, name = entry["path"], entry["name"]
            root = (
                UsdGeom.Xform.Define(stage, path) if entry["adding"] else UsdGeom.Xform(stage.GetPrimAtPath(path))
            )
            # A single world transform avoids inherited parent scale/rotation and
            # leaves all original transform attributes intact but unordered.
            xform = UsdGeom.Xformable(root.GetPrim())
            xform.ClearXformOpOrder()
            op = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "sceneObjects")
            op.Set(entry["transform"])
            xform.SetResetXformStack(True)
            if entry["adding"]:
                UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
                UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(entry["mass"])
                material = UsdShade.Material.Define(stage, path + "/PhysicsMaterial")
                physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                physics.CreateStaticFrictionAttr(entry["friction"]["static"])
                physics.CreateDynamicFrictionAttr(entry["friction"]["dynamic"])
                physics.CreateRestitutionAttr(entry["friction"]["restitution"])
                geometry = [("/Mesh", *entry["visual"], not entry["compound"])]
                if entry["compound"]:
                    geometry += [
                        (f"/Collision_{index:03d}", vertices, faces, None, True)
                        for index, (vertices, faces) in enumerate(entry["collisions"])
                    ]
                for suffix, vertices, faces, colors, collision in geometry:
                    mesh = UsdGeom.Mesh.Define(stage, path + suffix)
                    mesh.CreatePointsAttr(vertices.tolist())
                    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
                    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
                    mesh.CreateExtentAttr([vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()])
                    mesh.CreateSubdivisionSchemeAttr("none")
                    mesh.CreateOrientationAttr("rightHanded")
                    mesh.CreateDoubleSidedAttr(False)
                    if colors is None:
                        mesh.CreateVisibilityAttr("invisible")
                    else:
                        mesh.CreateDisplayColorAttr(colors.tolist())
                        mesh.GetDisplayColorPrimvar().SetInterpolation(
                            "constant" if len(colors) == 1 else "uniform"
                        )
                    if collision:
                        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
                        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
                        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                            material,
                            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                            materialPurpose="physics",
                        )
                result["objects"][name] = {}
            result["objects"][name].update(
                position=entry["position"].tolist(), quaternion_wxyz=entry["quaternion"].tolist()
            )
    return result
