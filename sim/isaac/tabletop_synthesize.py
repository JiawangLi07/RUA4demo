"""Calculate a Sharpa grasp and its closing trajectory on CPU from a registered metric mesh."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from .head_wrist_surface import RegisteredSurface
from .head_wrist_trial import ARM
from .paths import RIGHT_JOINTS, asset_path
from .scene_assets import collision_bounds, collision_support
from .scene_fixture import rigid_transform
from .scene_kinematics import KinematicTree
from .tabletop_catalog import load_registry
from .tabletop_control import Controller, NoPath
from .tabletop_handle import handle_sites
from .tabletop_layout import PARK_ARMS
from .tabletop_meshfit import carry_shape, closure_path
from .tabletop_vision import add_live_samples


def station_kinematics(source):
    from pxr import Usd, UsdGeom

    root = asset_path(source, "source")
    tree = KinematicTree.from_urdf(root / "robot_cad.urdf")
    initial = json.loads((root / "initial_pose.json").read_bytes())["joint_positions"]
    initial.update(dict(zip(ARM, PARK_ARMS["R"])))
    names = list(initial)
    joints = np.array(list(initial.values()), dtype=float)
    stage = Usd.Stage.Open(str(root / "scene.usdc"))
    robot = stage.GetPrimAtPath("/World/Robot")
    if not robot:
        raise ValueError("station USD has no /World/Robot")
    box = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"], False, True
    )
    support = {}
    for body in robot.GetChildren():
        name, path = body.GetName(), str(body.GetPath())
        if name.startswith("right_") and collision_bounds(stage, path, box) is not None:
            support[name] = collision_support(stage, path)
    limits = tree.joint_limits(RIGHT_JOINTS)
    model = SimpleNamespace(
        robot=SimpleNamespace(joint_names=names),
        tree=tree,
        hand_names={"right": RIGHT_JOINTS},
        hand_support=support,
        hand_bounds={"right": list(support)},
        limits={"right": limits},
    )
    return SimpleNamespace(
        commanded_joints=joints.copy,
        fk=lambda link: tree.pose(link, initial),
        model=model,
        tree=tree,
        palm_from_wrist=np.eye(4),
        arm=[names.index(n) for n in ARM],
        hand=[names.index(n) for n in RIGHT_JOINTS],
        time=lambda: 0.0,
    )


def synthesize(planner, name, surface, pose, obstacles=None, previous=()):
    """Use the same candidate search, arm reachability and surface continuation as the RGB-D controller."""
    planner.table = surface.table
    others = cKDTree(np.empty((0, 3)) if obstacles is None else obstacles)
    family, seeds = planner.shared.query(surface, pose)
    seeds.sort(key=lambda s: s["posture"] in previous)
    card, poses = planner.fit_registered(
        name, family, surface, pose, seeds, previous, others
    )
    q = np.asarray(card["q_touch"])
    if card["origin"] == "wrist_handle_cross_sections":
        sites = handle_sites(planner.hand_contacts, surface, poses["grasp"], q)
    else:
        sites = card["fit_sites"]
    path, depth, gaps = closure_path(
        planner.hand_contacts,
        poses["grasp"],
        q,
        sites,
        planner.io.model.limits["right"],
        surface.table,
    )
    return {
        "object": name,
        "frame": "station",
        "units": {"position": "m", "angle": "rad"},
        "family": family,
        "candidate_types": len(seeds),
        "posture": card["posture"],
        "origin": card["origin"],
        "joint_names": list(planner.hand_contacts.names),
        "palm_pre": poses["pre"].tolist(),
        "palm_grasp": poses["grasp"].tolist(),
        "q_open": np.asarray(card["q_open"]).tolist(),
        "q_touch": q.tolist(),
        "q_seated": path[-1].tolist(),
        "q_hold": carry_shape(
            planner.hand_contacts,
            poses["grasp"],
            path[-1],
            sites,
            planner.io.model.limits["right"],
            surface.table,
        ).tolist(),
        "q_close_path": path.tolist(),
        "contacts": {
            c["link"]: {
                "point_m": c["goal"].tolist(),
                "normal": c["normal"].tolist(),
                "source": c.get("source", "registered_handle_section"),
            }
            for c in sites
        },
        "normal_preload_m": depth,
        "predicted_signed_gap_m": gaps,
    }


def points(path):
    xyz = np.load(path, allow_pickle=False)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("expected a finite N by 3 world-coordinate .npy cloud")
    return xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--station", default="real2sim_0417")
    parser.add_argument(
        "--table-z",
        type=float,
        required=True,
        help="table plane elevation, metres in station frame",
    )
    pose_input = parser.add_mutually_exclusive_group(required=True)
    pose_input.add_argument(
        "--mesh-pose", type=Path, help="JSON 4x4, metric mesh to station"
    )
    pose_input.add_argument(
        "--wrist-points",
        type=Path,
        help="segmented object depth samples, N by 3 .npy in station frame",
    )
    parser.add_argument(
        "--obstacles", type=Path, help="other observed surfaces, N by 3 .npy"
    )
    parser.add_argument(
        "--previous", action="append", default=[], help="previously attempted posture"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    library, shared = load_registry(args.registry)
    if shared is None or args.object not in library or not np.isfinite(args.table_z):
        parser.error(
            "a registered metric object, shared catalog and finite table height are required"
        )
    io = station_kinematics(args.station)
    planner = Controller(io, {args.object: library[args.object]}, None, shared=shared)
    if args.mesh_pose:
        pose = rigid_transform(json.loads(args.mesh_pose.read_bytes()))
        surf = RegisteredSurface(planner.models.bundles[args.object], sample_faces=True)
        surf.place(pose[:3, :3], pose[:3, 3], args.table_z)
    else:
        xyz = points(args.wrist_points)
        surf, pose, _ = planner.models.fit(args.object, xyz, args.table_z)
        add_live_samples(surf, [], xyz)
    try:
        plan = synthesize(
            planner,
            args.object,
            surf,
            pose,
            points(args.obstacles) if args.obstacles else None,
            args.previous,
        )
    except NoPath as error:
        parser.exit(2, f"{error}\n")
    args.out.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
