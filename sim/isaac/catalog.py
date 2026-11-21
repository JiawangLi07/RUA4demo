"""Select full-precision Sharpa posture templates for a registered metric mesh."""

import argparse
import copy
import json
import os
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np

from .__main__ import CASES, load_case
from .head_wrist_surface import RegisteredSurface
from .paths import RIGHT_JOINTS
from .scene_fixture import rigid_transform
from .scene_meshes import load_mesh_bundle

DIGITS = ("index", "middle", "ring", "pinky")
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_INDEX = REPO_ROOT / "DexBridge/configs/skills/sharpa_grasp_templates.json"
# A standalone sim deployment may still supply the full DexBridge package via PYTHONPATH.
GRASP_PACKAGE = (
    "DexBridge.src.dexbridge.skills.grasp"
    if (REPO_ROOT / "DexBridge/src/dexbridge/skills/grasp").is_dir()
    else "dexbridge.skills.grasp"
)


def mesh_features(vertices, faces):
    """Use the same PCA sign, face sampling and silhouette convention as the source catalog."""
    shapes = import_module(f"{GRASP_PACKAGE}.shapes")
    silhouette = import_module(f"{GRASP_PACKAGE}.silhouette")

    mean = vertices.mean(axis=0)
    axes = np.linalg.svd(vertices - mean, full_matrices=False)[2].T
    for i in range(2):
        if axes[np.argmax(abs(axes[:, i])), i] < 0:
            axes[:, i] *= -1
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    points = (vertices - mean) @ axes
    lo, hi = points.min(0), points.max(0)
    dims = hi - lo
    box = np.eye(4)
    box[:3, :3], box[:3, 3] = axes, mean + axes @ ((lo + hi) / 2)
    return {
        "T_obj_bbox": box,
        "bbox_dims": dims,
        **shapes.fit_surface_primitives(vertices, faces, dims, box),
        "silhouette_features": silhouette.silhouette_features(vertices, faces, dims, box),
    }


def palm_contacts(card):
    intended = set(card["intended_contacts"])
    segments = [{part for part in ("PP", "MP") if f"right_{digit}_{part}" in intended} for digit in DIGITS]
    eligible = "right_hand_C_MC" in intended and sum(bool(s) for s in segments) >= 2
    return eligible, sum(len(s) == 2 for s in segments), "right_thumb_PP" in intended


def select(templates, features, world_from_mesh, goal):
    """Rank every type, then transfer size/support-compatible modes into the current world frame."""
    postures = import_module(f"{GRASP_PACKAGE}.postures")

    if goal not in ("palm_wrap", "any"):
        raise ValueError("expected palm_wrap or any")
    if list(templates["joint_names"]) != RIGHT_JOINTS:
        raise ValueError("catalog joint order differs from the Sharpa SDK order")
    world = rigid_transform(world_from_mesh)
    ranks = postures.rank_types(templates, features)
    candidates, rejected = [], {}
    for rank, item in enumerate(ranks):
        name = item["grasp_type"]
        card = templates["types"][name]
        eligible, paired_phalanges, thumb_proximal = palm_contacts(card)
        if goal == "palm_wrap" and not eligible:
            rejected[name] = "template lacks the requested palm and proximal finger contacts"
            continue
        try:
            plan = postures.choose_plan(templates, name, features, world)
        except ValueError as exc:
            if str(exc) != "no shape/support-compatible upper-hemisphere train mode":
                raise
            rejected[name] = str(exc)
            continue
        stages = {}
        for stage in ("pregrasp", "grasp", "squeeze"):
            frame = plan[stage][0] if stage == "pregrasp" else plan[stage]
            stages[stage] = {
                "T_world_palm": (world @ rigid_transform(frame["T_obj_palm"])).tolist(),
                "q_rad": frame["q"],
            }
        # For a palm wrap prefer distributed PP/MP targets and support from the thumb base.
        # The catalog's per-shape evidence breaks ties. No object names enter this ranking.
        key = (-paired_phalanges, -int(thumb_proximal), rank) if goal == "palm_wrap" else (rank,)
        candidates.append(
            (
                key,
                {
                    "posture": name,
                    "mode_id": plan["mode_id"],
                    "size_or_bbox_clamped": plan["clamped"],
                    "source_record_ids": plan["source_record_ids"],
                    "intended_contacts": card["intended_contacts"],
                    "shape_evidence_rank": rank + 1,
                    **stages,
                },
            )
        )
    for name in templates["types"].keys() - {r["grasp_type"] for r in ranks}:
        rejected[name] = "no recorded modes"
    ordered = [c for _, c in sorted(candidates, key=lambda entry: entry[0])]
    return {
        "types_inspected": len(templates["types"]),
        "goal": goal,
        "shape": ranks[0]["shape_features"]["shape_key"] if ranks else None,
        "bbox_dims_m": features["bbox_dims"].tolist(),
        "T_world_mesh": world.tolist(),
        "joint_names": RIGHT_JOINTS,
        "candidates": ordered,
        "rejected": rejected,
    }


def palm_case(config, selection, count):
    if selection["goal"] != "palm_wrap" or not selection["candidates"]:
        raise ValueError("no palm-wrap candidate available")
    if selection["shape"] not in ("slender_cylinder", "stout_cylinder"):
        raise ValueError("the native palm controller requires a cylindrical side wall")
    if not 1 <= count <= 8:
        raise ValueError("shortlist must contain one to eight types")
    result = copy.deepcopy(config)
    settings = result["head_wrist_validation"]
    shortlist = selection["candidates"][:count]
    settings.pop("catalog_grasps", None)
    settings["wrap_postures"] = [c["posture"] for c in shortlist]
    settings.setdefault("pinch_postures", {}).update({c["posture"]: c["grasp"]["q_rad"] for c in shortlist})
    settings["whole_palm"] = True
    settings["require_palm_hold"] = True
    settings["record_link_forces"] = True
    settings["max_grasp_attempts"] = len(shortlist)
    return result


def opposed_case(config, selection, count):
    """Transfer compatible catalog modes in object coordinates for a fresh camera-guided episode."""
    if selection["goal"] != "any" or not 1 <= count <= 8:
        raise ValueError("expected an unrestricted query and one to eight shortlisted modes")
    result = copy.deepcopy(config)
    settings = result["head_wrist_validation"]
    handle = settings.get("multi_finger_handle", False)
    choices = []
    for candidate in selection["candidates"]:
        contacts = candidate["intended_contacts"]
        digits = {d for d in ("thumb", *DIGITS) if any(k.startswith(f"right_{d}_") for k in contacts)}
        if {"thumb", "index"} <= digits and (not handle or len(digits) >= 3):
            choices.append((len(digits), candidate))
    if handle:
        choices.sort(key=lambda pair: -pair[0])
    shortlist = [candidate for _, candidate in choices[:count]]
    if not shortlist:
        return None
    mesh_from_world = np.linalg.inv(rigid_transform(selection["T_world_mesh"]))
    settings["catalog_grasps"] = {
        c["posture"]: {
            "mode_id": c["mode_id"],
            "q_pre": c["pregrasp"]["q_rad"],
            "T_mesh_palm": (mesh_from_world @ rigid_transform(c["grasp"]["T_world_palm"])).tolist(),
        }
        for c in shortlist
    }
    settings["pinch_postures"] = {c["posture"]: c["squeeze"]["q_rad"] for c in shortlist}
    settings.pop("gesture_order", None)
    settings["whole_palm"] = False
    settings["require_palm_hold"] = False
    settings["record_link_forces"] = True
    settings["max_grasp_attempts"] = len(shortlist)
    return result


def observe(bundle, files, table_z):
    surface = RegisteredSurface(bundle)
    fits = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            measured = np.asarray(data["vertices_world_m"], float)
        if (
            measured.ndim != 2
            or measured.shape[1] != 3
            or len(measured) < 10
            or not np.isfinite(measured).all()
        ):
            raise ValueError(f"invalid observed world mesh at {path}")
        fits.append(surface.register(measured, table_z))
    matrix = np.eye(4)
    matrix[:3, :3] = fits[-1]["pose_world_rotation"]
    matrix[:3, 3] = fits[-1]["pose_world_translation_m"]
    return matrix, fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=BUNDLED_INDEX, help="DexBridge v3 index (default: bundled catalog)"
    )
    parser.add_argument("--assets", type=Path, help="asset root for a station case or a metric shape prior")
    case = parser.add_mutually_exclusive_group(required=True)
    case.add_argument("--config", type=Path)
    case.add_argument("--case", choices=CASES)
    case.add_argument("--prior", choices=CASES, help="query only the metric shape prior for a named case")
    case.add_argument("--mesh", type=Path, help="standalone object mesh (e.g. a SAM3D GLB)")
    parser.add_argument("--meters-per-unit", type=float, help="physical scale of the standalone mesh")
    obs = parser.add_mutually_exclusive_group(required=True)
    obs.add_argument("--head-mesh", type=Path, help="observed D455 NPZ in world metres")
    obs.add_argument("--world-from-mesh", type=Path, help="registered metric mesh pose as a JSON 4x4 matrix")
    parser.add_argument("--wrist-mesh", type=Path, help="optional subsequent D405 observation")
    parser.add_argument("--table-z", type=float, help="measured horizontal support height in metres")
    parser.add_argument("--goal", choices=("any", "palm_wrap"), help="default: any for --prior, palm_wrap otherwise")
    parser.add_argument("--shortlist", type=int, default=3, choices=range(1, 9))
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if args.goal is None:
        args.goal = "any" if args.prior else "palm_wrap"
    if args.head_mesh and (args.table_z is None or not np.isfinite(args.table_z)):
        parser.error("--head-mesh requires a finite --table-z")
    if args.wrist_mesh and not args.head_mesh:
        parser.error("--wrist-mesh requires --head-mesh")
    if args.mesh:
        if not args.world_from_mesh:
            parser.error("--mesh requires the registered --world-from-mesh pose")
        if args.meters_per_unit is None or not np.isfinite(args.meters_per_unit) or args.meters_per_unit <= 0:
            parser.error("--mesh requires a positive --meters-per-unit (use 1 for metric vertices)")
    elif args.meters_per_unit is not None:
        parser.error("--meters-per-unit applies to --mesh; prepared priors use metres")
    elif not args.prior and not args.assets:
        parser.error("--case and --config require --assets")
    template_io = import_module(f"{GRASP_PACKAGE}.templates")

    start = perf_counter()
    templates = template_io.load_templates(args.index)
    config = None
    if args.mesh:
        import trimesh

        mesh = trimesh.load(args.mesh, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty or not np.isfinite(mesh.vertices).all():
            raise ValueError("expected a nonempty finite triangle mesh")
        vertices, faces = np.asarray(mesh.vertices) * args.meters_per_unit, np.asarray(mesh.faces)
    else:
        if args.prior:
            objects = json.loads((Path(__file__).with_name("config") / "objects.json").read_bytes())
            prior = objects[args.prior]["head_wrist_validation"]["metric_shape_prior"]
        else:
            os.environ["RUA_SIM_ASSETS"] = str(args.assets.resolve())
            config = load_case(args.config, args.case)
            prior = config["head_wrist_validation"]["metric_shape_prior"]
        bundle = load_mesh_bundle(prior, assets=args.assets or REPO_ROOT)
        vertices, faces, _ = bundle["visual"]
    features = mesh_features(vertices, faces)
    prepared = perf_counter()
    if args.head_mesh:
        files = [args.head_mesh, args.wrist_mesh] if args.wrist_mesh else [args.head_mesh]
        matrix, fits = observe(bundle, files, args.table_z)
    else:
        matrix, fits = rigid_transform(json.loads(args.world_from_mesh.read_bytes())), []
    registered = perf_counter()
    result = select(templates, features, matrix, args.goal)
    result["timing_ms"] = {
        "load_and_mesh_features": (prepared - start) * 1000,
        "registration": (registered - prepared) * 1000,
        "catalog_query": (perf_counter() - registered) * 1000,
    }
    result["registration"] = fits
    if args.mesh:
        result["mesh"] = {"file": args.mesh.name, "meters_per_unit": args.meters_per_unit}
    elif args.prior:
        result["mesh"] = {**prior, "meters_per_unit": 1.0}
    native = None
    # The palm controller fits cylindrical sides; opposed contacts use their own surface fit.
    cyl = result["shape"] in ("slender_cylinder", "stout_cylinder")
    axis = features["cylinder_fit"]["axis_bbox"]
    if config is not None and args.goal == "palm_wrap" and result["candidates"] and cyl:
        world_axis = (matrix @ features["T_obj_bbox"])[:3, axis]
        if abs(world_axis[2]) >= np.cos(np.deg2rad(25)):
            native = palm_case(config, result, args.shortlist)
    elif config is not None and args.goal == "any":
        native = opposed_case(config, result, args.shortlist)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "candidates.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if native is not None:
        (args.out / "case.json").write_text(json.dumps(native, indent=2, allow_nan=False) + "\n")
    print(
        f"{result['types_inspected']} types, {len(result['candidates'])} pose candidates", result["timing_ms"]
    )
    return 0 if result["candidates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
