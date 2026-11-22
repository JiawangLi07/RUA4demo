"""Post-run verification on the target's rigid-body trace, isolated from the controller."""

import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .scene_meshes import load_mesh_bundle


def path_minimum(pose, vertices):
    rot = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
    hull = vertices[ConvexHull(vertices).vertices]
    result = np.empty(len(rot))
    for i in range(0, len(rot), 512):
        result[i : i + 512] = (rot[i : i + 512, 2] @ hull.T).min(axis=1)
    return result + pose[:, 2]


def evaluate(scene, directory, lift_height=None):
    settings_path = directory / "trial_settings.json"
    settings = json.loads(settings_path.read_bytes()) if settings_path.is_file() else {}
    if lift_height is None:
        lift_height = settings.get("lift_height_m", 0.09)
    if not np.isfinite(lift_height) or lift_height <= 0:
        raise ValueError("lift height must be positive metres")
    explicit_closure = settings.get("closure_telemetry") == "completed_command"
    events = [
        json.loads(s) for s in (directory / "events.jsonl").read_text().splitlines()
    ]
    decisions = json.loads((directory / "decisions.json").read_bytes())
    recording = np.load(directory / "physics_audit.npz")
    dt, table = float(recording["dt"]), float(recording["table_z"])
    root, forces, palm = (
        recording[k] for k in ("pose_wxyz", "hand_load_N", "palm_wxyz")
    )
    if forces.shape != root.shape[:2]:
        raise ValueError("force/pose body count or time mismatch")
    length = len(palm)
    window = int(np.ceil(2 / dt)) + 1
    by_name = {o["name"]: o for o in scene["added_objects"]}
    names = []
    lower = []
    for i, name in enumerate(recording["object_names"].tolist()):
        obj = by_name[name]
        stem = Path(obj["mesh"]["file"]).stem
        aliases = scene.get("evaluation_aliases", {})
        names.append(aliases.get(stem, stem))
        pieces = load_mesh_bundle(obj["mesh"])["collisions"]
        lower.append(
            path_minimum(root[:, i], np.concatenate([v for v, _ in pieces])) - table
        )
    bottom = np.column_stack(lower)
    scores = {}

    def tick(t):
        return int(np.clip(round(t / dt) - 1, 0, length - 1))

    @cache
    def channel(side, key):
        name = side + "_" + key
        return recording[name] if name in recording.files else recording[key]

    @cache
    def hardware_clearance(side, part):
        key = side + "_" + part + "_tcp_vertices"
        if key in recording.files:
            return path_minimum(channel(side, "tcp_wxyz"), recording[key]) - table
        return None

    for name, decision in decisions.items():
        physical_name = scene.get("evaluation_aliases", {}).get(name, name)
        target = names.index(physical_name)
        outcomes = decision["outcomes"]
        hands = decision.get(
            "attempt_hands", [decision.get("hand", "right")] * len(outcomes)
        )
        if len(hands) != len(outcomes):
            raise ValueError("attempt and physical hand counts disagree")
        tries = []
        for index, (side, why) in enumerate(zip(hands, outcomes), 1):
            palm, forces = channel(side, "palm_wxyz"), channel(side, "hand_load_N")
            fixture_min = hardware_clearance(side, "fixture")
            flange_min = hardware_clearance(side, "flange")
            ev = [e for e in events if e["name"] == name and e["attempt"] == index]
            phases = {e["phase"]: e for e in ev}
            first, last = (
                tick(phases["attempt_start"]["time"]),
                tick(phases["attempt_end"]["time"]),
            )
            close_time = phases.get("close", {}).get("time")
            release_time = phases.get("lower_and_release", phases["attempt_end"])[
                "time"
            ]
            evidence = "none" if close_time is None else "requested_only"
            if close_time is not None:
                if "closure_completed" in phases:
                    evidence = "completed_command"
                elif not explicit_closure and "test_lift" in phases:
                    evidence = "test_lift"
                elif not explicit_closure and release_time - close_time >= 0.70:
                    evidence = "elapsed_only"
            row = {
                "attempt": index,
                "hand": side,
                "online": why,
                "closure": evidence
                in ("completed_command", "test_lift", "elapsed_only"),
                "closure_evidence": evidence,
                "closure_requested": close_time is not None,
                "wrist_refined": "registered_wrist" in phases,
                "head_only_grasp": "head_only_grasp" in phases,
                "left_observer_only_grasp": "left_observer_only_grasp" in phases,
                "stable": False,
                "raised": False,
                "held_through_end": False,
                "full_palm_held_2s": False,
                "start_s": round((first + 1) * dt, 3),
                "end_s": round((last + 1) * dt, 3),
            }
            if flange_min is not None:
                row["minimum_tool_flange_clearance_m"] = round(
                    float(flange_min[first : last + 1].min()), 5
                )
            if fixture_min is not None:
                row["minimum_fixture_table_clearance_m"] = round(
                    float(fixture_min[first : last + 1].min()), 5
                )
            if (
                "selected" in phases
                and phases["selected"].get("origin") == "lightning_grasp"
            ):
                row["predicted_contact_links"] = phases["selected"].get(
                    "physical_contact_links",
                    [
                        link.replace("right_", side + "_")
                        for link in phases["selected"]["native_contact_links"]
                    ],
                )
                row["predicted_thumb_height_difference_m"] = phases["selected"][
                    "thumb_above_mean_fingers_m"
                ]
            if "selected" in phases:
                row["solver_origin"] = phases["selected"].get("origin")
                if "approach" in phases:
                    u = tick(phases["approach"]["time"])
                    v = tick(release_time)
                    rot = Rotation.from_quat(palm[u : v + 1, [4, 5, 6, 3]])
                    radial_z = rot.as_matrix()[:, 2, 1] * (1 if side == "right" else -1)
                    park = Rotation.from_quat(palm[0, [4, 5, 6, 3]])
                    row["minimum_radial_axis_world_z"] = round(float(radial_z.min()), 4)
                    row["minimum_wrist_to_knuckles_world_z"] = round(
                        float(rot.as_matrix()[:, 2, 2].min()), 4
                    )
                    row["maximum_turn_from_park_deg"] = round(
                        float(np.rad2deg((park.inv() * rot).magnitude().max())), 2
                    )
            travel = np.linalg.norm(
                root[first : last + 1, :, :3] - root[first, :, :3], axis=2
            ).max(axis=0)
            row["other_motion_over_2cm"] = {
                n: round(float(x), 4)
                for n, x in zip(names, travel)
                if n != physical_name and x > 0.020
            }
            if row["closure"]:
                a = tick(phases["close"]["time"])
                end = (
                    tick(phases.get("lower_and_release", phases["attempt_end"])["time"])
                    + 1
                )
                sl = slice(a, end)
                t = root[sl, target]
                p = palm[sl]
                b = bottom[sl, target]
                load = forces[sl, target]
                rise = t[:, 2] - t[0, 2]
                bearing = None
                if "hold_start" in phases and "hold_end" in phases:
                    u = tick(phases["hold_start"]["time"]) - a
                    v = tick(phases["hold_end"]["time"]) - a
                    if "digit_load_N" in recording.files:
                        digits = channel(side, "digit_load_N")[
                            a + u : a + v + 1, :, target
                        ]
                        row["four_fingers_loaded_hold_fraction"] = round(
                            float(np.mean((digits[:, 1:] > 0.25).all(1))), 4
                        )
                        row["thumb_loaded_hold_fraction"] = round(
                            float(np.mean(digits[:, 0] > 0.25)), 4
                        )
                        row["hold_digit_mean_N"] = dict(
                            zip(
                                recording["digit_names"].tolist(),
                                np.round(digits.mean(0).astype(float), 3),
                            )
                        )
                    pad_key = side + "_pad_center_z_m"
                    if pad_key in recording.files:
                        heights = channel(side, "pad_center_z_m")[a + u : a + v + 1]
                        diff = heights[:, 0] - heights[:, 1:].mean(1)
                        row["actual_thumb_above_four_median_m"] = round(
                            float(np.median(diff)), 5
                        )
                    link_key = side + "_link_load_N"
                    if link_key in recording.files:
                        names_by_link = recording[side + "_link_names"].tolist()
                        if side + "_hand_C_MC" in names_by_link:
                            palm_load = channel(side, "link_load_N")[
                                a + u : a + v + 1,
                                names_by_link.index(side + "_hand_C_MC"),
                                target,
                            ]
                            row["palm_mean_hold_N"] = round(float(palm_load.mean()), 3)
                            if "digit_load_N" in recording.files:
                                bearing = (digits[:, 1:] > 0.25).all(1) & (
                                    palm_load > 0.25
                                )
                                row["four_and_palm_hold_fraction"] = round(
                                    float(bearing.mean()), 4
                                )
                                row["four_and_palm_all_hold_ticks"] = bool(
                                    bearing.all()
                                )
                    row.update(
                        hold_phase_s=round(float((v - u) * dt), 4),
                        hold_phase_minimum_rise_m=round(
                            float(rise[u : v + 1].min()), 5
                        ),
                        hold_phase_minimum_clearance_m=round(
                            float(b[u : v + 1].min()), 5
                        ),
                    )
                row.update(
                    posture=phases["close"]["posture"],
                    peak_rise_m=round(float(rise.max()), 4),
                    peak_bottom_clearance_m=round(float(b.max()), 4),
                    peak_hand_load_N=round(float(load.max()), 3),
                    hand_contact_s=round(float((load > 0.05).sum() * dt), 3),
                )
                rises = (root[sl, :, 2] - root[a, :, 2]).max(axis=0)
                row["other_lift_over_2cm"] = {
                    n: round(float(x), 4)
                    for n, x in zip(names, rises)
                    if n != physical_name and x > 0.02
                }
                met = (rise >= lift_height) & (b >= lift_height)
                row["raised"] = bool(np.any(met))
                spans = np.flatnonzero(
                    np.diff(np.r_[False, met, False].astype(int))
                ).reshape(-1, 2)
                row["longest_above_goal_s"] = round(
                    max([0.0, *[(v - u - 1) * dt for u, v in spans]]), 4
                )
                if lift_height == 0.09:
                    row["longest_9cm_s"] = row["longest_above_goal_s"]
                pr = Rotation.from_quat(p[:, [4, 5, 6, 3]])
                rr = Rotation.from_quat(t[:, [4, 5, 6, 3]])
                local = pr.inv().apply(t[:, :3] - p[:, :3])
                rotation = pr.inv() * rr
                if "hold_phase_s" in row:
                    stop = tick(phases["hold_end"]["time"]) - a + 1
                    start = stop - window
                    if (
                        start >= tick(phases["hold_start"]["time"]) - a
                        and met[start:stop].all()
                    ):
                        drift = np.linalg.norm(local[start:stop] - local[start], axis=1)
                        angle = (
                            rotation[start].inv() * rotation[start:stop]
                        ).magnitude()
                        row["held_through_end"] = bool(
                            np.mean(load[start:stop] > 0.05) >= 0.9
                            and drift.max() <= 0.03
                            and angle.max() <= 0.5
                        )
                        row["full_palm_held_2s"] = bool(
                            row["held_through_end"]
                            and bearing is not None
                            and len(bearing) >= window
                            and bearing[-window:].all()
                        )
                for u, v in spans:
                    for start in range(u, v - window + 1):
                        stop = start + window
                        if np.mean(load[start:stop] > 0.05) < 0.90:
                            continue
                        dp = np.max(
                            np.linalg.norm(local[start:stop] - local[start], axis=1)
                        )
                        dr = (
                            (rotation[start].inv() * rotation[start:stop])
                            .magnitude()
                            .max()
                        )
                        if dp <= 0.03 and dr <= 0.5:
                            if "digit_load_N" in recording.files:
                                window_loads = channel(side, "digit_load_N")[
                                    a + start : a + stop, :, target
                                ]
                                digits = recording["digit_names"].tolist()
                                row["minimum_loaded_digits"] = int(
                                    (window_loads > 0.25).sum(axis=1).min()
                                )
                                row["thumb_and_two_other_loaded_fraction"] = round(
                                    float(
                                        np.mean(
                                            (
                                                window_loads[:, digits.index("thumb")]
                                                > 0.25
                                            )
                                            & ((window_loads > 0.25).sum(axis=1) >= 3)
                                        )
                                    ),
                                    4,
                                )
                                row["mean_digit_load_N"] = dict(
                                    zip(
                                        digits,
                                        np.round(
                                            window_loads.mean(axis=0).astype(float), 3
                                        ).tolist(),
                                    )
                                )
                            row.update(
                                stable=True,
                                held_s=round((window - 1) * dt, 4),
                                minimum_held_rise_m=round(
                                    float(rise[start:stop].min()), 4
                                ),
                                minimum_held_clearance_m=round(
                                    float(b[start:stop].min()), 4
                                ),
                                relative_drift_m=round(float(dp), 4),
                                relative_rotation_rad=round(float(dr), 4),
                            )
                            break
                    if row["stable"]:
                        break
            tries.append(row)
        scores[physical_name] = tries
    return {
        "seed": scene["tabletop_seed"],
        "objects": len(scores),
        "required_lift_m": lift_height,
        "closed": sum(any(r["closure"] for r in rows) for rows in scores.values()),
        "confirmed_closures": sum(
            any(
                r["closure_evidence"] in ("completed_command", "test_lift")
                for r in rows
            )
            for rows in scores.values()
        ),
        "loaded": sum(
            any(r.get("hand_contact_s", 0) > 0 for r in rows)
            for rows in scores.values()
        ),
        "stable": sum(any(r["stable"] for r in rows) for rows in scores.values()),
        "raised": sum(any(r["raised"] for r in rows) for rows in scores.values()),
        "held_through_end": sum(
            any(r["held_through_end"] for r in rows) for rows in scores.values()
        ),
        "full_palm_held_objects": sum(
            any(r["full_palm_held_2s"] for r in rows) for rows in scores.values()
        ),
        "visual_pass": sum(
            any(r["online"] == "vision_pass" for r in rows) for rows in scores.values()
        ),
        "simulated_seconds": round(length * dt, 4),
        "per_object": scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--lift-height-m",
        type=float,
        help="override recorded physical acceptance height",
    )
    parser.add_argument(
        "--out", type=Path, help="optional path for scoring an alternative threshold"
    )
    args = parser.parse_args()
    result = evaluate(json.loads(args.scene.read_bytes()), args.run, args.lift_height_m)
    (args.out or args.run / "score.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print({k: v for k, v in result.items() if k != "per_object"})
