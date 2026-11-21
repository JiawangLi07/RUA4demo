"""Post-run verification on the target's rigid-body trace, isolated from the controller."""

import argparse
import json
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
    if lift_height is None:
        settings = directory / "trial_settings.json"
        lift_height = (
            json.loads(settings.read_bytes())["lift_height_m"]
            if settings.is_file()
            else 0.09
        )
    if not np.isfinite(lift_height) or lift_height <= 0:
        raise ValueError("lift height must be positive metres")
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
        names.append(Path(obj["mesh"]["file"]).stem)
        pieces = load_mesh_bundle(obj["mesh"])["collisions"]
        lower.append(
            path_minimum(root[:, i], np.concatenate([v for v, _ in pieces])) - table
        )
    bottom = np.column_stack(lower)
    scores = {}

    def tick(t):
        return int(np.clip(round(t / dt) - 1, 0, length - 1))

    for name, decision in decisions.items():
        physical_name = scene.get("evaluation_aliases", {}).get(name, name)
        target = names.index(physical_name)
        tries = []
        for index, why in enumerate(decision["outcomes"], 1):
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
            executed = close_time is not None and release_time - close_time >= 0.70
            row = {
                "attempt": index,
                "online": why,
                "closure": executed,
                "closure_requested": close_time is not None,
                "wrist_refined": "registered_wrist" in phases,
                "stable": False,
                "raised": False,
                "held_through_end": False,
                "start_s": round((first + 1) * dt, 3),
                "end_s": round((last + 1) * dt, 3),
            }
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
                if "hold_start" in phases and "hold_end" in phases:
                    u = tick(phases["hold_start"]["time"]) - a
                    v = tick(phases["hold_end"]["time"]) - a
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
                                window_loads = recording["digit_load_N"][
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
        "loaded": sum(
            any(r.get("hand_contact_s", 0) > 0 for r in rows)
            for rows in scores.values()
        ),
        "stable": sum(any(r["stable"] for r in rows) for rows in scores.values()),
        "raised": sum(any(r["raised"] for r in rows) for rows in scores.values()),
        "held_through_end": sum(
            any(r["held_through_end"] for r in rows) for rows in scores.values()
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
