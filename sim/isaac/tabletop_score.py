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


def evaluate(scene, directory):
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
        target = names.index(name)
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
                "start_s": round((first + 1) * dt, 3),
                "end_s": round((last + 1) * dt, 3),
            }
            travel = np.linalg.norm(
                root[first : last + 1, :, :3] - root[first, :, :3], axis=2
            ).max(axis=0)
            row["other_motion_over_2cm"] = {
                n: round(float(x), 4)
                for n, x in zip(names, travel)
                if n != name and x > 0.020
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
                    if n != name and x > 0.02
                }
                met = (rise >= 0.09) & (b >= 0.09)
                spans = np.flatnonzero(
                    np.diff(np.r_[False, met, False].astype(int))
                ).reshape(-1, 2)
                row["longest_9cm_s"] = round(
                    max([0.0, *[(v - u - 1) * dt for u, v in spans]]), 4
                )
                pr = Rotation.from_quat(p[:, [4, 5, 6, 3]])
                rr = Rotation.from_quat(t[:, [4, 5, 6, 3]])
                local = pr.inv().apply(t[:, :3] - p[:, :3])
                rotation = pr.inv() * rr
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
                            row.update(
                                stable=True,
                                held_s=round((window - 1) * dt, 4),
                                relative_drift_m=round(float(dp), 4),
                                relative_rotation_rad=round(float(dr), 4),
                            )
                            break
                    if row["stable"]:
                        break
            tries.append(row)
        scores[name] = tries
    return {
        "seed": scene["tabletop_seed"],
        "objects": len(scores),
        "closed": sum(any(r["closure"] for r in rows) for rows in scores.values()),
        "loaded": sum(
            any(r.get("hand_contact_s", 0) > 0 for r in rows)
            for rows in scores.values()
        ),
        "stable": sum(any(r["stable"] for r in rows) for rows in scores.values()),
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
    args = parser.parse_args()
    result = evaluate(json.loads(args.scene.read_bytes()), args.run)
    (args.run / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "per_object"})
