"""Audit a recorded grasp without loading Isaac or contacting a robot."""

import argparse
import json
from pathlib import Path

import numpy as np

from .handle_contacts import FINGERS, handle_observations, loaded_digits
from .power_wrap import envelope_contacts


def retry_distances(events, tracking):
    """Measure actual palm motion from failure to free rest, including any set-down."""
    first_move = next((e for e in events if e["stage"] == "move_to_pregrasp"), None)
    if not first_move or not tracking:
        return []
    home = np.asarray(first_move["start"])[:3, 3]
    times = np.asarray([r["time"] for r in tracking])
    xyz = np.asarray([r["palm"] for r in tracking])[:, :3, 3]

    def position(t):
        return np.array([np.interp(t, times, values) for values in xyz.T])

    measures = []
    for ready in (e for e in events if e["stage"] == "reobservation_ready"):
        attempt = ready["attempt"]
        fail = next(e for e in events if e["attempt"] == attempt and e["stage"] == "attempt_failed")
        start, end = fail["sim_time"], ready["sim_time"]
        path = np.vstack([position(start), xyz[(times > start) & (times < end)], position(end)])
        measurement = dict(
            attempt=attempt, route=ready.get("route"), start_s=start, end_s=end,
            maximum_displacement_m=float(np.linalg.norm(path - path[0], axis=1).max()),
            path_length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            end_distance_from_home_m=float(np.linalg.norm(path[-1] - home)),
        )
        next_attempt = [e for e in events if e.get("attempt") == attempt + 1]
        arrival = next((e for e in next_attempt if e["stage"] == "pregrasp_arrived"), None)
        exposure = next((e for e in next_attempt if e["stage"] == "wrist_mesh"), None)
        if arrival and exposure:
            palm = np.asarray(arrival["measured_palm"])
            view = np.vstack([path[0], xyz[(times > start) & (times <= exposure["sim_time"])], palm[:3, 3]])
            maximum = np.linalg.norm(view - path[0], axis=1).max()
            measurement["maximum_displacement_through_next_wrist_m"] = float(maximum)
            if "palm" in ready:
                rotation = np.asarray(ready["palm"])[:3, :3].T @ palm[:3, :3]
                cosine = np.clip((np.trace(rotation) - 1) / 2, -1, 1)
                measurement["turn_to_next_wrist_deg"] = float(np.rad2deg(np.arccos(cosine)))
        measures.append(measurement)
    return measures


def score(directory):
    def read(name):
        return json.loads((directory / name).read_text())

    config, result = read("config.json"), read("result.json")
    events = read("events.json") if (directory / "events.json").exists() else result["events"]
    physics, frames = read("physics.json"), read("frames.json")
    dt = config["physics_dt"]
    if (
        len(frames) != result["frames"]
        or [f["frame"] for f in frames] != list(range(len(frames)))
        or not np.allclose(np.diff([r["time"] for r in physics]), dt, rtol=0, atol=1e-7)
    ):
        raise ValueError("incomplete native frame or physics sequence")
    final_attempt = max((e.get("attempt", 1) for e in events), default=1)
    trial = [e for e in events if e.get("attempt", 1) == final_attempt]
    hold = [r for r in physics if r.get("attempt", 1) == final_attempt and r["stage"] == "hold"]
    summary = {
        "passed": result["passed"],
        "reason": result["reason"],
        "frames": len(frames),
        "attempts": result.get("attempts", []),
        "hold_steps": len(hold),
    }
    if any(e["stage"] == "reobservation_ready" for e in events):
        summary["recoveries"] = retry_distances(events, read("tracking.json"))
    if not hold:
        if result["passed"]:
            raise ValueError("successful episode is missing the hold window")
        return summary
    summary.update(
        minimum_rise_m=min(r["rise_m"] for r in hold),
        minimum_clearance_m=min(r["clearance_m"] for r in hold),
        maximum_slip_m=result.get("maximum_slip_m"),
        maximum_slip_rad=result.get("maximum_slip_rad"),
    )
    required_steps = round(2 / dt)
    if result["passed"] and (
        len(hold) != required_steps
        or summary["minimum_rise_m"] < 0.09
        or summary["minimum_clearance_m"] < 0.005
        or summary["maximum_slip_m"] > 0.03
        or summary["maximum_slip_rad"] > 0.5
    ):
        raise ValueError("reported success disagrees with native lift or hold measurements")
    grip = next(e for e in trial if e["stage"] == "refined_grasp")
    catalog = config.get("head_wrist_validation", {}).get("catalog_grasps")
    if catalog:
        method = catalog.get(grip["posture"])
        if method is None or method["mode_id"] != grip.get("catalog_mode_id"):
            raise ValueError("executed wrist grasp is missing its configured catalog mode")
        summary["catalog_mode_id"] = method["mode_id"]
        contacts = [loaded_digits(row["link_normal_N"]) for row in hold]
        summary["digit_ticks"] = {digit: sum(digit in c for c in contacts) for digit in FINGERS}
        summary["minimum_loaded_digits"] = min(map(len, contacts))
        if result["passed"] and not grip.get("tool_handle"):
            opposed = sum(
                bool(row["pads"].get("right_thumb_elastomer"))
                and any(row["pads"].get(f"right_{f}_elastomer") for f in FINGERS if f != "thumb")
                for row in hold
            )
            summary["opposed_pad_steps"] = opposed
            if opposed != required_steps:
                raise ValueError("hold lost the opposing thumb and finger pads")
    if grip.get("tool_handle"):
        measured = [loaded_digits(r["link_normal_N"]) for r in hold]
        summary["digit_ticks"] = {d: sum(d in contacts for contacts in measured) for d in FINGERS}
        summary["minimum_loaded_digits"] = min(map(len, measured))
        anchor = min(physics, key=lambda row: abs(row["time"] - grip["sim_time"]))
        if "handle_sections_m" not in grip:
            # Older episodes saved the registered mesh transform, before profiles were recorded.
            from .handle_grasp import handle_sections
            from .head_wrist_surface import RegisteredSurface
            from .scene_meshes import load_mesh_bundle

            mesh = RegisteredSurface(load_mesh_bundle(config["head_wrist_validation"]["metric_shape_prior"]))
            fit = next(e for e in trial if e["stage"] == "wrist_mesh")
            points = mesh.local @ np.asarray(fit["pose_world_rotation"]).T + fit["pose_world_translation_m"]
            origin, axis, _, bands = handle_sections(points)
            grip = dict(
                grip,
                handle_sections_m=bands,
                handle_origin_xy=origin,
                handle_axis_xy=axis,
                handle_bounds_m=bands[[0, -1], 0],
            )
        spatial = handle_observations(hold, grip, anchor["object_pose_wxyz"])
        summary.update(spatial)
        if result["passed"] and (
            not all("thumb" in c and len(c) >= 3 for c in measured)
            or spatial["opposed_on_handle_ticks"] != required_steps
        ):
            raise ValueError(f"hold lacks measured multi-finger opposition on the handle: {spatial}")
    elif grip.get("power_wrap"):
        summary["envelope_steps"] = sum(envelope_contacts(r["link_normal_N"]) for r in hold)
    if (
        result["passed"]
        and config.get("head_wrist_validation", {}).get("require_palm_hold")
        and summary.get("envelope_steps", 0) != required_steps
    ):
        raise ValueError("the required central palm and proximal finger loads did not persist throughout the hold")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    metrics = score(parser.parse_args().episode)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
