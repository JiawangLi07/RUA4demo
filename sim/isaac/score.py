"""Audit a recorded grasp without loading Isaac or contacting a robot."""

import argparse
import json
from pathlib import Path

import numpy as np

from .handle_contacts import FINGERS, handle_observations, loaded_digits
from .power_wrap import envelope_contacts


def score(directory):
    def read(name):
        return json.loads((directory / name).read_text())

    config, result, events = read("config.json"), read("result.json"), read("events.json")
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
