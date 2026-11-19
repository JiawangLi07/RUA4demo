"""Verify a Sharpa hand streams fresh, finite F6 frames with drive disabled."""

from argparse import ArgumentParser
from contextlib import ExitStack
import fcntl
import json
import time

import numpy as np

from cli import json_print, new_run
from hardware import DEPLOYMENT, ROOT, SIDES, hardware_owners, open_hand_sources


def measure_tactile(side):
    if side not in SIDES:
        raise ValueError("Unknown hand side")
    channels = range(5, 10) if side == "left" else range(5)
    frame_ids = {channel: set() for channel in channels}
    peak_force = {channel: np.zeros(6) for channel in channels}
    with ExitStack() as resources:
        for lock_path in (ROOT / ".capture.lock", DEPLOYMENT / "runs" / ".active.lock"):
            lock = resources.enter_context(lock_path.open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if hardware_owners()["robot_connections"]:
            raise RuntimeError("Robot devices already in use")
        hand = open_hand_sources(resources, (side,), tactile_sides=(side,))[side]
        error, enabled = hand._hand.get_enable_state()
        hand._raise_if_error(error, "Sharpa enable-state check failed")
        if enabled:
            raise RuntimeError("Expected the selected hand to be disabled")
        settings = {}
        for getter in ("get_control_mode", "get_control_source", "get_speed_coeff", "get_current_coeff"):
            err, value = getattr(hand._hand, getter)()
            hand._raise_if_error(err, getter)
            settings[getter[4:]] = str(value)
        read_protection = getattr(hand._hand, "is_collision_protection_enabled", None)
        settings["collision_protection_enabled"] = read_protection() if read_protection else None
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            for channel in channels:
                frame = hand.read_tactile_frame(channel, .08)
                if frame is None:
                    continue
                force = np.asarray(frame.f6)
                if force.shape == (6,) and np.isfinite(force).all():
                    frame_ids[channel].add(frame.frame_id)
                    peak_force[channel] = np.maximum(peak_force[channel], np.abs(force))
    return {
        "side": side,
        "motor_enabled": False,
        "readback": settings,
        "valid_f6_frames": {ch: len(ids) for ch, ids in frame_ids.items()},
        "peak_abs_f6": {ch: values.tolist() for ch, values in peak_force.items()},
        "ready": all(len(ids) >= 10 for ids in frame_ids.values()),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=SIDES)
    report = measure_tactile(parser.parse_args().side)
    out = new_run()
    report["run"] = str(out)
    (out / "tactile_check.json").write_text(json.dumps(report, indent=2) + "\n")
    json_print(report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
