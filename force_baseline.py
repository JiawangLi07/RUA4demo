"""Read static Tianji and Sharpa force diagnostics without enabling hardware."""

from argparse import ArgumentParser
from contextlib import ExitStack
import fcntl
import json
import time

import numpy as np

from hardware import ROOT, SIDES, hardware_owners, open_state_sources


def propose_limits(peak, k=4.0, floor_ratio=0.15):
    """Limit per channel: k times the measured resting peak, with a floor so
    a near-silent channel still gets a positive bound (peak > 0 assumed)."""
    peak = np.asarray(peak, dtype=float)
    floor = floor_ratio * float(np.max(peak)) if np.max(peak) > 0 else 1e-3
    return np.maximum(k * peak, floor)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--margin", type=float, default=4.0,
                        help="limit = margin x resting peak (per channel)")
    parser.add_argument("--out", help="Write the JSON report here")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 3:
        parser.error("sample at least 3 seconds so the peak is meaningful")
    if not np.isfinite(args.margin) or args.margin <= 0:
        parser.error("margin must be positive and finite")

    stacks = {("hand", side): [] for side in SIDES}
    stacks = {**stacks, **{("arm", side): [] for side in SIDES}}
    arm_torques = {side: [] for side in SIDES}
    with (ROOT / ".capture.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = hardware_owners()
        if owners["robot_connections"]:
            raise RuntimeError(f"Arm/hand devices are in use: {owners['robot_connections']}")
        with ExitStack() as resources:
            arm, hands = open_state_sources(resources)
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                for side, arm_id in zip(SIDES, ("A", "B")):
                    state = arm.read_arm_state(arm_id, "force_baseline")
                    stacks[("arm", side)].append(np.abs(np.asarray(state.est_cart_force, dtype=float)))
                    arm_torques[side].append(np.abs(np.asarray(state.joint_torque, dtype=float)))
                    hand = hands[side].read_state()
                    stacks[("hand", side)].append(np.abs(np.asarray(hand.joint_torque, dtype=float)))
                time.sleep(0.02)

    report = {"seconds": args.seconds, "channels": {}}
    patch = {}
    for side in SIDES:
        hand_peak = np.max(np.stack(stacks[("hand", side)]), axis=0)
        arm_peak = np.max(np.stack(stacks[("arm", side)]), axis=0)
        hand_limit = propose_limits(hand_peak, k=args.margin).tolist() if np.any(hand_peak) else None
        arm_limit = propose_limits(arm_peak, k=args.margin).tolist() if np.any(arm_peak) else None
        report["channels"][f"hand_{side}"] = {
            "resting_peak_abs_torque": hand_peak.round(4).tolist(),
            "proposed_limit": hand_limit,
        }
        report["channels"][f"arm_{side}"] = {
            "resting_peak_abs_est_wrench": arm_peak.round(4).tolist(),
            "resting_peak_abs_joint_torque": np.max(np.stack(arm_torques[side]), axis=0).tolist(),
            "proposed_limit": arm_limit,
        }
        if side == "right":
            if hand_limit is not None:
                patch["hand_max_abs_torque"] = hand_limit
            if arm_limit is not None:
                patch["arm_max_abs_est_wrench"] = arm_limit
    report["config_patch"] = patch
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        with open(args.out, "x") as file:
            file.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
