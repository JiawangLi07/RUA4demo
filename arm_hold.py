"""Sample the Tianji wrench in position hold, then verify return to idle."""

from argparse import ArgumentParser
from contextlib import ExitStack
import fcntl
import json
import signal
import time

import numpy as np

from cli import json_print, new_run
from exploration.real import idle_arm
from hardware import DEPLOYMENT, ROOT, SIDES, arm_settings, hardware_owners


def main():
    from tianji_command_adapter import ReliableTianjiRobot

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=SIDES)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    arm_id = {"left": "A", "right": "B"}[args.side]
    cfg = arm_settings()
    cfg.vel_ratio = cfg.acc_ratio = 5
    bounds = np.array([(joint.min_deg, joint.max_deg) for joint in cfg.joints])
    arm = ReliableTianjiRobot(cfg)
    out = new_run()
    print(f"ARM_HOLD_RUN {out}", flush=True)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("Position hold interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as resources:
        log = resources.enter_context((out / "arm_hold.jsonl").open("w", buffering=1))

        def trace(event, **data):
            log.write(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(),
                                  **data}, allow_nan=False) + "\n")

        for path in (ROOT / ".capture.lock", DEPLOYMENT / "runs" / ".active.lock"):
            lock = resources.enter_context(path.open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = hardware_owners()
        if owners["robot_connections"]:
            raise RuntimeError(f"Devices are in use: {owners}")
        resources.callback(arm.close)
        arm.connect()
        parked = "A" if arm_id == "B" else "B"
        for side in (arm_id, parked):
            state = arm.read_arm_status(side)
            if state.cur_state != 0 or state.err_code:
                raise RuntimeError(f"Arm {side} is not idle and fault-free: {state}")
        start = np.asarray(arm.read_joint_position_deg(arm_id))
        if not np.isfinite(start).all() or not np.all((bounds[:, 0] <= start) & (start <= bounds[:, 1])):
            raise RuntimeError("Initial joint position is outside Tianji limits")
        trace("enter_position_hold", start_deg=start.tolist())
        samples, torques, deviations = [], [], []
        with ExitStack() as control:
            control.callback(idle_arm, arm, arm_id, trace)
            arm.enter_position_mode(arm_id)
            end = time.monotonic() + 10
            last_serial = None
            last_fresh = time.monotonic()
            while time.monotonic() < end:
                state = arm.read_arm_state(arm_id, args.side)
                if state.cur_state != 1 or state.err_code:
                    raise RuntimeError(f"Arm left position hold: mode={state.cur_state}, error={state.err_code}")
                if state.frame_serial != last_serial:
                    last_serial, last_fresh = state.frame_serial, time.monotonic()
                    q = np.rad2deg(state.joint_pos)
                    force = np.asarray(state.est_cart_force)
                    torque = np.asarray(state.joint_torque)
                    if not all(np.isfinite(values).all() for values in (q, force, torque)):
                        raise RuntimeError("Nonfinite Tianji feedback")
                    drift = float(np.max(np.abs(q - start)))
                    trace("feedback", joint_deg=q.tolist(), est_cart_force=force.tolist(),
                          joint_torque=torque.tolist(), frame_serial=last_serial)
                    if drift > 0.5:
                        raise RuntimeError(f"Position hold drifted {drift:.3f} degrees")
                    samples.append(force)
                    torques.append(torque)
                    deviations.append(drift)
                elif time.monotonic() - last_fresh > 0.3:
                    raise RuntimeError("Tianji feedback stopped updating")
                time.sleep(0.025)

        report = {"run": str(out), "arm": args.side, "samples": len(samples),
                  "max_drift_deg": max(deviations), "idle_confirmed": True,
                  "peak_abs_est_wrench": np.max(np.abs(samples), axis=0).tolist(),
                  "peak_abs_joint_torque": np.max(np.abs(torques), axis=0).tolist(),
                  "wrench_has_signal": bool(np.any(samples))}
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        json_print(report)


if __name__ == "__main__":
    main()
