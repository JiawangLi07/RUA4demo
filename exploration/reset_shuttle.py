from argparse import ArgumentParser
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from cli import new_run
from exploration.live_cell import LiveGrasp
from exploration.live_pinch import LivePinch
from exploration.station import NoPlan
from hardware import DEPLOYMENT, ROOT, arm_settings, hardware_owners, load_extrinsics, open_state_sources
from rgbd import CameraGeometry, Snapshot, camera_stream
from robot import SnapshotBackend

parser = ArgumentParser()
parser.add_argument("observation", type=Path)
parser.add_argument("--hand-reference", type=Path, default=ROOT / ".local/station/reset_right_hand.json",
                    help="Local JSON array of 22 right-hand reset angles in radians")
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
profile = ROOT / "config/right_shuttlecock_station.json"
config = json.loads(profile.read_text())
transform = load_extrinsics(profile.parent / config["calibration"], config["camera_serial"])
home = np.asarray(config["start_deg"])
if not args.hand_reference.is_file():
    parser.error("Provide --hand-reference or configure .local/station/reset_right_hand.json")
opened = np.asarray(json.loads(args.hand_reference.read_text()), dtype=float)
if opened.shape != (22,) or not np.isfinite(opened).all():
    parser.error("Expected 22 finite right-hand angles in radians")
out = new_run()
print("RESET_RUN", out, flush=True)
reference = json.loads((args.observation / "observation.json").read_text())
stopped = False


def stop(*_):
    global stopped
    stopped = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def plan(robot):
    robot.prepare_motion()
    start = np.asarray(robot.arm.read_joint_position_deg("B"))
    robot.check_hand_path(home, [robot.fingers, opened])
    if np.max(np.abs(start - home)) <= .15:
        return []
    for height in (.06, .09):
        try:
            up = robot.plan_cartesian(height * robot.up, relative=True, near_table=True)
            count = max(1, int(np.ceil(np.max(np.abs(home - up[-1])) / .4)))
            back = np.linspace(up[-1], home, count + 1)
            robot.check_arm_path(back, robot.fingers, near_table=True)
        except NoPlan as exc:
            print("Rejected retreat height", height, str(exc), flush=True)
        else:
            return up, back
    raise NoPlan("No collision-checked route to the configured start pose")


result = {"run": str(out), "hand_start_reference": str(args.hand_reference),
          "arm_start_reference": str(profile), "success": False}
try:
    if not args.execute:
        snap = Snapshot(np.asarray(Image.open(args.observation / "head.png")),
                        np.asarray(Image.open(args.observation / "head_depth_native.png"), dtype=np.uint16),
                        CameraGeometry.from_metadata(reference["depth_geometry"]), reference["camera_serial"],
                        reference["captured_at_ns"], reference["captured_monotonic_ns"], "live")
        joints = reference["joints"]
        arm = SimpleNamespace(cfg=arm_settings(), read_joint_position_deg=lambda _: joints["arm_deg"]["right"])
        robot = LiveGrasp(config, transform, None, arm, {}, out, lambda: stopped, None)
        robot.observed = SnapshotBackend(snap, joints, transform, out)
        result["planned_waypoints"] = [len(path) - 1 for path in plan(robot)]
        result["success"] = True
    else:
        from tianji_command_adapter import ReliableTianjiRobot
        with ExitStack() as stack:
            for path in (ROOT / ".capture.lock", DEPLOYMENT / "runs/.active.lock"):
                lock = stack.enter_context(path.open("a"))
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any(hardware_owners().values()):
                raise RuntimeError("Robot or camera already in use")
            settings = arm_settings()
            settings.vel_ratio = settings.acc_ratio = 5
            arm, hands = open_state_sources(stack, ReliableTianjiRobot(settings))
            camera = stack.enter_context(camera_stream(config["camera_serial"]))
            robot = LiveGrasp(config, transform, camera, arm, hands, out, lambda: stopped, None)
            driver = LivePinch(robot)
            robot.pinch_driver = driver
            stack.callback(driver.release)
            robot.observed = robot.capture("before")
            for side in ("left", "right"):
                before = reference["joints"]
                now = robot.observed.joints
                if (np.max(np.abs(np.asarray(now["arm_deg"][side]) - before["arm_deg"][side])) > .15
                        or np.max(np.abs(np.asarray(now["hand_rad"][side]) - before["hand_rad"][side])) > .025):
                    raise RuntimeError(f"{side} pose changed since the reviewed frame")
            paths = plan(robot)
            for phase, path in zip(("retreat_from_cork", "restore_arm_start"), paths):
                robot.phase = phase
                robot.execute_cartesian(path)
            robot.phase = "restore_hand_start"
            robot.check_fingers(check_mode=True)
            start = driver.state()[1]
            if np.max(np.abs(opened - start)) > .015:
                driver.enable(start)
                count = int(np.ceil(np.max(np.abs(opened - start)) / np.deg2rad(1.5)))
                for target in np.linspace(start, opened, count + 1)[1:]:
                    measured = driver.state()[1]
                    if np.max(np.abs(target - measured)) <= .008:
                        continue
                    driver.drive(measured, target)
                    deadline = time.monotonic() + 1.2
                    while True:
                        measured = driver.state()[1]
                        residual = np.abs(measured - target)
                        if residual.max() <= .016:
                            break
                        if time.monotonic() >= deadline:
                            joint = int(np.argmax(residual))
                            raise RuntimeError(f"Opening feedback stalled at {driver.names[joint]} "
                                               f"with {np.rad2deg(residual[joint]):.2f} deg remaining")
                        time.sleep(.02)
            driver.release()
            after = robot.capture("after")
            joints = after.joints
            result.update(moves=robot.moves, arm_status=joints["arm_status"],
                          right_arm_error_deg=float(np.max(np.abs(home - joints["arm_deg"]["right"]))),
                          right_hand_error_deg=float(np.rad2deg(np.max(np.abs(opened - joints["hand_rad"]["right"])))),
                          right_hand_disabled=driver.disabled())
            result["success"] = (result["right_arm_error_deg"] <= .15
                                 and result["right_hand_error_deg"] <= 1.5
                                 and result["right_hand_disabled"]
                                 and all(s["cur_state"] == 0 and not s["err_code"] for s in joints["arm_status"].values()))
except (ValueError, RuntimeError, OSError) as exc:
    result["error"] = str(exc)
finally:
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("RESET_RESULT", json.dumps(result), flush=True)
raise SystemExit(0 if result["success"] else 2)
