"""One Python grasp cell using the live scene and measured arm/hand state."""

from contextlib import ExitStack
import fcntl
from types import SimpleNamespace

import numpy as np
from PIL import Image

from cells_real import ArmTrajectoryBackend
from hardware import DEPLOYMENT, ROOT, arm_settings, hardware_owners, load_extrinsics, open_state_sources, read_state
from rgbd import camera_stream
from robot import SnapshotBackend
from .cartesian import cartesian_path
from .clearance import HandClearance
from .cork_pinch import PadKinematics, fit_cork, grasp_candidates
from .live_pinch import LivePinch
from .pinch_contact import PINCH_DIGITS
from .pinch_retry import rank_finger_probes, retry_pinch
from .shuttlecock import detector_client, monitored_measure
from .station import NoPlan, validate_rig
from .station_diagnostics import TableNotVisible, tcp_report, workspace
from .wrist_view import open_wrist_view


LIFT_M = .25


class CorkNotVisible(NoPlan):
    """SAM3 detected the shuttlecock but could not separate its cork."""


CELL = """def run(robo):
    robo.capture_tactile_baseline()
    robo.locate_target()
    robo.prepare()
    robo.approach()
    if (robo.close_hand() or robo.retry_grasp()) and robo.tighten_grasp():
        robo.lift()


result = run(robo)
"""


class LiveGrasp:
    def __init__(self, config, transform, camera, arm, hands, out, cancelled, box, *,
                 pinch_driver=None, clear_view_first=False, wrist_view=None, wrist_unavailable=None):
        self.config, self.transform, self.camera = config, transform, camera
        self.arm, self.hands, self.out, self.cancelled = arm, hands, out, cancelled
        self.box = box
        self.pinch_driver = pinch_driver
        self.clear_view_first = clear_view_first
        self.wrist_view, self.wrist_unavailable = wrist_view, wrist_unavailable
        self.detector = None
        self.phase = "locate"
        self.moves = []
        self.grasp_plans = []
        self.finished = False
        self.capture_index = 0

    def guard(self):
        if self.cancelled():
            raise RuntimeError("Robot client disconnected")

    def capture(self, name):
        self.guard()
        snap = self.camera()
        joints = read_state(self.arm, self.hands, snap.monotonic_ns)
        if not joints["joints_aligned"]:
            raise RuntimeError("RGB-D and joints are not time aligned")
        destination = self.out / name
        destination.mkdir()
        return SnapshotBackend(snap, joints, self.transform, destination)

    def locate_target(self):
        self.phase = "locate"
        self.observed = self.capture("locate")
        if self.clear_view_first:
            self.prepare_motion()
            self.clear_view()
            self.phase = "locate"
            self.target = self.refresh_cork("initial-clear-view")
            return
        try:
            self.target = self.measure_cork(self.observed)
        except CorkNotVisible:
            self.prepare_motion()
            self.clear_view()
            self.phase = "locate"
            self.target = self.refresh_cork("initial-clear-view")

    def refresh_cork(self, phase):
        # Detector scores can change while the cork remains visible. Retry on fresh
        # frames from the same idle pose, without reusing an earlier 3D point.
        for attempt in range(3):
            self.capture_index += 1
            observed = self.capture(f"{phase}-{self.capture_index}")
            try:
                return self.measure_cork(observed)
            except NoPlan:
                if attempt == 2:
                    raise

    def relocalize_for_regrasp(self):
        try:
            return self.refresh_cork("retry")
        except CorkNotVisible:
            self.clear_view()
        self.phase = "relocalize"
        return self.refresh_cork("clear-view")

    def clear_view(self):
        self.phase = "clear_view"
        self.check_fingers(check_mode=True)
        # Camera-right takes the right hand away from the cork in the image.
        right = self.transform[:3, 0].copy()
        right -= (right @ self.up) * self.up
        length = np.linalg.norm(right)
        if not np.isfinite(length) or length < 1e-6:
            raise NoPlan("Camera-right direction is unavailable in the table plane")
        arm = self.arm.read_joint_position_deg("B")
        self.geometry.set_pose(arm, self.fingers)
        low = min(gap for i, gap in zip(self.geometry.meshes, self.geometry.table_gaps())
                  if i in self.geometry.digit_meshes)
        if low <= self.geometry.finger_margin:
            rise = self.geometry.finger_margin - low + .006
            path = self.plan_cartesian(rise * self.up, relative=True, near_table=True, retreat=True)
            self.execute_cartesian(path)
        displacement = .040 * right / length + .030 * self.up
        path = self.plan_cartesian(displacement, relative=True, near_table=True)
        self.execute_cartesian(path)

    def measure_cork(self, observed):
        point, self.detection = self.measure_views(observed, self.guard, cork=True)
        self.surface = self.detection.pop("cork_surface", None)
        if point is None or not np.isfinite(point).all():
            reason = self.detection["reason"]
            error = CorkNotVisible if reason in ("cork_missing_or_ambiguous", "weak_shuttlecock_candidate") else NoPlan
            raise error(f"No reliable SAM3 cork depth: {reason}")
        return np.asarray(point)

    def measure_views(self, observed, guard, *, cork=False):
        self.detector = self.detector or detector_client()
        up = getattr(self, "up", None)
        if up is None:
            up = np.asarray(self.config["up_axis_mount"])
        wrist_failure = self.wrist_unavailable
        if self.wrist_view is not None:
            guard()
            try:
                wrist = self.wrist_view.capture(self.arm, self.hands, observed.out / "right-wrist")
            except (OSError, RuntimeError) as exc:
                wrist_failure = str(exc)
            else:
                height, width = wrist.snapshot.rgb.shape[:2]
                station = SimpleNamespace(roi=[0, 0, width, height], mount_from_color=wrist.extrinsics, up=up,
                                          min_depth_m=.07, max_depth_m=.50,
                                          image_rotation_quarters=self.wrist_view.rotation_quarters)
                point, detection = self.measure_view(wrist, station, guard, cork=cork)
                wrist_failure = detection["reason"]
                if point is not None and np.isfinite(point).all():
                    return point, {**detection, "camera": "right_wrist", "camera_serial": wrist.snapshot.serial}

        roi = self.box if cork and self.box is not None else self.config["target_roi_xyxy"]
        if cork:
            x1, y1, x2, y2 = roi
            height, width = observed.snapshot.rgb.shape[:2]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise NoPlan("SAM3 search region is outside the live image")
        station = SimpleNamespace(roi=roi, mount_from_color=self.transform, up=up)
        point, detection = self.measure_view(observed, station, guard, cork=cork)
        return point, {**detection, "camera": "head", "right_wrist_fallback_reason": wrist_failure}

    def measure_view(self, observed, station, guard, *, cork):
        options = {"cork": True, "include_surface": True} if cork else {}
        point, mask, detection = monitored_measure(self.detector, observed.snapshot, station, guard, **options)
        if mask is not None:
            Image.fromarray(mask).save(observed.out / "target.png")
        points = detection.pop("surface_mount_m", None)
        if points is not None:
            geometry = getattr(self, "geometry", None)
            table = geometry.height if geometry is not None else self.config["clearance"]["plane_height_m"]
            try:
                detection["cork_surface"] = fit_cork(points, station.up, table)
            except NoPlan as exc:
                return None, {**detection, "reason": str(exc)}
        return point, detection

    def prepare(self):
        self.phase = "prepare"
        if not hasattr(self, "backend"):
            self.prepare_motion()
        if getattr(self, "surface", None) is None:
            raise NoPlan("No fitted cork surface for two-pad planning")
        start = np.asarray(self.arm.read_joint_position_deg("B"))
        joints = self.hands["right"].cfg.joints
        bounds = [(joint.min_rad, joint.max_rad) for joint in joints]
        pads = PadKinematics(self.geometry, start, self.fingers, bounds)
        reasons = []
        rise_paths = {}
        for plan in self.approach_options(pads):
            opened = plan.finger_path[0]
            try:
                self.check_hand_path(start, [self.fingers, opened])
                paths = self.plan_approach(plan.wrist, start, opened, rise_paths=rise_paths)
                self.check_hand_path(paths[-1][-1], plan.finger_path)
            except NoPlan as exc:
                reasons.append(str(exc))
                continue
            self.grasp_plan, self.approach_paths = plan, paths
            self.grasp_plans.append({**plan.report(self.surface),
                                     "approach_route": "raised"})
            return
        detail = "; ".join(dict.fromkeys(reasons)) if reasons else "no matching pad shape and wrist pose"
        raise NoPlan(f"No executable two-pad cork plan: {detail}")

    def approach_options(self, pads):
        tried = set()
        for expanded in (False, True):
            # Path checks use the same MuJoCo data as the generator.
            self.geometry.set_pose(pads.arm, pads.fingers)
            for plan in grasp_candidates(self.surface, pads, self.up, self.guard, expanded=expanded):
                key = tuple(np.round(np.r_[plan.wrist, plan.finger_path.ravel()], 8))
                if key in tried:
                    continue
                tried.add(key)
                yield plan

    def check_hand_path(self, arm, waypoints):
        for a, b in zip(waypoints, waypoints[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(b - a)) / .025)))
            moving = np.flatnonzero(a != b)
            for q in np.linspace(a, b, count + 1):
                self.guard()
                self.geometry.pose(arm, q, near_table=True, moving_fingers=moving)

    def plan_approach(self, goal, start, opened, *, rise_paths):
        origin = self.ik.forward_jacobian(np.deg2rad(start))[0][:3, 3]
        height = max(origin @ self.up, goal[:3] @ self.up) + .06
        above_start = origin + (height - origin @ self.up) * self.up
        above_goal = goal.copy()
        above_goal[:3] += (height - goal[:3] @ self.up) * self.up
        key = (height, opened.tobytes())
        if key not in rise_paths:
            rise_paths[key] = self.plan_cartesian(above_start, start=start, fingers=opened, cork=self.surface)
        paths = [rise_paths[key]]
        start = paths[0][-1]
        for waypoint in (above_goal, goal):
            path = self.plan_cartesian(waypoint, start=start, fingers=opened, cork=self.surface)
            paths.append(path)
            start = path[-1]
        return paths

    def prepare_motion(self):
        from async_policy import FastLeftIK
        from kinematics import URDFKinematics

        height, width = self.observed.snapshot.rgb.shape[:2]
        clearance = self.config["clearance"]
        try:
            table = workspace(self.observed, [0, 0, width, height], self.config)
        except TableNotVisible:
            table = {
                "plane_normal": self.config["up_axis_mount"],
                "plane_height_m": clearance["plane_height_m"],
                "observed_xy_bounds_m": clearance.get("table_xy_bounds_m", [[-np.inf] * 2, [np.inf] * 2]),
            }
            self.table_source = "station_calibration"
        else:
            self.table_source = "live_head_depth"
        station = SimpleNamespace(arm="right", up=np.asarray(table["plane_normal"]),
                                  clearance={**clearance, "plane_height_m": table["plane_height_m"]})
        self.up = station.up
        self.xy_bounds = np.asarray(table["observed_xy_bounds_m"])
        initial = self.observed.joints
        self.fingers = np.asarray(initial["hand_rad"]["right"])
        self.parked_fingers = np.asarray(initial["hand_rad"]["left"])
        self.geometry = HandClearance(station, initial["arm_deg"]["left"], self.parked_fingers,
                                      buffered=False)
        self.limits = np.array([(j.min_deg, j.max_deg) for j in self.arm.cfg.joints])
        self.ik = FastLeftIK(URDFKinematics(DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf"),
                             self.limits, "right")
        motion = self.out / "motion"
        motion.mkdir()
        self.backend = ArmTrajectoryBackend(
            self.observed.snapshot, initial, self.transform, motion,
            self.arm, self.hands, self.camera, excursion_deg=None)

    def check_fingers(self, *, check_mode=False):
        self.guard()
        right_enabled = self.pinch_driver is not None and self.pinch_driver.owned
        for side, expected in (("left", self.parked_fingers), ("right", self.fingers)):
            hand = self.hands[side]
            if check_mode:
                error, enabled = hand._hand.get_enable_state()
                hand._raise_if_error(error, "Hand mode check failed")
                if bool(enabled) != (right_enabled and side == "right"):
                    raise RuntimeError(f"{side} hand drive mode changed during the arm path")
            q = np.asarray(hand.read_state().joint_pos)
            if q.shape != (22,) or not np.isfinite(q).all() or np.max(np.abs(q - expected)) > .035:
                raise RuntimeError(f"{side} hand moved during the arm path")

    def move_cartesian(self, phase, target, *, relative=False, holding=False):
        self.phase = phase
        path = self.plan_cartesian(target, relative=relative, near_table=holding)
        return self.execute_cartesian(path, holding=holding)

    def plan_cartesian(self, target, *, relative=False, near_table=False, retreat=False,
                       start=None, fingers=None, cork=None):
        if start is None:
            start = self.arm.read_joint_position_deg("B")
        if fingers is None:
            fingers = self.fingers
        retreat_gaps = None
        if retreat:
            rise = target @ self.up
            if not (near_table and relative and rise > 0 and np.allclose(target, rise * self.up)):
                raise NoPlan("Near-table retreat must move straight up")
            self.geometry.set_pose(start, fingers)
            retreat_gaps = {i: gap for i, gap in zip(self.geometry.meshes, self.geometry.table_gaps())
                            if i in self.geometry.digit_meshes and gap <= self.geometry.finger_margin}
        path, _ = cartesian_path(self.ik, start, target, relative, verify_steps=False)
        self.check_arm_path(path, fingers, near_table=near_table, retreat_gaps=retreat_gaps, cork=cork)
        if retreat:
            self.geometry.pose(path[-1], fingers, near_table=True, moving_fingers=())
        return path

    def check_arm_path(self, path, fingers, *, near_table=False, retreat_gaps=None, cork=None):
        if np.any(path < self.limits[:, 0]) or np.any(path > self.limits[:, 1]):
            raise NoPlan("Projected path exceeds native arm limits")
        if cork is not None:
            radius_sq = max(0., cork.radius - cork.residual_m) ** 2
        model, data = self.geometry.model, self.geometry.data
        arm_meshes = []
        for link in range(1, 8):
            geom = model.geom(f"Link{link}_R_visual")
            mesh = geom.dataid[0]
            lo = model.mesh_vertadr[mesh]
            arm_meshes.append((link, geom.id, model.mesh_vert[lo:lo + model.mesh_vertnum[mesh]]))
        # A low grasp usually rules out the final pose first.
        for joints in path[::-1]:
            self.guard()
            self.geometry.pose(joints, fingers, near_table=near_table,
                               retreat_gaps=retreat_gaps, moving_fingers=())
            if cork is not None:
                for i, mesh in self.geometry.meshes.items():
                    local = (cork.center - data.geom_xpos[i]) @ data.geom_xmat[i].reshape(3, 3)
                    if np.any(np.sum((mesh - local) ** 2, axis=1) < radius_sq):
                        raise NoPlan("Open hand path crosses the fitted cork")
            for link, g, vertices in arm_meshes:
                if self.geometry.other_arm_gap(g) <= self.geometry.separation:
                    raise NoPlan(f"Right link {link} too close to parked arm")
                points = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                tabletop = points[np.all((points[:, :2] >= self.xy_bounds[0]) &
                                         (points[:, :2] <= self.xy_bounds[1]), axis=1)]
                if len(tabletop) and min(tabletop @ self.up) - self.geometry.height <= self.geometry.margin:
                    raise NoPlan(f"Right link {link} too close to tabletop")

    def execute_cartesian(self, path, *, holding=False):
        def check_motion():
            self.check_fingers()
            if holding and not self.pinch_driver.verify_retention():
                raise RuntimeError("Pinch contact lost during lift")

        self.check_fingers(check_mode=True)
        move = self.backend.run_path("right", "B", path[-1], planned_path=path, guard=check_motion)
        self.moves.append({"phase": self.phase, **move})
        return move

    def approach(self):
        self.phase = "precontact"
        self.check_fingers(check_mode=True)
        held = self.pinch_driver.prepare_open(self.grasp_plan.finger_path[0])
        self.pinch_driver.record_open_gap()
        self.grasp_plans[-1]["precontact_measured_rad"] = self.fingers.tolist()
        self.grasp_plans[-1]["precontact_held_joints"] = held
        self.grasp_plans[-1]["measured_precontact_gap_m"] = self.pinch_driver.precontact_gap_m
        if held:
            for path in self.approach_paths:
                self.check_arm_path(path, self.fingers, cork=self.surface)
        self.phase = "approach"
        for path in self.approach_paths:
            self.execute_cartesian(path)

    def align_in_place(self, target):
        self.target = target
        self.prepare()
        self.approach()
        self.phase = "contact"

    def close_hand(self):
        self.phase = "contact"
        self.check_fingers(check_mode=True)
        self.contact_decision = self.pinch_driver.close_and_measure()
        self.reason = self.contact_decision["reason"]
        return self.contact_decision["confirmed"]

    def capture_tactile_baseline(self):
        self.phase = "baseline"
        self.pinch_driver.capture_baseline()
        self.tactile_channels = [0, 1]

    def retry_grasp(self):
        self.contact_decision = retry_pinch(self.contact_decision, self.pinch_driver,
                                                select_probe=self.next_finger_probe)
        self.reason = self.contact_decision.get("retry_blocked", self.contact_decision["reason"])
        return self.contact_decision["confirmed"]

    def tighten_grasp(self):
        self.phase = "tighten"
        self.light_contact_decision = self.contact_decision
        arm = self.arm.read_joint_position_deg("B")
        fingers = self.pinch_driver.state()[1]
        points = tcp_report(self.geometry, arm, fingers, "right", {})["modeled_pad_points_m"]
        self.contact_decision = self.pinch_driver.tighten_and_measure(points, self.next_finger_probe)
        self.reason = self.contact_decision["reason"]
        return self.contact_decision["confirmed"]

    def next_finger_probe(self, anchor, free, tried, *, blocked=(), target=None, inward_only=False):
        self.guard()
        arm = self.arm.read_joint_position_deg("B")
        hand = self.hands["right"]
        if anchor is None and target is None:
            target = self.grasp_plan.closing_points[PINCH_DIGITS.index(free)]
        candidates = rank_finger_probes(
            self.geometry, arm, hand.read_state().joint_pos,
            [joint.name for joint in hand.cfg.joints],
            [(joint.min_rad, joint.max_rad) for joint in hand.cfg.joints],
            free, target=target, anchor=anchor, near_table=True, blocked=blocked, inward_only=inward_only,
        )
        return next((candidate for candidate in candidates if candidate not in tried), None)

    def lift(self):
        self.phase = "lift"
        decision = getattr(self, "contact_decision", {})
        if (self.pinch_driver is None or not decision.get("confirmed")
                or not decision.get("finished_tightening")):
            raise RuntimeError("Lift requires confirmed pinch force")
        self.fingers = np.asarray(self.hands["right"].read_state().joint_pos)
        self.check_fingers(check_mode=True)
        if not self.pinch_driver.verify_retention():
            raise RuntimeError("Pinch contact lost before lift")

        start = self.arm.read_joint_position_deg("B")
        move = self.move_cartesian("lift", LIFT_M * self.up, relative=True, holding=True)
        self.check_fingers(check_mode=True)
        if not self.pinch_driver.verify_retention():
            raise RuntimeError("Pinch contact lost after lift")
        initial = self.ik.forward_jacobian(np.deg2rad(start))[0][:3, 3]
        self.lift_start = initial.copy()
        final = self.ik.forward_jacobian(np.deg2rad(move["joints_deg"]))[0][:3, 3]
        wrist_rise = float((final - initial) @ self.up)
        self.lift_result = {"requested_rise_m": LIFT_M, "measured_wrist_rise_m": wrist_rise,
                            "object_height_above_table_m": None}
        if not move["moved"] or not .20 <= wrist_rise <= .30:
            raise RuntimeError("Measured wrist lift outside the requested range")

        observed = self.capture("lifted")
        try:
            self.detector = self.detector or detector_client()
        except (OSError, RuntimeError) as exc:
            self.reason = f"Lift executed; object height unavailable: {exc}"
            return

        def watch_lift():
            self.check_fingers()
            if not self.pinch_driver.verify_retention():
                raise RuntimeError("Pinch contact lost during lift observation")

        point, meta = self.measure_views(observed, watch_lift)
        self.lift_result["detection"] = meta
        low_height = meta.get("low_height_m")
        height = float(low_height - self.geometry.height) if low_height is not None else None
        self.lift_result["object_height_above_table_m"] = height
        self.finished = point is not None and height is not None and .20 <= height <= .30
        self.reason = "shuttlecock_lift_confirmed" if self.finished else "object_lift_unverified"


def run_live_cell(config, profile, out, cancelled, box=None, *, clear_view_first=False):
    path = out / "grasp_shuttlecock.py"
    path.write_text(CELL)
    result = {"run": str(out), "execution_mode": "single_cell", "cell": str(path),
              "status": "blocked", "success": False}
    validate_rig(config["camera_serial"], config["rig"])
    transform = load_extrinsics(profile.parent / config["calibration"], config["camera_serial"])
    from tianji_command_adapter import ReliableTianjiRobot

    with ExitStack() as resources:
        for lock_path in (ROOT / ".capture.lock", DEPLOYMENT / "runs" / ".active.lock"):
            lock = resources.enter_context(lock_path.open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(hardware_owners().values()):
            raise RuntimeError("Robot or camera already in use")
        settings = arm_settings()
        settings.vel_ratio = settings.acc_ratio = 5
        arm, hands = open_state_sources(resources, ReliableTianjiRobot(settings), tactile_sides=("right",))
        camera = resources.enter_context(camera_stream(config["camera_serial"]))
        wrist_view, wrist_unavailable = open_wrist_view(resources, config, profile, camera_stream)
        for hand in hands.values():
            hand.read_state()
        robot = LiveGrasp(config, transform, camera, arm, hands, out, cancelled, box,
                          clear_view_first=clear_view_first, wrist_view=wrist_view,
                          wrist_unavailable=wrist_unavailable)
        robot.pinch_driver = LivePinch(robot)
        resources.callback(robot.pinch_driver.release)
        try:
            exec(compile(path.read_text(), str(path), "exec"), {"__builtins__": {}, "robo": robot})
            if hasattr(robot, "lift_start"):
                robot.move_cartesian("set_down", robot.lift_start, holding=True)
                robot.pinch_driver.reopen_hand()
        except (ValueError, RuntimeError, OSError) as exc:
            robot.reason = str(exc)
            if robot.phase == "set_down":
                robot.finished = False
        result.update(phase=robot.phase, reason=robot.reason, moves=robot.moves,
                      success=robot.finished, status="success" if robot.finished else "blocked",
                      target_m=robot.target.tolist() if hasattr(robot, "target") else None,
                      detection=getattr(robot, "detection", None),
                      table_source=getattr(robot, "table_source", None),
                      tactile_channels=getattr(robot, "tactile_channels", []),
                      grasp_plans=robot.grasp_plans,
                      light_contact_decision=getattr(robot, "light_contact_decision", None),
                      contact_decision=getattr(robot, "contact_decision", None),
                      lift=getattr(robot, "lift_result", None),
                      arm_status={side: vars(arm.read_arm_status(aid))
                                  for side, aid in (("left", "A"), ("right", "B"))})
    return result
