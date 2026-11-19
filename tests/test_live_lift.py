from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest

from exploration import live_cell
from exploration import wrist_view
from exploration.cartesian import cartesian_path
from exploration.station import NoPlan


def test_live_cartesian_path_can_subdivide_a_larger_ik_step():
    def forward(q):
        pose = np.eye(4)
        pose[0, 3] = np.rad2deg(q[0]) * .001
        return pose, None

    def solve(pose, seed):
        result = seed.copy()
        result[0] = np.deg2rad(pose[0, 3] / .001)
        return result

    ik = SimpleNamespace(forward_jacobian=forward, solve=solve)
    start, goal = np.zeros(7), [.002, 0., 0.]
    with pytest.raises(NoPlan, match="IK discontinuity"):
        cartesian_path(ik, start, goal, relative=False)

    path, _ = cartesian_path(ik, start, goal, relative=False, verify_steps=False)
    assert np.max(np.abs(np.diff(path, axis=0))) <= .4 + 1e-8
    assert path[-1, 0] == pytest.approx(2.)


def lift_robot(tmp_path, *, confirmed=True, contacts=(True, True, True)):
    contact_states = iter(contacts)
    fingers = SimpleNamespace(read_state=lambda: SimpleNamespace(joint_pos=[0.] * 22))
    driver = SimpleNamespace(verify_retention=lambda: next(contact_states))
    arm = SimpleNamespace(read_joint_position_deg=lambda side: [0.] * 7)
    robot = live_cell.LiveGrasp(
        {"target_roi_xyxy": [0, 0, 640, 480]}, np.eye(4), None, arm,
        {"right": fingers}, tmp_path, lambda: False, None, pinch_driver=driver,
    )
    robot.up = np.array([0., .6, .8])
    robot.contact_decision = {"confirmed": confirmed, "finished_tightening": True}
    robot.geometry = SimpleNamespace(height=-.33)
    robot.detector = object()
    robot.check_fingers = lambda *, check_mode=False: None
    robot.capture = lambda name: SimpleNamespace(out=tmp_path, snapshot=object())
    robot.moves = []

    def cartesian(phase, target, **options):
        robot.moves.append((phase, target, options))
        return {"moved": True, "joints_deg": [1.] + [0.] * 6}

    def forward(q):
        pose = np.eye(4)
        if q[0] > 0:
            pose[:3, 3] = .25 * robot.up
        return pose, None

    robot.move_cartesian = cartesian
    robot.ik = SimpleNamespace(forward_jacobian=forward)
    return robot


def test_lift_raises_from_current_pose_along_table_normal_and_checks_object(tmp_path, monkeypatch):
    robot = lift_robot(tmp_path)

    def measure(detector, snapshot, station, guard):
        guard()
        return np.ones(3), None, {"low_height_m": -.08}

    monkeypatch.setattr(live_cell, "monitored_measure", measure)
    robot.lift()
    phase, target, options = robot.moves[0]
    assert phase == "lift"
    assert np.allclose(target, [.0, .15, .20])
    assert options == {"relative": True, "holding": True}
    assert robot.lift_result["object_height_above_table_m"] == pytest.approx(.25)
    assert robot.finished


def test_lift_does_not_move_without_confirmed_retention(tmp_path):
    for robot in (lift_robot(tmp_path, confirmed=False), lift_robot(tmp_path, contacts=(False,))):
        with pytest.raises(RuntimeError):
            robot.lift()
        assert not robot.moves


def test_tightening_uses_joint_feedback_without_camera(tmp_path, monkeypatch):
    robot = lift_robot(tmp_path)
    robot.capture = lambda *_: pytest.fail("unexpected camera capture")
    robot.pinch_driver.state = lambda: (None, np.zeros(22))
    points = [[-.014, 0., 0.], [.014, 0., 0.]]
    monkeypatch.setattr(live_cell, "tcp_report", lambda *_: {"modeled_pad_points_m": points})

    def tighten(measured_points, select):
        assert measured_points == points
        return {"confirmed": True, "finished_tightening": True, "reason": "bilateral_contact_confirmed"}

    robot.pinch_driver.tighten_and_measure = tighten
    assert robot.tighten_grasp()


def test_lift_rejects_light_contact_without_completed_tightening(tmp_path):
    robot = lift_robot(tmp_path)
    robot.contact_decision.pop("finished_tightening")
    with pytest.raises(RuntimeError, match="confirmed pinch force"):
        robot.lift()
    assert not robot.moves


def test_wrist_motion_alone_does_not_confirm_object_lift(tmp_path, monkeypatch):
    robot = lift_robot(tmp_path)
    monkeypatch.setattr(live_cell, "monitored_measure", lambda *args: (None, None, {}))
    robot.lift()
    assert robot.lift_result["measured_wrist_rise_m"] == pytest.approx(.25)
    assert robot.reason == "object_lift_unverified" and not robot.finished


@pytest.mark.parametrize("wrist_reason", [None, "insufficient_depth", "target_missing_or_ambiguous"])
def test_cork_location_uses_head_only_after_wrist_miss(tmp_path, monkeypatch, wrist_reason):
    robot = lift_robot(tmp_path)
    robot.up = np.array([0., 0., 1.])
    head = SimpleNamespace(snapshot=SimpleNamespace(rgb=np.zeros((480, 640, 3)), serial="head"), out=tmp_path)
    wrist_transform = np.eye(4)
    wrist_transform[:3, 0] = [0, 1, 0]
    wrist = SimpleNamespace(
        snapshot=SimpleNamespace(rgb=np.zeros((480, 640, 3)), serial="wrist"),
        extrinsics=wrist_transform, out=tmp_path,
    )
    robot.wrist_view = SimpleNamespace(capture=lambda *args: wrist, rotation_quarters=1)
    calls = []

    def measure(detector, snapshot, station, guard, *, cork, include_surface):
        assert cork and include_surface
        calls.append(snapshot.serial)
        if snapshot.serial == "wrist":
            assert station.mount_from_color is wrist_transform
            assert station.min_depth_m == .07 and station.max_depth_m == .50
            if wrist_reason is not None:
                return None, None, {"reason": wrist_reason}
        return np.array([0., 0., -.31]), None, {"reason": "surface_measured"}

    monkeypatch.setattr(live_cell, "monitored_measure", measure)
    assert np.allclose(robot.measure_cork(head), [0., 0., -.31])
    if wrist_reason is None:
        assert calls == ["wrist"] and robot.detection["camera"] == "right_wrist"
    else:
        assert calls == ["wrist", "head"] and robot.detection["camera"] == "head"
        assert robot.detection["right_wrist_fallback_reason"] == wrist_reason


def test_wrist_transform_uses_joint_feedback_of_the_paired_frame(tmp_path, monkeypatch):
    path = Path(live_cell.ROOT) / "config" / "d405_right_wrist_tcp_from_color.json"
    calibration = json.loads(path.read_text())
    state = {"joints_aligned": True, "arm_deg": {"right": [10.] * 7},
             "arm_status": {"right": {"cur_state": 0, "err_code": 0}}}
    frame = SimpleNamespace(
        serial=calibration["camera_serial"], monotonic_ns=1234,
        geometry=SimpleNamespace(metadata=lambda: {"color": calibration["native_intrinsics"]}),
    )
    wrist = wrist_view.WristView(lambda: frame, calibration, calibration["camera_serial"])
    arm = SimpleNamespace(read_joint_position_deg=lambda _arm: [10.] * 7)

    def read_state(_arm, _hands, stamp):
        assert stamp == frame.monotonic_ns
        return state

    monkeypatch.setattr(wrist_view, "read_state", read_state)
    monkeypatch.setattr(wrist_view, "SnapshotBackend",
                        lambda snap, joints, transform, path, **kwargs: transform)
    transform = wrist.capture(arm, {}, tmp_path / "wrist")
    expected = wrist.model.forward(np.deg2rad([[10.] * 7]), wrist.joint_names,
                                   wrist_view.MOUNT_FRAME, "TCP_Link_R")[0]
    assert np.allclose(transform, expected @ calibration["T_tcp_from_color"])
    state["arm_deg"]["right"][0] += .2
    with pytest.raises(RuntimeError, match="time-aligned arm pose"):
        wrist.capture(arm, {}, tmp_path / "moving")


def test_raised_approach_rotates_at_height_and_descends_with_final_orientation(tmp_path):
    robot = lift_robot(tmp_path)
    robot.up = np.array([0., 0., 1.])
    robot.surface = SimpleNamespace(center=np.array([0., -.65, -.33]))
    goal = np.array([0., -.45, -.33, 0., .6, 0., .8])
    fingers, arm = np.zeros(22), np.zeros(7)
    tried = []

    def plan(target, *, start, fingers, cork):
        assert cork is robot.surface and fingers.shape == (22,)
        tried.append(target)
        return np.vstack((start, start + 1))

    robot.plan_cartesian = plan
    cache = {}
    paths = robot.plan_approach(goal, arm, fingers, rise_paths=cache)
    assert len(paths) == 3
    assert all(np.array_equal(a[-1], b[0]) for a, b in zip(paths, paths[1:]))
    assert np.allclose(tried[0], [0., 0., .06])
    assert np.allclose(tried[1], [0., -.45, .06, *goal[3:]])
    assert np.array_equal(tried[2], goal)
    robot.plan_approach(goal, arm, fingers, rise_paths=cache)
    assert len(tried) == 5


def test_approach_tries_each_wrist_goal_once(tmp_path, monkeypatch):
    robot = lift_robot(tmp_path)
    robot.surface = object()
    robot.geometry.set_pose = lambda *args: None
    pads = SimpleNamespace(arm=np.zeros(7), fingers=np.zeros(22))
    first = SimpleNamespace(wrist=np.zeros(7), finger_path=np.zeros((2, 22)))
    second = SimpleNamespace(wrist=np.ones(7), finger_path=np.zeros((2, 22)))

    def candidates(*args, expanded):
        return [first, second] if expanded else [first]

    monkeypatch.setattr(live_cell, "grasp_candidates", candidates)
    assert [id(plan) for plan in robot.approach_options(pads)] == [id(first), id(second)]


def test_regrasp_moves_only_when_fresh_frames_cannot_separate_cork(tmp_path):
    robot = lift_robot(tmp_path)
    robot.up = np.array([0., 0., 1.])
    robot.fingers = np.zeros(22)
    robot.geometry = SimpleNamespace(meshes={0: None}, digit_meshes={0}, finger_margin=.004,
                                     set_pose=lambda *_: None, table_gaps=lambda:[.05])
    target = np.array([1., 2., 3.])
    actions = []

    def measure(phase):
        actions.append(phase)
        if phase == "retry":
            raise live_cell.CorkNotVisible("cork missing")
        return target

    def plan(displacement, *, relative, near_table):
        assert relative and near_table
        assert np.allclose(displacement, [.040, 0., .030])
        actions.append("plan")
        return "checked path"

    robot.refresh_cork = measure
    robot.plan_cartesian = plan
    robot.execute_cartesian = lambda path: actions.append(path)
    assert np.array_equal(robot.relocalize_for_regrasp(), target)
    assert actions == ["retry", "plan", "checked path", "clear-view"]


def test_regrasp_does_not_move_for_good_measurement_or_camera_error(tmp_path):
    robot = lift_robot(tmp_path)
    target = np.array([.1, .2, .3])
    robot.refresh_cork = lambda phase: target
    assert np.array_equal(robot.relocalize_for_regrasp(), target)

    def camera_error(phase):
        raise RuntimeError("camera unavailable")

    robot.refresh_cork = camera_error
    with pytest.raises(RuntimeError, match="camera unavailable"):
        robot.relocalize_for_regrasp()
    assert not robot.moves


def test_initial_locate_clears_view_before_remeasuring_cork(tmp_path):
    robot = lift_robot(tmp_path)
    actions = []
    target = np.array([.1, .2, .3])

    def miss(_):
        raise live_cell.CorkNotVisible("shuttlecock visible without cork")

    robot.measure_cork = miss
    robot.prepare_motion = lambda: actions.append("prepare")
    robot.clear_view = lambda: actions.append("clear_view")
    robot.refresh_cork = lambda phase: actions.append(phase) or target
    robot.locate_target()
    assert actions == ["prepare", "clear_view", "initial-clear-view"]
    assert np.array_equal(robot.target, target)


def test_clear_view_first_retreats_vertically_if_start_is_below_margin(tmp_path):
    robot = lift_robot(tmp_path)
    robot.up = np.array([0., 0., 1.])
    robot.fingers = np.zeros(22)
    robot.geometry = SimpleNamespace(meshes={0: None}, digit_meshes={0}, finger_margin=.004,
                                     set_pose=lambda *_: None, table_gaps=lambda:[.0038])
    calls = []
    robot.plan_cartesian = lambda delta, **opts: calls.append((delta, opts)) or "checked"
    robot.execute_cartesian = lambda path: None
    robot.clear_view()
    assert np.allclose(calls[0][0], [0, 0, .0062]) and calls[0][1]["retreat"]
    assert np.allclose(calls[1][0], [.04, 0, .03]) and not calls[1][1].get("retreat")
