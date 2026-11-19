import json
from types import SimpleNamespace

import numpy as np
import pytest

from exploration import live_pinch
from exploration.pinch_contact import PinchSample
from exploration.station import NoPlan


def make_driver(tmp_path):
    events = []
    q = np.zeros(22)
    sdk = SimpleNamespace(
        get_enable_state=lambda: (0, events.count("enable") > events.count("disable")),
        get_control_mode=lambda: (0, "position"),
        get_control_source=lambda: (0, "sdk"),
        get_speed_coeff=lambda: (0, .15),
        get_current_coeff=lambda: (0, .20),
        enable_collision_protection=lambda value: 0,
        is_collision_protection_enabled=lambda: True,
        set_control_source=lambda value: 0,
        set_speed_coeff=lambda value: 0,
        set_current_coeff=lambda value: 0,
        set_joint_position=lambda pose, interpolate: 0,
        set_enable_state=lambda enable: events.append("enable" if enable else "disable") or 0,
    )
    names = ["right_thumb_IP", "right_index_PIP"] + [str(i) for i in range(20)]

    def send(target):
        events.append("command")
        q[:] = target

    hand = SimpleNamespace(
        _hand=sdk, _enabled=False,
        _symbols={"ControlMode": SimpleNamespace(POSITION="position"),
                  "ControlSource": SimpleNamespace(SDK="sdk")},
        _raise_if_error=lambda err, name: None,
        cfg=SimpleNamespace(joints=[SimpleNamespace(name=name, min_rad=-1., max_rad=1.) for name in names]),
        read_state=lambda: SimpleNamespace(joint_pos=q.tolist()),
        send_joint_position_rad=send,
    )
    arm = SimpleNamespace(
        read_arm_status=lambda side: SimpleNamespace(cur_state=0, err_code=0),
        read_joint_position_deg=lambda side: [0.] * 7,
    )
    geometry = SimpleNamespace(pose=lambda *_, **__: None)
    path = np.zeros((3, 22))
    path[1:, :2] = [[.01, .01], [.02, .02]]
    robot = SimpleNamespace(hands={"right": hand}, arm=arm, geometry=geometry, out=tmp_path,
                            grasp_plan=SimpleNamespace(finger_path=path), guard=lambda: None)
    driver = live_pinch.LivePinch(robot)
    seq = iter(range(40))

    def sample(*, measure_gap=True):
        n = next(seq)
        pressure = .1 if "command" in events else 0.
        return PinchSample((n, n), (n * .05, n * .05), ((pressure, 0, 0), (0, pressure, 0)),
                           (.024 if pressure else .04) if measure_gap else None, n)

    driver.sample = sample
    driver.capture_baseline()
    driver.precontact_gap_m = .04
    return driver, events


def test_live_driver_sends_checked_position_and_requires_both_contacts(tmp_path):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))
    result = driver.close_and_measure()
    assert result["confirmed"] and events == ["enable", "command"]
    assert driver.verify_retention()
    assert (tmp_path / "pinch_commands.jsonl").read_text()


def test_retry_reuses_the_first_baseline_even_with_load_at_restart(tmp_path):
    driver, events = make_driver(tmp_path)
    baseline = driver.baseline
    log = tmp_path / "tactile_baseline.json"
    initial = log.read_text()
    assert len(json.loads(initial)) == 8 and not events
    driver.prepare_open(np.zeros(22))
    assert driver.close_and_measure()["confirmed"]

    driver.capture_baseline()
    assert driver.close_and_measure()["confirmed"]
    assert driver.baseline is baseline and log.read_text() == initial


def test_unchecked_finger_path_sends_no_position(tmp_path):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))

    def reject(*_, **__):
        raise NoPlan("finger contacts tabletop")

    driver.robot.geometry.pose = reject
    with pytest.raises(NoPlan, match="contacts tabletop"):
        driver.close_and_measure()
    assert events == ["enable"]


def test_thumb_stops_when_it_touches_and_index_follows_its_path(tmp_path):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))
    seq = iter(range(8, 20))

    def sample():
        n = next(seq)
        steps = events.count("command")
        return PinchSample((n, n), (n * .05, n * .05),
                           ((.1 if steps else 0., 0, 0), (0, .1 if steps > 1 else 0., 0)),
                           .04 - .005 * steps, n)

    driver.sample = sample
    result = driver.close_and_measure()
    assert result["confirmed"]
    assert events == ["enable", "command", "command"]
    assert driver.state()[1][:2].tolist() == [.01, .02]


def test_light_touch_leaves_inward_waypoints_available(tmp_path):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))
    seq = iter(range(8, 20))

    def sample():
        n = next(seq)
        steps = events.count("command")
        pressure = (0., .000005, .1)[steps]
        return PinchSample((n, n), (.05 * n, .05 * n), ((pressure, 0., 0.), (pressure, 0., 0.)),
                           .04 - .005 * steps, n)

    driver.sample = sample
    assert driver.close_and_measure()["confirmed"]
    assert events == ["enable", "command", "command"]
    assert driver.state()[1][:2].tolist() == [.02, .02]


def test_tightening_targets_move_inward_and_require_firm_contact(tmp_path):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))
    seq = iter(range(8, 40))

    def sample():
        n = next(seq)
        q = driver.state()[1][:2]
        thumb, index = [0. if angle == 0. else .0002 if angle <= .011 else .03 for angle in q]
        return PinchSample((n, n), (.05 * n, .05 * n), ((thumb, 0., 0.), (index, 0., 0.)),
                           .04 if not thumb and not index else .03, n)

    driver.sample = sample
    assert driver.close_and_measure()["confirmed"]
    assert not driver.verify_retention()
    targets = {}

    def select(anchor, digit, tried, *, target, blocked, inward_only):
        assert anchor is None and inward_only
        candidate = (driver.names[0 if digit == "thumb" else 1], .75)
        targets[digit] = target.tolist()
        return candidate if candidate not in tried else None

    result = driver.tighten_and_measure([[-.014, 0., 0.], [.014, 0., 0.]], select)
    assert result["confirmed"] and result["finished_tightening"]
    assert result["contact_threshold_f6"] == [.02, .02]
    assert targets == {"thumb": [-.01, 0., 0.], "index": [.01, 0., 0.]}
    assert events == ["enable", "command", "command", "command"]


def test_closure_waits_past_old_timeout_while_pad_keeps_moving(tmp_path, monkeypatch):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))
    target = np.r_[.04, .04, np.zeros(20)]
    measured = np.r_[.01, .04, np.zeros(20)]
    driver.robot.grasp_plan.finger_path = np.vstack((np.zeros(22), target))
    original_send = driver.hand.send_joint_position_rad
    driver.hand.send_joint_position_rad = lambda _: original_send(measured)
    seq = iter(range(8, 20))
    now = 0.
    monkeypatch.setattr(live_pinch, "time", SimpleNamespace(monotonic=lambda: now))

    def sample():
        nonlocal now
        n = next(seq)
        now += .25
        if "command" in events:
            measured[0] = min(target[0], measured[0] + .005)
            driver.hand.read_state = lambda: SimpleNamespace(joint_pos=measured.copy())
        pressure = .1 if measured[0] >= .03 - 1e-8 else 0.
        return PinchSample((n, n), (n * .05, n * .05), ((pressure, 0, 0), (0, pressure, 0)),
                           .024 if pressure else .04, n)

    driver.sample = sample
    result = driver.close_and_measure()
    assert driver.state()[1][0] == target[0]
    assert events == ["enable", "command"]
    assert result["confirmed"]


def test_no_progress_records_the_blocked_joint_without_pushing_again(tmp_path, monkeypatch):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))
    target = np.r_[.04, .04, np.zeros(20)]
    measured = np.r_[.04, .01, np.zeros(20)]
    driver.robot.grasp_plan.finger_path = np.vstack((np.zeros(22), target))
    send = driver.hand.send_joint_position_rad
    targets = []

    def send_with_stall(q):
        targets.append(q)
        send(measured)

    driver.hand.send_joint_position_rad = send_with_stall
    seq = iter(range(8, 30))
    now = 0.
    monkeypatch.setattr(live_pinch, "time", SimpleNamespace(monotonic=lambda: now))

    def sample():
        nonlocal now
        n = next(seq)
        now += .25
        return PinchSample((n, n), (n * .05, n * .05), ((0., 0., 0.), (0., 0., 0.)), .04, n)

    driver.sample = sample
    result = driver.close_and_measure()
    assert targets == [tuple(target)]
    trace = json.loads((tmp_path / "pinch_commands.jsonl").read_text())
    assert trace["stalled_digits"] == ["index"]
    assert trace["stalled_joints"] == {"right_index_PIP": .03}
    assert result["reason"] == "waypoint_tracking_stalled" and result["contact_pattern"] == "none"


def test_relief_moves_the_stalled_joint_away_and_maintains_anchor_setpoint(tmp_path):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))
    driver.hand.send_joint_position_rad(np.r_[.03, .05, np.zeros(20)])
    driver.last_target = np.r_[.045, .06, np.zeros(20)]
    driver.baseline = [PinchSample((n, n), (.05 * n, .05 * n), ((0., 0., 0.), (0., 0., 0.)), .04, n)
                       for n in range(8)]
    seq = iter(range(8, 25))

    def sample():
        n = next(seq)
        return PinchSample((n, n), (.05 * n, .05 * n), ((0., 0., 0.), (.1, 0., 0.)), .034, n)

    driver.sample = sample
    result = driver.relieve_stall({"contacting_digits": ["index"], "stalled_digits": ["thumb"],
                                  "stalled_joints": {"right_thumb_IP": .015}})
    assert driver.state()[1][:2].tolist() == pytest.approx([.015, .06])
    assert result["contacting_digits"] == ["index"]
    assert events == ["enable", "command", "command"]


def test_index_holds_and_thumb_continues_after_submillimeter_closure(tmp_path):
    driver, events = make_driver(tmp_path)
    driver.prepare_open(np.zeros(22))
    seq = iter(range(8, 24))

    def sample():
        n = next(seq)
        steps = events.count("command")
        thumb = .000007 if steps == 1 and n % 3 == 0 else .00003
        gap = .03932 if steps == 1 else .038
        return PinchSample((n, n), (.05 * n, .05 * n), ((thumb, 0., 0.), (.00003, 0., 0.)), gap, n)

    driver.sample = sample
    assert driver.close_and_measure()["confirmed"]
    assert driver.state()[1][:2].tolist() == [.02, .01]


def test_preshape_holds_stalled_joint_and_finishes_other_joints(tmp_path, monkeypatch):
    driver, events = make_driver(tmp_path)
    driver.enable(np.zeros(22))
    now = [0.]
    monkeypatch.setattr(live_pinch, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt)))
    send = driver.hand.send_joint_position_rad
    def stalled_thumb(target):
        actual = np.array(target)
        actual[0] = 0.
        send(tuple(actual))
    driver.hand.send_joint_position_rad = stalled_thumb
    held = driver.prepare_open(np.r_[.055, .055, np.zeros(20)])
    assert held == ["right_thumb_IP"]
    assert driver.open_pose[:2].tolist() == [0., .055]
    assert driver.last_target[0] == 0.
    assert driver.owned


def test_unexpected_enable_during_preload_is_owned_for_shutdown(tmp_path):
    driver, events = make_driver(tmp_path)

    def preload(*_):
        events.append("enable")
        return 0

    driver.sdk.set_joint_position = preload
    with pytest.raises(RuntimeError, match="enabled during preload"):
        driver.enable(np.zeros(22))
    assert driver.owned
    driver.release()
    assert events == ["enable", "disable"] and driver.disabled()
