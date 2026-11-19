from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from exploration import real
from exploration.grasp_program import Program
from exploration.station import NoPlan, Planner, Station, arm_limits_deg, find_red_cube, validate_rig
from hardware import ROOT
from rgbd import HEAD_SERIAL


@pytest.fixture
def station(tmp_path):
    # Feasible synthetic IK seed, generated from joint limits. Used only by the fake SDK.
    seed = [75.047, -22.875, 96.94, -72.697, -59.968, -10.863, 52.533]
    (tmp_path / "calibration.json").write_text(json.dumps({
        "camera_serial": HEAD_SERIAL, "target_frame": "dual_arm_mount",
        "T_mount_from_color": np.eye(4).tolist(),
    }))
    path = tmp_path / "station.json"
    path.write_text(json.dumps({
        "camera_serial": HEAD_SERIAL, "calibration": "calibration.json", "arm": "right",
        "start_deg": seed, "parked_deg": seed,
        "joints_deg": arm_limits_deg().tolist(), "workspace_m": [[-1, 1]] * 3,
        "up_axis_mount": [0, 0, 1], "cube_m": [0, 0, 0], "tcp_offset_m": [0, 0, 0],
        "red_cube_roi_xyxy": [0, 0, 640, 480], "hand_open_rad": [0] * 22,
        "hand_grasps_rad": {"pinch": [0] * 22}, "hand_max_abs_torque": [1] * 22,
        "arm_max_abs_est_wrench": [20] * 6,
    }))
    parsed = Station.read(path)
    tcp, _ = Planner(parsed).ik.forward_jacobian(np.deg2rad(parsed.start_deg))
    start = tcp[:3, 3].copy()
    return replace(parsed, cube_m=start + [0.005, 0.002, -0.105],
                   workspace_m=np.column_stack((start - [0.035, 0.035, 0.13],
                                                 start + [0.035, 0.035, 0.03])))


def test_measured_joint_ik_plan_and_workspace_refusal_without_hardware(station):
    planner = Planner(station)
    paths = planner.plan(station.start_deg, station.cube_m, (0, 0, 0), 0.065)
    assert len(paths) == 3
    for path in paths:
        assert np.max(np.abs(np.diff(path, axis=0))) <= 1.5
    lifted, _ = planner.ik.forward_jacobian(np.deg2rad(paths[-1][-1]))
    assert lifted[:3, 3] == pytest.approx(station.cube_m + [0, 0, 0.065], abs=0.0002)
    with pytest.raises(NoPlan, match="workspace"):
        planner.plan(station.start_deg, station.cube_m + [0.09, 0, 0], (0, 0, 0), 0.065)


def test_4090_profile_matches_the_deployed_rig():
    profile = json.loads((ROOT / "config" / "cube_station.json").read_text())
    validate_rig(profile["camera_serial"], profile["rig"])
    other_rig = {**profile["rig"], "sharpa_serials": {"left": "other", "right": "other"}}
    with pytest.raises(ValueError, match="Station identity"):
        validate_rig(profile["camera_serial"], other_rig)


def test_red_cube_deprojection_uses_mask_and_rejects_ambiguity(station):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[100:112, 100:112, 0] = 200
    v, u = np.mgrid[100:112, 100:112]
    pixels = np.column_stack((u.ravel(), v.ravel()))
    xyz = np.repeat([[0.1, 0.2, 0.6]], len(pixels), axis=0)
    snapshot = SimpleNamespace(
        rgb=rgb, raw_depth=None,
        geometry=SimpleNamespace(register=lambda _: (pixels, xyz)),
    )
    assert find_red_cube(snapshot, station) == pytest.approx([0.1, 0.2, 0.6])
    rgb[200:212, 200:212, 0] = 200
    assert find_red_cube(snapshot, station) is None


def test_wrench_fault_prevents_target_and_acknowledges_idle(station, tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(real.time, "sleep", lambda _: None)
    serials = iter(range(1, 8))

    def sample(_, phase):
        mode = 0 if phase in ("idle_requested", "idle") else 1
        return SimpleNamespace(frame_serial=next(serials), err_code=0, cur_state=mode,
                               cmd_state=mode, est_cart_force=[30]*6)

    arm = SimpleNamespace(
        _robot=SimpleNamespace(
            clear_set=lambda: True, set_state=lambda **args: events.append(args) or True,
            send_cmd=lambda: True,
        ),
        read_arm_state=sample,
        read_arm_status=lambda _: SimpleNamespace(cur_state=0, cmd_state=0, err_code=0),
        read_joint_position_deg=lambda arm_id: station.start_deg if arm_id == "B" else station.parked_deg,
        send_joint_position_deg=lambda *_, **__: events.append("move"),
    )
    trace = lambda event, **data: events.append(event)
    with ExitStack() as resources:
        resources.callback(real.idle_arm, arm, "B", trace)
        robo = real.GraspRobot(station, Planner(station), arm, "B", None, None, resources,
                              tmp_path, trace, lambda: False)
        robo.moving = True
        with pytest.raises(RuntimeError, match="wrench"):
            robo.move([station.start_deg, station.start_deg + 0.1])
    assert events == ["arm_wrench_limit", {"arm": "B", "state": 0}, "idle_confirmed"]


@pytest.mark.parametrize("target, disconnect, attached", [("cube",False,True), ("cube",True,True),
                                                        ("shuttlecock",False,True), ("shuttlecock",False,False), ("drift", False, False), ("mode_switch", False, True)])
def test_serial_real_trial_through_fake_sdk(
    station, tmp_path, monkeypatch, target, disconnect, attached,
):
    import tianji_command_adapter
    from unirobot_infra.robots.end_effector import sharpa
    from unirobot_infra.utils import sdk_paths

    inject_drift = target == "drift"
    needs_mode_change = target == "mode_switch"
    if inject_drift or needs_mode_change:
        target = "cube"
    station = replace(station, target=target, hand_targets={"pinch": np.r_[0.15, np.zeros(21)]})
    planner = Planner(station)
    initial_hand = station.hand_open.copy()
    initial_hand[5] = 0.22
    state = SimpleNamespace(arm=station.start_deg.copy(), hand=initial_hand.copy(),
                            mode=0, arm_frame=0, hand_frame=0, hand_enabled=False)
    events = []

    def set_state(**args):
        assert args == {"arm": "B", "state": 0}
        state.mode = 0
        events.append("idle")
        return True

    class Arm:
        def __init__(self, cfg):
            assert (cfg.vel_ratio, cfg.acc_ratio) == (5, 5)
            self._robot = SimpleNamespace(clear_set=lambda: True, set_state=set_state,
                                          send_cmd=lambda: True)

        def connect(self):
            events.append("arm_connect")

        def close(self):
            events.append("arm_disconnect")

        def read_arm_status(self, arm_id):
            mode = state.mode if arm_id == "B" else 0
            return SimpleNamespace(cur_state=mode, cmd_state=mode, err_code=0)

        def read_arm_state(self, arm_id, _name):
            state.arm_frame += 1
            return SimpleNamespace(frame_serial=state.arm_frame, cur_state=state.mode, cmd_state=state.mode,
                                   err_code=0, est_cart_force=[0] * 6)

        def read_joint_position_deg(self, arm_id):
            return state.arm if arm_id == "B" else station.parked_deg

        def enter_position_mode(self, arm_id):
            assert arm_id == "B"
            state.mode = 1
            events.append("enable_arm")

        def send_joint_position_deg(self, arm_id, target, *, blocking):
            assert arm_id == "B" and not blocking and state.mode == 1
            state.arm = np.asarray(target)
            events.append("arm_command")

    class Hand:
        def __init__(self, cfg, **args):
            self.parked = args["side"] == "left"
            self.cfg = cfg
            assert not self.cfg.enable_on_connect
            self._enabled = False
            self._symbols = {"ControlSource": SimpleNamespace(SDK=1),
                             "ControlMode": SimpleNamespace(POSITION=1)}
            self._hand = SimpleNamespace(
                get_enable_state=lambda: (True, state.hand_enabled),
                set_control_source=lambda _: True,
                set_joint_position=self.preload,
                enable_collision_protection=lambda value: events.append("protect") or value,
                is_collision_protection_enabled=lambda: "protect" in events,
                get_control_mode=lambda: (True, 0 if needs_mode_change and not state.hand_enabled else 1),
                set_control_mode=self.change_mode,
                set_speed_coeff=lambda _: events.append("speed") or True,
                set_current_coeff=lambda _: events.append("current") or True,
                set_enable_state=self.enable,
            )

        def connect(self):
            events.append("hand_connect")

        def close(self):
            if self._enabled:
                state.hand_enabled = False
            if not self.parked:
                events.append("hand_disconnect")

        def _raise_if_error(self, ack, label):
            assert ack, label

        def preload(self, target, interpolate):
            assert not state.hand_enabled and interpolate
            assert target == pytest.approx(state.hand)
            events.append("hand_preload")
            return True

        def change_mode(self, mode):
            assert needs_mode_change and mode == 1 and not state.hand_enabled
            return self.enable(True)

        def enable(self, enabled):
            assert not state.hand_enabled and all(k in events for k in ("speed", "current", "protect"))
            assert enabled and "hand_preload" in events
            state.hand_enabled = True
            if inject_drift:
                state.hand = state.hand.copy()
                state.hand[3] += 0.04
            events.append("enable_hand")
            return True

        def read_state(self):
            state.hand_frame += 1
            return SimpleNamespace(sequence=state.hand_frame, joint_pos=initial_hand if self.parked else state.hand,
                                   joint_torque=[0] * 22)

        def send_joint_position_rad(self, target):
            assert state.hand_enabled
            state.hand = np.asarray(target)
            events.append("hand_command")

    @contextmanager
    def camera(_serial):
        from itertools import count
        counter = count(1)
        def read():
            i = next(counter)
            return SimpleNamespace(rgb=np.zeros((480,640,3), dtype=np.uint8),
                                   raw_depth=np.zeros((480,640), dtype=np.uint16),
                                   monotonic_ns=i*300_000_000, color_sequence=i, depth_sequence=i+10)
        yield read

    def observed_cube(*_):
        tcp, _ = planner.ik.forward_jacobian(np.deg2rad(state.arm))
        if attached and state.hand[0] >= 0.14:
            return station.cube_m + [0, 0, max(0, tcp[2, 3] - station.cube_m[2])]
        return station.cube_m

    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(real, "ROOT", tmp_path)
    monkeypatch.setattr(real, "DEPLOYMENT", tmp_path)
    monkeypatch.setattr(real, "hardware_owners", lambda: {})
    monkeypatch.setattr(real, "camera_stream", camera)
    monkeypatch.setattr(real, "find_red_cube", observed_cube)
    if target == "shuttlecock":
        from exploration import shuttlecock
        monkeypatch.setattr(shuttlecock, "detector_client", lambda: object())
        def measure(_, snapshot, __):
            if snapshot.color_sequence == 1:
                assert "enable_arm" not in events and "enable_hand" not in events
                return None, None, {"reason":"target_missing_or_ambiguous"}
            point = observed_cube()
            return point, np.ones((12,12),dtype=bool), {
                "reason":"surface_measured","score":.95,"pixels":144,"low_height_m":float(point[2])}
        monkeypatch.setattr(shuttlecock, "measure", measure)
    monkeypatch.setattr(real.time, "sleep", lambda _: None)
    monkeypatch.setattr(tianji_command_adapter, "ReliableTianjiRobot", Arm)
    monkeypatch.setattr(sharpa, "SharpaHand", Hand)
    monkeypatch.setattr(sdk_paths, "sharpa_manus_sdk_paths", lambda _: None)
    monkeypatch.setattr(sdk_paths, "configure_sharpa_runtime", lambda _: None)
    program = Program.parse({"offset_m": [0, 0, 0], "lift_m": 0.065,
                             "grasp": "pinch", "recheck": True}, ["pinch"])
    def validate_path(paths, opened, closed):
        assert opened == pytest.approx(initial_hand)
        assert len(paths) == 3
        assert "enable_arm" not in events
        events.append("clearance")
    monkeypatch.setattr(real, "HandClearance", lambda *_: SimpleNamespace(path=validate_path))
    backend = real.RealTrials(replace(station, start_deg=station.start_deg + 2))
    out = tmp_path / "trial"
    out.mkdir()
    if inject_drift:
        with pytest.raises(RuntimeError, match="clearance-checked"):
            backend.run(program, out, 0, cancelled=lambda: False)
        assert "arm_command" not in events and "hand_command" not in events
        assert not state.hand_enabled
        return
    if disconnect:
        with pytest.raises(RuntimeError, match="client disconnected"):
            backend.run(program, out, 0, cancelled=lambda: "arm_command" in events)
        assert events.count("arm_command") == 1
        assert events[-3:] == ["idle", "hand_disconnect", "arm_disconnect"]
        return
    result = backend.run(program, out, 0, cancelled=lambda: False)
    from exploration.physical import independent_measurement
    checked = independent_measurement(out, station.up, target)
    if target == "shuttlecock":
        assert (out / "before_miss_0.png").is_file()
    assert checked["success"] is attached and result["reset_ok"]
    assert (out / f"grasp_{target}.py").read_text() == program.source()
    assert checked["measured_actions"]["hand_waypoint"] and checked["measured_actions"]["arm_waypoint"]
    if not attached:
        assert checked["reason"] == "shuttlecock_did_not_lift"
        return
    assert result["success"]
    assert result["object_lift_m"] == pytest.approx([0.065, 0.065], abs=0.001)
    assert state.arm == pytest.approx(station.start_deg)
    assert state.hand == pytest.approx(initial_hand)
    assert events.index("hand_preload") < events.index("enable_hand") < events.index("arm_command")
    assert not state.hand_enabled
    assert events[-3:] == ["idle", "hand_disconnect", "arm_disconnect"]
    assert (out / "hold_0.png").exists()


def test_busy_station_refused_before_camera_or_sdk_connection(station, tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(real, "ROOT", tmp_path)
    monkeypatch.setattr(real, "DEPLOYMENT", tmp_path)
    monkeypatch.setattr(real, "hardware_owners", lambda: {"camera_pids": [1234]})
    monkeypatch.setattr(real, "camera_stream", lambda _: pytest.fail("opened the camera"))
    program = Program.parse({"offset_m": [0, 0, 0], "lift_m": 0.065,
                             "grasp": "pinch", "recheck": False}, ["pinch"])
    with pytest.raises(RuntimeError, match="Devices are in use"):
        real.RealTrials(station).run(program, tmp_path, 0, cancelled=lambda: False)

def test_whole_hand_clearance_rejects_midstroke_and_table(station):
    from exploration.clearance import HandClearance
    checker = HandClearance(replace(station, clearance={"plane_height_m": -5,
                            "table_margin_m": .03, "interarm_margin_m": .04}),
                            station.parked_deg, station.hand_open)
    assert len(checker.meshes) >= 20
    checker.height = 5
    with pytest.raises(NoPlan, match="table clearance"):
        checker.pose(station.start_deg, station.hand_open)
    def intermediate_obstruction(arm, hand):
        if 0.02 < hand[5] < 0.05:
            raise NoPlan("obstruction during closure")
    checker.pose = intermediate_obstruction
    closed = station.hand_open.copy()
    closed[5] += 0.1
    with pytest.raises(NoPlan, match="during closure"):
        checker.path([[station.start_deg]] * 3, station.hand_open, closed)


def test_near_table_margin_applies_only_to_digits(station):
    from exploration.clearance import HandClearance
    checker = HandClearance(replace(station, clearance={"plane_height_m": -5,
                            "table_margin_m": .0313, "finger_table_margin_m": .004,
                            "interarm_margin_m": .04}), station.parked_deg, station.hand_open)
    checker.opposite = []
    checker.pairs = []
    checker.error_bounds = {}
    gaps = {index: .1 for index in checker.meshes}
    palm = next(index for index in gaps if index not in checker.digit_meshes)
    digit = next(iter(checker.digit_meshes))
    checker.table_gaps = lambda: [gaps[index] for index in checker.meshes]
    gaps[digit] = .012
    with pytest.raises(NoPlan, match="table clearance"):
        checker.pose(station.start_deg, station.hand_open)
    checker.pose(station.start_deg, station.hand_open, near_table=True)
    gaps[palm] = .030
    with pytest.raises(NoPlan, match="hand_C_MC"):
        checker.pose(station.start_deg, station.hand_open, near_table=True)
    gaps[palm] = .1
    gaps[digit] = -.001
    with pytest.raises(NoPlan, match="table clearance"):
        checker.pose(station.start_deg, station.hand_open, near_table=True)


def test_live_clearance_uses_error_margin_for_moving_finger_pairs(station, monkeypatch):
    from exploration import clearance

    checker = clearance.HandClearance(
        replace(station, clearance={"plane_height_m": -5, "table_margin_m": .0313,
                                   "interarm_margin_m": .04}),
        station.parked_deg, station.hand_open,
    )
    prefix = station.arm
    middle, ring = [checker.model.geom(name).id for name in
                    (f"{prefix}_middle_DP_visual", f"{prefix}_ring_elastomer_visual")]
    checker.opposite = []
    checker.pairs = [(middle, ring)]
    checker.error_bounds = {middle: .004, ring: .004}
    checker.separating_gap = lambda *_: -.00075
    gap = .0053
    monkeypatch.setattr(clearance.mj, "mj_geomDistance", lambda *_: gap)
    arm, hand = station.start_deg, station.hand_open
    with pytest.raises(NoPlan, match="Unverified self clearance"):
        checker.pose(arm, hand)
    checker.pose(arm, hand, moving_fingers=())
    checker.pose(arm, hand, moving_fingers=(0,))
    ring_joint = min(checker.finger_dependencies[ring] - checker.finger_dependencies[middle])
    with pytest.raises(NoPlan, match="Unverified self clearance"):
        checker.pose(arm, hand, moving_fingers=(ring_joint,))
    gap = -.001
    with pytest.raises(NoPlan, match="Hand self collision"):
        checker.pose(arm, hand, moving_fingers=())


def test_unbuffered_cell_keeps_nominal_intersection_checks(station, monkeypatch):
    from exploration import clearance

    profile = replace(station, clearance={"plane_height_m": -5, "table_margin_m": .0313,
                                         "finger_table_margin_m": .004, "interarm_margin_m": .04})
    checker = clearance.HandClearance(profile, station.parked_deg, station.hand_open,
                                      buffered=False)
    table_gap = .002
    pair_gap = .0005
    checker.table_gaps = lambda: [table_gap] * len(checker.meshes)
    monkeypatch.setattr(clearance.mj, "mj_geomDistance", lambda *_: pair_gap)
    checker.pose(station.start_deg, station.hand_open, moving_fingers=(0,))
    table_gap = -.001
    with pytest.raises(NoPlan, match="table clearance"):
        checker.pose(station.start_deg, station.hand_open)
    table_gap = .002
    pair_gap = -.001
    with pytest.raises(NoPlan, match="too close"):
        checker.pose(station.start_deg, station.hand_open)
    checker.opposite = []
    with pytest.raises(NoPlan, match="Hand self collision"):
        checker.pose(station.start_deg, station.hand_open, moving_fingers=(0,))
