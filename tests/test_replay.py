"""Regression checks for replay geometry and joint timestamps on 4090-zhenji."""

import pytest

from cli import REFERENCE_EPISODE
from hardware import read_recorded_state
from rgbd import HEAD_SERIAL, replay
from robot import RobotTools, SnapshotBackend


@pytest.fixture(scope="module")
def observation():
    snapshot = replay(REFERENCE_EPISODE, 1, HEAD_SERIAL)
    return snapshot, read_recorded_state(REFERENCE_EPISODE, snapshot.monotonic_ns)


def test_replay_pairs_color_depth_and_monotonic_joint_feedback(observation):
    snapshot, joints = observation
    assert snapshot.rgb.shape == (480, 640, 3)
    assert snapshot.raw_depth.shape == (480, 640)
    assert joints["joints_aligned"]
    assert joints["max_camera_joint_skew_ms"] < 100
    assert [len(joints["arm_deg"][side]) for side in ("left", "right")] == [7, 7]
    assert [len(joints["hand_rad"][side]) for side in ("left", "right")] == [22, 22]


def test_rgb_box_registration_and_calibration_gate(observation, tmp_path):
    snapshot, joints = observation
    backend = SnapshotBackend(snapshot, joints, None, tmp_path)
    tools = RobotTools(backend)
    inputs = {
        "frame_id": backend.frame_id,
        "label": "shuttlecock",
        "bbox_xyxy": [379, 329, 429, 370],
    }
    match = tools.execute_tool("locate", inputs).result
    assert match["found"]
    assert match["point_camera_m"] == pytest.approx((0.144, 0.217, 0.746), abs=0.005)
    assert "point_mount_m" not in match
    specs = {spec["name"] for spec in tools.get_tools_spec()}
    assert {"observe", "locate"} <= specs and "preview_reach" not in specs
    stale = tools.execute_tool("locate", {**inputs, "frame_id": "previous-frame"}).result
    assert "Frame mismatch" in stale["error"]
