"""station_capture helpers against the replay episode; no hardware needed."""

import json
import sys

import numpy as np
import pytest

from cli import REFERENCE_EPISODE
from hardware import MOUNT_FRAME
from kinematics import URDFKinematics
from rgbd import HEAD_SERIAL, replay
from station_capture import aligned_tcp_m, main, suggest_roi, surface_point


@pytest.fixture(scope="module")
def snapshot():
    return replay(REFERENCE_EPISODE, 1, HEAD_SERIAL)


def test_surface_point_and_roi_suggestion(snapshot):
    point, mad, samples = surface_point(snapshot, [360, 310, 450, 390])
    assert samples > 20
    assert 0.2 < point[2] < 2.5
    assert mad < 0.05
    roi = suggest_roi([360, 310, 450, 390], snapshot.rgb.shape[1], snapshot.rgb.shape[0])
    assert roi[0] < 360 and roi[1] < 310 and roi[2] > 450 and roi[3] > 390
    assert 0 <= roi[0] < roi[2] <= 640 and 0 <= roi[1] < roi[3] <= 480
    with pytest.raises(ValueError, match="widen it"):
        surface_point(snapshot, [0, 0, 4, 4])


def test_aligned_tcp_matches_urdf_forward(snapshot):
    tcp, measured_deg = aligned_tcp_m(REFERENCE_EPISODE, "right", snapshot.monotonic_ns)
    model = URDFKinematics(
        __import__("hardware").DEPLOYMENT / "assets/robot/marvin_dual_arm.urdf")
    expected = model.forward(
        np.deg2rad(np.asarray(measured_deg)[None, :]),
        [f"Joint{i}_R" for i in range(1, 8)], MOUNT_FRAME, "TCP_Link_R")[0][:3, 3]
    assert tcp == pytest.approx(expected.tolist(), abs=1e-9)


def test_capture_entry_writes_report(snapshot, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "station-capture", "--bbox", "360", "310", "450", "390",
        "--episode", str(REFERENCE_EPISODE), "--frame", "1",
        "--tcp-episode", str(REFERENCE_EPISODE),
        "--tcp-mono", str(snapshot.monotonic_ns), "--tcp-side", "right",
    ])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["source"] == "replay"
    assert report["frame"] == "head_color_optical"
    assert report["config_patch"]["calibration"] is None
    assert report["config_patch"]["cube_m"] is None
    assert report["aligned_tcp_m"] is not None
