from types import SimpleNamespace

import numpy as np
import pytest
from exploration.diagnostics import validate_cell
from exploration.station_diagnostics import fixed_arm_escape, workspace


def test_workspace_marks_missing_and_raised_returns_separately():
    y,x = np.mgrid[0:60,0:60]
    uv = np.column_stack((x.ravel(), y.ravel()))
    p = np.column_stack((uv*.002, np.zeros(uv.shape[0])))
    p[(x.ravel()>29)&(x.ravel()<40),2] = .035
    valid = x.ravel()<50
    snap = SimpleNamespace(rgb=np.zeros((60,60,3)), geometry=SimpleNamespace(
        register=lambda _: (uv[valid],p[valid])), raw_depth=None)
    result = workspace(SimpleNamespace(snapshot=snap,extrinsics=np.eye(4),frame_id="fixture"),
                       [0,0,60,60], {"up_axis_mount":[0,0,1],"clearance":{"plane_height_m":0}})
    assert result["inliers"] > 100 and result["rms_mm"] == 0
    assert [7,2] in result["raised_cells_xy"] and [7,2] not in result["table_cells_xy"]
    assert [11,2] not in result["table_cells_xy"]
    with pytest.raises(ValueError):
        workspace(SimpleNamespace(snapshot=snap,extrinsics=np.eye(4)),[1,1,2,2],
                  {"up_axis_mount":[0,0,1],"clearance":{"plane_height_m":0}})


def test_escape_only_changes_fingers_and_checks_other_arm(monkeypatch):
    import exploration.station_diagnostics as diag
    monkeypatch.setattr(diag,"hand_settings",lambda:SimpleNamespace(
        joints=[SimpleNamespace(min_rad=0,max_rad=1)]*22))
    class Geometry:
        margin = .031
        separation = .04
        meshes = {0:None}
        def set_pose(self, arm, hand):
            assert not any(arm)
            self.q = hand
        def finger_error_bounds(self, radians):
            return np.zeros(2)
        def audit_finger_path(self, *args):
            return {}
        def table_gaps(self):
            return np.array([.025 + self.q[0]*.1,.15])
        def other_arm_gap(self, _):
            return .041
    geom = Geometry()
    result = fixed_arm_escape(geom, np.zeros(7), np.zeros(22))
    assert result["found"] and result["end_min_mm"] > 31 and not result["motion_issued"]
    geom.other_arm_gap = lambda _: .039
    assert not fixed_arm_escape(geom,np.zeros(7),np.zeros(22))["found"]


def test_cell_api_exposes_measurements_without_motion():
    for name in ("workspace", "locate"):
        validate_cell(f"robo.inspect('{name}', [1,2,3,4])")
    for name in ("fixed_arm_escape", "tcp", "contact_stop"):
        validate_cell(f"robo.inspect('{name}')")
    with pytest.raises(ValueError):
        validate_cell("robo.inspect('execute_escape')")
