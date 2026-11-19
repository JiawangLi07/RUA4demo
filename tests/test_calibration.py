from contextlib import contextmanager
from types import SimpleNamespace as Obj

import numpy as np
import pytest

from exploration import calibration as cal
from exploration.diagnostics import validate_cell


def test_trial_cell_requires_motion_authorization():
    validate_cell("robo.calibrate()",True)
    for code, authorized in [("robo.calibrate()",False),("robo.calibrate(2)",True),
                             ("robo.calibrate()\nrobo.calibrate()",True)]:
        with pytest.raises(ValueError):
            validate_cell(code,authorized)


@pytest.mark.parametrize("refuse", [False, True])
def test_probe_protected_preload_and_shutdown(tmp_path, monkeypatch, refuse):
    (tmp_path/"runs").mkdir()
    monkeypatch.setattr(cal,"ROOT",tmp_path)
    monkeypatch.setattr(cal,"DEPLOYMENT",tmp_path)
    monkeypatch.setattr(cal,"validate_rig",lambda *_: None)
    monkeypatch.setattr(cal,"hardware_owners",lambda: {})
    snap=Obj(rgb=np.zeros((3,3,3)),monotonic_ns=1)
    state={"hand_rad": {"right":[0.]*22,"left":[0.]*22}, "arm_deg":{"right":[0.]*7,"left":[0.]*7},
           "arm_status":{"right":{"cur_state":0,"err_code":0}},"joints_aligned":True}
    @contextmanager
    def camera(_):
        yield lambda:snap
    monkeypatch.setattr(cal,"camera_stream",camera)
    monkeypatch.setattr(cal,"read_state",lambda *_:state)
    monkeypatch.setattr(cal,"SnapshotBackend",lambda *_:None)
    monkeypatch.setattr(cal,"HandClearance",lambda *_:Obj(report=lambda *_:{"passed":True},pose=lambda *_:None))
    class Hand:
        def __init__(self):
            self._hand=self
            self._symbols={"ControlMode":Obj(POSITION=1),"ControlSource":Obj(SDK=2)}
            self.cfg=Obj(joints=[Obj(name="right_index_MCP_FE" if i==5 else str(i),
                                     min_rad=-1,max_rad=1) for i in range(22)])
            self.enabled=False
            self.protected=False
            self.q=np.zeros(22)
            self.seq=0
        def _raise_if_error(self,ok,name):
            if not ok: raise RuntimeError(name+" collision rejection")
        def read_state(self):
            self.seq+=1
            return Obj(sequence=self.seq,joint_pos=self.q)
        def read_tactile_frame(self,*_):
            self.seq+=1
            return Obj(frame_id=self.seq,f6=np.zeros(6))
        def set_joint_position(self,q,interpolate):
            assert self.protected
            if refuse:return False
            self.q=np.array(q)
            return True
        def get_enable_state(self):return True,self.enabled
        def get_control_mode(self):return True,1
        def get_control_source(self):return True,2
        def get_speed_coeff(self):return True,.15
        def get_current_coeff(self):return True,.2
        def enable_collision_protection(self,v):self.protected=v;return True
        def is_collision_protection_enabled(self):return self.protected
        def set_enable_state(self,v):self.enabled=v;return True
    h=Hand()
    arm=Obj(read_arm_state=lambda *_:Obj(joint_pos=np.zeros(7),cur_state=0,err_code=0))
    monkeypatch.setattr(cal,"open_state_sources",lambda *a,**kw:(arm,{"right":h}))
    result=cal.index_trial({"arm":"right","rig":{},"camera_serial":"x","up_axis_mount":[0,0,1],"clearance":{}},
                           Obj(snapshot=snap,extrinsics=None,joints=state),tmp_path/"trial",lambda:False)
    assert result["disabled_confirmed"] and not h.enabled and not result["loaded_contact_test"]
    assert result["step_sent"] != refuse
    assert result["status"] == ("blocked" if refuse else "unloaded_step_measured")
