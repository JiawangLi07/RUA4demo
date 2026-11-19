from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import perception_backend as pb
from cli import REFERENCE_EPISODE
from hardware import read_recorded_state
from robot import SnapshotBackend, RobotTools
from rgbd import HEAD_SERIAL, replay


@pytest.fixture
def backend(tmp_path):
    snapshot = replay(REFERENCE_EPISODE, 1, HEAD_SERIAL)
    joints = read_recorded_state(REFERENCE_EPISODE, snapshot.monotonic_ns)
    return SnapshotBackend(snapshot, joints, np.eye(4), tmp_path)


def test_mask_excludes_box_background_and_cannot_be_reused_on_stale_frame(backend, monkeypatch):
    mask = np.zeros((480,640),bool)
    mask[300,350:360] = True
    seen = []
    class Client:
        def __init__(self,endpoint):
            seen.append(endpoint)
        def segment(self,image,*,text_prompt,point):
            seen.append(point)
            return SimpleNamespace(found=True,score=.93,mask=mask)
    monkeypatch.setenv("PYRUA_SAM3_ENDPOINT","http://127.0.0.1:8790")
    monkeypatch.setattr(pb,"sam3_client_class",lambda:Client)
    tools = RobotTools(backend)
    reply = tools.execute_tool("segment", {"frame_id":backend.frame_id, "model":"sam3",
                                           "prompt":{"point_xy":[355,300]}}).result
    assert reply["mask_pixels"] == 10 and seen[-1] == [300,355]
    assert (backend.out / (reply["mask_id"]+".png")).is_file()
    # Inject registered points, including background depth and an out-of-bounds pixel.
    backend.registered = (np.array([[x,300] for x in range(345,360)]+[[-1,300]]),
                          np.array([[0,0,9]]*5+[[0,0,.7]]*10+[[0,0,9]]))
    result = tools.robot.locate_mask(backend.frame_id,reply["mask_id"])
    assert result["samples"] == 10 and result["point_mount_m"] == [0.,0.,.7]
    with pytest.raises((ValueError, RuntimeError), match="Frame mismatch"):
        backend.perception.locate_mask("stale",reply["mask_id"])
    with pytest.raises(ValueError, match="Unknown mask"):
        backend.perception.rank_labels(backend.frame_id,["table"],"other")


def test_invalid_prompts_do_not_invoke_model(backend):
    for args in [("sam",{"text":"ball"}), ("sam3",{"bbox_xyxy":[0,1,5,6]}),
                 ("sam",{"point_xy":[640,480]}), ("sam",{"point_xy":[0,0],"text":"ball"})]:
        with pytest.raises(ValueError):
            backend.perception.segment(backend.frame_id,*args)


def test_upstream_program_attaches_requested_frame_and_records_task(backend, tmp_path):
    from pyrualean._backend import Ledger
    from pyrualean.arms import ArmToolkit
    from cells_robot import StationRobot
    from cells_server import run_script
    robot = StationRobot(backend,ledger=Ledger())
    toolkit = ArmToolkit(robot,arm="program",workspace=tmp_path/"w",images="on-demand")
    result = toolkit.execute_tool("run_program",{"code":"def run(robo):\n    robo.show('head')"})
    assert result.result["status"] == "completed" and len(result.images)==1
    summary = run_script(toolkit,robot,{"task":"check error", "programs":["def run(robo):\n    raise ValueError('bad')"],
                        "finish":{"status":"success","summary":"incorrect"}},tmp_path)
    assert summary["task"] == "check error" and summary["finish"]["status"] == "stuck"
