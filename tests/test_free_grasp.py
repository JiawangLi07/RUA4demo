from pathlib import Path
from types import SimpleNamespace as NS
import json
import pytest
from exploration.free_grasp import finish_and_verify
from exploration.physical import PlacementNotChanged
from task_library import atomic_json


class Program:
    def source(self):return "def run(robo):\n    return robo.lift()\n"


def test_frozen_candidate_automatic_verification_and_resume(tmp_path):
    calls=[]
    campaign=NS(trial=lambda *args: {"success":True,"before_m":[0,0,0],"run_dir":str(tmp_path)})
    def verify(campaign,parent):
        p=parent/"result.json";s=json.loads(p.read_text())
        calls.append(len(s["trials"]))
        s["trials"].append({"case":len(s["trials"]),"success":True})
        s["status"]="validated" if len(s["trials"])==3 else "awaiting_new_placement"
        atomic_json(p,s)
        return s
    physical=NS(skill="grasp.py",env={"fixture":True},verify_next=verify)
    result=finish_and_verify(campaign,physical,Program(),tmp_path)
    assert result["status"]=="validated" and calls==[1,2]
    finish_and_verify(campaign,physical,Program(),tmp_path)
    assert calls==[1,2]


def test_same_placement_returns_action_and_does_not_publish(tmp_path):
    campaign=NS(trial=lambda *args: {"success":True,"before_m":[0,0,0]})
    def verify(*_):raise PlacementNotChanged("restage object")
    result=finish_and_verify(campaign,NS(skill="grasp.py",env={},verify_next=verify),Program(),tmp_path)
    assert result["status"]=="awaiting_new_placement" and result["required_action"]=="restage object"
    assert len(result["trials"])==1


def test_failed_development_and_uncertain_attempt_do_not_replay(tmp_path):
    campaign=NS(trial=lambda *_:{"success":False,"reason":"object not lifted"})
    physical=NS(skill="grasp.py",env={},verify_next=lambda *_:pytest.fail("verification after failure"))
    assert finish_and_verify(campaign,physical,Program(),tmp_path)["status"]=="development_failed"
    state=json.loads((tmp_path/"result.json").read_text());state["status"]="development_in_progress"
    atomic_json(tmp_path/"result.json",state)
    with pytest.raises(ValueError,match="Interrupted"):
        finish_and_verify(campaign,physical,Program(),tmp_path)
