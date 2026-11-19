import json

from task_router import TaskRouter
from task_server import scripted_proposal

DELTA = {"kind": "joint_delta", "arm": "right", "object": "none", "values": [0]*6+[3]}
LOCATE = {"kind": "locate", "arm": "none", "object": "tube", "values": []}


def test_composition_learns_only_missing_then_reuses_whole_program(tmp_path):
    calls = []
    def propose(kind, request):
        calls.append(request["contract"]["kind"])
        return scripted_proposal(kind, request)
    def router(name):
        return TaskRouter(tmp_path/"library", tmp_path/name, resolve=lambda _: None,
                          propose=propose, legacy_root=tmp_path/"empty")
    goal = {"kind": "sequence", "steps": [LOCATE, DELTA, DELTA]}
    result = router("first").run("locate and move twice", goal)
    assert result["status"] == "success", result
    assert result["route"] == "composed" and calls == ["locate", "joint_delta"]
    assert [s["route"] for s in result["dependencies"]] == ["learned", "learned", "reused"]
    for trial in result["trials"]:
        a,b,c = trial["steps"]
        assert a["after"] == b["before"] and b["after"] == c["before"]
        assert c["after"]["arm_deg"]["right"][6] - a["before"]["arm_deg"]["right"][6] == 2*b["contract"]["values"][6]
    other = router("again").run("locate and move twice")
    assert other["route"] == "reused" and other["revision"] == result["revision"]
    assert calls == ["locate", "joint_delta"]


def test_whole_sequence_fails_if_second_step_exceeds_limits(tmp_path):
    # Both deltas are feasible from reset. Only the first is feasible in sequence.
    from hardware import arm_settings
    from task_trials import initial_state
    _, joints = initial_state()
    delta = (arm_settings().joints[6].max_deg-joints["arm_deg"]["right"][6])*0.7
    step = {**DELTA,"values":[0]*6+[delta]}
    contract = {"kind":"sequence","steps":[step, step, LOCATE]}
    router=TaskRouter(tmp_path/"library", tmp_path/"run", resolve=lambda _:contract,
                      propose=scripted_proposal, legacy_root=tmp_path/"empty")
    result=router.run("impossible cumulative motion")
    assert result["status"]=="failed"
    trial=result["trials"][-1]
    assert trial["completed_steps"]==1 and len(trial["steps"])==2
    assert router.library.lookup(result["contract"],router.environment) is None


def test_unverifiable_step_blocks_whole_task_before_any_trial(tmp_path):
    router=TaskRouter(tmp_path/"library",tmp_path/"run",resolve=None,propose=None)
    r=router.run("locate then grasp shuttlecock",{"kind":"sequence","steps":[LOCATE,
                 {"kind":"unsupported","reason":"requires shuttlecock contact feedback"}]})
    assert r["status"]=="blocked" and r["trials"]==[]


def test_existing_skill_revalidated_across_offline_environment_update(tmp_path):
    from task_library import SkillLibrary
    lib=SkillLibrary(tmp_path/"library")
    router=TaskRouter(lib.root,tmp_path/"new",resolve=lambda _:DELTA,propose=None)
    previous={**router.environment,"dependencies":"previous"}
    source=scripted_proposal("cells",{"contract":DELTA})["source"]
    lib.publish("old",DELTA,previous,source,[{"case":n,"success":True} for n in range(3)],"old")
    r=router.run("new")
    assert r["route"]=="imported" and len(r["trials"])==3


def test_refines_component_using_previous_step_actual_state(tmp_path):
    # This deliberately brittle skill works for all three atomic trials but
    # fails after the first delta changes joint 7 in the shared scene.
    from task_trials import initial_state
    _, joints = initial_state()
    limit = joints["arm_deg"]["right"][-1] + 1
    calls=[]
    def propose(kind, request):
        calls.append(request)
        reply=scripted_proposal(kind, request)
        if len(calls)==1:
            reply["source"]=reply["source"].replace("    target =", f"    assert obs['joints']['arm_deg']['right'][6] < {limit!r}\n    target =")
        return reply
    router=TaskRouter(tmp_path/"library",tmp_path/"run",resolve=None,propose=propose)
    r=router.run("two moves",{"kind":"sequence","steps":[DELTA,DELTA]})
    assert r["status"]=="success" and len(calls)==2
    assert calls[1]["sequence_step"]==1 and not calls[1]["evidence"][0]["success"]
    assert r["dependencies"][1]["embedded_revision"]!=r["dependencies"][1]["revision"]


def test_unknown_task_can_investigate_but_cannot_promote_itself(tmp_path):
    goal = {"kind": "unsupported", "reason": "Missing physical-contact evidence"}
    def research(brief):
        return {"action":"experiment","hypothesis":"inspect available observation",
                "instruction":"read state","expected_observation":"replay"}
    router=TaskRouter(tmp_path/"library",tmp_path/"run",resolve=lambda _:goal,
                      propose=scripted_proposal,researcher=research,rounds=1)
    r=router.run("new unmeasurable goal")
    assert r["status"]=="blocked" and r["route"]=="investigated"
    assert r["trials"][0]["reason"]=="goal_unverified"
    assert (tmp_path/"run/investigation-0.py").is_file()
    assert not (tmp_path/"library/index.json").exists()
