from types import SimpleNamespace
from research_loop import ResearchLoop, ResearchStop
import pytest


def test_stopped_research_retained_for_next_engineer(tmp_path):
    lib = SimpleNamespace(root=tmp_path / "library")
    contract = {"kind": "joint_delta", "arm": "right", "object": "none"}
    first = tmp_path / "first"
    first.mkdir()
    decision = {"action": "stop", "hypothesis": "missing signal",
                "instruction": "inspect feedback", "expected_observation": "fresh sample"}
    def unexpected(*args):
        pytest.fail("stopped experiment invoked engineer")
    loop = ResearchLoop(lib, first, "move", contract, "offline", lambda _: decision, unexpected)
    with pytest.raises(ResearchStop):
        loop.propose("cells", {"round": 0})
    loop.finish({"status": "blocked", "reason": "inspect feedback"})
    second = tmp_path / "second"
    second.mkdir()
    def researcher(brief):
        assert brief["prior_experiments"][0]["status"] == "blocked"
        return {**decision, "action": "experiment"}
    def engineer(kind, request):
        assert request["research_plan"]["hypothesis"] == "missing signal"
        return {"reason": "test revised assumption"}
    resumed = ResearchLoop(lib, second, "move", contract, "offline", researcher, engineer)
    assert resumed.propose("cells", {"round": 0})["reason"] == "test revised assumption"


def test_selected_cross_task_skill_supplied_to_engineer(tmp_path):
    from task_library import SkillLibrary
    library = SkillLibrary(tmp_path/"library")
    env = {"backend":"offline"}
    source = "def run(robo, params):\n    return robo.observe()\n"
    old = {"kind":"locate","arm":"none","object":"tube"}
    entry = library.publish("tube",old,env,source,[{"success":True,"case":i} for i in range(3)],"test")
    def researcher(brief):
        assert brief["environment_facts"]["execution"] == "offline"
        assert entry["skill_key"] in [r["id"] for r in brief["available_implementations"]]
        return {"action":"experiment","hypothesis":"reuse localization","instruction":"adapt to cork",
                "expected_observation":"measured depth", "skill_references":[entry["skill_key"]]}
    def engineer(kind,request):
        assert request["implementations"][0]["source"] == source
        return {"source":source}
    out=tmp_path/"new"
    out.mkdir()
    loop=ResearchLoop(library,out,"cork",{**old,"object":"cork"},env,researcher,engineer)
    loop.propose("cells", {"round":0})
