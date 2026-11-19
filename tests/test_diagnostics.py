import pytest
from exploration.diagnostics import Diagnostics, diagnose, validate_cell


@pytest.mark.parametrize("ready", [False, True])
def test_readonly_loop_before_motion(tmp_path, ready):
    class Backend(Diagnostics):
        def profile_state(self):
            return {"valid":ready,"blockers":[] if ready else ["missing TCP"]}
        def inspect(self, name, box):
            if name == "clearance":
                return {"passed":False,"reason":"candidate rejected"}
            return {"tool":name}
    backend = Backend(None, "right", "shuttlecock", tmp_path)
    class Channel:
        def request(self, kind, packet):
            assert kind == "diagnostic_request"
            if not packet["history"]:
                return {"action":"cell","source":"robo.inspect('profile')\nrobo.inspect('clearance')","reason":"measure"}
            if len(packet["history"]) == 1:
                assert len(packet["history"][0]["calls"]) == 2
                assert not packet["history"][0]["error"]
                return {"action":"proceed","reason":"evaluate gate","source":""}
            if len(packet["history"]) == 2:
                entry = packet["history"][-1]
                assert entry["error"] is None and entry["blockers"] == ["missing TCP"]
                return {"action":"cell","source":"robo.inspect('fixed_arm_escape')","reason":"explore"}
            assert packet["history"][-1]["calls"][0]["tool"] == "fixed_arm_escape"
            return {"action":"stop","source":"","reason":"missing TCP"}
    result = diagnose(backend, Channel(), tmp_path)
    assert (result is None) == ready
    assert (tmp_path/"workspace/cell_001.py").exists()
    assert len((tmp_path/"calls.jsonl").read_text().splitlines()) == (2 if ready else 3)


@pytest.mark.parametrize("source", ["robo.set_hand('right', [])", "import os", "robo.inspect('force', 1)",
                                    "robo.inspect('locate', [1,2,3,robo.move()])"])
def test_readonly_cell_rejects_other_capabilities(source):
    with pytest.raises(ValueError):
        validate_cell(source)
