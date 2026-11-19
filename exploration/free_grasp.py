"""Execute a frozen free-exploration grasp proposal and verify it before publication."""
from hashlib import sha256
import json
from pathlib import Path

from task_library import atomic_json, atomic_text
from .grasp_program import Program
from .physical import PlacementNotChanged


def finish_and_verify(campaign, physical, program, parent):
    """Crash-resumable handoff to the existing independent verifier and skill library."""
    parent = Path(parent)
    result_path = parent/"result.json"
    pending = parent/"pending"/physical.skill
    if not result_path.exists():
        atomic_text(pending, program.source())
        atomic_json(result_path, {"status":"development_in_progress", "environment":physical.env,
                                  "program_sha256":sha256(program.source().encode()).hexdigest(),
                                  "candidate":str(pending), "trials":[], "development_positions":[]})
        first = campaign.trial(program,"free-development",0)
        state = json.loads(result_path.read_text())
        if not first.get("success"):
            state.update(status="development_failed",reason=first.get("reason"))
            atomic_json(result_path,state)
            return state
        state.update(status="awaiting_new_placement",trials=[{**first,"case":0}],
                     development_positions=[first["before_m"]])
        atomic_json(result_path,state)
    state = json.loads(result_path.read_text())
    if (state["environment"] != physical.env
            or state["program_sha256"] != sha256(program.source().encode()).hexdigest()):
        raise ValueError("Frozen candidate or execution environment changed")
    if state["status"] == "development_in_progress":
        # An uncertain hardware attempt cannot be rerun automatically.
        raise ValueError("Interrupted development trial requires evidence recovery")
    while state["status"] in ("awaiting_new_placement","verification_in_progress"):
        try:
            state = physical.verify_next(campaign,parent)
        except PlacementNotChanged as exc:
            state = json.loads(result_path.read_text())
            state["required_action"] = str(exc)
            atomic_json(result_path,state)
            break
    return state


def requested_program(plan, station):
    if station.contact_policy is None:
        raise ValueError("Contact proposal requires commissioned contact_policy")
    from .contact import ContactClosure
    import numpy as np
    ContactClosure(station.contact_policy,np.zeros(22))
    return Program.parse(plan,station.hand_targets)
