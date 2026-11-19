"""File-only task trials with an independent, host-owned goal checker.

This worker is invoked in a separate process with a hard timeout. All motion
uses the existing virtual-joint backend or deterministic cube fixture.
"""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import ast
import base64
import json
import sys

import numpy as np
from PIL import Image

from pyrualean._backend import Ledger
from pyrualean.arms import ArmToolkit
from pyrualean.prompt import api_reference

from cells_robot import SimMotionBackend, StationRobot
from autonomous_scene import observation, surface_point
from exploration.grasp_program import Program
from exploration.offline import OfflineTrials
from hardware import arm_settings, hand_settings

from task_library import atomic_json
from perception_backend import capabilities


KINDS = ("joint_delta", "joint_target", "hand_target", "locate", "grasp_cube", "unsupported")
OBJECTS = ("shuttlecock", "tube")


def contract_from(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a goal object")
    kind = value.get("kind")
    if kind == "sequence":
        raw = value.get("steps")
        if not isinstance(raw, list) or not raw or any(
                not isinstance(v, dict) or v.get("kind") == "sequence" for v in raw):
            raise ValueError("Expected a nonempty ordered list of atomic goals")
        steps = [contract_from(v) for v in raw]
        if len(steps) == 1:
            return steps[0]
        if any(step["kind"] == "grasp_cube" for step in steps):
            raise ValueError("Cube fixture resets the scene; a shared object-state backend is needed for composition")
        return {"kind": "sequence", "object": "none", "arm": "none", "steps": steps}
    arm = value.get("arm", "none")
    obj = value.get("object", "none")
    values = value.get("values", [])
    if kind not in KINDS:
        raise ValueError("No independent goal checker for this task kind")
    if kind == "unsupported":
        raise ValueError(value.get("reason") or "Task has no offline backend and goal checker")
    if arm not in ("left", "right", "none"):
        raise ValueError("Expected left, right or none for arm")
    if kind in ("joint_delta", "joint_target", "hand_target"):
        length = 22 if kind == "hand_target" else 7
        if (arm == "none" or not isinstance(values, list) or len(values) != length
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)):
            raise ValueError(f"{kind} needs an arm and {length} finite goal values")
        obj = "none"
        values = [float(v) for v in values]
    elif kind == "locate":
        if obj not in OBJECTS:
            raise ValueError("Procedural localization supports shuttlecock and tube")
        arm, values = "none", []
    else:
        if arm != "right":
            raise ValueError("The current cube fixture represents a right-hand grasp")
        obj, values = "cube", []
    return {"kind": kind, "arm": arm, "object": obj, "values": values}


def validate_source(source, *, legacy_location=False):
    if not isinstance(source, str) or len(source.encode()) > 16384:
        raise ValueError("Expected a Python skill of at most 16 KiB")
    tree = ast.parse(source)
    body = tree.body
    if legacy_location:
        body = [node for node in body if not isinstance(node, (ast.Import, ast.ImportFrom))
                and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                         and isinstance(node.value.value, str))]
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != "run":
        raise ValueError("Skill must contain one function: def run(robo, params)")
    function = body[0]
    legacy = (legacy_location and [arg.arg for arg in function.args.args] == ["robo"]
              and [arg.arg for arg in function.args.kwonlyargs] == ["label", "bbox_xyxy"])
    normal = ([arg.arg for arg in function.args.args] == ["robo", "params"]
              and not function.args.kwonlyargs and len(tree.body) == 1)
    if (not (normal or legacy)
            or function.decorator_list or function.args.defaults
            or any(value is not None for value in function.args.kw_defaults)
            or function.args.posonlyargs
            or function.args.vararg or function.args.kwarg or function.returns
            or any(arg.annotation for arg in function.args.args)):
        raise ValueError("Use the plain function signature run(robo, params)")
    return legacy


class ObservedSimBackend(SimMotionBackend):
    def __init__(self, *args):
        super().__init__(*args)
        self.readings = []

    def execute_tool(self, name, input_dict):
        result = super().execute_tool(name, input_dict)
        self.readings.append({"tool": name, "inputs": deepcopy(input_dict),
                              "result": deepcopy(result)})
        return deepcopy(result)


def initial_state(case=0):
    return observation(case)


def describe(archived=None):
    if archived:
        from rgbd import archived_observation
        snapshot, joints, _ = archived_observation(Path(archived["run"]), archived["frame_id"])
    else:
        snapshot, joints = initial_state()
    image = BytesIO()
    Image.fromarray(snapshot.rgb).save(image, format="JPEG", quality=70)
    return {"source": snapshot.source if archived else "procedural_scene", "archive": archived, "joints": joints,
            "api": api_reference(StationRobot), "perception_backends":capabilities(),
            "image_base64": base64.b64encode(image.getvalue()).decode()}


def held_out(contract, case):
    result = deepcopy(contract)
    if result["kind"] == "sequence":
        result["steps"] = [held_out(step, case) for step in result["steps"]]
    elif case and result["kind"] == "joint_delta":
        result["values"] = [v * (0.5 if case == 1 else 0.75) for v in result["values"]]
    elif case and result["kind"] in ("joint_target", "hand_target"):
        bounds = (np.array([(j.min_rad, j.max_rad) for j in hand_settings().joints])
                  if result["kind"] == "hand_target" else
                  np.array([(j.min_deg, j.max_deg) for j in arm_settings().joints]))
        # Held-out goals remain within native limits and differ from training.
        center = bounds.mean(axis=1)
        result["values"] = (np.array(result["values"]) * (1 - 0.01 * case)
                            + center * (0.01 * case)).tolist()
    return result


def run_trial(source, requested, case, out):
    contract = held_out(requested, case)
    if contract["kind"] == "grasp_cube":
        path = out / "skill.py"
        path.write_text(source)
        program = Program.read(path, OfflineTrials.grasps)
        result = OfflineTrials().run(program, out, case, cancelled=lambda: False)
        return {**result, "case": case, "contract": contract}

    snapshot, joints = initial_state(case)
    # Reuse must respond to a fresh initial state, not a memorized joint vector.
    for side in ("left", "right"):
        joints["arm_deg"][side][0] += (0, 0.7, -0.9)[case]
    backend = ObservedSimBackend(snapshot, joints, None, out)
    robot = StationRobot(backend, ledger=Ledger())
    sequence = contract["kind"] == "sequence"
    steps = contract["steps"] if sequence else [contract]
    toolkit = ArmToolkit(robot, arm="cells", workspace=out / "workspace", max_turns=len(steps))
    reports = []
    for i, step in enumerate(steps):
        before = backend.execute_tool("view_env_state", {"step": -1})["joints"]
        backend.readings.clear()
        if sequence:
            # Execute the frozen composition itself. Each yield returns control
            # to the host verifier before the next skill can change the scene.
            code = (source + f"\nsequence_iter = run_steps(robo, {contract!r})\n" if i == 0 else "")
            code += "next(sequence_iter)"
        else:
            validate_source(source)
            invoke = f"run(robo, {step!r})"
            code = source + "\n" + invoke
        execution = toolkit.execute_tool("python", {"code": code}).result
        after = backend.execute_tool("view_env_state", {"step": -1})["joints"]
        report = check_step(step, backend, before, after, execution, toolkit.sandbox_flags, case)
        reports.append(report)
        if not report["success"]:
            break
    robot._ledger.write_jsonl(out / "calls.jsonl")
    common = {"case": case, "contract": contract, "source": "procedural_scene"}
    if not sequence:
        return {**reports[0], **common}
    success = len(reports) == len(steps) and all(row["success"] for row in reports)
    return {**common, "success": success, "reason": "ordered_goals_reached" if success else reports[-1]["reason"],
            "steps": reports, "completed_steps": sum(r["success"] for r in reports)}


def check_step(contract, backend, before, after, execution, sandbox_flags, case):
    reason, passed = "goal_not_reached", False
    error = execution.get("exception") or execution.get("error")
    if error or sandbox_flags:
        reason = "program_error"
    elif contract["kind"] == "locate":
        reference = surface_point(backend.snapshot, contract["object"], case)
        matches = [row["result"] for row in backend.readings
                   if row["tool"] in ("locate","locate_mask") and row["result"].get("found")]
        try:
            selected = ast.literal_eval(execution.get("value") or "None")
        except (ValueError, SyntaxError):
            selected = None
        if isinstance(selected, dict) and selected.get("found") is True:
            point = selected.get("point_camera_m")
            passed = (any(point == row.get("point_camera_m") for row in matches)
                      and np.linalg.norm(np.array(point) - reference) < 0.025 and after == before)
        if passed:
            reason = "measured_target_localized"
        elif matches and (not isinstance(selected, dict) or "point_camera_m" not in selected):
            reason = "return_one_selected_locate_or_locate_mask_result"
        else:
            reason = "target_not_localized"
    else:
        side = contract["arm"]
        group = "hand_rad" if contract["kind"] == "hand_target" else "arm_deg"
        start = np.array(before[group][side])
        expected = np.array(contract["values"])
        if contract["kind"] == "joint_delta":
            expected += start
        limits = (np.array([(j.min_rad, j.max_rad) for j in hand_settings().joints])
                  if group == "hand_rad" else
                  np.array([(j.min_deg, j.max_deg) for j in arm_settings().joints]))
        passed = (np.all(expected >= limits[:, 0]) and np.all(expected <= limits[:, 1])
                  and np.allclose(after[group][side], expected, atol=0.001, rtol=0))
        for other_group in ("arm_deg", "hand_rad"):
            for other_side in ("left", "right"):
                if (other_group, other_side) != (group, side):
                    passed = passed and after[other_group][other_side] == before[other_group][other_side]
        reason = "measured_joint_goal_reached" if passed else "joint_goal_or_other_limb_mismatch"
    return {"success": bool(passed), "reason": reason, "contract": contract,
            "before": before, "after": after, "execution": execution, "sandbox_flags": sandbox_flags}



if __name__ == "__main__":
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    try:
        report = run_trial(request["source"], contract_from(request["contract"]),
                           request["case"], request_path.parent)
    except Exception as exc:
        report = {"success": False, "reason": "trial_error", "error": str(exc)}
    atomic_json(request_path.parent / "result.json", report)
