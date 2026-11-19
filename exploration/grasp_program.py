"""Compile bounded grasp proposals to the Python skill that a trial executes."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path


OFFSET_LIMIT_M = 0.012
LIFT_RANGE_M = (0.06, 0.08)
MIN_OBJECT_LIFT_M = 0.04


def vector(values, length, name):
    if not isinstance(values, (tuple, list)) or len(values) != length:
        raise ValueError(f"{name}: expected {length} values")
    if any(type(value) not in (int, float) or not isfinite(value) for value in values):
        raise ValueError(f"{name}: values must be finite numbers")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class Program:
    offset_m: tuple[float, float, float]
    lift_m: float
    grasp: str
    recheck: bool

    @classmethod
    def parse(cls, value, grasps):
        if not isinstance(value, dict) or set(value) != {"offset_m", "lift_m", "grasp", "recheck"}:
            raise ValueError("Expected offset_m, lift_m, grasp and recheck")
        offset = vector(value["offset_m"], 3, "offset_m")
        if any(abs(delta) > OFFSET_LIMIT_M for delta in offset):
            raise ValueError(f"offset_m must be within ±{OFFSET_LIMIT_M} m per axis")
        lift, = vector([value["lift_m"]], 1, "lift_m")
        if not LIFT_RANGE_M[0] <= lift <= LIFT_RANGE_M[1]:
            raise ValueError(f"lift_m must be in {LIFT_RANGE_M}")
        grasp = value["grasp"]
        if not isinstance(grasp, str) or grasp not in grasps or type(value["recheck"]) is not bool:
            raise ValueError("Use a configured grasp name and a boolean recheck")
        return cls(tuple(round(delta, 4) or 0.0 for delta in offset),
                   round(lift, 4), grasp, value["recheck"])

    def source(self, *, single_cell=False):
        steps = [
            f"PLAN = {asdict(self)!r}",
            "",
            "",
            "def run(robo):",
        ]
        if single_cell:
            steps.append("    robo.locate_target()")
        steps += [
            '    robo.prepare(PLAN["offset_m"], PLAN["lift_m"], PLAN["grasp"])',
            "    robo.approach()",
        ]
        if self.recheck:
            steps.append("    robo.confirm_target()")
        steps += ["    robo.descend()", "    robo.close_hand()", "    return robo.lift()"]
        if single_cell:
            steps += ["", "", "result = run(robo)"]
        return "\n".join(steps) + "\n"

    @classmethod
    def from_proposal(cls, source, grasps, *, single_cell=False):
        """Accept equivalent Python formatting and literal plans, never arbitrary SDK code."""
        if not isinstance(source, str) or not 0 < len(source.encode()) <= 4096:
            raise ValueError("Supply a Python grasp program of at most 4096 bytes")
        try:
            body = ast.parse(source).body
            if len(body) != (3 if single_cell else 2) or not isinstance(body[0], ast.Assign):
                raise ValueError("Expected PLAN assignment and def run(robo)")
            assignment = body[0]
            if (len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name)
                    or assignment.targets[0].id != "PLAN"):
                raise ValueError("Use the literal variable PLAN")
            program = cls.parse(ast.literal_eval(assignment.value), grasps)
            expected = ast.parse(program.source(single_cell=single_cell)).body[1:]
            if [ast.dump(node) for node in body[1:]] != [ast.dump(node) for node in expected]:
                raise ValueError("run(robo) must follow the documented bounded primitive sequence")
            return program
        except (SyntaxError, RecursionError, TypeError) as exc:
            raise ValueError("Invalid Python grasp proposal") from exc

    @classmethod
    def read(cls, path: Path, grasps):
        if path.stat().st_size > 4096:
            raise ValueError("Grasp program exceeds 4096 bytes")
        source = path.read_text()
        try:
            first = ast.parse(source).body[0]
            if not isinstance(first, ast.Assign):
                raise ValueError("The first statement must define the grasp plan")
            program = cls.parse(ast.literal_eval(first.value), grasps)
        except (SyntaxError, IndexError, RecursionError) as exc:
            raise ValueError("Invalid grasp program") from exc
        if source not in (program.source(), program.source(single_cell=True)):
            raise ValueError("Grasp program must match the compiled primitive sequence")
        return program

    def run(self, robo, *, single_cell=False, path=None):
        namespace = {"__builtins__": {}, "robo": robo}
        source = self.source(single_cell=single_cell)
        if path is not None and path.read_text() != source:
            raise ValueError("Saved grasp cell differs from the checked program")
        exec(compile(source, str(path or "grasp_cell.py"), "exec"), namespace)
        return namespace["result"] if single_cell else namespace["run"](robo)


def verdict(before, lifted, up=(0, 0, 1)):
    """Judge the sampled object positions without reading the program or its claims."""
    if len(lifted) < 2 or any(point is None for point in lifted):
        return {"success": False, "reason": "cube_unobservable", "object_lift_m": None}
    rises = [sum((after - start) * axis for after, start, axis in zip(point, before, up))
             for point in lifted]
    if rises[0] < MIN_OBJECT_LIFT_M:
        reason = "cube_did_not_lift"
    elif min(rises) < MIN_OBJECT_LIFT_M or max(rises) - min(rises) > 0.015:
        reason = "cube_slipped"
    else:
        reason = "stable_lift"
    drift = [
        sum((after - start - rise * axis) ** 2
            for after, start, axis in zip(point, before, up)) ** 0.5
        for point, rise in zip(lifted, rises)
    ]
    if reason == "stable_lift" and max(drift) > 0.025:
        reason = "cube_moved_sideways"
    return {"success": reason == "stable_lift", "reason": reason,
            "object_lift_m": [round(float(rise), 4) for rise in rises]}
