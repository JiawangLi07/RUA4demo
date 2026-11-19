"""Cells-mode loop against the replay episode with the simulated motion backend."""

import json

import pytest

from pyrualean._backend import Ledger, ToolError
from pyrualean.arms import ArmToolkit, arm_contract
from pyrualean.prompt import api_reference

from cells_robot import ROBOT_WORDS, SimMotionBackend, StationRobot
from cells_server import run_script
from cli import REFERENCE_EPISODE
from hardware import read_recorded_state
from rgbd import HEAD_SERIAL, replay
from robot import SnapshotBackend


@pytest.fixture(scope="module")
def observation():
    snapshot = replay(REFERENCE_EPISODE, 1, HEAD_SERIAL)
    return snapshot, read_recorded_state(REFERENCE_EPISODE, snapshot.monotonic_ns)


@pytest.fixture()
def sim_session(observation, tmp_path):
    snapshot, joints = observation
    backend = SimMotionBackend(snapshot, joints, None, tmp_path)
    robot = StationRobot(backend, ledger=Ledger())
    toolkit = ArmToolkit(
        robot, arm="cells", workspace=tmp_path / "workspace", max_turns=8
    )
    return robot, toolkit, backend


def test_sim_motion_updates_virtual_joints_and_counts_steps(observation, tmp_path):
    snapshot, joints = observation
    backend = SimMotionBackend(snapshot, joints, None, tmp_path)
    target = list(joints["arm_deg"]["right"])
    target[0] += 5.0
    result = backend.execute_tool("move_joints", {"arm": "right", "joints_deg": target})
    assert result["moved"] and result["simulated"]
    state = backend.execute_tool("view_env_state", {"step": -1})
    assert state["joints"]["arm_deg"]["right"] == pytest.approx(target)
    assert backend.env_steps() == result["env_steps"]

    hand = [0.1] * 22
    result = backend.execute_tool("set_hand", {"side": "left", "joints_rad": hand})
    assert result["moved"]
    state = backend.execute_tool("view_env_state", {"step": 0})
    assert state["joints"]["hand_rad"]["left"] == pytest.approx(hand)


def test_motion_refused_on_readonly_backend(observation, tmp_path):
    snapshot, joints = observation
    robot = StationRobot(SnapshotBackend(snapshot, joints, None, tmp_path), ledger=Ledger())
    with pytest.raises((ToolError, ValueError), match="Unknown tool"):
        robot.move_joints("right", joints["arm_deg"]["right"])

    bad = list(joints["arm_deg"]["right"])
    bad[0] = 999.0
    robot = StationRobot(SimMotionBackend(snapshot, joints, None, tmp_path), ledger=Ledger())
    with pytest.raises(ToolError, match="joint limits"):
        robot.move_joints("right", bad)


def test_cells_persist_and_motion_feedback(sim_session):
    robot, toolkit, backend = sim_session
    first = toolkit.execute_tool("python", {"code": "start = robo.observe()\nhelper = 41"})
    assert first.result["cell"] == 1
    second = toolkit.execute_tool("python", {"code": "helper + 1"})
    assert second.result["value"] == "42"
    third = toolkit.execute_tool("python", {
        "code": (
            "import json\n"
            "target = start['joints']['arm_deg']['right']\n"
            "robo.move_joints('right', target)"
        )
    })
    result = third.result
    assert result["moved"] is True
    assert result["images"] == ["head.png"]
    tools = [call["tool"] for call in result["primitive_calls"]]
    assert "move_joints" in tools
    assert backend.env_steps() > 0
    assert toolkit.turns[2]["moved"] is True


def test_sandbox_blocks_os_import(sim_session):
    robot, toolkit, backend = sim_session
    result = toolkit.execute_tool("python", {"code": "import os"})
    assert "not available inside the policy sandbox" in result.result["exception"]
    assert toolkit.sandbox_flags and toolkit.sandbox_flags[-1]["flags"]


def test_contract_and_reference_render():
    contract = arm_contract(
        "cells", budget_s=240, max_env_steps=10000, max_programs=None,
        budget_mode="calls", robot=ROBOT_WORDS,
    )
    assert "python(code)" in contract
    assert "Tianji dual-arm" in contract
    reference = api_reference(StationRobot)
    assert "move_joints" in reference and "set_hand" in reference
    assert "preview_reach" in reference
    assert "world_map" not in reference and "terminated |" not in reference


def test_scripted_episode_end_to_end(sim_session, tmp_path):
    robot, toolkit, backend = sim_session
    script = {
        "task": "把右臂移动 5 度并报告关节",
        "cells": [
            "obs = robo.observe()",
            (
                "target = obs['joints']['arm_deg']['right']\n"
                "target[0] += 5.0\n"
                "out = robo.move_joints('right', target)\n"
                "print(out['max_joint_delta_deg'])"
            ),
            "robo.state()['joints']['arm_deg']['right'][0]",
        ],
        "finish": {"status": "success", "summary": "右臂已移动，虚拟关节回读一致"},
    }
    summary = run_script(toolkit, robot, script, tmp_path)
    assert summary["finish"]["status"] == "success"
    assert [turn["tool"] for turn in summary["turns"]] == ["python"] * 3 + ["finish"]
    assert all(turn["ok"] for turn in summary["turns"])
    cells = sorted((tmp_path / "workspace").glob("cell_*.py"))
    assert len(cells) == 3
    recorded = json.loads((tmp_path / "result.json").read_text())
    assert recorded["finish"]["summary"].startswith("右臂已移动")
    assert (tmp_path / "calls.jsonl").is_file()
