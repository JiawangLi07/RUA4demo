import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from sim.isaac import __main__ as cli


def test_rejected_grasp_is_reported_after_kit_exits_cleanly(tmp_path, monkeypatch):
    out = tmp_path / "episode"
    monkeypatch.setattr(cli, "load_case", lambda *_: {"physics_dt": 1 / 240})

    def kit(command):
        assert command[1:3] == ["-m", "sim.isaac.native_runner"]
        assert command[4:] == ["--headless", "--device", "cuda:2"]
        for filename, data in {
            "result": {"passed": False, "reason": "failed trial lift", "frames": 0},
            "events": [],
            "physics": [],
            "frames": [],
        }.items():
            (Path(command[3]) / (filename + ".json")).write_text(json.dumps(data))
        return CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", kit)
    assert (
        cli.main(
            [
                "--case",
                "screwdriver",
                "--assets",
                str(tmp_path),
                "--out",
                str(out),
                "--headless",
                "--device",
                "cuda:2",
            ]
        )
        == 1
    )


def test_kit_shutdown_cannot_hide_a_native_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_case", lambda *_: {})

    def kit(command):
        (Path(command[3]) / "failure.json").write_text('{"traceback": "native error"}')
        return CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", kit)
    with pytest.raises(RuntimeError, match="Isaac process failed"):
        cli.main(["--case", "screwdriver", "--assets", str(tmp_path), "--out", str(tmp_path / "episode")])
