"""Run GPT-6.2 Astra locally against DexHumanid-RUA on 4090-zhenji."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from urllib.request import getproxies
import json
import os
import shlex
import socket
import subprocess
import tempfile
import time


HOST = "4090-zhenji"
REMOTE = "/home/admin2026/lianjh/DexHumanid-RUA"
READY = "PYRUA_SHARPA_READY "
MODEL = "gpt-6.2-astra"
MODEL_REPLY_TIMEOUT_S = 300
MODEL_FLAGS = ("-m", MODEL, "-c", 'model_reasoning_effort="medium"')


# The model only needs the supplied robot tools and its own observations.
MODEL_TOOL_FLAGS = ["-c", "features.shell_tool=false", "-c", "features.unified_exec=false",
                    "-c", "features.computer_use=false", "-c", 'web_search="disabled"']

def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_until_ready(ssh, stdout, stderr, ready_prefix: str = READY):
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        for line in stdout.read_text().splitlines():
            if line.startswith(ready_prefix):
                return json.loads(line.removeprefix(ready_prefix))
        if ssh.poll() is not None:
            raise RuntimeError(stderr.read_text().strip() or "Robot tool server exited")
        time.sleep(0.2)
    raise TimeoutError(f"Robot tool server startup timed out. See {stderr}")


def model_environment():
    env = os.environ.copy()
    proxies = getproxies()
    for protocol in ("http", "https"):
        if protocol in proxies:
            env.setdefault(f"{protocol.upper()}_PROXY", proxies[protocol])
    no_proxy = ",".join(filter(None, [proxies.get("no"), "127.0.0.1", "localhost"]))
    env["NO_PROXY"] = env["no_proxy"] = no_proxy
    return env


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Natural-language observation or localization task")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Use one fresh RGB-D snapshot")
    mode.add_argument("--episode", help="Path to a RealSense recording on the server")
    parser.add_argument("--frame", default=1, type=int)
    parser.add_argument("--calibration", help="Path to the camera-to-robot transform on the server")
    parser.add_argument("--codex", default="codex", help="Local Codex CLI")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "local_runs")
    args = parser.parse_args()
    if args.episode:
        parser.error("Autonomous observation accepts a procedural scene or live input")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y%m%dT%H%M%S-")
    out = Path(tempfile.mkdtemp(prefix=prefix, dir=args.output.resolve()))
    port = free_port()
    remote_args = [f"{REMOTE}/run.sh", "serve", "--port", str(port), "--seconds", "170",
                   "--frame", str(args.frame)]
    if args.live:
        remote_args.append("--live")
    if args.episode:
        remote_args += ["--episode", args.episode]
    if args.calibration:
        remote_args += ["--calibration", args.calibration]
    ssh_command = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=8",
        "-L", f"127.0.0.1:{port}:127.0.0.1:{port}", HOST, "exec " + shlex.join(remote_args),
    ]
    ssh_stdout, ssh_stderr = out / "ssh.stdout", out / "ssh.stderr"
    with ssh_stdout.open("w") as stdout, ssh_stderr.open("w") as stderr:
        ssh = subprocess.Popen(ssh_command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
        try:
            ready = wait_until_ready(ssh, ssh_stdout, ssh_stderr)
            print(f"Server run directory {ready['run']}", flush=True)
            model_command = [
                args.codex, "-a", "never", "exec", "--ignore-user-config", "--ignore-rules",
                *MODEL_TOOL_FLAGS, "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                *MODEL_FLAGS,
                "-c", f"mcp_servers.pyrua_sharpa.url={json.dumps(ready['url'])}",
                "-c", "mcp_servers.pyrua_sharpa.required=true",
                "-c", "project_doc_max_bytes=0", "--json",
                "-o", str(out / "answer.md"), "-",
            ]
            task = (
                "Use the pyrua_sharpa MCP tools for this robot task. Call observe first. "
                "For a visible target you can call segment on the observed frame, optionally rank candidate masks with rank_labels, then use locate_mask. If using a box, draw it in the image's reported "
                "pixel dimensions, then call locate with the exact frame_id and bbox_xyxy. "
                "Use the measured tool return for metric positions and report its coordinate frame. "
                "Skip objects you cannot identify. These tools perform observation, RGB-D "
                "localization, and optional kinematic preview. Report the source as procedural or live. "
                "Use Chinese for the final answer.\n\n" + args.task
            )
            with (out / "model.jsonl").open("w") as events, (out / "model.stderr").open("w") as errors:
                result = subprocess.run(
                    model_command, input=task, text=True, stdout=events, stderr=errors,
                    cwd=out, env=model_environment(), timeout=145,
                )
            if result.returncode:
                raise RuntimeError(f"Codex exited with code {result.returncode}. See {out}")
            print((out / "answer.md").read_text().strip())
            print(f"Local model logs {out}", flush=True)
        finally:
            ssh.stdin.close()
            try:
                ssh.wait(timeout=8)
            except subprocess.TimeoutExpired:
                ssh.terminate()
                ssh.wait(timeout=8)


if __name__ == "__main__":
    main()
