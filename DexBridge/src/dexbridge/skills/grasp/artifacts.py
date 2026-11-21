"""Content-addressed local cache; restoration is explicit and never automatic."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from ...paths import CONFIGS, REPO_ROOT

CACHE_DIR = "build/dexonomy/artifacts"


def load_policy():
    policy = json.loads((CONFIGS / "skills/artifacts.json").read_bytes())
    if policy.get("schema") != "dexbridge.artifact-policy.v1" or policy.get("cache_dir") != CACHE_DIR:
        raise ValueError("expected artifact-policy.v1 with build/dexonomy/artifacts cache")
    if "archive_uri" in policy:
        _archive_source(policy["archive_uri"])
    return policy


def _archive_source(uri):
    parsed = urlsplit(uri)
    path = unquote(parsed.path)
    if parsed.query or parsed.fragment or not path.startswith("/") or ".." in PurePosixPath(path).parts:
        raise ValueError("invalid archive URI path")
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        return "file", path
    if (parsed.scheme == "ssh" and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", parsed.netloc)
            and re.fullmatch(r"/[A-Za-z0-9_./-]+", path)):
        return "ssh", f"{parsed.netloc}:{path}"
    raise ValueError("archive URI must be file:// or ssh:// with a safe host and path")


def cache_path(reference, *, root=REPO_ROOT):
    digest = reference["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid artifact SHA256")
    if type(reference["size_bytes"]) is not int or reference["size_bytes"] < 0:
        raise ValueError("invalid artifact byte size")
    expected = f"{CACHE_DIR}/{digest}"
    if reference["cache_path"] != expected:
        raise ValueError("artifact cache path must be repository-relative and content-addressed")
    root = Path(root).resolve()
    cache = root / CACHE_DIR
    if cache.resolve() != cache:
        raise ValueError("artifact cache directory escapes through a symlink")
    path = cache / digest
    if path.resolve() != path:
        raise ValueError("artifact cache path escapes through a symlink")
    if "archive_uri" in reference:
        _archive_source(reference["archive_uri"])
        if PurePosixPath(unquote(urlsplit(reference["archive_uri"]).path)).name != digest:
            raise ValueError("archive blob must be named by SHA256")
    return path


def restore_command(reference):
    """A portable isolated command, run from the repository root."""
    return shlex.join(["python", "-I", "-c",
                       "import sys; sys.path.insert(0, 'src'); from dexbridge.skills.grasp.artifacts import main; main()",
                       "restore", "--sha256", reference["sha256"],
                       "--size-bytes", str(reference["size_bytes"]), "--cache-path", reference["cache_path"],
                       "--archive-uri", reference["archive_uri"]])


def _verify(content, reference):
    if len(content) != reference["size_bytes"]:
        raise ValueError(f"artifact byte size mismatch: {reference['cache_path']}")
    if hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise ValueError(f"artifact hash mismatch: {reference['cache_path']}")


def read_bytes(reference, *, root=REPO_ROOT):
    path = cache_path(reference, root=root)
    try:
        content = path.read_bytes()
    except FileNotFoundError as error:
        if "archive_uri" not in reference:
            raise FileNotFoundError(f"Missing bundled template component: {path}") from error
        raise FileNotFoundError(f"Missing artifact {reference['sha256']}. Restore from repository root:\n"
                                f"{restore_command(reference)}") from error
    _verify(content, reference)
    return content


def read_json(reference, *, root=REPO_ROOT):
    return json.loads(gzip.decompress(read_bytes(reference, root=root)))


def _temporary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".artifact-", dir=path.parent)
    os.close(fd)
    return Path(name)


def store_json(value, *, root=REPO_ROOT):
    """Write deterministic gzip bytes to the cache, without uploading anything."""
    content = gzip.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                       allow_nan=False).encode("utf-8"), compresslevel=6, mtime=0)
    digest = hashlib.sha256(content).hexdigest()
    policy = load_policy()
    reference = {"sha256": digest, "size_bytes": len(content), "cache_path": f"{CACHE_DIR}/{digest}"}
    if "archive_uri" in policy:
        reference["archive_uri"] = policy["archive_uri"].rstrip("/") + "/" + digest
        reference["restore_command"] = restore_command(reference)
    path = cache_path(reference, root=root)
    if path.exists():
        read_bytes(reference, root=root)
        return reference
    temporary = _temporary(path)
    try:
        temporary.write_bytes(content)
        os.replace(temporary, cache_path(reference, root=root))
    finally:
        temporary.unlink(missing_ok=True)
    return reference


def restore(reference):
    """Fetch once, verify before installing, and preserve existing bytes on failure."""
    path = cache_path(reference)
    if path.exists():
        read_bytes(reference)
        return path
    temporary = _temporary(path)
    try:
        scheme, source = _archive_source(reference["archive_uri"])
        if scheme == "file":
            shutil.copyfile(source, temporary)
        else:
            subprocess.run(["scp", "-B", "--", source, str(temporary)], check=True)
        _verify(temporary.read_bytes(), reference)
        os.replace(temporary, cache_path(reference))
    finally:
        temporary.unlink(missing_ok=True)
    return path


def query(templates, grasp_type, mode_id=None, max_nodes=3, features=None):
    """Return one real mode, or an SDK plan; numpy is needed only for plans."""
    if not 1 <= max_nodes <= 3:
        raise ValueError("max_nodes must be between 1 and 3")
    modes = templates["types"][grasp_type]["modes"]
    mode = None
    if mode_id is not None:
        mode = next((entry for entry in modes if entry["mode_id"] == mode_id), None)
        if mode is None:
            raise ValueError(f"unknown mode: {mode_id}")
    if features is not None:
        from .postures import choose_plan, plan_from_mode
        return (plan_from_mode(mode, features) if mode is not None
                else choose_plan(templates, grasp_type, features))
    if mode is None:
        mode = min(modes, key=lambda entry: (not entry["validated"], -entry["support_records"],
                                            entry["mode_id"]), default=None)
    if mode is None:
        raise ValueError(f"no measured modes for {grasp_type}")
    knots = mode["knots"]
    return {"grasp_type": grasp_type, "node_count": len(knots), "node_offset": 0,
            "mode": {**mode, "knots": knots[:max_nodes]},
            "omitted_nodes": max(0, len(knots) - max_nodes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("restore", help="Explicitly restore a single verified blob")
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--size-bytes", type=int, required=True)
    fetch.add_argument("--cache-path", required=True)
    fetch.add_argument("--archive-uri", required=True)
    inspect = commands.add_parser("query", help="Read one mode or generate a full-precision bbox plan")
    inspect.add_argument("--templates", type=Path, default=CONFIGS / "skills/sharpa_grasp_templates.json")
    inspect.add_argument("--type", dest="grasp_type", required=True)
    inspect.add_argument("--mode")
    inspect.add_argument("--max-nodes", type=int, choices=(1, 2, 3), default=3)
    inspect.add_argument("--features", type=Path, help="JSON with bbox_dims, T_obj_bbox, shape_key; produces a plan")
    args = parser.parse_args()
    if args.command == "restore":
        reference = {key: getattr(args, key) for key in ("sha256", "size_bytes", "cache_path", "archive_uri")}
        print(restore(reference))
    else:
        from .templates import load_templates
        features = json.loads(args.features.read_bytes()) if args.features else None
        print(json.dumps(query(load_templates(args.templates), args.grasp_type, args.mode,
                               args.max_nodes, features), ensure_ascii=False, allow_nan=False))
