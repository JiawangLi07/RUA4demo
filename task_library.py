"""Persistent, versioned Python skills and exact task aliases.

Entries are selected by task semantics and execution environment. Publication
holds only this library's lock, never a camera or robot lock.
"""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import os
import tempfile
import unicodedata


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(pending, path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def alias_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).strip().casefold().split())


def skill_key(contract, environment):
    # Numerical goals are runtime parameters. Every reuse is checked against
    # the new request; held-out trials also vary them before publication.
    identity = {key: contract[key] for key in ("kind", "arm", "object")}
    if contract["kind"] == "sequence":
        identity["steps"] = [skill_key(step, environment) for step in contract["steps"]]
    return digest({"task": identity, "environment": environment})


class SkillLibrary:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self, *, write=False):
        with (self.root / ".library.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            path = self.root / "index.json"
            index = json.loads(path.read_text()) if path.exists() else {
                "schema_version": 1, "skills": {}, "aliases": {},
            }
            if index.get("schema_version") != 1:
                raise ValueError("Unsupported skill-library index version")
            yield index

    def resolve_alias(self, task, environment):
        with self.locked() as index:
            row = index["aliases"].get(digest([alias_key(task), environment]))
            return None if row is None else row["contract"]

    def lookup(self, contract, environment):
        key = skill_key(contract, environment)
        with self.locked() as index:
            row = index["skills"].get(key)
            if row is None:
                return None
            revision = row["active"]
            if len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("Invalid skill revision")
            folder = self.root / "skills" / key
            metadata = json.loads((folder / f"{revision}.json").read_text())
            source = (folder / f"{revision}.py").read_text()
            if (sha256(source.encode()).hexdigest() != revision
                    or metadata.get("revision") != revision
                    or metadata["environment"] != environment
                    or metadata["skill_key"] != key):
                raise ValueError("Skill checksum or execution environment mismatch")
            return {**metadata, "source": source, "path": str(folder / f"{revision}.py")}

    def previous_candidate(self, contract, environment):
        """An older offline revision may seed a new validation, never a cache hit."""
        with self.locked() as index:
            for key, row in reversed(list(index["skills"].items())):
                revision = row["active"]
                if not all(len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                           for v in (key, revision)):
                    continue
                folder = self.root / "skills" / key
                try:
                    meta = json.loads((folder / f"{revision}.json").read_text())
                    old = meta["environment"]
                    if (old == environment or old.get("input_policy") != environment.get("input_policy")
                            or old["backend"] != environment["backend"]
                            or skill_key(contract, old) != key or meta["revision"] != revision):
                        continue
                    path = folder / f"{revision}.py"
                    if sha256(path.read_bytes()).hexdigest() == revision:
                        return path
                except (OSError, ValueError, KeyError):
                    continue
        return None

    def remember(self, task, contract, environment):
        with self.locked(write=True) as index:
            index["aliases"][digest([alias_key(task), environment])] = {
                "task": task, "contract": contract,
            }
            atomic_json(self.root / "index.json", index)

    def publish(self, task, contract, environment, source, trials, origin):
        if (len(trials) != 3 or {row.get("case") for row in trials} != {0, 1, 2}
                or not all(row.get("success") is True for row in trials)):
            raise ValueError("Publication requires a successful trial and two held-out checks")
        key = skill_key(contract, environment)
        revision = sha256(source.encode()).hexdigest()
        folder = self.root / "skills" / key
        with self.locked(write=True) as index:
            folder.mkdir(parents=True, exist_ok=True)
            program = folder / f"{revision}.py"
            if program.exists() and program.read_text() != source:
                raise ValueError("Immutable skill revision has changed")
            if not program.exists():
                atomic_text(program, source)
            metadata = {
                "skill_key": key, "revision": revision, "environment": environment,
                "contract": contract, "origin": origin, "validation": trials,
            }
            atomic_json(folder / f"{revision}.json", metadata)
            row = index["skills"].setdefault(key, {"revisions": []})
            if revision not in row["revisions"]:
                row["revisions"].append(revision)
            row["active"] = revision
            index["aliases"][digest([alias_key(task), environment])] = {
                "task": task, "contract": contract,
            }
            atomic_json(self.root / "index.json", index)
        return {**metadata, "path": str(program)}
