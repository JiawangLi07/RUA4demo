"""Environment facts (L0), relevant experiment reports (L1), and frozen code (L2)."""
from pathlib import Path
import json
from task_library import atomic_json, digest, skill_key


def affinity(a, b):
    def features(c):
        steps = c.get("steps", [c])
        return {(key, s.get(key)) for s in steps for key in ("kind","object","arm")
                if s.get(key) not in (None, "none")}
    return len(features(a) & features(b))


def gather(library, environment, contract):
    sig = digest(environment)
    physical = isinstance(environment, dict) and environment.get("backend") in ("physical-cube", "physical-shuttlecock")
    facts = {"environment":environment,
             "units":{"arm":"degrees","hand":"radians","position":"meters"},
             "execution":"physical" if physical else "offline", "source":"execution_environment and deployed action interface"}
    atomic_json(library.root / "facts" / (sig + ".json"), facts)
    memories = []
    for path in (library.root / "research").glob("*/*.json"):
        record = json.loads(path.read_text())
        old = record.get("contract", {})
        # The key protects older reports lacking an explicit environment id too.
        if path.parent.name != skill_key(old, environment):
            continue
        score = affinity(contract, old)
        if score:
            memories.append((score, path.stat().st_mtime_ns, str(path), record))
    memories.sort(key=lambda r: (r[0],r[1]), reverse=True)

    skills = {}
    # Test fixtures and older libraries may have only reports.
    if (library.root / "index.json").exists():
        with library.locked() as index:
            active = [(k,v["active"]) for k,v in index["skills"].items()]
        ranked = []
        for key, rev in active:
            meta_path = library.root / "skills" / key / (rev+".json")
            meta = json.loads(meta_path.read_text())
            old = meta["contract"]
            if meta["environment"] == environment and affinity(contract, old):
                ranked.append((affinity(contract, old), meta_path.stat().st_mtime_ns, old))
        for _,_,old in sorted(ranked, key=lambda r:(r[0],r[1]), reverse=True)[:5]:
            row = library.lookup(old, environment)
            if row:
                skills[row["skill_key"]] = row
    return facts, [{"path":path,"report":r} for _,_,path,r in memories[:3]], skills
