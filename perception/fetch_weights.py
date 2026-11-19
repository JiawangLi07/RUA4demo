"""Fetch pinned public Transformers checkpoints; verify blobs before atomic rename."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "sam": ("facebook/sam-vit-base", "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"),
    "siglip": ("google/siglip-base-patch16-224", "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"),
}


def verify(path, entry):
    lfs = entry.get("lfs")
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest() == (lfs["oid"] if lfs else entry["oid"])


def fetch(name, repo, rev):
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    target = ROOT / ".local/perception" / name
    target.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{base}/api/models/{repo}/tree/{rev}", timeout=30) as response:
        files = json.load(response)
    for entry in files:
        filename = entry["path"]
        if filename not in {"model.safetensors", "config.json", "preprocessor_config.json",
                             "special_tokens_map.json", "spiece.model", "tokenizer.json",
                             "tokenizer_config.json"}:
            continue
        path = target / filename
        if path.is_file() and verify(path, entry):
            continue
        with tempfile.NamedTemporaryFile(dir=target, delete=False) as f:
            tmp = Path(f.name)
        try:
            url = f"{base}/{repo}/resolve/{rev}/{filename}"
            subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "12",
                            "--max-time", "600", "--limit-rate", "12M", "-o", str(tmp), url], check=True)
            if not verify(tmp, entry):
                raise ValueError(f"Bad digest for {repo}/{filename}")
            tmp.replace(path)
            print(f"Verified {repo} {filename}", flush=True)
        finally:
            tmp.unlink(missing_ok=True)
    if not (target/"model.safetensors").is_file():
        raise ValueError(f"Missing weights for {repo}")


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        for job in [pool.submit(fetch, name, *model) for name, model in MODELS.items()]:
            job.result()
