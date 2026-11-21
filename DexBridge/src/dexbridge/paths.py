"""Single place that resolves repository and third-party locations.

Third-party code and robot assets are never tracked. They live under ``third_party/<name>`` (clones or
symlinks, see ``third_party/manifest.json`` and ``tools/setup_third_party.py``). ``DEXBRIDGE_THIRD_PARTY``
relocates the whole directory; ``DEXJOCO_ROOT`` keeps its historical meaning for the Panda/Allegro source.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
REPORTS = REPO_ROOT / "reports"
THIRD_PARTY = Path(os.environ.get("DEXBRIDGE_THIRD_PARTY", REPO_ROOT / "third_party"))
SHARPA_SDK_SAMPLE = Path("/opt/sharpa-wave-sdk/sample/python/sharpa_wave_example.py")
ASTRA_ENV = Path.home() / ".config/dexbridge/astra.env"


def third_party(name: str, *parts: str) -> Path:
    root = THIRD_PARTY / name
    if not root.exists():
        raise FileNotFoundError(f"missing third_party/{name}; run `make third-party` (see third_party/manifest.json) "
                                "or set DEXBRIDGE_THIRD_PARTY")
    return root.joinpath(*parts)


def dexjoco_root() -> Path:
    override = os.environ.get("DEXJOCO_ROOT")
    return Path(override) if override else third_party("dexjoco")


def dx2_root() -> Path:
    """Frozen dx2 baseline (``experiments/dexjoco_xmimic``); used only by the Allegro regression."""
    return third_party("dexjoco_xmimic")
