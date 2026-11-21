"""Small v3 indices with verified, per-type lazy external numerical components."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ...paths import REPO_ROOT
from .artifacts import read_json, store_json


def _cache_root(index):
    directory = Path(index).resolve().parent
    if directory.name == "skills" and directory.parent.name == "configs":
        return directory.parent.parent
    return REPO_ROOT


class LazySequence(Sequence):
    """Materialize a component once on access, never on len()/truth testing."""

    def __init__(self, reference, count, *, root=REPO_ROOT):
        if type(count) is not int or count < 0:
            raise ValueError("invalid template component count")
        self.reference = reference
        self.count = count
        self.root = root
        self._values = None

    def __len__(self):
        return self.count

    def _load(self):
        if self._values is None:
            values = read_json(self.reference, root=self.root)
            if not isinstance(values, list) or len(values) != self.count:
                raise ValueError("template component count mismatch")
            self._values = values
        return self._values

    def __getitem__(self, index):
        return self._load()[index]

    def __iter__(self):
        return iter(self._load())


def save_templates(path, templates):
    """Write full-precision components and their v3 index."""
    path = Path(path)
    root = _cache_root(path)
    index = {key: value for key, value in templates.items()
             if key not in ("types", "visual_shape_evidence", "selection")}
    index.update(schema="dexbridge.grasp-postures.v3", storage="external-gzip-v1")
    if "visual_shape_evidence" in templates:
        index["visual_shape_evidence_file"] = store_json(templates["visual_shape_evidence"], root=root)
    index["selection_file"] = store_json(list(templates["selection"]), root=root)
    index["selection_count"] = len(templates["selection"])
    index["types"] = {}
    for name, card in templates["types"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("type name must be a filename component")
        modes = list(card["modes"])
        entry = {key: value for key, value in card.items() if key != "modes"}
        entry["mode_count"] = len(modes)
        entry["validated_shape_keys"] = sorted({mode["shape_key"] for mode in modes if mode["validated"]})
        entry["modes_file"] = store_json(modes, root=root)
        index["types"][name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_templates(path):
    """Read only the index; missing unused types do not prevent selection."""
    templates = json.loads(Path(path).read_bytes())
    if templates.get("schema") != "dexbridge.grasp-postures.v3" or templates.get("storage") != "external-gzip-v1":
        raise ValueError("expected a frozen v3 external-template index")
    root = _cache_root(path)
    templates["selection"] = LazySequence(templates["selection_file"], templates["selection_count"], root=root)
    for card in templates["types"].values():
        if not isinstance(card["validated_shape_keys"], list):
            raise ValueError("expected validated_shape_keys in type index")
        card["modes"] = LazySequence(card["modes_file"], card["mode_count"], root=root)
    return templates
