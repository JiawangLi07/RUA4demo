"""Combine independently generated Lightning Grasp proposals while preserving each source and row."""

import argparse
import json
from pathlib import Path

import numpy as np


def merge(files, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    array_keys = (
        "q",
        "object_pose",
        "q_mask",
        "target_pos",
        "target_normal",
        "contact_pos",
        "contact_normal",
        "contact_link_id",
    )
    pieces = {key: [] for key in array_keys}
    sources, indices, records = [], [], {}
    joint_names = link_names = None
    common = None
    for file in files:
        with np.load(file, allow_pickle=False) as z:
            receipt = json.loads(str(z["receipt"]))
            identity = tuple(
                receipt[k]
                for k in (
                    "upstream",
                    "input_sha256",
                    "hand_sha256",
                    "prior_sha256",
                    "source_frame",
                )
            )
            if common is not None and (
                common != identity
                or not np.array_equal(z["joint_names"], joint_names)
                or not np.array_equal(z["link_names"], link_names)
            ):
                raise ValueError("incompatible Lightning Grasp banks")
            common = identity
            joint_names, link_names = z["joint_names"], z["link_names"]
            label = file.parent.name
            if label in records:
                raise ValueError("duplicate bank label")
            records[label] = receipt
            n = len(z["q"])
            if n:
                for key in array_keys:
                    pieces[key].append(z[key])
                sources.extend([label] * n)
                indices.extend(range(n))
    if not records:
        raise ValueError("no input banks")
    data = {k: np.concatenate(v) for k, v in pieces.items() if v}
    data.setdefault("q", np.empty((0, len(joint_names)), dtype=np.float32))
    data.setdefault("object_pose", np.empty((0, 4, 4), dtype=np.float32))
    receipt = dict(
        upstream=common[0],
        input_sha256=common[1],
        hand_sha256=common[2],
        prior_sha256=common[3],
        source_frame=common[4],
        candidates=len(data["q"]),
        sources=records,
    )
    tmp = out.with_suffix(".part.npz")
    np.savez_compressed(
        tmp,
        **data,
        link_names=link_names,
        joint_names=joint_names,
        candidate_source=np.array(sources, dtype=str),
        source_id=indices,
        receipt=json.dumps(receipt),
    )
    tmp.replace(out)
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directories", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    groups = {}
    for directory in a.directories:
        for path in sorted(directory.glob("*.npz")):
            groups.setdefault(path.name, []).append(path)
    for name, files in groups.items():
        merge(files, a.out / name)


if __name__ == "__main__":
    main()
