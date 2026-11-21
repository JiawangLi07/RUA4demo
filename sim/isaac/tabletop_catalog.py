"""A shared 36-posture library indexed only by measured geometry and support orientation."""

import argparse
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from .handle_grasp import handle_sections
from .head_wrist_trial import Rejected
from .paths import RIGHT_JOINTS


def load_registry(path, comparison=False):
    data = json.loads(path.read_bytes())
    if "schema" not in data:
        if not data or any("seeds" not in row for row in data.values()):
            raise ValueError(
                "expected a geometry registry or a historical posture manifest"
            )
        return data, None
    if data["schema"] != "rua.tabletop-geometries.v1" or comparison:
        raise ValueError("expected the shared geometry registry for mesh synthesis")
    objects = data["objects"]
    if not objects or any(set(row) != {"prior"} for row in objects.values()):
        raise ValueError("generic evaluation accepts only a metric mesh per object")
    return objects, SharedPostures(path.resolve().parent / data["shared_postures"])


def geometry_family(surface):
    points = getattr(surface, "handle_world", surface.world)
    _, axes = np.linalg.eigh(np.cov(points[:, :2].T))
    span = np.sort(
        np.ptp(np.quantile(points[:, :2] @ axes, [0.01, 0.99], axis=0), axis=0)
    )
    narrow, long = span
    high = float(np.quantile(points[:, 2], 0.99) - surface.table)
    if long > 0.090 and long > 2.5 * max(narrow, high):
        t = (points[:, :2] - points[:, :2].mean(0)) @ axes[:, -1]
        widths = []
        # A thick handle may have many more triangles than its narrow shaft.
        # Equal-length sections preserve the shaft in the width profile.
        edges = np.linspace(*np.quantile(t, [0.004, 0.996]), 17)
        for lo, hi in itertools.pairwise(edges):
            band = (t >= lo) & (t < hi)
            if np.count_nonzero(band) > 5:
                widths.append(
                    np.ptp(np.quantile(points[band, :2] @ axes[:, 0], [0.04, 0.96]))
                )
        if widths and np.quantile(widths, 0.85) > 1.45 * max(
            np.quantile(widths, 0.20), 0.005
        ):
            try:
                _, _, _, handle = handle_sections(points)
                if np.ptp(handle[:, 0]) >= 0.045 and np.median(handle[:, 3]) >= 0.008:
                    return "handled"
            except Rejected:
                pass
        return "slender"
    if (
        high > 0.061
        and high > 0.73 * long
        and narrow > 0.025
        and long < 0.13
        and long < 1.75 * narrow
    ):
        return "wrap"
    if high < max(0.48 * narrow, 0.012) or long > 0.13:
        return "edge"
    if long < 0.045 and high < 0.047:
        return "small"
    return "compact"


def export_shared(index, output):
    """Export a numerical subset of the training catalog without evaluation-object references."""
    from dexbridge.skills.grasp.templates import load_templates

    db = load_templates(index)
    if db["joint_names"] != RIGHT_JOINTS:
        raise ValueError("Sharpa joint order mismatch")
    result = {
        "schema": "rua.shared-hand-postures.v1",
        "joint_names": RIGHT_JOINTS,
        "types": {},
    }
    for name, card in db["types"].items():
        bins = {}
        for mode in card["modes"]:
            up = np.asarray(mode["support_up_bbox"])
            axis = int(np.argmax(np.abs(up)))
            for knot in mode["knots"]:
                np.array(knot["bbox_dims"])
                size = float(knot["feature_size_m"])
                key = (
                    mode["shape_key"],
                    axis,
                    up[axis] > 0,
                    int(np.digitize(size, [0.022, 0.043, 0.077, 0.13])),
                )
                quality = (
                    bool(mode["validated"]),
                    mode["support_records"],
                    -abs(np.log(size / 0.06)),
                )
                if key in bins and bins[key][0] >= quality:
                    continue
                sample = {
                    k: knot[k]
                    for k in (
                        "bbox_dims",
                        "pregrasp",
                        "grasp",
                        "squeeze",
                        "q_pre",
                        "q_grasp",
                        "q_squeeze",
                    )
                }
                sample.update(
                    mode=mode["mode_id"],
                    shape=mode["shape_key"],
                    up=up.tolist(),
                    validated=mode["validated"],
                )
                bins[key] = (quality, sample)
        result["types"][name] = {
            "roles": card["intended_contacts"],
            "samples": [v[1] for v in bins.values()],
        }
    with gzip.open(output, "wt", encoding="utf8") as f:
        json.dump(result, f, separators=(",", ":"))


class SharedPostures:
    def __init__(self, path):
        with gzip.open(path, "rt", encoding="utf8") as f:
            data = json.load(f)
        if (
            data["schema"] != "rua.shared-hand-postures.v1"
            or data["joint_names"] != RIGHT_JOINTS
            or len(data["types"]) != 36
        ):
            raise ValueError("expected the shared 36-posture Sharpa catalog")
        self.types = data["types"]

    def query(self, surface, pose):
        p = surface.world
        center = p.mean(0)
        _, _, vt = np.linalg.svd(p - center, full_matrices=False)
        rotation = vt.T
        for i in (0, 1):
            if rotation[np.argmax(np.abs(rotation[:, i])), i] < 0:
                rotation[:, i] *= -1
        rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
        coords = (p - center) @ rotation
        lo, hi = np.quantile(coords, [0.002, 0.998], axis=0)
        dims = np.maximum(hi - lo, 0.003)
        bbox = np.eye(4)
        bbox[:3, :3] = rotation
        bbox[:3, 3] = center + rotation @ ((lo + hi) / 2)
        family = geometry_family(surface)
        kinds = {
            "handled": {"elongated", "slender_cylinder"},
            "slender": {"elongated", "slender_cylinder"},
            "wrap": {"slender_cylinder", "stout_cylinder", "box_like"},
            "edge": {"flat_plate", "box_like"},
            "small": {"small_item", "rounded", "elongated"},
            "compact": {"rounded", "sphere", "stout_cylinder", "box_like"},
        }[family]
        inv = np.linalg.inv(pose)
        rows = []
        for name, card in self.types.items():
            choices = card["samples"]
            if not choices:
                continue
            roles = card["roles"]
            axial = [c for c in choices if c["shape"] in kinds] or choices

            def cost(c):
                orientation = np.clip(np.dot(c["up"], rotation[2]), -1, 1)
                return (
                    np.linalg.norm(np.log(dims / np.array(c["bbox_dims"])))
                    + 0.4 * (1 - orientation)
                    + 0.25 * (not c["validated"])
                )

            c = min(axial, key=cost)

            def stage(key, c=c):
                part = c[key]
                old = np.asarray(c["bbox_dims"])
                off = np.array(part["offset_bbox"]) * old
                anchor = np.clip(off, -old / 2, old / 2)
                tf = np.eye(4)
                tf[:3, :3] = part["rotation"]
                tf[:3, 3] = anchor / old * dims + off - anchor
                return inv @ bbox @ tf

            pre = stage("pregrasp")
            close = stage("grasp")
            tight = stage("squeeze")
            view = pre.copy()
            # The observation pose is a suggestion. The reachable camera bank can supply alternatives.
            view[:3, 3] += pose[:3, :3].T @ [-0.045, -0.015, 0.07]
            digits = {
                x.split("_")[1]
                for x in roles
                if x.startswith("right_") and len(x.split("_")) > 2
            }
            rank = cost(c)
            if not any("_thumb_" in x for x in roles):
                rank += 3
            if family == "wrap":
                rank += 2.0 * ("right_hand_C_MC" not in roles) + 0.2 * max(
                    0, 3 - sum(x.endswith("_PP") for x in roles)
                )
            elif family == "handled":
                rank += 0.6 * max(0, 4 - len(digits))
            else:
                rank += 0.30 * abs(
                    len(digits) - (2 if family in ("slender", "small") else 3)
                )
            rows.append(
                (
                    rank,
                    {
                        "posture": name,
                        "origin": c["mode"],
                        "kind": "shared_training_mode",
                        "intended_contacts": roles,
                        "q_open": c["q_pre"],
                        "q_touch": c["q_grasp"],
                        "q_tight": c["q_squeeze"],
                        "pre": pre.tolist(),
                        "grasp": close.tolist(),
                        "tight": tight.tolist(),
                        "view": view.tolist(),
                    },
                )
            )
        rows.sort(key=lambda r: r[0])
        return family, [row for _, row in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export_shared(args.index, args.out)
