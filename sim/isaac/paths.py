"""Station assets are supplied by the operator and kept outside the checkout."""

import os
from pathlib import Path

RIGHT_JOINTS = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]


def asset_path(source, *parts):
    if not source or Path(source).parts != (source,) or source == "..":
        raise ValueError(f"invalid asset source {source!r}")
    base = os.environ.get("RUA_SIM_ASSETS")
    if not base:
        raise ValueError("pass --assets or set RUA_SIM_ASSETS to the station asset directory")
    root = (Path(base) / source).resolve()
    result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"asset escapes source directory {source!r}")
    if not result.exists():
        raise FileNotFoundError(result)
    return result
