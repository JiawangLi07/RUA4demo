"""Track visible object texture in calibrated RGB-D across a verification lift."""

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .tabletop_vision import outside_robot, world_image


def initialize(frame, io, points):
    xyz, valid = world_image(frame)
    lo = points.min(0) - 0.009
    hi = points.max(0) + 0.009
    v, u = np.nonzero(valid & np.all((xyz >= lo) & (xyz <= hi), axis=2))
    if len(v) < 12:
        return None
    pts = xyz[v, u]
    near = (cKDTree(points).query(pts)[0] < 0.0065) & outside_robot(pts, io, frame["q"])
    mask = np.zeros(valid.shape, np.uint8)
    mask[v[near], u[near]] = 255
    gray = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2GRAY)
    uv = cv2.goodFeaturesToTrack(gray, 200, 0.005, 2.4, mask=mask, blockSize=3)
    if uv is None or len(uv) < 5:
        return None
    ij = np.rint(uv[:, 0]).astype(int)
    return (
        gray,
        uv,
        xyz[ij[:, 1], ij[:, 0]],
        frame["rgb"][ij[:, 1], ij[:, 0]].astype(float),
    )


def pose_fit(a, b):
    ma, mb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ma).T @ (b - mb))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    return rot, mb - rot @ ma


def measure(reference, frame, io, world, tcp_delta):
    if reference is None:
        return {}
    gray0, uv, points, color = reference
    gray = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2GRAY)
    guess = points @ tcp_delta[:3, :3].T + tcp_delta[:3, 3]
    cam = (guess - frame["pose"][:3, 3]) @ frame["pose"][:3, :3]
    k = frame["K"]
    first = (cam[:, :2] / np.maximum(cam[:, 2, None], 0.01)) @ k[:2, :2].T + k[:2, 2]
    options = {
        "winSize": (19, 19),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    dst, ok, _ = cv2.calcOpticalFlowPyrLK(
        gray0,
        gray,
        uv,
        first.astype(np.float32).reshape(-1, 1, 2),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        **options,
    )
    if dst is None:
        return {}
    back, valid, _ = cv2.calcOpticalFlowPyrLK(
        gray, gray0, dst, uv.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **options
    )
    h, w = gray.shape
    ij = np.rint(dst[:, 0]).astype(int)
    valid = (
        (valid[:, 0] > 0)
        & (ok[:, 0] > 0)
        & (np.linalg.norm(back[:, 0] - uv[:, 0], axis=1) < 1.1)
        & (ij[:, 0] >= 0)
        & (ij[:, 0] < w)
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < h)
    )
    if valid.sum() < 5:
        return {}
    u, v = ij[valid].T
    xyz, depth = world_image(frame)
    b = xyz[v, u]
    a = points[valid]
    keep = (
        depth[v, u]
        & (np.linalg.norm(frame["rgb"][v, u].astype(float) - color[valid], axis=1) < 75)
        & outside_robot(b, io, frame["q"])
    )
    a, b = a[keep], b[keep]
    if len(a) < 5:
        return {}
    rng = np.random.default_rng(47)
    best = np.zeros(len(a), bool)
    for _ in range(80):
        j = rng.choice(len(a), 3, replace=False)
        if np.linalg.norm(np.cross(a[j[1]] - a[j[0]], a[j[2]] - a[j[0]])) < 0.000014:
            continue
        rot, shift = pose_fit(a[j], b[j])
        res = np.linalg.norm(a @ rot.T + shift - b, axis=1)
        keep = res < 0.005
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < 5 or best.mean() < 0.55:
        return {}
    rot, shift = pose_fit(a[best], b[best])
    support = np.linalg.svd(a[best] - a[best].mean(0), compute_uv=False) / np.sqrt(
        best.sum()
    )
    if support[0] < 0.008 or support[1] < 0.002:
        return {}
    transformed = world @ rot.T + shift
    return {
        "texture_inliers": int(best.sum()),
        "texture_span_m": round(float(support[0]), 4),
        "tracked_rise_m": round(float(transformed.mean(0)[2] - world.mean(0)[2]), 5),
        "tracked_bottom_rise_m": round(
            float(transformed[:, 2].min() - world[:, 2].min()), 5
        ),
    }
