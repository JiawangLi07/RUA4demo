"""Measure one visible shuttlecock using the station ROI and registered RGB-D."""
import math
import os
import queue
import threading

import numpy as np
from scipy import ndimage

from perception_backend import sam3_client_class
from .station import mask_samples


def detector_client():
    client = sam3_client_class()(os.environ.get("PYRUA_SAM3_ENDPOINT"), timeout_s=4)
    if not client.alive():
        raise RuntimeError("Shuttlecock segmentation service is unavailable")
    return client


def measure(client, snapshot, station, *, cork=False, include_surface=False):
    x1, y1, x2, y2 = station.roi
    turns = getattr(station, "image_rotation_quarters", 0)
    crop = np.rot90(snapshot.rgb[y1:y2, x1:x2], turns)
    found = client.segment(crop, text_prompt="badminton shuttlecock", top_k=2,
                           min_score=.48 if cork else .60)
    candidates = [candidate for candidate in client.last_candidates if candidate["score"] >= .60]
    if cork and not candidates and len(client.last_candidates) == 1:
        weak = client.last_candidates[0]
        mask = np.asarray(weak["mask"], dtype=bool)
        if (found.found and .48 <= weak["score"] < .60 and mask.shape == crop.shape[:2]
                and mask.sum() >= 48 and not (mask[0].any() or mask[-1].any()
                                             or mask[:, 0].any() or mask[:, -1].any())):
            return None, None, {"instances": 1, "score": float(weak["score"]),
                                "reason": "weak_shuttlecock_candidate"}
    if not found.found or len(candidates) != 1:
        return None, None, {"instances":len(candidates), "reason":"target_missing_or_ambiguous",
                            "detector_reason": getattr(found, "reason", None)}
    only = candidates[0]
    mask = np.asarray(only["mask"], dtype=bool)
    score = only["score"]
    if (mask.shape != crop.shape[:2] or not math.isfinite(score) or score < 0.6
            or mask.sum() < 48 or mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()):
        return None, None, {"reason":"incomplete_target_mask"}
    metadata = {"score":float(score), "pixels":int(mask.sum()), "roi_xyxy":list(station.roi),
                "target_part":"cork" if cork else "shuttlecock"}
    if cork:
        mask, cork_score = cork_mask(client, crop, mask)
        if mask is None:
            return None, None, {**metadata, "reason":"cork_missing_or_ambiguous"}
        metadata.update(cork_score=cork_score, cork_pixels=int(mask.sum()))
    mask = np.rot90(mask, -turns)
    mount, interior_size = surface_samples(snapshot, station, mask)
    if len(mount) < max(20, .35 * interior_size):
        return None, mask, {**metadata, "reason":"insufficient_depth"}
    camera_depth = (mount - station.mount_from_color[:3, 3]) @ station.mount_from_color[:3, 2]
    median = np.median(camera_depth)
    if np.median(np.abs(camera_depth - median)) > .02:
        return None, mask, {**metadata, "reason":"inconsistent_depth"}
    if include_surface:
        metadata["surface_mount_m"] = mount
    return np.median(mount, axis=0), mask, {
        **metadata, "reason":"surface_measured", "depth_pixels":len(mount),
        "low_height_m":float(np.quantile(mount @ station.up, .05)),
    }


def surface_samples(snapshot, station, mask):
    """Return the frontmost registered depth samples inside a native RGB mask."""
    interior = ndimage.binary_erosion(mask)
    pixels, points = mask_samples(snapshot, station, interior)
    valid = np.isfinite(points).all(axis=1)
    points, pixels = points[valid], pixels[valid]
    if len(points):
        # Multiple depth pixels can project onto one color pixel. Use its frontmost surface.
        order = np.argsort(points[:,2])
        _, first = np.unique(pixels[order], axis=0, return_index=True)
        points = points[order[first]]
    mount = points @ station.mount_from_color[:3,:3].T + station.mount_from_color[:3,3]
    return mount, interior.sum()


def cork_mask(client, image, shuttlecock):
    """Run the part prompt only around the detected object and reject whole-object hits."""
    y, x = np.nonzero(shuttlecock)
    pad = max(12, int(.3 * max(np.ptp(x), np.ptp(y))))
    h, w = shuttlecock.shape
    x1, x2 = max(0, x.min()-pad), min(w, x.max()+pad+1)
    y1, y2 = max(0, y.min()-pad), min(h, y.max()+pad+1)
    whole = shuttlecock[y1:y2, x1:x2]
    support = ndimage.binary_dilation(whole, iterations=2)
    crop = image[y1:y2, x1:x2]
    accepted = []
    for prompt in ("cork tip of a badminton shuttlecock", "shuttlecock head", "white rounded cork base"):
        client.segment(crop, text_prompt=prompt, top_k=3, min_score=.30)
        for candidate in client.last_candidates:
            part = np.asarray(candidate["mask"], dtype=bool)
            score = candidate["score"]
            if part.shape != whole.shape or not math.isfinite(score) or score < .30:
                continue
            count = part.sum()
            if not 24 <= count <= .55 * whole.sum() or (part & support).sum() < .9 * count:
                continue
            accepted.append((float(score), part & support))
        if accepted:
            break
    if not accepted:
        return None, None
    score, best = max(accepted, key=lambda item: item[0])
    if any((best & part).sum() < .7 * (best | part).sum() for _, part in accepted):
        return None, None
    result = np.zeros_like(shuttlecock)
    result[y1:y2, x1:x2] = best
    return result, score


def monitored_measure(client, snapshot, station, check, *, cork=False, include_surface=False):
    # The worker owns only the captured pixels and HTTP call. The SDK remains on the trial thread.
    reply = queue.Queue(maxsize=1)
    def compute():
        try:
            if include_surface:
                reply.put(measure(client, snapshot, station, cork=cork, include_surface=True))
            elif cork:
                reply.put(measure(client, snapshot, station, cork=True))
            else:
                reply.put(measure(client, snapshot, station))
        except Exception as exc:
            reply.put(exc)
    threading.Thread(target=compute, daemon=True).start()
    while True:
        check()
        try:
            value = reply.get(timeout=.04)
        except queue.Empty:
            continue
        check()
        if isinstance(value, Exception):
            raise RuntimeError("Shuttlecock observation failed") from value
        return value
