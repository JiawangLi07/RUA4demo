"""Segment, score the target crop against the bare table, then measure its surface."""
def run(robo, *, label, bbox_xyxy):
    obs = robo.observe()
    frame = obs["frame_id"]
    mask = robo.segment(frame, "sam", {"bbox_xyxy": list(bbox_xyxy)})
    if not mask["found"]:
        raise ValueError("SAM did not find a mask")
    target_label = "a photo of a " + label
    scores = robo.rank_labels(frame, [target_label, "a photo of an empty table"], mask["mask_id"])
    point = robo.locate_mask(frame, mask["mask_id"])
    if not point["found"]:
        raise ValueError("Insufficient registered depth on the target mask")
    print({"mask": mask, "siglip": scores, "position": point})
    return point
