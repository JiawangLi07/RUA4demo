"""A replayable Python skill with a shared observation and localization contract."""

import json


def run(robo, *, label, bbox_xyxy):
    observation = robo.observe()
    point = robo.locate(observation["frame_id"], label, bbox_xyxy)
    if not point["found"]:
        raise RuntimeError(f"{label}: {point['status']}")
    print(json.dumps(point, ensure_ascii=False))
    return point
