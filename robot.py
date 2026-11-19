"""Sharpa adapter for PyRUA-Lean's RobotBase and MCP runtime."""

from pathlib import Path
from uuid import uuid4
import json

import numpy as np
from PIL import Image

from pyrualean._backend import Ledger
from pyrualean._robot import RobotBase
from pyrualean.arms import ArmResult

from hardware import MOUNT_FRAME, SIDES, preview_reach
from perception_backend import FramePerception, capabilities


class SharpaRobot(RobotBase):
    NAME = "tianji_sharpa"
    CAMERAS = ("head",)
    IMAGE_ARTIFACTS = {("head", "high"): "head.png"}
    ABSENT_MEMBERS = frozenset({"done", "world_map"})
    SUMMARY_KEYS = ("found", "frame_id", "point_camera_m", "point_mount_m", "samples", "status")

    def _state_from_envelope(self, envelope):
        return envelope

    def _summary_dict(self):
        return self.observe()

    def observe(self):
        """Observe the RGB-D frame identifier, calibration status and measured joints."""
        return self.state()

    def tactile(self, frame_id: str, side: str):
        """Return recorded three-axis force or live F6 samples associated with this snapshot."""
        return self._readonly("tactile", {"frame_id": frame_id, "side": side})

    def locate(self, frame_id: str, label: str, bbox_xyxy: list):
        """Estimate a surface point in meters from the RGB box in the named frame."""
        return self._readonly("locate", {
            "frame_id": frame_id, "label": label, "bbox_xyxy": bbox_xyxy,
        })

    def segment(self, frame_id: str, model: str, prompt: dict):
        """Segment current RGB. sam: {point_xy:[x,y]} or {bbox_xyxy:[l,t,r,b]}; sam3: text or point_xy. Returns mask_id."""
        return self._readonly("segment", {"frame_id": frame_id, "model": model, "prompt": prompt})

    def locate_mask(self, frame_id: str, mask_id: str):
        """Measure the visible RGB-D surface inside a mask returned for this frame."""
        return self._readonly("locate_mask", {"frame_id": frame_id, "mask_id": mask_id})

    def rank_labels(self, frame_id: str, labels: list, mask_id: str = ""):
        """Score English label descriptions with SigLIP for the image or a current-frame mask."""
        return self._readonly("rank_labels", {"frame_id": frame_id, "labels": labels, "mask_id": mask_id})

    def preview_reach(self, frame_id: str, label: str, bbox_xyxy: list, arm: str):
        """Preview a position IK solution at the localized point with the current TCP rotation."""
        return self._readonly("preview_reach", {
            "frame_id": frame_id, "label": label, "bbox_xyxy": bbox_xyxy, "arm": arm,
        })


class SnapshotBackend:
    def __init__(self, snapshot, joints, extrinsics, out: Path, *, camera_frame="head_color_optical"):
        self.snapshot = snapshot
        self.extrinsics = extrinsics
        self.out = out
        self.perception = FramePerception(self)
        self.registered = None
        self.frame_id = getattr(snapshot, "archived_frame_id", None) or uuid4().hex
        self.joints = joints
        self.info = {
            "frame_id": self.frame_id,
            "source": snapshot.source,
            "perception_backends": capabilities(),
            "camera_serial": snapshot.serial,
            "captured_at_ns": snapshot.stamp_ns,
            "captured_monotonic_ns": snapshot.monotonic_ns,
            "image_width": int(snapshot.rgb.shape[1]),
            "image_height": int(snapshot.rgb.shape[0]),
            "camera_frame": camera_frame,
            "mount_frame_calibrated": extrinsics is not None,
            "joints": {k:v for k,v in joints.items() if k != "tactile"},
            "tactile_sides": [side for side,row in joints.get("tactile", {}).items() if row.get("samples")],
        }
        Image.fromarray(snapshot.rgb).save(out / "head.png")
        Image.fromarray(snapshot.raw_depth).save(out / "head_depth_native.png")
        (out / "observation.json").write_text(json.dumps({
            **self.info,
            "tactile": joints.get("tactile", {}),
            "depth_geometry": snapshot.geometry.metadata(),
            "T_mount_from_color": None if extrinsics is None else extrinsics.tolist(),
        }, indent=2, allow_nan=False) + "\n")

    def execute_tool(self, name, input_dict):
        if name == "view_env_state" and input_dict["step"] in (-1, 0):
            return self.info
        if name == "tactile":
            if input_dict["frame_id"] != self.frame_id or input_dict["side"] not in SIDES:
                raise ValueError("Use the current frame_id and a valid hand side")
            row = self.joints.get("tactile", {}).get(input_dict["side"], {})
            return {"frame_id": self.frame_id, "side": input_dict["side"],
                    "available": bool(row.get("samples")), "source": row.get("source"),
                    "samples": row.get("samples", [])}
        if name in ("segment", "locate_mask", "rank_labels"):
            return getattr(self.perception, name)(**input_dict)
        if name == "locate":
            return self.locate(**input_dict)
        if name == "preview_reach":
            return self.plan(**input_dict)
        raise ValueError(f"Unknown tool or observation step: {name}")

    def solved(self):
        return False

    def env_steps(self):
        return None

    def cancel(self):
        pass

    def artifact(self, name, step=-1):
        if name != "head.png" or step not in (-1, 0):
            raise ValueError("The episode has one head RGB image")
        return (self.out / name).read_bytes()

    def locate(self, frame_id, label, bbox_xyxy):
        if frame_id != self.frame_id:
            raise ValueError("Frame mismatch. Call observe to get this episode's RGB frame.")
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise ValueError("Use a non-empty object label of at most 80 characters")
        box = np.asarray(bbox_xyxy, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("bbox_xyxy must contain four finite RGB pixel coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= self.info["image_width"]
                and 0 <= y1 < y2 <= self.info["image_height"]):
            raise ValueError("Expected [x1,y1,x2,y2] in the observed RGB image")
        if self.registered is None:
            self.registered = self.snapshot.geometry.register(self.snapshot.raw_depth)
        pixels, color_points = self.registered
        # Use the central part of the RGB box to reduce leakage from the background.
        dx, dy = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
        u, v = pixels.T
        points = color_points[(u >= x1 + dx) & (u < x2 - dx)
                              & (v >= y1 + dy) & (v < y2 - dy)]
        return self.surface_result(points, {"frame_id": frame_id, "label": label, "bbox_xyxy":box.tolist()})

    def surface_result(self, points, metadata):
        result = {**metadata, "camera_frame":"head_color_optical", "samples":len(points)}
        if len(points) < 8:
            return {**result, "found": False, "status": "insufficient_depth"}
        median_depth = np.median(points[:, 2])
        mad_m = float(np.median(np.abs(points[:, 2] - median_depth)))
        keep = np.abs(points[:, 2] - median_depth) <= max(0.01, 3 * mad_m)
        point = np.median(points[keep], axis=0)
        result.update(
            found=True, status="surface_point", point_camera_m=point.tolist(),
            depth_mad_m=mad_m, inlier_samples=int(keep.sum()),
        )
        if self.extrinsics is not None:
            result["mount_frame"] = MOUNT_FRAME
            result["point_mount_m"] = (self.extrinsics @ np.r_[point, 1])[:3].tolist()
        return result

    def plan(self, frame_id, label, bbox_xyxy, arm):
        if arm not in SIDES:
            raise ValueError("arm must be left or right")
        if self.extrinsics is None:
            raise ValueError("Missing measured head_color_optical -> dual_arm_mount extrinsics")
        if not self.joints or not self.joints["joints_aligned"]:
            raise ValueError("Need camera-aligned joint measurements to seed IK")
        target = self.locate(frame_id, label, bbox_xyxy)
        if not target["found"]:
            raise ValueError("The object box has insufficient depth for IK")
        return {
            "frame_id": frame_id, "label": label,
            **preview_reach(arm, target["point_mount_m"], self.joints["arm_deg"][arm]),
        }


BOX_SCHEMA = {
    "frame_id": {"type": "string", "description": "Exact frame_id returned by observe"},
    "label": {"type": "string", "maxLength": 80},
    "bbox_xyxy": {
        "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
        "description": "[left, top, right, bottom] in pixels of the returned RGB image",
    },
}


class RobotTools:
    """Typed model-facing tools backed by the same API as Python skills."""

    def __init__(self, backend):
        self.ledger = Ledger()
        self.robot = SharpaRobot(backend, ledger=self.ledger)
        self.out = backend.out
        self.calibrated = backend.extrinsics is not None
        self.tactile_sides = backend.info.get("tactile_sides", [])
        self.perception_backends = backend.info["perception_backends"]

    def get_tools_spec(self):
        specs = [
            ("observe", "Return the head RGB image and its frame ID. Read this before locating.", {}),
            ("locate", "Project an object box into a measured RGB-D camera surface point.", BOX_SCHEMA),
        ]
        available = self.perception_backends
        models = [k for k, ok in (("sam",available["sam"]),("sam3",available["sam3_endpoint_configured"])) if ok]
        frame = {"type": "string"}
        if self.tactile_sides:
            specs.append(("tactile", "Read per-finger force samples for the current snapshot.",
                          {"frame_id":frame,"side":{"type":"string","enum":self.tactile_sides}}))
        if models:
            specs.extend([
                ("segment", "Segment the current RGB image. Supply point_xy, bbox_xyxy (SAM) or text (SAM3).",
                 {"frame_id":frame,"model":{"type":"string","enum":models},"prompt":{"type":"object"}}),
                ("locate_mask", "Measure 3D from registered depth inside the segmentation mask.",
                 {"frame_id":frame,"mask_id":{"type":"string"}}),
            ])
        if available["siglip"]:
            specs.append(("rank_labels", "SigLIP scores for English labels; use empty mask_id for the whole image.",
                          {"frame_id":frame,"labels":{"type":"array","items":{"type":"string"}},
                           "mask_id":{"type":"string"}}))
        if self.calibrated:
            specs.append(("preview_reach", "Preview position IK for a detected object.", {
                **BOX_SCHEMA, "arm": {"type": "string", "enum": list(SIDES)},
            }))
        return [{
            "name": name, "description": description,
            "input_schema": {
                "type": "object", "properties": fields, "required": list(fields),
                "additionalProperties": False,
            },
        } for name, description, fields in specs]

    def execute_tool(self, name, inputs):
        allowed = {spec["name"] for spec in self.get_tools_spec()}
        try:
            if name not in allowed:
                raise ValueError(f"Unknown tool: {name}")
            result = getattr(self.robot, name)(**inputs)
            images = [self.robot.artifact("head.png")] if name == "observe" else []
        except (ValueError, TypeError, RuntimeError) as exc:
            result, images = {"error": str(exc)}, []
        self.ledger.write_jsonl(self.out / "calls.jsonl")
        with (self.out / "tool_results.jsonl").open("a") as report:
            report.write(json.dumps({"tool": name, "result": result}, ensure_ascii=False,
                                    allow_nan=False) + "\n")
        return ArmResult(result, images)
