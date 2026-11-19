"""REAL's read-only diagnostic phase, with model-authored Lean cells."""
import ast
import base64
from io import BytesIO
import json
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pyrualean.arms import ArmToolkit
from pyrualean._backend import Ledger

from hardware import ROOT, load_extrinsics, observe_hardware
from robot import SharpaRobot, SnapshotBackend
from .clearance import HandClearance
from .physical import check_profile_origin
from .station import Station
from .station_diagnostics import fixed_arm_escape, tcp_report, workspace, pad_measurement, preview_approach
from task_library import atomic_json


TOOLS = ("profile", "capture", "clearance", "locate", "force", "fixed_arm_escape", "workspace", "tcp", "contact_stop", "pad_measurement", "preview_approach")


class DiagnosticRobot(SharpaRobot):
    def calibrate(self):
        return self._stateful("calibrate", {}, build=dict)

    def inspect(self, name, box=None):
        return self._readonly("inspect", {"name": name, "box": box})


class Diagnostics:
    def __init__(self, profile, arm, target, out, *, calibration=False, cancelled=lambda:False):
        self.calibration, self.cancelled = calibration, cancelled
        self.probe_attempted = False
        self.profile, self.arm, self.target, self.out = profile, arm, target, out
        self.snapshot = None
        self.calls = []
        self.frames = 0

    def solved(self):
        return False

    def env_steps(self):
        return 0

    def profile_state(self):
        config = json.loads(self.profile.read_text())
        errors = []
        if (config.get("arm") != self.arm or config.get("target", "cube") != self.target):
            errors.append("Task and station identity differ")
        for check in (lambda: Station.read(self.profile), lambda: check_profile_origin(config)):
            try:
                check()
            except (ValueError, KeyError, OSError) as exc:
                errors.append(str(exc))
        return {"valid": not errors, "blockers": errors,
                "contact_calibration": {k:config.get(k) for k in
                    ("tcp_offset_m", "workspace_m", "hand_max_abs_torque", "arm_max_abs_est_wrench")},
                "grasp_names": list((config.get("hand_grasps_rad") or {}).keys()),
                "contact_policy":config.get("contact_policy"),
                "hand_target_origin": config.get("input_sources", {}).get("hand_targets")}

    def execute_tool(self, tool, inputs):
        if tool == "calibrate":
            if not self.calibration or self.snapshot is None or self.probe_attempted or self.arm != "right":
                raise ValueError("Calibration requires explicit run authorization, a fresh capture and one right-hand probe")
            from .calibration import index_trial
            self.probe_attempted = True
            try:
                from .hand_session import SessionDialogue
                dialogue=SessionDialogue(self.channel,self.out,self.profile_state())
                dialogue.profile_path=self.profile
                result = index_trial(json.loads(self.profile.read_text()), self.snapshot, self.out/"index-trial", self.cancelled, dialogue=dialogue)
            finally:
                self.snapshot = None
            self.session_result=result
            self.calls.append({"tool":tool,"result":result})
            return result
        name, box = inputs["name"], inputs["box"]
        if tool != "inspect" or name not in TOOLS:
            raise ValueError("Unknown diagnostic tool")
        try:
            result = self.inspect(name, box)
        except Exception as exc:
            result = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        self.calls.append({"tool":name,"result":result})
        return result

    def inspect(self, name, box):
        if name == "profile":
            return self.profile_state()
        c = json.loads(self.profile.read_text())
        if name == "capture":
            self.snapshot = None
            self.frames += 1
            dest = self.out / f"frame-{self.frames}"
            dest.mkdir()
            transform = load_extrinsics(self.profile.parent / c["calibration"], c["camera_serial"])
            with observe_hardware(c["camera_serial"], camera_only=False) as (snap, joints):
                if not joints.get("joints_aligned"):
                    raise ValueError("Camera and joints are not time aligned")
                self.snapshot = SnapshotBackend(snap, joints, transform, dest)
            return {"frame_id": self.snapshot.frame_id, "arm_status": joints["arm_status"],
                    "arm_deg": joints["arm_deg"], "hand_rad": joints["hand_rad"],
                    "camera_joint_skew_ms":joints["max_camera_joint_skew_ms"]}
        if name == "force":
            dest = self.out / f"static-force-{len(self.calls)}.json"
            subprocess.run([str(ROOT/"run.sh"), "force-baseline", "--seconds", "3", "--out", str(dest)],
                           stdout=subprocess.DEVNULL, check=True, timeout=22)
            channels = json.loads(dest.read_text())["channels"]
            return {"resting_arm_wrench_peak":channels["arm_"+self.arm]["resting_peak_abs_est_wrench"],
                    "resting_hand_torque_peak":channels["hand_"+self.arm]["resting_peak_abs_torque"],
                    "measurement":"stationary baseline only; cannot establish contact limits"}
        if name == "contact_stop":
            from tactile_check import measure_tactile
            force = self.inspect("force", None)
            try:
                tactile = measure_tactile(self.arm)
            except Exception as exc:
                tactile = {"ready": False, "reason": str(exc)}
            return {"static_feedback":force, "fingertip_readback":tactile,
                    "hand_limit":c.get("hand_max_abs_torque"), "arm_limit":c.get("arm_max_abs_est_wrench"),
                    "contact_stop_tested":False,
                    "missing_evidence":"loaded response plus threshold crossing, stop acknowledgment and measured stopping response; passive samples alone cannot test stopping"}
        if self.snapshot is None:
            return {"available":False, "reason":"Capture a fresh scene first"}
        if name == "locate":
            return {**self.snapshot.locate(self.snapshot.frame_id, self.target, box),
                    "scope":"RGB-box depth estimate; background pixels may contribute. Not a verified grasp or TCP point."}
        if name == "workspace":
            report = workspace(self.snapshot, box, c)
            artifact = self.out / f"workspace-grid-{len(self.calls)}.json"
            atomic_json(artifact, report)
            for key in ("table_cells_xy","raised_cells_xy"):
                report[key + "_count"] = len(report.pop(key))
            return {**report,"grid_file":artifact.name}
        side = self.arm
        other = "left" if side == "right" else "right"
        joints = self.snapshot.joints
        geometry = HandClearance(SimpleNamespace(arm=side, up=np.asarray(c["up_axis_mount"]),
                                 clearance=c.get("clearance")), joints["arm_deg"][other], joints["hand_rad"][other])
        arm, hand = joints["arm_deg"][side], joints["hand_rad"][side]
        if name == "fixed_arm_escape":
            report = fixed_arm_escape(geometry, arm, hand)
            artifact = self.out / f"escape-{len(self.calls)}.json"
            atomic_json(artifact, {"frame_id":self.snapshot.frame_id, **report})
            report.pop("joint_waypoints_rad", None)
            return {**report, "frame_id":self.snapshot.frame_id,"plan_file":artifact.name}
        if name == "pad_measurement":
            return pad_measurement(self.snapshot,box,geometry,arm,hand,side)
        if name == "preview_approach":
            return preview_approach(self.snapshot,box,geometry,arm,hand,side,c)
        if name == "tcp":
            return {**tcp_report(geometry, arm, hand, side, c), "frame_id":self.snapshot.frame_id}
        report = geometry.report(arm, hand)
        return {**report, "frame_id":self.snapshot.frame_id}

    def image(self):
        if self.snapshot is None:
            return None
        buf = BytesIO()
        Image.fromarray(self.snapshot.snapshot.rgb).save(buf,format="JPEG",quality=78)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def validate_cell(source, calibration=False):
    if len(source) > 1800:
        raise ValueError("Cell too large")
    body = ast.parse(source).body
    if not 1 <= len(body) <= 4:
        raise ValueError("Use one to four diagnostic calls")
    for stmt in body:
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if (calibration and isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "robo"
                and call.func.attr == "calibrate" and not call.args and not call.keywords and len(body)==1):
            continue
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "robo"
                and call.func.attr == "inspect" and not call.keywords and 1 <= len(call.args) <= 2):
            raise ValueError("Only literal robo.inspect calls are supported")
        args = [ast.literal_eval(arg) for arg in call.args]
        if args[0] not in TOOLS or ((args[0] in ("locate", "workspace","preview_approach","pad_measurement")) != (len(args) == 2)):
            raise ValueError("Invalid diagnostic tool or argument count")
        if len(args) == 2 and not (isinstance(args[1], list) and len(args[1]) == (8 if args[0]=="pad_measurement" else 4) and
                                   all(type(x) in (int,float) for x in args[1])):
            raise ValueError("Unexpected pixel box format")


def diagnose(backend, channel, out):
    backend.channel=channel
    ledger = Ledger()
    toolkit = ArmToolkit(DiagnosticRobot(backend, ledger=ledger), arm="cells", workspace=out/"workspace",
                         images=False, feedback="pure", max_turns=6)
    history = []
    for turn in range(7):
        profile = backend.profile_state()
        request = {"turn":turn,"task":f"{backend.arm} hand grasp {backend.target}",
                   "calibration_authorized":backend.calibration,"profile":profile,"history":history,"last_turn":turn == 6,
                   "image_base64":backend.image()}
        reply = channel.request("diagnostic_request",request)
        action, source, reason = reply["action"], reply["source"], reply["reason"]
        atomic_json(out/f"decision-{turn}.json",reply)
        # The executor checks fresh feedback and the complete proposed path before enabling.
        if action == "proceed" and profile["valid"]:
            return None
        if action == "stop" or turn == 6:
            return {"status":"blocked","success":False,"reason":reason,
                    "run":str(out),"phase":"diagnostics","calibration_authorized":backend.calibration,"profile":profile,"history":history}
        if action == "proceed":
            history.append({"turn":turn, "reason":reason, "calls":[], "error":None,
                            "unavailable_action":"grasp", "blockers":profile["blockers"]})
            atomic_json(out/"diagnostics.json",history)
            continue
        start = len(backend.calls)
        try:
            if action != "cell":
                raise ValueError("Unknown diagnostic action")
            validate_cell(source, backend.calibration)
            result = toolkit.execute_tool("python",{"code":source}).result
            error = result.get("exception") or result.get("error")
        except (ValueError,SyntaxError,TypeError) as exc:
            error = str(exc)
        history.append({"turn":turn,"reason":reason,"calls":backend.calls[start:],"error":error})
        ledger.write_jsonl(out/"calls.jsonl")
        atomic_json(out/"diagnostics.json",history)
        session=getattr(backend,"session_result",None)
        if session and session.get("dialogue_started"):
            if session.get("status")=="session_complete" and (backend.out/"grasp-request.json").is_file():
                return None
            return {"status":"blocked","success":False,"phase":"calibration","run":str(out),
                    "reason":session.get("reason") or session.get("conclusion"),"profile":profile,"history":history}
