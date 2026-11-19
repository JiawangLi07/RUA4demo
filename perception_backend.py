"""Snapshot-scoped perception tools. Heavy models stay outside the robot interpreter."""
import atexit
import base64
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import threading
from uuid import uuid4

import numpy as np
from PIL import Image
from hardware import DEPLOYMENT, ROOT


def folder(kind):
    return Path(os.environ.get("PYRUA_" + kind.upper() + "_DIR", ROOT / ".local/perception" / kind))


def interpreter():
    return Path(os.environ.get("PYRUA_VISION_PYTHON", DEPLOYMENT / "envs/easywam/bin/python"))


def capabilities():
    return {k: (folder(k) / "model.safetensors").is_file() and interpreter().is_file()
            for k in ("sam","siglip")} | {"sam3_endpoint_configured": bool(os.getenv("PYRUA_SAM3_ENDPOINT"))}


class CPUWorker:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        atexit.register(self.close)

    def close(self):
        proc, self.proc = self.proc, None
        if proc is not None:
            proc.stdin.close()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc.stdout.close()

    def request(self, kind, image, **params):
        with self.lock:
            if not capabilities()[kind]:
                raise ValueError(f"{kind} offline weights or vision interpreter missing")
            if self.proc is None or self.proc.poll() is not None:
                self.close()
                env = {**os.environ, "CUDA_VISIBLE_DEVICES":"", "HF_HUB_OFFLINE":"1",
                       "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "PYTHONNOUSERSITE":"1",
                       "TOKENIZERS_PARALLELISM":"false"}
                env.pop("PYTHONPATH", None)
                self.proc = subprocess.Popen([str(interpreter()), "-I", str(ROOT / "perception/worker.py")],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             text=True, env=env)
            try:
                self.proc.stdin.write(json.dumps({"kind":kind, "folder":str(folder(kind)),
                                                  "image":str(image), **params}) + "\n")
                self.proc.stdin.flush()
                if not select.select([self.proc.stdout], [], [], 90)[0]:
                    raise TimeoutError(f"{kind} inference timed out")
                reply = json.loads(self.proc.stdout.readline())
            except (OSError, ValueError) as exc:
                self.close()
                raise RuntimeError(f"Perception worker unavailable: {exc}") from exc
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return reply


cpu_worker = CPUWorker()


@lru_cache(maxsize=1)
def sam3_client_class():
    # The RSIJudge package may be first on sys.path; use the pinned Lean HTTP transport.
    name = "pyrualean.hosts.sharpa_sam3_http"
    path = ROOT / "upstream/src/pyrualean/hosts/sam3_http.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Sam3Http


def checked_prompt(prompt, backend, width, height):
    if not isinstance(prompt,dict) or len(prompt)!=1:
        raise ValueError("Supply a single prompt: point_xy, bbox_xyxy or text")
    if "text" in prompt and backend == "sam3":
        value = prompt["text"]
        if isinstance(value,str) and 0 < len(value.strip()) <= 200:
            return {"text":value.strip()}
    if "point_xy" in prompt:
        p=prompt["point_xy"]
        if (isinstance(p,list) and len(p)==2 and all(type(v) is int for v in p)
                and 0 <= p[0] < width and 0 <= p[1] < height):
            return prompt
    if "bbox_xyxy" in prompt and backend == "sam":
        p=prompt["bbox_xyxy"]
        if (isinstance(p,list) and len(p)==4 and all(type(v) in (int,float) for v in p)
                and 0 <= p[0] < p[2] <= width and 0 <= p[1] < p[3] <= height):
            return prompt
    raise ValueError(f"Invalid {backend} prompt for a {width}x{height} image")


class FramePerception:
    def __init__(self, backend):
        self.backend = backend
        self.masks = {}

    def check(self, frame_id):
        if self.backend.frame_id != frame_id:
            raise ValueError("Frame mismatch; observe and segment the current image")

    def segment(self, frame_id, model, prompt):
        self.check(frame_id)
        rgb = self.backend.snapshot.rgb
        h,w = rgb.shape[:2]
        if model not in ("sam", "sam3"):
            raise ValueError("Expected sam or sam3")
        prompt = checked_prompt(prompt, model, w,h)
        if model == "sam":
            result = cpu_worker.request(model, self.backend.out / "head.png", prompt=prompt)
            if not result["found"]:
                return {"found": False, "frame_id": frame_id, "model": model}
            if result["shape"] != [h,w]:
                raise ValueError("SAM mask dimensions differ from source image")
            bits = np.frombuffer(base64.b64decode(result["bits"],validate=True), dtype=np.uint8)
            if len(bits)!=(h*w+7)//8:
                raise ValueError("Incomplete SAM mask")
            mask = np.unpackbits(bits)[:h*w].reshape(h,w).astype(bool)
            score = result["score"]
        else:
            endpoint = os.getenv("PYRUA_SAM3_ENDPOINT")
            if not endpoint:
                raise ValueError("SAM3 service endpoint is not configured")
            client = sam3_client_class()(endpoint)
            point = prompt.get("point_xy")
            result = client.segment(rgb, text_prompt=prompt.get("text"), point=point[::-1] if point else None)
            if not result.found:
                return {"found":False, "frame_id":frame_id, "model":model, "reason":result.reason}
            mask = np.asarray(result.mask,dtype=bool)
            score = result.score
        if mask.shape!=(h,w) or not mask.any() or not np.isfinite(score):
            raise ValueError("Invalid model mask or score")
        v,u=np.nonzero(mask)
        ident=uuid4().hex
        self.masks[ident]=mask
        path=self.backend.out / (ident + ".png")
        Image.fromarray(mask.astype(np.uint8)*255).save(path)
        return {"found":True,"model":model,"score":float(score),"mask_id":ident,"frame_id":frame_id,
                "mask_pixels":len(v),"bbox_xyxy":[int(u.min()),int(v.min()),int(u.max()+1),int(v.max()+1)]}

    def mask(self, frame_id, mask_id):
        self.check(frame_id)
        if mask_id not in self.masks:
            raise ValueError("Unknown mask for this frame")
        return self.masks[mask_id]

    def locate_mask(self, frame_id, mask_id):
        mask=self.mask(frame_id,mask_id)
        b=self.backend
        if b.registered is None:
            b.registered=b.snapshot.geometry.register(b.snapshot.raw_depth)
        pixels,points=b.registered
        uv=np.rint(pixels).astype(int)
        x,y=uv.T
        h,w=mask.shape
        valid=(x>=0)&(x<w)&(y>=0)&(y<h)
        selected=points[valid][mask[y[valid],x[valid]]]
        return b.surface_result(selected, {"frame_id":frame_id,"mask_id":mask_id})

    def rank_labels(self, frame_id, labels, mask_id=""):
        self.check(frame_id)
        if not isinstance(labels,list) or not (1 <= len(labels) <= 64) or not all(
                isinstance(x,str) and 0 < len(x.strip()) <= 200 for x in labels):
            raise ValueError("Provide 1..64 nonempty text labels")
        mask_path=None
        if mask_id:
            self.mask(frame_id,mask_id)
            mask_path=str(self.backend.out/(mask_id+".png"))
        return {"frame_id":frame_id,"mask_id":mask_id,"model":"siglip",
                **cpu_worker.request("siglip",self.backend.out/"head.png",labels=labels,mask=mask_path)}
