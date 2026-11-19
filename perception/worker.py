"""CPU-only, file-local SAM / SigLIP inference in a separate Transformers process."""
import base64
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@lru_cache(maxsize=2)
def load(kind, folder):
    from transformers import AutoProcessor, SamModel, SiglipModel
    cls = SamModel if kind == "sam" else SiglipModel
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
    model = cls.from_pretrained(folder, local_files_only=True).eval()
    return processor, model


@torch.inference_mode()
def predict(req):
    image = Image.open(req["image"]).convert("RGB")
    kind = req["kind"]
    processor, model = load(kind, req["folder"])
    if kind == "siglip":
        if req.get("mask"):
            mask = Image.open(req["mask"]).convert("L")
            if image.size != mask.size or mask.getbbox() is None:
                raise ValueError("Invalid mask for this image")
            image = Image.composite(image, Image.new("RGB", image.size, (127,127,127)), mask)
            image = image.crop(mask.getbbox())
        texts = req["labels"]
        inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")
        scores = model(**inputs).logits_per_image[0].sigmoid().tolist()
        return {"ranking": sorted([{"label": t, "score": float(s)} for t,s in zip(texts,scores)],
                                  key=lambda row: -row["score"])}
    prompt = req["prompt"]
    kwargs = ({"input_points": [[prompt["point_xy"]]]} if "point_xy" in prompt
              else {"input_boxes": [[prompt["bbox_xyxy"]]]})
    inputs = processor(images=image, return_tensors="pt", **kwargs)
    shapes = [inputs.pop(k) for k in ("original_sizes","reshaped_input_sizes")]
    outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), *shapes)[0]
    scores = outputs.iou_scores[0,0]
    i = scores.argmax().item()
    mask = np.asarray(masks[0,i], dtype=bool)
    return {"found": bool(mask.any()), "score": float(scores[i]), "shape": list(mask.shape),
            "bits": base64.b64encode(np.packbits(mask).tobytes()).decode()}


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            answer = predict(json.loads(line))
        except Exception as exc:
            answer = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(answer, allow_nan=False), flush=True)
