"""Test-set metrics (mAP + per-class AP) and CPU FPS (onnxruntime) for one or more ONNX exports.

Usage: uv run python evaluate.py runs/nano/r576/rfdetr-nano.onnx runs/nano/r448/rfdetr-nano.onnx ...
Writes results.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import supervision as sv
from PIL import Image
from supervision.metrics import MeanAveragePrecision

TEST = Path("data/coco/test")
CLASSES = ["unripe", "turning", "ripe"]
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class Detector:
    """Minimal ONNX runner: resize -> normalise -> boxes (cxcywh, 0-1) + per-class logits."""

    def __init__(self, path: str, threads: int = 0):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.size = self.inp.shape[2]

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR), np.float32) / 255
        return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

    def __call__(self, rgb: np.ndarray, conf: float = 0.3) -> sv.Detections:
        h, w = rgb.shape[:2]
        boxes, logits = self.sess.run(None, {self.inp.name: self.preprocess(rgb)})[:2]
        # Same as RF-DETR's PostProcess: sigmoid scores, top-300 over all (query, class) pairs.
        probs = 1 / (1 + np.exp(-logits[0][:, : len(CLASSES)]))
        top = np.argsort(probs.ravel())[::-1][:300]
        q, cls = np.unravel_index(top, probs.shape)
        score = probs[q, cls]
        keep = score >= conf
        q, cls, score = q[keep], cls[keep], score[keep]
        cx, cy, bw, bh = boxes[0][q].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        return sv.Detections(xyxy=xyxy, confidence=score, class_id=cls)


def evaluate(onnx_path: str, images: list, targets: list) -> dict:
    det = Detector(onnx_path)
    preds = [det(rgb, conf=0.01) for rgb in images]
    m = MeanAveragePrecision().update(preds, targets).compute()
    ap50 = {CLASSES[c]: m.ap_per_class[i][0] for i, c in enumerate(m.matched_classes)}
    # CPU speed: model only (preprocessed tensor), all threads, 20 warm-up + 200 timed runs.
    x = det.preprocess(images[0])
    for _ in range(20):
        det.sess.run(None, {det.inp.name: x})
    t = time.perf_counter()
    for _ in range(200):
        det.sess.run(None, {det.inp.name: x})
    ms = (time.perf_counter() - t) / 200 * 1000
    return {"size": det.size, "map50": m.map50, "map": m.map50_95, "ap50": ap50, "ms": ms}


def main(paths: list[str]) -> None:
    coco = json.loads((TEST / "_annotations.coco.json").read_text())
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    images, targets = [], []
    for im in coco["images"]:
        images.append(np.asarray(Image.open(TEST / im["file_name"]).convert("RGB")))
        anns = by_img.get(im["id"], [])
        xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)]).reshape(-1, 4)
        targets.append(sv.Detections(xyxy=xyxy, class_id=np.array([a["category_id"] for a in anns], int)))
    n = {c: sum(a["category_id"] == i for a in coco["annotations"]) for i, c in enumerate(CLASSES)}

    rows = [evaluate(p, images, targets) for p in paths]
    lines = [
        f"Test set: {len(images)} frames from a separate recording pass (never seen in training); "
        + ", ".join(f"{n[c]} {c}" for c in CLASSES) + " boxes.",
        "",
        "| input | mAP50 | mAP50-95 | " + " | ".join(f"AP50 {c}" for c in CLASSES) + " | CPU ms/frame | CPU FPS |",
        "|---" * (5 + len(CLASSES)) + "|",
        *(f"| {r['size']} px | {r['map50']:.3f} | {r['map']:.3f} | "
          + " | ".join(f"{r['ap50'].get(c, 0):.3f}" for c in CLASSES)
          + f" | {r['ms']:.0f} | {1000 / r['ms']:.1f} |" for r in rows),
        "",
        f"CPU: model only, onnxruntime {ort.__version__} CPUExecutionProvider, all threads, "
        "mean of 200 runs after 20 warm-up, laptop Intel Core i9-13900H, FP32.",
    ]
    Path("results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
