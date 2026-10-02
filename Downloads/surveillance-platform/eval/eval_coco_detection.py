#!/usr/bin/env python3
"""Detection mAP on COCO val2017 using Ultralytics' validator.

    python eval/eval_coco_detection.py --weights yolo11n.pt

First run downloads COCO val (about 1 GB images + labels) via Ultralytics.
Published baseline for YOLO11n: mAP50-95 about 39.5 at imgsz 640.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="yolo11n.pt")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default="0")
a = ap.parse_args()

m = YOLO(a.weights).val(data="coco.yaml", imgsz=a.imgsz, device=a.device, plots=False)
res = {"weights": a.weights, "imgsz": a.imgsz, "mAP50-95": round(m.box.map * 100, 2),
       "mAP50": round(m.box.map50 * 100, 2), "precision": round(float(m.box.mp) * 100, 2),
       "recall": round(float(m.box.mr) * 100, 2)}
out = Path(__file__).parent / "runs" / f"coco_val_{Path(a.weights).stem}_{a.imgsz}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(res)
