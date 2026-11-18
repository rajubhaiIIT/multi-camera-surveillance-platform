#!/usr/bin/env python3
"""Run detection + ByteTrack on the MOT17 train sequences and write MOTChallenge result files.

    python eval/run_mot17.py --run-name yolo11n_640 --imgsz 640
    python eval/run_mot17.py --run-name yolo11n_1280 --imgsz 1280

MOT17 has three detector variants (DPM/FRCNN/SDP) of the SAME images and ground truth.
We run our own detector, so only one variant is needed (default SDP): 7 unique sequences.
Afterwards:  python eval/eval_mot17.py --run-name <same name>
"""
from __future__ import annotations

import argparse
import configparser
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from ai.mot_io import write_mot  # noqa: E402
from ai.yolo_tracker import Tracker, resolve_device  # noqa: E402


def find_mot_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    base = ROOT / "data" / "raw" / "mot17"
    for p in [base / "MOT17", base]:
        if (p / "train").is_dir():
            return p
    sys.exit("MOT17 not found. Run: python scripts/download_data.py fetch mot17")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--mot-root")
    ap.add_argument("--detector", default="SDP", choices=["DPM", "FRCNN", "SDP"])
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-frames", type=int, help="debug: only the first N frames per sequence")
    a = ap.parse_args()

    mot = find_mot_root(a.mot_root)
    seqs = sorted(p for p in (mot / "train").iterdir() if p.is_dir() and p.name.endswith(f"-{a.detector}"))
    if not seqs:
        sys.exit(f"no *-{a.detector} sequences under {mot / 'train'}")
    out_dir = ROOT / "eval" / "runs" / a.run_name
    (out_dir / "tracks").mkdir(parents=True, exist_ok=True)

    # warm-up so CUDA start-up cost is not counted in the FPS numbers
    warm = Tracker("warmup", a.weights, imgsz=a.imgsz, conf=a.conf, classes=[0], device=a.device)
    first = cv2.imread(str(sorted((seqs[0] / "img1").glob("*.jpg"))[0]))
    for i in range(3):
        warm.update(first, 0.0, i)
    device, half = warm.device, warm.half
    del warm

    per_seq, total_frames, total_time = {}, 0, 0.0
    for sq in seqs:
        info = configparser.ConfigParser()
        info.read(sq / "seqinfo.ini")
        fps = float(info["Sequence"]["frameRate"])
        imgs = sorted((sq / "img1").glob("*.jpg"))[: a.max_frames]
        tr = Tracker(sq.name, a.weights, imgsz=a.imgsz, conf=a.conf, classes=[0], device=a.device)  # persons only
        rows = []
        t0 = time.perf_counter()
        for i, p in enumerate(imgs):
            img = cv2.imread(str(p))
            for r in tr.update(img, i / fps, i):
                rows.append((int(p.stem), r.track_id, *r.bbox, r.conf))
        dt = time.perf_counter() - t0
        write_mot(out_dir / "tracks" / f"{sq.name}.txt", rows)
        per_seq[sq.name] = {"frames": len(imgs), "seconds": round(dt, 2), "fps": round(len(imgs) / dt, 1)}
        total_frames += len(imgs)
        total_time += dt
        print(f"{sq.name}: {len(imgs)} frames, {len(imgs) / dt:.1f} FPS, {len(rows)} boxes")

    meta = {
        "date": str(date.today()), "weights": a.weights, "imgsz": a.imgsz, "conf": a.conf,
        "detector_variant": a.detector, "device": device, "fp16": half, "classes": "person only",
        "tracker": "ByteTrack (ultralytics bytetrack.yaml, default params)",
        "overall_fps_incl_image_read": round(total_frames / total_time, 1),
        "frames": total_frames, "per_sequence": per_seq, "python": platform.python_version(),
    }
    try:
        import torch, ultralytics
        meta.update(torch=torch.__version__, ultralytics=ultralytics.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nOverall: {meta['overall_fps_incl_image_read']} FPS on {total_frames} frames "
          f"({device}, fp16={half}).\nNext: python eval/eval_mot17.py --run-name {a.run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
