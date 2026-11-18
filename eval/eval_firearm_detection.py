"""Comprehensive evaluation of Phase 5 Firearm & Weapon Detection on Videos and Images.

Supports:
- Video files (.mp4, .avi, .mkv, .mov) and Image files (.jpg, .png, .jpeg)
- Folder structure:
    data/raw/weapons/firearms/     (positive class: clips with firearms)
    data/raw/weapons/non_weapons/  (negative class: normal/control clips)
  OR flat files in data/raw/weapons/ with 'gun' or 'firearm' in the filename.

Architecture Evaluated:
- YOLOv8n Firearm Detector (models/firearm_yolov8n.pt)
- Spatial Threat & Person Association (WeaponDetector)

Metrics Evaluated:
- Sensitivity / Recall (Firearm videos correctly alarmed)
- Specificity (Control videos without false alarms)
- Precision (PPV)
- Overall Accuracy
- F1-Score
- Mean Detection Latency (seconds from clip start to weapon alarm)
- Processing Throughput (FPS)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ai.weapons.detector import WeaponDetector
from ai.weapons.schema import WeaponAlert, WeaponRecord, WeaponType


def ensure_dataset(data_dir: Path, weights_path: Path):
    """Ensure benchmark files and firearm weights exist locally."""
    has_weights = weights_path.exists() and weights_path.stat().st_size > 1000000

    if not has_weights:
        print("Downloading Firearm model weights...")
        script = ROOT / "scripts" / "download_firearm_data.ps1"
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)], check=True)


def find_weapon_samples(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Discover firearm samples and non-weapon control samples."""
    extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.jpg", "*.jpeg", "*.png", "*.webp")
    all_files = []
    for ext in extensions:
        all_files.extend(list(data_dir.rglob(ext)))

    firearms = []
    controls = []

    for f in all_files:
        stem_lower = f.stem.lower()
        full_path_lower = str(f).lower()

        # Distinguish controls first (e.g. non_weapon, ctrl, normal, harmless)
        if any(k in full_path_lower for k in ["non_weapon", "nonweapon", "ctrl", "normal", "safe", "control"]):
            controls.append(f)
        elif any(k in full_path_lower for k in ["firearm", "gun", "pistol", "revolver", "rifle", "weapon", "threat"]):
            firearms.append(f)
        else:
            if "gun" in stem_lower:
                firearms.append(f)
            else:
                controls.append(f)

    # Fallback for controls: automatically include ADL videos if none in weapons dir
    if len(controls) == 0:
        urfall_dir = ROOT / "data" / "raw" / "urfall"
        if urfall_dir.exists():
            adls = sorted(list(urfall_dir.glob("adl-*.mp4")))
            if adls:
                controls.extend(adls[:5])

    firearms = sorted(list(set(firearms)))
    controls = sorted(list(set(controls)))
    return firearms, controls


def evaluate_sample(
    file_path: Path,
    detector: WeaponDetector,
    out_dir: Path,
    conf_thresh: float = 0.25,
    save_vis: bool = True,
) -> dict:
    """Evaluate either a video clip or a static image."""
    is_video = file_path.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov", ".webm"]

    if is_video:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {file_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        first_alert_time = None
        alerts_emitted = []
        max_conf = 0.0
        writer = None

        t_start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_idx / max(fps, 1.0)
            weapons, alerts = detector.process(frame, camera_id="bench", frame_idx=frame_idx, timestamp=ts)

            for w in weapons:
                if w.conf > max_conf:
                    max_conf = w.conf

            gun_alerts = [a for a in alerts if a.weapon_type in [WeaponType.GUN, WeaponType.OTHER] or "gun" in str(a.weapon_type).lower()]
            if gun_alerts:
                alerts_emitted.extend(gun_alerts)
                if first_alert_time is None:
                    first_alert_time = ts

            if save_vis:
                vis = detector.draw(frame, weapons, alerts)
                if writer is None:
                    h, w = vis.shape[:2]
                    out_path = out_dir / f"det_{file_path.stem}.mp4"
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)

            frame_idx += 1

        elapsed = max(time.perf_counter() - t_start, 1e-4)
        cap.release()
        if writer:
            writer.release()

        return {
            "type": "video",
            "frames": frame_idx,
            "elapsed_sec": elapsed,
            "fps": frame_idx / elapsed,
            "detected_weapon": len(alerts_emitted) > 0,
            "max_conf": max_conf,
            "latency_sec": first_alert_time,
        }

    else:
        # Static image
        img = cv2.imread(str(file_path))
        if img is None:
            raise RuntimeError(f"Cannot read image {file_path}")

        t0 = time.perf_counter()
        weapons, alerts = detector.process(img, camera_id="bench", frame_idx=0, timestamp=0.0)
        elapsed = max(time.perf_counter() - t0, 1e-5)

        gun_found = any(w.weapon_type in [WeaponType.GUN, WeaponType.OTHER] or "gun" in str(w.weapon_type).lower() for w in weapons)
        max_conf = max([w.conf for w in weapons], default=0.0)

        if save_vis:
            vis = detector.draw(img, weapons, alerts)
            cv2.imwrite(str(out_dir / f"det_{file_path.name}"), vis)

        return {
            "type": "image",
            "frames": 1,
            "elapsed_sec": elapsed,
            "fps": 1.0 / elapsed,
            "detected_weapon": gun_found,
            "max_conf": max_conf,
            "latency_sec": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 Firearm Detection on Videos and Images.")
    parser.add_argument("--weights", type=str, default="models/firearm_yolov8n.pt", help="Path to firearm YOLO weights")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('cpu', 'cuda:0', 'auto')")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--data-dir", type=str, default="data/raw/weapons", help="Directory storing benchmark files")
    parser.add_argument("--num-firearms", type=int, default=5, help="Number of firearm samples to evaluate")
    parser.add_argument("--num-controls", type=int, default=5, help="Number of control samples to evaluate")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    weights_path = ROOT / args.weights
    ensure_dataset(data_dir, weights_path)

    out_vis_dir = ROOT / "eval" / "runs" / "weapons"
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    firearms, controls = find_weapon_samples(data_dir)

    print("\n" + "=" * 80)
    print("  PHASE 5: FIREARM & WEAPON DETECTION BENCHMARK (VIDEOS & IMAGES)")
    print(f"  Model Weights : {args.weights}")
    print(f"  Target Device : {args.device} | Confidence: {args.conf}")
    print(f"  Found Samples : {len(firearms)} Firearm sample(s), {len(controls)} Control sample(s)")
    print("=" * 80)

    if len(firearms) == 0:
        print("\n[!] No firearm video or image files found in data/raw/weapons/.")
        print(f"    Please place at least {args.num_firearms} firearm clips/images into: {data_dir.resolve()}")
        sys.exit(1)

    firearm_samples = firearms[: args.num_firearms]
    control_samples = controls[: args.num_controls]

    print(f"\nEvaluating {len(firearm_samples)} Firearm samples and {len(control_samples)} Control samples...")

    detector = WeaponDetector(weights=str(weights_path), device=args.device, conf=args.conf)

    print("\n" + "-" * 80)
    print(f"{'Sample Name':<28} {'Kind':<8} {'GT Label':<10} {'Detected':<10} {'Conf':<8} {'Result':<10} {'Speed':<8}")
    print("-" * 80)

    tp, fn, fp, tn = 0, 0, 0, 0
    latencies = []
    fps_list = []

    # 1. Evaluate Firearm Samples (Positive Class)
    for p in firearm_samples:
        res = evaluate_sample(p, detector, out_vis_dir, conf_thresh=args.conf)
        fps_list.append(res["fps"])

        is_tp = res["detected_weapon"]
        if is_tp:
            tp += 1
            status = "PASS (TP)"
            if res["latency_sec"] is not None:
                latencies.append(res["latency_sec"])
        else:
            fn += 1
            status = "FAIL (FN)"

        det_str = "GUN" if is_tp else "NONE"
        print(f"{p.name:<28} {res['type']:<8} {'FIREARM':<10} {det_str:<10} {res['max_conf']:<8.2f} {status:<10} {res['fps']:<6.1f} FPS")

    # 2. Evaluate Control Samples (Negative Class)
    for p in control_samples:
        res = evaluate_sample(p, detector, out_vis_dir, conf_thresh=args.conf)
        fps_list.append(res["fps"])

        is_fp = res["detected_weapon"]
        if is_fp:
            fp += 1
            status = "FAIL (FP)"
        else:
            tn += 1
            status = "PASS (TN)"

        det_str = "GUN" if is_fp else "NONE"
        print(f"{p.name:<28} {res['type']:<8} {'NORMAL':<10} {det_str:<10} {res['max_conf']:<8.2f} {status:<10} {res['fps']:<6.1f} FPS")

    total_samples = tp + fn + fp + tn
    sensitivity = (tp / max(1, tp + fn)) * 100.0
    specificity = (tn / max(1, tn + fp)) * 100.0
    precision = (tp / max(1, tp + fp)) * 100.0
    accuracy = ((tp + tn) / max(1, total_samples)) * 100.0
    f1 = 2 * (precision * sensitivity) / max(1e-6, precision + sensitivity)
    mean_lat = np.mean(latencies) if latencies else 0.0
    mean_fps = np.mean(fps_list) if fps_list else 0.0

    print("=" * 80)
    print("  FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Total Test Samples   : {total_samples} ({tp + fn} Firearms, {tn + fp} Controls)")
    print(f"  True Positives (TP)  : {tp}")
    print(f"  False Negatives (FN) : {fn}")
    print(f"  True Negatives (TN)  : {tn}")
    print(f"  False Positives (FP) : {fp}")
    print("  --------------------------------------------------")
    print(f"  Sensitivity (Recall) : {sensitivity:.1f}%")
    print(f"  Specificity          : {specificity:.1f}%")
    print(f"  Precision (PPV)      : {precision:.1f}%")
    print(f"  Overall Accuracy     : {accuracy:.1f}%")
    print(f"  F1-Score             : {f1:.1f}%")
    print(f"  Mean Detection Time  : {mean_lat:.2f} s")
    print(f"  Mean Processing Speed: {mean_fps:.1f} FPS")
    print(f"  Visual Annotations   : {out_vis_dir.resolve()}")
    print("=" * 80)

    # Markdown format for results.md
    print("\n### Markdown Summary for eval/results.md:")
    print(
        f"| 5 | Firearm detection | Firearm & Threat Video/Image Suite, {total_samples} samples ({tp+fn} firearms, {tn+fp} controls) | "
        f"Sensitivity / Specificity / Accuracy / F1 | {sensitivity:.1f}% / {specificity:.1f}% / {accuracy:.1f}% / {f1:.1f}% | "
        f"YOLOv8n Firearm Detector, latency {mean_lat:.2f}s, {mean_fps:.1f} FPS |"
    )


if __name__ == "__main__":
    main()
