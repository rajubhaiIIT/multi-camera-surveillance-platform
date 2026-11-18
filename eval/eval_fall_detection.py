"""Comprehensive evaluation of Phase 4 YOLO-Pose Biomechanical Fall Detection.

Benchmark Dataset: UR Fall Detection Dataset (URFD)
- University of Rzeszow multimodal benchmark (Michal Kepski & Bogdan Kwolek)
- Ground-truth fall events (tripping, fainting, slipping, loss of balance)
- Ground-truth ADL activities (walking, sitting, bending to pick up objects, tying shoes, lying down)

Metrics Evaluated:
- Sensitivity / Recall (Falls correctly alarmed)
- Specificity (Normal activities without false alarms)
- Precision (Positive Predictive Value)
- Overall Accuracy
- F1-Score
- Mean Detection Latency (seconds from onset to alarm)
- Inference Throughput (FPS)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ai.pose.detector import PoseDetector
from ai.pose.fall import FallDetector
from ai.pose.schema import FallState


def ensure_dataset(data_dir: Path, num_falls: int = 5, num_adls: int = 5):
    """Ensure URFD benchmark clips exist locally, downloading via PowerShell if needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    falls_present = list(data_dir.glob("fall-*.mp4"))
    adls_present = list(data_dir.glob("adl-*.mp4"))

    if len(falls_present) < num_falls or len(adls_present) < num_adls:
        print(f"Downloading URFD benchmark clips ({num_falls} falls, {num_adls} ADLs)...")
        script = ROOT / "scripts" / "download_urfd.ps1"
        cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Falls",
            str(num_falls),
            "-ADLs",
            str(num_adls),
        ]
        subprocess.run(cmd, check=True)


def evaluate_sequence(
    video_path: Path,
    detector: PoseDetector,
    inactivity_seconds: float = 0.6,
    descent_velocity_threshold: float = 1.5,
) -> dict:
    """Process a single video sequence and return detection metrics."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fall_detector = FallDetector(
        inactivity_seconds=inactivity_seconds,
        descent_velocity_threshold=descent_velocity_threshold,
    )

    frame_idx = 0
    first_fall_time = None
    alerts_emitted = []
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / max(fps, 1.0)
        poses = detector.detect(frame, frame_idx=frame_idx)
        if poses:
            # Associate to primary tracked subject
            main_p = max(poses, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))
            main_p.track_id = 1
            alert = fall_detector.update(main_p, timestamp=timestamp)
            if alert and alert.state == FallState.FALLEN:
                alerts_emitted.append(alert)
                if first_fall_time is None:
                    first_fall_time = timestamp

        frame_idx += 1

    elapsed = max(time.perf_counter() - t_start, 1e-4)
    cap.release()

    return {
        "frames": frame_idx,
        "elapsed_sec": elapsed,
        "fps": frame_idx / elapsed,
        "detected_fall": len(alerts_emitted) > 0,
        "alert_count": len(alerts_emitted),
        "latency_sec": first_fall_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Fall Detection on UR Fall Dataset.")
    parser.add_argument("--num-falls", type=int, default=5, help="Number of fall sequences to test")
    parser.add_argument("--num-adls", type=int, default=5, help="Number of ADL non-fall sequences to test")
    parser.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO-Pose weights")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('cpu', 'cuda:0', 'auto')")
    parser.add_argument("--data-dir", type=str, default="data/raw/urfall", help="Directory storing URFD videos")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    ensure_dataset(data_dir, num_falls=args.num_falls, num_adls=args.num_adls)

    print("\n" + "=" * 80)
    print("  PHASE 4: YOLO-POSE BIOMECHANICAL FALL DETECTION BENCHMARK")
    print(f"  Dataset: UR Fall Detection Dataset (URFD)")
    print(f"  Model Weights: {args.weights} | Device: {args.device}")
    print("=" * 80)

    fall_paths = sorted(data_dir.glob("fall-*.mp4"))[: args.num_falls]
    adl_paths = sorted(data_dir.glob("adl-*.mp4"))[: args.num_adls]

    print(f"\nEvaluating {len(fall_paths)} Fall sequences and {len(adl_paths)} ADL sequences...")

    # Initialize Pose & Fall Detection Engine
    detector = PoseDetector(weights=args.weights, device=args.device)

    print("\n" + "-" * 80)
    print(f"{'Sequence':<16} {'Type':<10} {'GT Label':<12} {'Prediction':<12} {'Result':<10} {'Latency':<10} {'FPS':<8}")
    print("-" * 80)

    tp, fn, fp, tn = 0, 0, 0, 0
    latencies = []
    fps_list = []

    # Evaluate Falls (Positive class)
    for path in fall_paths:
        res = evaluate_sequence(path, detector)
        fps_list.append(res["fps"])
        is_tp = res["detected_fall"]
        if is_tp:
            tp += 1
            status = "PASS (TP)"
            lat = f"{res['latency_sec']:.2f}s"
            if res["latency_sec"] is not None:
                latencies.append(res["latency_sec"])
        else:
            fn += 1
            status = "FAIL (FN)"
            lat = "—"

        pred_str = "FALL" if res["detected_fall"] else "NORMAL"
        print(f"{path.stem:<16} {'Fall':<10} {'FALL':<12} {pred_str:<12} {status:<10} {lat:<10} {res['fps']:<8.1f}")

    # Evaluate ADLs (Negative class)
    for path in adl_paths:
        res = evaluate_sequence(path, detector)
        fps_list.append(res["fps"])
        is_fp = res["detected_fall"]
        if is_fp:
            fp += 1
            status = "FAIL (FP)"
            lat = f"{res['latency_sec']:.2f}s"
        else:
            tn += 1
            status = "PASS (TN)"
            lat = "—"

        pred_str = "FALL" if res["detected_fall"] else "NORMAL"
        print(f"{path.stem:<16} {'ADL':<10} {'NORMAL':<12} {pred_str:<12} {status:<10} {lat:<10} {res['fps']:<8.1f}")

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
    print(f"  Total Test Sequences : {total_samples} ({tp + fn} Falls, {tn + fp} ADLs)")
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
    print("=" * 80)

    # Markdown format for results.md
    print("\n### Markdown Summary for eval/results.md:")
    print(
        f"| 4 | Fall detection | UR Fall Detection (URFD), {total_samples} sequences ({tp+fn} falls, {tn+fp} ADLs) | "
        f"Sensitivity / Specificity / Accuracy / F1 | {sensitivity:.1f}% / {specificity:.1f}% / {accuracy:.1f}% / {f1:.1f}% | "
        f"YOLO11n-Pose + Biomechanical state machine, latency {mean_lat:.2f}s, {mean_fps:.1f} FPS |"
    )


if __name__ == "__main__":
    main()
