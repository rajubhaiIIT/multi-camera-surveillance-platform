"""Comprehensive evaluation of Phase 4 Multi-Person Violence and Fight Detection on Actual Video Footage.

Dataset Supported:
- Real-Life Violence Situations (RLVS) / RWF-2000 / AIRTLab / Hockey Fights
- Supports any .mp4, .avi, or .mkv files placed in data/raw/violence/

Folder Structure Supported:
Option A (Flat):
  data/raw/violence/fight_01.mp4 (or V_1.mp4)
  data/raw/violence/fight_02.mp4
  ...
  data/raw/violence/nonfight_01.mp4 (or NV_1.mp4)
  data/raw/violence/nonfight_02.mp4
  ...

Option B (Subfolders):
  data/raw/violence/fights/   (contains 5 fight clips)
  data/raw/violence/nonfights/ (contains 5 non-fight clips)

Pipeline Evaluated:
- YOLO11n-Pose (17 COCO Keypoints)
- Frame-to-frame Multi-Person IoU Tracker
- Spatio-temporal Violence & Struggle Engine (ViolenceDetector)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ai.pose.detector import PoseDetector, _box_iou
from ai.pose.schema import FightAlert, PoseRecord
from ai.pose.violence import ViolenceDetector


class SimpleIoUTracker:
    """Lightweight single-camera multi-person tracker for frame-to-frame pose association."""

    def __init__(self, iou_thresh: float = 0.25, max_age: int = 15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.active_tracks: Dict[int, Tuple[tuple, int]] = {}  # tid -> (bbox, age)

    def update(self, poses: List[PoseRecord]) -> List[PoseRecord]:
        if not poses:
            for tid in list(self.active_tracks):
                bbox, age = self.active_tracks[tid]
                if age + 1 > self.max_age:
                    del self.active_tracks[tid]
                else:
                    self.active_tracks[tid] = (bbox, age + 1)
            return []

        assigned_poses = []
        unmatched_poses = []
        used_tids = set()

        for p in poses:
            best_iou = 0.0
            best_tid = None
            for tid, (bbox, _) in self.active_tracks.items():
                if tid in used_tids:
                    continue
                iou = _box_iou(p.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_thresh:
                used_tids.add(best_tid)
                p.track_id = best_tid
                self.active_tracks[best_tid] = (p.bbox, 0)
                assigned_poses.append(p)
            else:
                unmatched_poses.append(p)

        for p in unmatched_poses:
            new_id = self.next_id
            self.next_id += 1
            p.track_id = new_id
            self.active_tracks[new_id] = (p.bbox, 0)
            assigned_poses.append(p)

        for tid in list(self.active_tracks):
            if tid not in used_tids and tid not in [p.track_id for p in unmatched_poses]:
                bbox, age = self.active_tracks[tid]
                if age + 1 > self.max_age:
                    del self.active_tracks[tid]
                else:
                    self.active_tracks[tid] = (bbox, age + 1)

        return assigned_poses


def evaluate_video_file(
    video_path: Path,
    detector: PoseDetector,
    proximity_thresh: float = 1.2,
    energy_thresh: float = 1.2,
    consecutive_frames: int = 4,
) -> dict:
    """Process an actual video file through YOLO-Pose + ViolenceDetector."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = SimpleIoUTracker(iou_thresh=0.25)
    vdetector = ViolenceDetector(
        proximity_threshold=proximity_thresh,
        motion_energy_threshold=energy_thresh,
        consecutive_frames=consecutive_frames,
    )

    frame_idx = 0
    first_fight_time = None
    alerts_emitted: List[FightAlert] = []
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / max(fps, 1.0)
        poses = detector.detect(frame, frame_idx=frame_idx)
        tracked_poses = tracker.update(poses)

        alerts = vdetector.update(tracked_poses, timestamp=timestamp)
        if alerts:
            alerts_emitted.extend(alerts)
            if first_fight_time is None:
                first_fight_time = timestamp

        frame_idx += 1

    elapsed = max(time.perf_counter() - t_start, 1e-4)
    cap.release()

    return {
        "frames": frame_idx,
        "elapsed_sec": elapsed,
        "fps": frame_idx / elapsed,
        "detected_fight": len(alerts_emitted) > 0,
        "alert_count": len(alerts_emitted),
        "latency_sec": first_fight_time,
    }


def find_benchmark_clips(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Automatically discover fight and non-fight video clips in data_dir at any directory depth."""
    extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm")
    all_files = []
    for ext in extensions:
        all_files.extend(list(data_dir.rglob(ext)))

    fights = []
    nonfights = []

    for f in all_files:
        stem_lower = f.stem.lower()
        parent_lower = f.parent.name.lower()
        full_path_lower = str(f).lower()

        # Distinguish non-fight first (e.g. nonfight, nonviolence, nv, normal, no_fight, non-fight)
        if any(k in full_path_lower for k in ["nonfight", "non_fight", "non-fight", "nonviolence", "non_violence", "normal", "nv_"]):
            nonfights.append(f)
        elif any(k in full_path_lower for k in ["fight", "violence", "violent", "v_"]):
            fights.append(f)
        else:
            if stem_lower.startswith("v") and not stem_lower.startswith("nv"):
                fights.append(f)
            else:
                nonfights.append(f)

    fights = sorted(list(set(fights)))
    nonfights = sorted(list(set(nonfights)))

    # Fallback for non-fights: automatically use existing ADL surveillance clips if none in violence dir
    if len(nonfights) == 0:
        urfall_dir = ROOT / "data" / "raw" / "urfall"
        if urfall_dir.exists():
            adls = sorted(list(urfall_dir.glob("adl-*.mp4")))
            if adls:
                nonfights.extend(adls)

    return fights, nonfights


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Violence / Fight Detection on Real Video Data.")
    parser.add_argument("--num-fights", type=int, default=5, help="Number of fight sequences to evaluate")
    parser.add_argument("--num-nonfights", type=int, default=5, help="Number of non-fight sequences to evaluate")
    parser.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO-Pose weights")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('cpu', 'cuda:0', 'auto')")
    parser.add_argument("--data-dir", type=str, default="data/raw/violence", help="Directory storing video clips")
    parser.add_argument("--proximity-thresh", type=float, default=1.2, help="Proximity threshold in fraction of body height")
    parser.add_argument("--energy-thresh", type=float, default=1.2, help="Kinetic energy threshold in h/s")
    parser.add_argument("--consecutive-frames", type=int, default=4, help="Consecutive frames required for alert")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    fights, nonfights = find_benchmark_clips(data_dir)

    print("\n" + "=" * 80)
    print("  PHASE 4: YOLO-POSE MULTI-PERSON VIOLENCE / FIGHT DETECTION BENCHMARK")
    print(f"  Target Directory: {data_dir}")
    print(f"  Found Files     : {len(fights)} Fight clip(s), {len(nonfights)} Non-Fight clip(s)")
    print(f"  Model Weights   : {args.weights} | Device: {args.device}")
    print(f"  Config          : proximity={args.proximity_thresh}h, energy_thresh={args.energy_thresh} h/s, min_frames={args.consecutive_frames}")
    print("=" * 80)

    if len(fights) == 0 or len(nonfights) == 0:
        print("\n[!] Video clips not found in data/raw/violence/.")
        print(f"    Please place at least {args.num_fights} fight clips and {args.num_nonfights} non-fight clips into:")
        print(f"    -> {data_dir.resolve()}")
        print("\n    Accepted naming:")
        print("      Fight files     : fight_01.mp4, fight_02.mp4 ... OR V_1.mp4, V_2.mp4 ... OR inside 'fights/' subfolder")
        print("      Non-fight files : nonfight_01.mp4, nonfight_02.mp4 ... OR NV_1.mp4, NV_2.mp4 ... OR inside 'nonfights/' subfolder")
        sys.exit(1)

    fight_paths = fights[: args.num_fights]
    nonfight_paths = nonfights[: args.num_nonfights]

    print(f"\nEvaluating {len(fight_paths)} Fight clips and {len(nonfight_paths)} Non-Fight clips...")

    detector = PoseDetector(weights=args.weights, device=args.device)

    print("\n" + "-" * 80)
    print(f"{'Sequence':<24} {'Type':<10} {'GT Label':<10} {'Prediction':<10} {'Result':<10} {'Latency':<10} {'FPS':<8}")
    print("-" * 80)

    tp, fn, fp, tn = 0, 0, 0, 0
    latencies = []
    fps_list = []

    # 1. Evaluate Fights (Positive Class)
    for path in fight_paths:
        res = evaluate_video_file(
            path,
            detector,
            proximity_thresh=args.proximity_thresh,
            energy_thresh=args.energy_thresh,
            consecutive_frames=args.consecutive_frames,
        )
        fps_list.append(res["fps"])

        is_tp = res["detected_fight"]
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

        pred_str = "FIGHT" if res["detected_fight"] else "NORMAL"
        print(f"{path.name:<24} {'Fight':<10} {'FIGHT':<10} {pred_str:<10} {status:<10} {lat:<10} {res['fps']:<8.1f}")

    # 2. Evaluate Non-Fights (Negative Class)
    for path in nonfight_paths:
        res = evaluate_video_file(
            path,
            detector,
            proximity_thresh=args.proximity_thresh,
            energy_thresh=args.energy_thresh,
            consecutive_frames=args.consecutive_frames,
        )
        fps_list.append(res["fps"])

        is_fp = res["detected_fight"]
        if is_fp:
            fp += 1
            status = "FAIL (FP)"
            lat = f"{res['latency_sec']:.2f}s"
        else:
            tn += 1
            status = "PASS (TN)"
            lat = "—"

        pred_str = "FIGHT" if res["detected_fight"] else "NORMAL"
        print(f"{path.name:<24} {'NonFight':<10} {'NORMAL':<10} {pred_str:<10} {status:<10} {lat:<10} {res['fps']:<8.1f}")

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
    print(f"  Total Test Sequences : {total_samples} ({tp + fn} Fights, {tn + fp} NonFights)")
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
        f"| 4 | Violence / Fight detection | Real-World Video Benchmark, {total_samples} sequences ({tp+fn} fights, {tn+fp} non-fights) | "
        f"Sensitivity / Specificity / Accuracy / F1 | {sensitivity:.1f}% / {specificity:.1f}% / {accuracy:.1f}% / {f1:.1f}% | "
        f"YOLO11n-Pose + Pairwise kinetic energy engine, latency {mean_lat:.2f}s, {mean_fps:.1f} FPS |"
    )


if __name__ == "__main__":
    main()
