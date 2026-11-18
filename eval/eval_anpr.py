"""Comprehensive evaluation of Phase 5 Automatic Number Plate Recognition (ANPR).

Evaluates the two-stage ANPR pipeline:
1. Vehicle Detection: YOLO11n (cars, buses, trucks, motorcycles)
2. License Plate Localization: Morphological Top-Hat + Sobel edge filtering
3. Optical Character Recognition (OCR): EasyOCR (CRAFT + CRNN / ResNet-BiLSTM) with CLAHE
4. Watchlist Matching: Exact match + Levenshtein fuzzy edit distance (<= 1) against stolen vehicles

Metrics Evaluated:
- Plate Detection & Localization Rate (%)
- OCR Character Recognition Accuracy (%)
- Watchlist Hotlist Matching Recall (%)
- Processing Latency (ms) & Throughput (FPS)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ai.vehicles.anpr import LicensePlateReader, clean_plate_text
from ai.vehicles.index import PlateIndex, normalize_plate
from ai.vehicles.tracker import VehicleTracker


def generate_benchmark_plates(out_dir: Path) -> List[Tuple[Path, str]]:
    """Generate standardized benchmark plates with ground truth labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_data = [
        ("plate_bench_01_MH12DE1433.jpg", "MH12DE1433", "White Car (Front)"),
        ("plate_bench_02_DL8CAF5030.jpg", "DL8CAF5030", "Silver Sedan"),
        ("plate_bench_03_KA01MJ5014.jpg", "KA01MJ5014", "Blue SUV"),
        ("plate_bench_04_HR26DQ5551.jpg", "HR26DQ5551", "Commercial Van"),
        ("plate_bench_05_MH02CB1234.jpg", "MH02CB1234", "Black Sedan (Clean)"),
    ]

    generated = []
    for filename, plate_str, desc in bench_data:
        p_path = out_dir / filename
        if not p_path.exists():
            # Create a realistic vehicle front crop with a mounted license plate
            h, w = 360, 640
            # Vehicle bumper canvas (dark metallic gray / slate)
            canvas = np.full((h, w, 3), (45, 48, 55), dtype=np.uint8)
            # Add vehicle grille textures
            for y_line in range(40, 160, 16):
                cv2.line(canvas, (100, y_line), (540, y_line), (25, 28, 32), 4)

            # Center vehicle emblem
            cv2.circle(canvas, (320, 100), 22, (180, 185, 190), 3)

            # Plate dimensions (standard 4:1 aspect ratio)
            pw, ph = 300, 75
            px1 = (w - pw) // 2
            py1 = 190
            px2 = px1 + pw
            py2 = py1 + ph

            # Outer black plate frame
            cv2.rectangle(canvas, (px1 - 4, py1 - 4), (px2 + 4, py2 + 4), (10, 10, 10), -1)
            # High-reflectance clean plate background
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (250, 250, 252), -1)
            # Blue registration strip on left edge
            cv2.rectangle(canvas, (px1, py1), (px1 + 28, py2), (180, 70, 30), -1)
            # Inner border margin
            cv2.rectangle(canvas, (px1 + 2, py1 + 2), (px2 - 2, py2 - 2), (20, 20, 20), 2)

            # High-contrast bold plate characters
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.15
            thickness = 3
            (tw, th), _ = cv2.getTextSize(plate_str, font, scale, thickness)
            tx = px1 + 36 + (pw - 42 - tw) // 2
            ty = py1 + (ph + th) // 2
            cv2.putText(canvas, plate_str, (tx, ty), font, scale, (5, 5, 5), thickness, cv2.LINE_AA)

            cv2.imwrite(str(p_path), canvas)
        generated.append((p_path, plate_str))

    return generated


def discover_plate_samples(data_dir: Path) -> List[Tuple[Path, Optional[str]]]:
    """Discover plate test files in directory or fallback to generated benchmark."""
    samples = []
    if data_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for f in data_dir.glob(ext):
                # Try to extract ground truth from filename if present (e.g. *_MH12DE1433.jpg)
                stem = f.stem.upper()
                gt = None
                for token in stem.split("_"):
                    clean = normalize_plate(token)
                    if len(clean) >= 6:
                        gt = clean
                        break
                samples.append((f, gt))

    # If real images are present, ignore old mock synthetic plates
    real_samples = [s for s in samples if not s[0].name.startswith("plate_bench_")]
    if real_samples:
        samples = real_samples

    if not samples:
        print("[*] No existing plate images found in data/raw/plates/. Initializing standard benchmark suite...")
        samples = generate_benchmark_plates(data_dir)

    return sorted(samples, key=lambda s: s[0].name)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 ANPR License Plate Extraction Model.")
    parser.add_argument("--data-dir", type=str, default="data/raw/plates", help="Path to plate images directory")
    parser.add_argument("--watchlist", type=str, default="models/vehicle_watchlist.json", help="Path to hotlist JSON")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('cpu', 'cuda:0', 'auto')")
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "fast_alpr", "easyocr"], help="ANPR engine")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--out-dir", type=str, default="eval/runs/anpr", help="Output directory for annotated crops")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    watchlist_path = ROOT / args.watchlist
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Watchlist Index
    watchlist = PlateIndex.load(watchlist_path) if watchlist_path.exists() else PlateIndex()

    # Initialize ANPR components
    print("\n" + "=" * 80)
    print("  PHASE 5: AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) BENCHMARK")
    print(f"  Target Device : {args.device} | Requested Engine: {args.engine}")
    print(f"  Watchlist DB  : {args.watchlist} ({watchlist.size} active targets)")
    print("=" * 80)

    print("Initializing vehicle detector & plate recognizer engine...")
    t_init = time.perf_counter()
    tracker = VehicleTracker(device=args.device, conf=args.conf)
    reader = LicensePlateReader(engine=args.engine, gpu=(args.device != "cpu"))
    init_time = time.perf_counter() - t_init
    engine_desc = "Fast-ALPR (YOLO-v9-t + MobileViT-v2)" if reader.fast_alpr is not None else "EasyOCR (Morphological + CRAFT/CRNN)"
    print(f"Engine active : {engine_desc}")
    print(f"Initialized in: {init_time:.2f}s.")

    samples = discover_plate_samples(data_dir)
    print(f"\nEvaluating {len(samples)} license plate benchmark sequence(s)...")

    print("\n" + "-" * 88)
    print(f"{'Sample Image':<28} {'GT Plate':<13} {'Extracted Plate':<16} {'Format':<8} {'Hotlist Match':<16} {'Latency':<8}")
    print("-" * 88)

    total_samples = len(samples)
    plates_located = 0
    exact_matches = 0
    watchlist_hits = 0
    latencies = []

    for img_path, gt_plate in samples:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        # 1. Detect vehicle or use full crop
        vehicles, _ = tracker.update(frame, camera_id="bench", frame_idx=0, timestamp=0.0)
        veh_bbox = vehicles[0].bbox if vehicles else (0.0, 0.0, float(w), float(h))

        # 2. Localize plate candidate & extract characters
        plate_rec = reader.process_vehicle(frame, veh_bbox)
        if plate_rec is None and hasattr(reader, "read_frame"):
            frame_records = reader.read_frame(frame)
            if frame_records:
                plate_rec = frame_records[0]

        elapsed_sec = time.perf_counter() - t0
        latencies.append(elapsed_sec)

        extracted_text = plate_rec.text if plate_rec else "NOT_FOUND"
        country_fmt = plate_rec.country_format if plate_rec else "—"

        is_located = plate_rec is not None
        if is_located:
            plates_located += 1

        # Compare with Ground Truth if available
        gt_display = gt_plate or "N/A"
        is_exact = False
        if gt_plate and is_located:
            norm_extracted = normalize_plate(extracted_text)
            norm_gt = normalize_plate(gt_plate)
            is_exact = (norm_extracted == norm_gt)
            if is_exact:
                exact_matches += 1

        # Check Hotlist / Watchlist matching
        match_entry = watchlist.match(extracted_text) if is_located else None
        if match_entry:
            watchlist_hits += 1
            hotlist_str = f"ALERT ({match_entry.reason[:10]})"
        else:
            hotlist_str = "CLEAN (No Hit)"

        # Save visual output
        vis = frame.copy()
        if plate_rec:
            px1, py1, px2, py2 = (int(round(v)) for v in plate_rec.bbox)
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 2)
            tag = f"{plate_rec.text} ({int(plate_rec.confidence * 100)}%)"
            cv2.rectangle(vis, (px1, max(0, py1 - 22)), (px1 + 180, py1), (0, 255, 0), -1)
            cv2.putText(vis, tag, (px1 + 4, max(16, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        if match_entry:
            cv2.rectangle(vis, (0, 0), (w, 32), (0, 0, 200), -1)
            cv2.putText(vis, f"WATCHLIST ALERT: {match_entry.plate} - {match_entry.reason}", (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out_vis_path = out_dir / f"anpr_{img_path.name}"
        cv2.imwrite(str(out_vis_path), vis)

        latency_str = f"{elapsed_sec * 1000:.1f} ms"
        print(f"{img_path.name:<28} {gt_display:<13} {extracted_text:<16} {country_fmt:<8} {hotlist_str:<16} {latency_str:<8}")

    # Summary
    loc_rate = (plates_located / max(1, total_samples)) * 100.0
    rec_acc = (exact_matches / max(1, plates_located)) * 100.0 if plates_located > 0 else 0.0
    mean_lat_ms = (sum(latencies) / max(1, len(latencies))) * 1000.0
    mean_fps = 1000.0 / max(1e-3, mean_lat_ms)

    print("\n" + "=" * 80)
    print("  FINAL ANPR BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Total Test Sequences       : {total_samples}")
    print(f"  Plates Successfully Located: {plates_located} / {total_samples} ({loc_rate:.1f}%)")
    print(f"  Character Exact Accuracy   : {exact_matches} / {plates_located} ({rec_acc:.1f}%)")
    print(f"  Watchlist Matches Emitted  : {watchlist_hits}")
    print("  --------------------------------------------------")
    print(f"  Plate Localization Rate    : {loc_rate:.1f}%")
    print(f"  Character Recognition Acc  : {rec_acc:.1f}%")
    print(f"  Mean Processing Latency    : {mean_lat_ms:.1f} ms")
    print(f"  Mean Throughput Speed      : {mean_fps:.1f} FPS")
    print("=" * 80)
    print(f"Visual inference overlays saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
