"""CLI Runner for Weapon Detection and Threat Object Tracking.

Usage:
  python -m ai.weapons.runner demo --source data/sample_videos/walk.mp4 --save-video out_weapons.mp4
  python -m ai.weapons.runner image --photo threat.jpg --out-photo detected.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import WeaponDetector


def cmd_image(a) -> int:
    """Run weapon detection on a single image."""
    img = cv2.imread(a.photo)
    if img is None:
        print(f"error: cannot read image {a.photo}")
        return 1

    det = WeaponDetector(weights=a.weights, device=a.device, conf=a.conf)
    weapons, alerts = det.process(img, camera_id="photo", frame_idx=0, timestamp=0.0)

    print(f"Detected {len(weapons)} threat/weapon object(s):")
    for w in weapons:
        print(f"  - {w.weapon_type.value}: conf={w.conf:.2f}, bbox={w.bbox}")

    vis = det.draw(img, weapons, alerts)
    out_p = a.out_photo or "detected_weapons.jpg"
    cv2.imwrite(out_p, vis)
    print(f"Saved annotated image -> {out_p}")
    return 0


def cmd_demo(a) -> int:
    """Run weapon detection on video file or live stream."""
    src = a.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        print(f"error: cannot open source {src}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = WeaponDetector(weights=a.weights, device=a.device, conf=a.conf)

    writer = None
    frame_idx = 0
    t0 = time.time()
    all_alerts = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_idx / fps
            weapons, alerts = det.process(frame, camera_id="demo", frame_idx=frame_idx, timestamp=ts)

            for al in alerts:
                all_alerts.append(al.to_dict())
                print(f"[FRAME {frame_idx:05d}] {al.details}")

            vis = det.draw(frame, weapons, alerts)
            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)

            if a.show:
                try:
                    cv2.imshow("Weapon Detection (q to quit)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    if frame_idx == 0:
                        print("warning: headless OpenCV detected (cv2.imshow unavailable). Use --save-video out.mp4 instead.")
                    a.show = False

            frame_idx += 1
            if a.max_frames and frame_idx >= a.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if a.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = time.time() - t0
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(1e-6, elapsed):.1f} FPS)")
    print(f"Total weapon alerts emitted: {len(all_alerts)}")
    if a.out:
        Path(a.out).write_text(json.dumps(all_alerts, indent=2))
        print(f"Wrote alerts -> {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    im = sub.add_parser("image", help="run weapon detection on an image")
    im.add_argument("--photo", required=True)
    im.add_argument("--weights", default="yolo11n.pt")
    im.add_argument("--conf", type=float, default=0.25)
    im.add_argument("--device", default="auto")
    im.add_argument("--out-photo", default="detected_weapons.jpg")
    im.set_defaults(fn=cmd_image)

    d = sub.add_parser("demo", help="run weapon detection on video")
    d.add_argument("--source", required=True)
    d.add_argument("--weights", default="yolo11n.pt")
    d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--device", default="auto")
    d.add_argument("--show", action="store_true")
    d.add_argument("--save-video")
    d.add_argument("--out")
    d.add_argument("--max-frames", type=int)
    d.set_defaults(fn=cmd_demo)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
