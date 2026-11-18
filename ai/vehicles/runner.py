"""CLI Runner for Vehicle Tracking, ANPR and Watchlist Management.

Commands:
  demo       run live vehicle tracking and ANPR on video/stream
  watchlist  manage hotlist plates (add, remove, list)
  run-eval   run vehicle tracking and ANPR on an existing run directory
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .anpr import LicensePlateReader
from .index import PlateIndex
from .pipeline import VehicleANPRPipeline
from .tracker import VehicleTracker

DEFAULT_WATCHLIST = "models/vehicle_watchlist.json"


def cmd_watchlist_add(a) -> int:
    idx = PlateIndex.load(a.file) if Path(a.file).exists() else PlateIndex()
    entry = idx.enroll(a.plate, reason=a.reason, owner=getattr(a, "owner", "") or "", notes=getattr(a, "notes", "") or "")
    idx.save(a.file)
    print(f"Added plate '{entry.plate}' ({entry.reason}) to watchlist. Total entries: {idx.size}")
    return 0


def cmd_watchlist_remove(a) -> int:
    if not Path(a.file).exists():
        print("error: watchlist file not found")
        return 1
    idx = PlateIndex.load(a.file)
    if idx.remove(a.plate):
        idx.save(a.file)
        print(f"Removed plate '{a.plate}'. Total entries: {idx.size}")
        return 0
    else:
        print(f"error: plate '{a.plate}' not in watchlist")
        return 1


def cmd_watchlist_list(a) -> int:
    if not Path(a.file).exists():
        print("watchlist is empty (file does not exist)")
        return 0
    idx = PlateIndex.load(a.file)
    print(f"{'PLATE':15s}  {'REASON':25s}  OWNER")
    for k, v in idx._entries.items():
        print(f"{k:15s}  {v.reason:25s}  {v.owner}")
    print(f"\nTotal: {idx.size} entries")
    return 0


def cmd_demo(a) -> int:
    src = a.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        print(f"error: cannot open source {src}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    watchlist = PlateIndex.load(a.watchlist) if (a.watchlist and Path(a.watchlist).exists()) else PlateIndex()
    tracker = VehicleTracker(weights=a.weights, device=a.device, conf=a.conf)
    reader = LicensePlateReader(gpu=(a.device != "cpu"))
    pipe = VehicleANPRPipeline(tracker=tracker, plate_reader=reader, watchlist=watchlist)

    writer = None
    frame_idx = 0
    t0 = time.time()
    all_events = []
    all_alerts = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_idx / fps
            vehicles, alerts = pipe.process(frame, camera_id="demo", frame_idx=frame_idx, timestamp=ts)

            for v in vehicles:
                all_events.append(v.to_dict())
            for al in alerts:
                all_alerts.append(al.to_dict())
                print(f"[FRAME {frame_idx:05d}] {al.details}")

            vis = pipe.draw(frame, vehicles, alerts)
            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)

            if a.show:
                try:
                    cv2.imshow("Vehicle Tracking & ANPR (q to quit)", vis)
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
    print(f"Total vehicle events: {len(all_events)}, Total alerts: {len(all_alerts)}")
    if a.out:
        out_data = {"vehicle_events": all_events, "alerts": all_alerts}
        Path(a.out).write_text(json.dumps(out_data, indent=2))
        print(f"Wrote results -> {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Watchlist subcommand
    w = sub.add_parser("watchlist", help="manage hotlist/stolen license plates")
    w_sub = w.add_subparsers(dest="wcmd", required=True)

    wa = w_sub.add_parser("add", help="add plate to watchlist")
    wa.add_argument("--plate", required=True)
    wa.add_argument("--reason", required=True)
    wa.add_argument("--owner", default="")
    wa.add_argument("--notes", default="")
    wa.add_argument("--file", default=DEFAULT_WATCHLIST)
    wa.set_defaults(fn=cmd_watchlist_add)

    wr = w_sub.add_parser("remove", help="remove plate from watchlist")
    wr.add_argument("--plate", required=True)
    wr.add_argument("--file", default=DEFAULT_WATCHLIST)
    wr.set_defaults(fn=cmd_watchlist_remove)

    wl = w_sub.add_parser("list", help="list watchlist plates")
    wl.add_argument("--file", default=DEFAULT_WATCHLIST)
    wl.set_defaults(fn=cmd_watchlist_list)

    # Demo subcommand
    d = sub.add_parser("demo", help="run vehicle tracking & ANPR on video/camera")
    d.add_argument("--source", required=True)
    d.add_argument("--weights", default="yolo11n.pt")
    d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--watchlist", default=DEFAULT_WATCHLIST)
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
