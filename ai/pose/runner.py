"""CLI Runner for Phase 4 Pose, Fall, and Violence detection.

Usage:
  python -m ai.pose.runner demo --source data/sample_videos/walk.mp4 --save-video out.mp4
  python -m ai.pose.runner run-eval runs/epfl_terrace_rtdetr --render
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import PoseDetector
from .fall import FallDetector
from .pipeline import PosePipeline
from .violence import ViolenceDetector


def cmd_demo(a) -> int:
    """Run live pose, fall, and violence detection on a video file or stream."""
    src = a.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        print(f"error: cannot open video source {src}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = PoseDetector(weights=a.weights, device=a.device, conf=a.conf)
    pipe = PosePipeline(det, run_every=a.run_every)

    writer = None
    frame_idx = 0
    t0 = time.time()
    all_fall_alerts, all_fight_alerts = [], []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts = frame_idx / fps
            poses, falls, fights = pipe.process(frame, camera_id="demo", frame_idx=frame_idx, timestamp=ts)

            for fa in falls:
                all_fall_alerts.append(fa.to_dict())
                print(f"[FRAME {frame_idx:05d}] {fa.details}")
            for fta in fights:
                all_fight_alerts.append(fta.to_dict())
                print(f"[FRAME {frame_idx:05d}] {fta.details}")

            vis = pipe.draw(frame, poses, falls, fights)

            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)

            if a.show:
                cv2.imshow("Phase 4: Pose, Fall & Violence (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if a.max_frames and frame_idx >= a.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if a.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(1e-6, elapsed):.1f} FPS)")
    print(f"Emitted {len(all_fall_alerts)} fall alerts, {len(all_fight_alerts)} fight alerts.")
    if a.out:
        out_data = {"fall_alerts": all_fall_alerts, "fight_alerts": all_fight_alerts}
        Path(a.out).write_text(json.dumps(out_data, indent=2))
        print(f"Wrote alerts -> {a.out}")
    return 0


def cmd_run_eval(a) -> int:
    """Run pose, fall, and violence detection on an existing multi-camera run directory."""
    from ..recording import load_run
    from ..sources import MediaReader

    run_dir = Path(a.run)
    run = load_run(run_dir)
    meta_path = run.dir / "meta.json"
    if not meta_path.exists():
        print(f"error: {meta_path} not found")
        return 1

    meta = json.loads(meta_path.read_text())
    videos = meta.get("videos", {})
    if not videos and "cameras" in meta:
        videos = {cam: cdata["video"] for cam, cdata in meta["cameras"].items() if "video" in cdata}
    if not videos:
        print("error: no video paths in meta.json")
        return 1

    det = PoseDetector(weights=a.weights, device=a.device, conf=a.conf)
    pipe = PosePipeline(det, run_every=a.run_every)

    render_dir = None
    if getattr(a, "render", False):
        render_dir = run.dir / "pose_render"
        render_dir.mkdir(parents=True, exist_ok=True)

    events_by_cam: dict = {}
    for ev in run.events:
        for r in ev.recs:
            events_by_cam.setdefault((ev.cam, ev.idx), []).append(r)

    all_fall_alerts = []
    all_fight_alerts = []

    for cam_id, vid_path in videos.items():
        cam_info = meta.get("cameras", {}).get(cam_id, {})
        stride = int(cam_info.get("stride", meta.get("stride", 1)))
        folder_fps = float(cam_info.get("fps", 25.0))
        last_idx = cam_info.get("last_idx", float("inf"))

        try:
            reader = MediaReader(vid_path, stride=stride, folder_fps=folder_fps)
        except Exception as e:
            print(f"  warning: cannot open media for camera {cam_id} ({vid_path}): {e}")
            continue

        print(f"  evaluating pose + falls on camera {cam_id} ...")
        writer = None
        for frame_idx, frame in reader.frames():
            if frame_idx > last_idx:
                break
            ts = frame_idx / folder_fps
            tracks = events_by_cam.get((cam_id, frame_idx), [])

            poses, falls, fights = pipe.process(frame, camera_id=cam_id, frame_idx=frame_idx,
                                                timestamp=ts, tracks=tracks)
            for fa in falls:
                all_fall_alerts.append(fa.to_dict())
            for fta in fights:
                all_fight_alerts.append(fta.to_dict())

            if render_dir is not None:
                vis = pipe.draw(frame, poses, falls, fights)
                if writer is None:
                    h, w = vis.shape[:2]
                    out_path = render_dir / f"{cam_id}.mp4"
                    fps_out = max(reader.fps / stride, 1.0)
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
                writer.write(vis)

        if writer:
            writer.release()

    alerts_path = run.dir / "pose_alerts.json"
    alerts_data = {
        "fall_alerts": all_fall_alerts,
        "fight_alerts": all_fight_alerts,
        "summary": {
            "total_falls": len(all_fall_alerts),
            "total_fights": len(all_fight_alerts),
        }
    }
    alerts_path.write_text(json.dumps(alerts_data, indent=2))
    print(f"\nPose evaluation complete:")
    print(f"  Total fall alerts: {len(all_fall_alerts)}")
    print(f"  Total fight alerts: {len(all_fight_alerts)}")
    print(f"  Wrote alerts -> {alerts_path}")
    if render_dir:
        print(f"  Rendered annotated videos -> {render_dir}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run pose, fall and violence detection on video")
    d.add_argument("--source", required=True)
    d.add_argument("--weights", default="yolo11n-pose.pt")
    d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--run-every", type=int, default=1)
    d.add_argument("--device", default="auto")
    d.add_argument("--show", action="store_true")
    d.add_argument("--save-video")
    d.add_argument("--out")
    d.add_argument("--max-frames", type=int)
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("run-eval", help="evaluate pose, falls and violence on a run directory")
    r.add_argument("run")
    r.add_argument("--weights", default="yolo11n-pose.pt")
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--run-every", type=int, default=1)
    r.add_argument("--device", default="auto")
    r.add_argument("--render", action="store_true")
    r.set_defaults(fn=cmd_run_eval)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
