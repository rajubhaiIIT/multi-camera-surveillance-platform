"""Ingestion worker: read a camera, detect + track, emit TrackRecords.

    python -m ai.worker --camera-id cam1 --source rtsp://localhost:8554/cam1 --show
    python -m ai.worker --source data/sample_videos/sample.mp4 --jsonl out.jsonl --max-frames 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .sources import open_source
from .viz import draw


def run(source, tracker, *, min_conf: float = 0.3, jsonl_path: str | None = None,
        show: bool = False, save_video: str | None = None, max_frames: int | None = None,
        log_every: int = 30, log=print) -> dict:
    """Core loop. `tracker` is anything with .update(image, ts, idx) -> list[TrackRecord]."""
    import cv2

    out = open(jsonl_path, "w") if jsonl_path else None
    writer = None
    n, n_records, ema_fps, t_prev = 0, 0, None, time.perf_counter()
    t_start = t_prev
    try:
        for fr in source:
            recs = [r for r in tracker.update(fr.image, fr.ts, fr.idx) if r.conf >= min_conf]
            n += 1
            n_records += len(recs)
            if out:
                for r in recs:
                    out.write(json.dumps(r.to_dict()) + "\n")

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            ema_fps = inst if ema_fps is None else 0.9 * ema_fps + 0.1 * inst

            if show or save_video:
                vis = draw(fr.image.copy(), recs, ema_fps)
                if save_video:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                    writer.write(vis)
                if show:
                    cv2.imshow("tracking (press q to quit)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if log_every and n % log_every == 0:
                log(f"frames={n} fps={ema_fps:.1f} tracks_in_frame={len(recs)}")
            if max_frames and n >= max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if out:
            out.close()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        if hasattr(source, "close"):
            source.close()
    elapsed = time.perf_counter() - t_start
    return {"frames": n, "records": n_records, "seconds": round(elapsed, 2),
            "avg_fps": round(n / elapsed, 2) if elapsed else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-id", default="cam1")
    ap.add_argument("--source", required=True, help="rtsp://... URL or a video file path")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    ap.add_argument("--min-conf", type=float, default=0.3, help="drop output tracks below this confidence")
    ap.add_argument("--jsonl", help="write one JSON line per track record to this file")
    ap.add_argument("--show", action="store_true", help="live preview window (needs opencv-python, not headless)")
    ap.add_argument("--save-video", help="save the annotated video to this .mp4")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)

    from .yolo_tracker import Tracker  # heavy import (torch) only when actually running

    tracker = Tracker(a.camera_id, weights=a.weights, imgsz=a.imgsz, device=a.device)
    print(f"device={tracker.device} half={tracker.half} imgsz={a.imgsz} source={a.source}")
    stats = run(open_source(a.source), tracker, min_conf=a.min_conf, jsonl_path=a.jsonl,
                show=a.show, save_video=a.save_video, max_frames=a.max_frames)
    print(f"done: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
