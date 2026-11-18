"""Stage 1 (slow, uses the GPU): run detection + tracking + Re-ID on each camera video and save everything."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2

from .multicam import FrameEmbedder
from .runcache import RunWriter
from .sources import MediaReader

CROP_SLOTS = (1, 4, 8, 12)      # which embedding numbers also save a picture of the person (for labeling)


def extract_run(videos: dict, offsets: dict, fe: FrameEmbedder, out_dir, stride: int = 2,
                max_seconds: float | None = None, max_frames: int | None = None, folder_fps: float = 2.0,
                meta_extra: dict | None = None, log=print) -> Path:
    """videos {camera: video file | image folder | list of image paths}, offsets {camera: seconds >= 0}
    (from ai.sync). Every `stride`-th frame is processed."""
    out = Path(out_dir)
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "stride": stride, "cameras": {},
            **(meta_extra or {})}
    writer = RunWriter(out, meta)

    def sink(cam, tid, n, crop):
        if n in CROP_SLOTS:
            d = out / "crops" / cam
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{tid}_{n}.jpg"), crop)
    fe.crop_sink = sink

    for cam, media in videos.items():
        rd = MediaReader(media, stride, folder_fps)
        off = float(offsets.get(cam, 0.0))
        if off < 0:
            raise ValueError(f"offset for {cam} must be >= 0 (got {off}); use `python -m ai.sync`")
        last, t0, n_done = -1, time.time(), 0
        for idx, img in rd.frames():
            if (max_seconds and idx / rd.fps > max_seconds) or (max_frames and idx >= max_frames):
                break
            ts = idx / rd.fps + off
            recs, embs = fe.run(cam, img, ts, idx)
            writer.add(cam, ts, idx, recs, embs)
            last, n_done = idx, n_done + 1
            if log and n_done % 100 == 0:
                log(f"{cam}: {n_done} frames, {n_done / (time.time() - t0):.1f} fps")
        meta["cameras"][cam] = {"video": rd.source, "offset": off, "fps": rd.fps, "stride": stride,
                                "frames_total": rd.total, "last_idx": last, "frames_processed": n_done}
        if log:
            log(f"{cam}: done, {n_done} frames processed")
    writer.close()
    return out
