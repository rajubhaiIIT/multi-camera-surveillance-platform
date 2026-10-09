"""Re-run identity matching on a saved run (fast, no GPU), and draw the result on the videos."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

from .crosscam import GlobalMatcher, MatcherConfig, Topology
from .multicam import MultiCamPipeline
from .runcache import Run
from .sources import MediaReader
from .tracklets import TrackletBook
from .viz import draw


def replay(run: Run, cfg: MatcherConfig | None = None, topology: Topology | None = None,
           no_reid: bool = False, stale_after_s: float = 3.0, person_class: str = "person") -> list:
    """Feed every saved camera frame, in time order, through the matcher.

    no_reid=True replaces every embedding by the same vector (appearance carries no information), which shows
    how much the Re-ID model adds on top of timing/layout rules alone.
    Returns all records; person records get their tracklet's FINAL global_id back-filled (offline evaluation).
    """
    matcher = GlobalMatcher(cfg or MatcherConfig(), topology or Topology())
    pipe = MultiCamPipeline(None, None, matcher, book=TrackletBook(stale_after_s=stale_after_s), person_class=person_class)
    out = []
    for ev in sorted(run.events, key=lambda e: (e.ts, e.cam)):
        embs = ev.embs
        if no_reid and embs:
            d = len(next(iter(embs.values())))
            const = np.ones(d, np.float32) / np.sqrt(d)
            embs = {t: const for t in embs}
        out.extend(pipe.ingest(ev.cam, ev.ts, ev.recs, embs))
    pipe.flush()
    gid = matcher.bound
    return [dataclasses.replace(r, global_id=gid.get((r.camera_id, r.track_id))) if r.cls == person_class else r
            for r in out]


def render_run(run: Run, records: list, out_dir: Path | None = None) -> list[Path]:
    """Write one annotated video per camera (boxes labelled P<global id>). Videos must still be at their original path."""
    out_dir = Path(out_dir or run.dir / "render")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict = {}
    for r in records:
        by_frame.setdefault((r.camera_id, r.frame_idx), []).append(r)
    paths = []
    for cam, info in run.meta["cameras"].items():
        try:
            rd = MediaReader(info["video"], int(info["stride"]), float(info["fps"]))
        except IOError as e:
            print(f"skip {cam}: {e}")
            continue
        writer, p = None, None
        for idx, img in rd.frames():
            if idx > info["last_idx"]:
                break
            vis = draw(img, by_frame.get((cam, idx), []))
            cv2.putText(vis, f"{cam}  t={idx / rd.fps + info['offset']:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                h, w = vis.shape[:2]
                p = out_dir / f"{cam}.mp4"
                writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), max(rd.fps / rd.stride, 1.0), (w, h))
            writer.write(vis)
        if writer:
            writer.release()
            paths.append(p)
    return paths
