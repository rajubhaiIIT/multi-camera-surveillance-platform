"""Read/write the MOTChallenge text format: frame,id,x,y,w,h,conf,-1,-1,-1 (frames are 1-based)."""
from __future__ import annotations

from pathlib import Path


def write_mot(path: Path, rows) -> None:
    """rows: iterable of (frame, track_id, x1, y1, x2, y2, conf)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, tid, x1, y1, x2, y2, conf in rows:
            f.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1,-1\n")


def read_mot(path: Path) -> list[tuple]:
    """Return rows as (frame, track_id, x1, y1, x2, y2, conf)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        frame, tid = int(p[0]), int(p[1])
        x, y, w, h, conf = (float(v) for v in p[2:7])
        rows.append((frame, tid, x, y, x + w, y + h, conf))
    return rows
