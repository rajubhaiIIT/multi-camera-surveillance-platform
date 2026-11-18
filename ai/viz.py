"""Draw boxes, track IDs and an FPS counter on a frame."""
from __future__ import annotations

import cv2


def color_for(track_id: int) -> tuple:
    """Stable, distinct-looking BGR colour per track id."""
    h = (track_id * 47) % 180
    hsv = cv2.cvtColor(__import__("numpy").uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def draw(frame, records, fps: float | None = None):
    for r in records:
        x1, y1, x2, y2 = (int(v) for v in r.bbox)
        ident = r.global_id if r.global_id is not None else r.track_id
        c = color_for(ident + (1000 if r.global_id is not None else 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"P{r.global_id}" if r.global_id is not None else f"{r.cls} #{r.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), c, -1)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"{fps:.1f} FPS | {len(records)} tracks", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame
