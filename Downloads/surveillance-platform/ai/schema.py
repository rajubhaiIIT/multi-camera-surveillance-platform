"""The output contract every downstream module (Re-ID, storage, API) relies on.

One TrackRecord = one tracked object in one frame of one camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackRecord:
    camera_id: str
    frame_ts: float      # seconds since epoch (live) or since stream start (files)
    frame_idx: int
    track_id: int        # unique within one camera only; cross-camera IDs come in Phase 2
    bbox: tuple          # (x1, y1, x2, y2) in pixels
    cls: str             # class name, e.g. "person", "car"
    conf: float          # detector confidence, 0..1

    def __post_init__(self):
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        if not (isinstance(self.track_id, int) and self.track_id >= 0):
            raise ValueError("track_id must be a non-negative int")
        if not (isinstance(self.frame_idx, int) and self.frame_idx >= 0):
            raise ValueError("frame_idx must be a non-negative int")
        if not (math.isfinite(self.frame_ts) and self.frame_ts >= 0):
            raise ValueError("frame_ts must be a finite, non-negative number")
        if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox):
            raise ValueError("bbox must be 4 finite numbers")
        x1, y1, x2, y2 = self.bbox
        if not (x2 > x1 and y2 > y1):
            raise ValueError("bbox must have x2 > x1 and y2 > y1")
        if not (isinstance(self.cls, str) and self.cls):
            raise ValueError("cls must be a non-empty string")
        if not (0.0 <= self.conf <= 1.0):
            raise ValueError("conf must be within 0..1")

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_ts": round(self.frame_ts, 4),
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "class": self.cls,
            "conf": round(float(self.conf), 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrackRecord":
        return cls(d["camera_id"], d["frame_ts"], d["frame_idx"], d["track_id"],
                   tuple(d["bbox"]), d["class"], d["conf"])
