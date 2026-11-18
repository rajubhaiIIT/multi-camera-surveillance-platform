"""Per-frame face detection and matching integrated with the tracking pipeline.

For each tracked person, it finds their face, extracts an embedding, and queries the
enrollment index. The result is attached to the TrackRecord as face_id and face_name.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict

import numpy as np

from ..schema import TrackRecord


@dataclasses.dataclass
class FaceHit:
    track_id: int
    camera_id: str
    frame_idx: int
    person_id: str
    name: str
    similarity: float
    face_bbox: tuple
    det_score: float


class FacePipeline:
    """Run face detection on person crops and match against the enrollment index.

    Parameters
    ----------
    detector : FaceDetector
    index : FaceIndex
    run_every : run face detection every N frames per track (saves GPU time)
    min_hits : require this many successful matches before considering a track identified
    """

    def __init__(self, detector, index, run_every: int = 10, min_hits: int = 2):
        self.detector = detector
        self.index = index
        self.run_every = run_every
        self.min_hits = min_hits
        self._frame_count: dict = defaultdict(int)
        self._hits: dict = defaultdict(list)           # track key -> list of MatchResult

    def process(self, frame, recs: list[TrackRecord]) -> list[FaceHit]:
        """Process one frame. Returns any face identifications made."""
        hits = []
        h, w = frame.shape[:2]
        for r in recs:
            if r.cls != "person":
                continue
            key = (r.camera_id, r.track_id)
            self._frame_count[key] += 1
            if (self._frame_count[key] - 1) % self.run_every != 0:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in r.bbox)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if y2 <= y1 or x2 <= x1:
                continue
            crop = frame[y1:y2, x1:x2]
            faces = self.detector.detect(crop)
            if not faces:
                continue
            best_face = max(faces, key=lambda f: f.det_score)
            match = self.index.identify(best_face.embedding)
            if match is None:
                continue
            self._hits[key].append(match)
            if len(self._hits[key]) >= self.min_hits:
                # take the majority vote over collected hits
                from collections import Counter
                pid = Counter(m.person_id for m in self._hits[key]).most_common(1)[0][0]
                best = next(m for m in reversed(self._hits[key]) if m.person_id == pid)
                # translate face bbox from crop coordinates to full-frame coordinates
                face_bb = (x1 + best_face.bbox[0], y1 + best_face.bbox[1],
                           x1 + best_face.bbox[2], y1 + best_face.bbox[3])
                hits.append(FaceHit(r.track_id, r.camera_id, r.frame_idx,
                                    pid, best.name, best.similarity,
                                    face_bb, best_face.det_score))
        return hits

    def reset_track(self, camera_id: str, track_id: int):
        key = (camera_id, track_id)
        self._frame_count.pop(key, None)
        self._hits.pop(key, None)
