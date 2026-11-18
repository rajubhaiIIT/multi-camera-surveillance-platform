"""Lightweight single-camera multi-person tracker for frame-to-frame pose association."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .detector import _box_iou
from .schema import PoseRecord


class SimpleIoUTracker:
    """Lightweight single-camera multi-person tracker for frame-to-frame pose association."""

    def __init__(self, iou_thresh: float = 0.25, max_age: int = 15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.active_tracks: Dict[int, Tuple[tuple, int]] = {}  # tid -> (bbox, age)

    def update(self, poses: List[PoseRecord]) -> List[PoseRecord]:
        if not poses:
            for tid in list(self.active_tracks):
                bbox, age = self.active_tracks[tid]
                if age + 1 > self.max_age:
                    del self.active_tracks[tid]
                else:
                    self.active_tracks[tid] = (bbox, age + 1)
            return []

        assigned_poses = []
        unmatched_poses = []
        used_tids = set()

        for p in poses:
            best_iou = 0.0
            best_tid = None
            for tid, (bbox, _) in self.active_tracks.items():
                if tid in used_tids:
                    continue
                iou = _box_iou(p.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_thresh:
                used_tids.add(best_tid)
                p.track_id = best_tid
                self.active_tracks[best_tid] = (p.bbox, 0)
                assigned_poses.append(p)
            else:
                unmatched_poses.append(p)

        for p in unmatched_poses:
            new_id = self.next_id
            self.next_id += 1
            p.track_id = new_id
            self.active_tracks[new_id] = (p.bbox, 0)
            assigned_poses.append(p)

        for tid in list(self.active_tracks):
            if tid not in used_tids and tid not in [p.track_id for p in unmatched_poses]:
                bbox, age = self.active_tracks[tid]
                if age + 1 > self.max_age:
                    del self.active_tracks[tid]
                else:
                    self.active_tracks[tid] = (bbox, age + 1)

        return assigned_poses
