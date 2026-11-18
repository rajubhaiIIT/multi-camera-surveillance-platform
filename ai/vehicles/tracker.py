"""Multi-vehicle detection and tracking engine."""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .schema import COCO_VEHICLE_MAP, VehicleAlert, VehicleRecord, VehicleType


def _box_iou(box1: tuple, box2: tuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / (area1 + area2 - inter)


def _box_center(b: tuple) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


class VehicleTracker:
    """Detect and track vehicles across video frames.

    Parameters
    ----------
    weights : YOLO weights (default: 'yolo11n.pt')
    device : 'cuda:0', 'cpu', or 'auto'
    conf : detection threshold
    parking_limit_seconds : stationary duration threshold to emit illegal parking alert (default: 30.0)
    """

    def __init__(self, weights: str = "yolo11n.pt", device: str = "auto",
                 conf: float = 0.25, parking_limit_seconds: float = 30.0):
        import torch
        from ultralytics import YOLO

        self.conf = conf
        self.parking_limit_seconds = parking_limit_seconds
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(weights)
        self.weights = weights

        self._next_track_id = 1
        # Active tracks: track_id -> dict(bbox, last_seen_frame, vehicle_type, conf, history)
        self._active_tracks: Dict[int, dict] = {}
        # Stationary timer: track_id -> (start_ts, initial_center)
        self._stationary_info: Dict[int, dict] = {}

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detect vehicles in frame."""
        res = self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]
        detections = []
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            cls_id = int(classes[i])
            if cls_id in COCO_VEHICLE_MAP:
                vtype = COCO_VEHICLE_MAP[cls_id]
                detections.append({
                    "bbox": tuple(float(v) for v in boxes[i]),
                    "conf": float(confs[i]),
                    "vehicle_type": vtype,
                })
        return detections

    def update(self, frame: np.ndarray, camera_id: str, frame_idx: int,
               timestamp: float) -> Tuple[List[VehicleRecord], List[VehicleAlert]]:
        """Track vehicles across frames and check for parking/loitering alerts."""
        dets = self.detect(frame)
        records = []
        alerts = []

        # IoU tracking association
        matched_track_ids = set()
        unmatched_dets = []

        for det in dets:
            best_iou = 0.0
            best_tid = None
            for tid, tinfo in self._active_tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = _box_iou(det["bbox"], tinfo["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= 0.3:
                # Match to existing track
                matched_track_ids.add(best_tid)
                self._active_tracks[best_tid]["bbox"] = det["bbox"]
                self._active_tracks[best_tid]["conf"] = det["conf"]
                self._active_tracks[best_tid]["last_seen_frame"] = frame_idx
                self._active_tracks[best_tid]["history"].append((timestamp, det["bbox"]))
                tid_to_use = best_tid
            else:
                unmatched_dets.append(det)

        # Allocate new IDs for unmatched detections
        for det in unmatched_dets:
            new_id = self._next_track_id
            self._next_track_id += 1
            self._active_tracks[new_id] = {
                "bbox": det["bbox"],
                "conf": det["conf"],
                "vehicle_type": det["vehicle_type"],
                "last_seen_frame": frame_idx,
                "history": deque([(timestamp, det["bbox"])], maxlen=60),
            }
            matched_track_ids.add(new_id)

        # Prune dead tracks (not seen for > 30 frames)
        for tid in list(self._active_tracks):
            if frame_idx - self._active_tracks[tid]["last_seen_frame"] > 30:
                self._active_tracks.pop(tid, None)
                self._stationary_info.pop(tid, None)

        # Analyze speed and stationary duration for active tracks in this frame
        for tid in matched_track_ids:
            tinfo = self._active_tracks[tid]
            hist = tinfo["history"]
            c_curr = _box_center(tinfo["bbox"])

            # Stationary tracking
            if tid not in self._stationary_info:
                self._stationary_info[tid] = {
                    "start_ts": timestamp,
                    "anchor_center": c_curr,
                    "alert_emitted": False,
                }
            sinfo = self._stationary_info[tid]
            dist_from_anchor = math.hypot(c_curr[0] - sinfo["anchor_center"][0],
                                          c_curr[1] - sinfo["anchor_center"][1])

            if dist_from_anchor > 30.0:  # vehicle has moved significantly
                sinfo["start_ts"] = timestamp
                sinfo["anchor_center"] = c_curr
                sinfo["alert_emitted"] = False
                stationary_sec = 0.0
            else:
                stationary_sec = timestamp - sinfo["start_ts"]

            # Emit parking violation alert if stationary > limit
            if stationary_sec >= self.parking_limit_seconds and not sinfo["alert_emitted"]:
                sinfo["alert_emitted"] = True
                alerts.append(VehicleAlert(
                    camera_id=camera_id,
                    frame_idx=frame_idx,
                    track_id=tid,
                    alert_type="parking_violation",
                    plate_text=None,
                    confidence=0.85,
                    timestamp=timestamp,
                    details=f"Parking/Loitering alert: Vehicle #{tid} ({tinfo['vehicle_type'].value}) stationary for {stationary_sec:.1f}s",
                ))

            # Approximate speed (pixel distance per second)
            speed = 0.0
            if len(hist) >= 2:
                past_ts, past_box = hist[0]
                dt = timestamp - past_ts
                if dt >= 0.2:
                    past_c = _box_center(past_box)
                    px_dist = math.hypot(c_curr[0] - past_c[0], c_curr[1] - past_c[1])
                    speed = (px_dist / dt) * 0.1  # rough px to km/h conversion scale

            records.append(VehicleRecord(
                camera_id=camera_id,
                frame_idx=frame_idx,
                track_id=tid,
                bbox=tinfo["bbox"],
                vehicle_type=tinfo["vehicle_type"],
                conf=tinfo["conf"],
                speed_kmh=speed,
                stationary_seconds=stationary_sec,
            ))

        return records, alerts
