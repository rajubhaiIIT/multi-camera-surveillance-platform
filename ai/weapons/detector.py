"""Weapon and threat object detector wrapper."""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .schema import WeaponAlert, WeaponRecord, WeaponType

# COCO 80 classes mapping for threat/weapon objects:
COCO_WEAPON_CLASSES = {
    43: WeaponType.KNIFE,
    34: WeaponType.BAT,
    76: WeaponType.SCISSORS,
}


def _box_center(b: tuple) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _point_in_box(pt: Tuple[float, float], box: tuple, margin: float = 0.1) -> bool:
    """Check if point is inside box with optional fractional margin."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (box[0] - margin * w <= pt[0] <= box[2] + margin * w and
            box[1] - margin * h <= pt[1] <= box[3] + margin * h)


class WeaponDetector:
    """Detect weapons and identify if they are brandished/held by tracked persons.

    Parameters
    ----------
    weights : YOLO weights file (default: 'yolo11n.pt')
    device : 'cuda:0', 'cpu', or 'auto'
    conf : detection confidence threshold
    """

    def __init__(self, weights: str = "yolo11n.pt", device: str = "auto", conf: float = 0.25):
        import torch
        from ultralytics import YOLO

        self.conf = conf
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(weights)
        self.weights = weights

    def detect(self, frame: np.ndarray) -> List[WeaponRecord]:
        """Detect weapon candidates in a BGR frame."""
        res = self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]
        records = []
        if res.boxes is None or len(res.boxes) == 0:
            return records

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        names = getattr(self.model, "names", {})
        for i in range(len(boxes)):
            cls_id = int(classes[i])
            cname = names.get(cls_id, "").lower() if isinstance(names, dict) else (names[cls_id].lower() if cls_id < len(names) else "")

            if cls_id in COCO_WEAPON_CLASSES:
                wtype = COCO_WEAPON_CLASSES[cls_id]
                box = tuple(float(v) for v in boxes[i])
                records.append(WeaponRecord(
                    bbox=box,
                    weapon_type=wtype,
                    conf=float(confs[i]),
                ))
            elif any(k in cname for k in ["gun", "pistol", "handgun", "firearm", "rifle", "revolver", "weapon"]):
                box = tuple(float(v) for v in boxes[i])
                records.append(WeaponRecord(
                    bbox=box,
                    weapon_type=WeaponType.GUN,
                    conf=float(confs[i]),
                ))
        return records

    def associate_with_persons(self, weapons: List[WeaponRecord],
                               person_tracks: list) -> List[WeaponRecord]:
        """Determine if a weapon is held or carried by any tracked person."""
        for w in weapons:
            wc = _box_center(w.bbox)
            best_dist = float("inf")
            best_track_id = None
            for p in person_tracks:
                pbox = getattr(p, "bbox", None)
                if pbox is None:
                    continue
                # If weapon center is inside or near the person bounding box
                if _point_in_box(wc, pbox, margin=0.15):
                    pc = _box_center(pbox)
                    d = math.hypot(wc[0] - pc[0], wc[1] - pc[1])
                    if d < best_dist:
                        best_dist = d
                        best_track_id = getattr(p, "track_id", None)
            w.held_by_track_id = best_track_id
        return weapons

    def process(self, frame: np.ndarray, camera_id: str, frame_idx: int,
                timestamp: float, person_tracks: Optional[list] = None) -> Tuple[List[WeaponRecord], List[WeaponAlert]]:
        """Run weapon detection and generate alerts."""
        weapons = self.detect(frame)
        if person_tracks:
            weapons = self.associate_with_persons(weapons, person_tracks)

        alerts = []
        for w in weapons:
            tid = w.held_by_track_id
            gid = None
            if tid is not None and person_tracks:
                matched_p = next((p for p in person_tracks if getattr(p, "track_id", None) == tid), None)
                if matched_p:
                    gid = getattr(matched_p, "global_id", None)

            is_held = (tid is not None)
            detail = f"Threat detected: {w.weapon_type.value} ({w.conf:.2f})"
            if is_held:
                detail += f" held by Track #{tid}"

            alerts.append(WeaponAlert(
                camera_id=camera_id,
                frame_idx=frame_idx,
                track_id=tid,
                global_id=gid,
                weapon_type=w.weapon_type,
                confidence=w.conf,
                weapon_bbox=w.bbox,
                timestamp=timestamp,
                is_held=is_held,
                details=detail,
            ))
        return weapons, alerts

    def draw(self, frame: np.ndarray, weapons: List[WeaponRecord],
             alerts: Optional[List[WeaponAlert]] = None) -> np.ndarray:
        """Render weapon bounding boxes and warning banners onto the frame."""
        vis = frame.copy()
        for w in weapons:
            x1, y1, x2, y2 = (int(round(v)) for v in w.bbox)
            color = (0, 0, 255) # Bright red
            label = f"WEAPON: {w.weapon_type.value} ({w.conf:.2f})"
            if w.held_by_track_id is not None:
                label += f" | #{w.held_by_track_id}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if alerts:
            h, w_img = vis.shape[:2]
            y_offset = 25
            for a in alerts:
                banner_text = f"CRITICAL: {a.details.upper()}"
                cv2.rectangle(vis, (0, y_offset - 20), (w_img, y_offset + 10), (0, 0, 220), -1)
                cv2.putText(vis, banner_text, (15, y_offset + 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 40
        return vis
