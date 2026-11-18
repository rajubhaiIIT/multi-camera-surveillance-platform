"""Weapon and threat object detector wrapper with size validation and temporal confirmation."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _box_area(b: tuple) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _point_in_box(pt: Tuple[float, float], box: tuple, margin: float = 0.15) -> bool:
    """Check if point is inside box with optional fractional margin."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (box[0] - margin * w <= pt[0] <= box[2] + margin * w and
            box[1] - margin * h <= pt[1] <= box[3] + margin * h)


class WeaponDetector:
    """Detect weapons and identify if they are brandished/held by tracked persons.

    Includes physical proportion filtering and temporal persistence confirmation
    to eliminate false positive detections on background architecture, shadows,
    and single-frame transient noise.

    Parameters
    ----------
    weights : YOLO weights file (default: 'yolo11n.pt')
    device : 'cuda:0', 'cpu', or 'auto'
    conf : detection confidence threshold (default: 0.55)
    min_consecutive : required consecutive detection cycles before alerting (default: 2)
    """

    def __init__(self, weights: str = "yolo11n.pt", device: str = "auto",
                 conf: float = 0.55, min_consecutive: int = 2):
        import torch
        from ultralytics import YOLO

        self.conf = conf
        self.min_consecutive = min_consecutive
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(weights)
        self.weights = weights
        # Temporal persistence tracker: track_key -> {"hits": count, "last_frame": frame_idx}
        self._persistence: Dict[str, dict] = {}

    def detect(self, frame: np.ndarray) -> List[WeaponRecord]:
        """Detect weapon candidates in a BGR frame with geometric plausibility filtering."""
        res = self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]
        records = []
        if res.boxes is None or len(res.boxes) == 0:
            return records

        fh, fw = frame.shape[:2]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        names = getattr(self.model, "names", {})
        for i in range(len(boxes)):
            cls_id = int(classes[i])
            cname = names.get(cls_id, "").lower() if isinstance(names, dict) else (names[cls_id].lower() if cls_id < len(names) else "")
            box = tuple(float(v) for v in boxes[i])
            w_w = box[2] - box[0]
            w_h = box[3] - box[1]
            c = float(confs[i])

            # Sanity Filter 1: Size plausibility
            # A real weapon does not span > 35% of the frame height or > 40% of frame width
            # (which would be architectural columns, walls, ceilings, or doorframes)
            if w_h > 0.35 * fh or w_w > 0.40 * fw:
                continue
            # Minimum dimension to filter microscopic noise
            if w_h < 12 or w_w < 12:
                continue

            # Class check
            if cls_id in COCO_WEAPON_CLASSES:
                # COCO classes (knife, bat, scissors) are prone to false positives on tools/phones/poles
                # Require higher threshold for COCO threat objects
                min_coco_conf = 0.65 if cls_id == 43 else 0.70
                if c >= min_coco_conf:
                    wtype = COCO_WEAPON_CLASSES[cls_id]
                    records.append(WeaponRecord(
                        bbox=box,
                        weapon_type=wtype,
                        conf=c,
                    ))
            elif any(k in cname for k in ["gun", "pistol", "handgun", "firearm", "rifle", "revolver", "weapon"]):
                records.append(WeaponRecord(
                    bbox=box,
                    weapon_type=WeaponType.GUN,
                    conf=c,
                ))

        return records

    def associate_with_persons(self, weapons: List[WeaponRecord],
                               person_tracks: list) -> List[WeaponRecord]:
        """Determine if a weapon is held or carried by any tracked person.

        Enforces realistic human-to-weapon physical proportion bounds.
        """
        for w in weapons:
            wc = _box_center(w.bbox)
            w_area = _box_area(w.bbox)
            w_h = w.bbox[3] - w.bbox[1]

            best_dist = float("inf")
            best_track_id = None

            for p in person_tracks:
                pbox = getattr(p, "bbox", None)
                if pbox is None:
                    continue

                p_area = _box_area(pbox)
                p_h = pbox[3] - pbox[1]

                # Sanity: A handheld weapon cannot be larger than the person!
                # Even a rifle is < 70% of person bounding box area
                if w_area > 0.65 * p_area or w_h > 1.1 * p_h:
                    continue

                # Check if weapon center is inside or immediately adjacent to person
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
        """Run weapon detection, temporal confirmation, and generate verified alerts."""
        raw_weapons = self.detect(frame)
        if person_tracks:
            weapons = self.associate_with_persons(raw_weapons, person_tracks)
        else:
            weapons = raw_weapons

        # Clean stale persistence entries older than 25 frames
        active_keys = set()
        alerts = []

        for w in weapons:
            tid = w.held_by_track_id
            is_held = (tid is not None)

            # Unheld object filter:
            # If persons are tracked in the scene, weapons on the floor/wall must have very high
            # confidence (>= 0.75) to prevent background textures from alarming.
            if person_tracks and not is_held:
                if w.conf < 0.75:
                    continue

            # Temporal Confirmation:
            # Generate a spatial/track key to track persistence across consecutive checks
            wc = _box_center(w.bbox)
            grid_x = int(wc[0] // 60)
            grid_y = int(wc[1] // 60)
            track_ident = f"t{tid}" if tid is not None else f"pos_{grid_x}_{grid_y}"
            pkey = f"{camera_id}_{track_ident}_{w.weapon_type.value}"
            active_keys.add(pkey)

            entry = self._persistence.get(pkey)
            if entry and (frame_idx - entry["last_frame"] <= 20):
                entry["hits"] += 1
                entry["last_frame"] = frame_idx
            else:
                entry = {"hits": 1, "last_frame": frame_idx}
                self._persistence[pkey] = entry

            # Require at least `min_consecutive` hits before firing alert
            if entry["hits"] < self.min_consecutive:
                continue

            gid = None
            if tid is not None and person_tracks:
                matched_p = next((p for p in person_tracks if getattr(p, "track_id", None) == tid), None)
                if matched_p:
                    gid = getattr(matched_p, "global_id", None)

            detail = f"Threat detected: {w.weapon_type.value.upper()} ({w.conf:.2f})"
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

        # Evict inactive persistence records
        self._persistence = {
            k: v for k, v in self._persistence.items()
            if frame_idx - v["last_frame"] <= 30
        }

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
