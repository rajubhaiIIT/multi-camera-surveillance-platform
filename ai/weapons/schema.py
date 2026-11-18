"""Data schemas for weapon and threat object detection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class WeaponType(str, Enum):
    KNIFE = "knife"
    GUN = "gun"
    BAT = "baseball_bat"
    SCISSORS = "scissors"
    OTHER = "weapon"


@dataclass
class WeaponRecord:
    """One detected weapon object in a frame."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    weapon_type: WeaponType
    conf: float
    held_by_track_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "bbox": [round(float(v), 1) for v in self.bbox],
            "weapon_type": self.weapon_type.value,
            "conf": round(float(self.conf), 4),
            "held_by_track_id": self.held_by_track_id,
        }


@dataclass
class WeaponAlert:
    """Alert triggered when a weapon is detected or held by an individual."""
    camera_id: str
    frame_idx: int
    track_id: Optional[int]
    global_id: Optional[int]
    weapon_type: WeaponType
    confidence: float
    weapon_bbox: Tuple[float, float, float, float]
    timestamp: float
    is_held: bool
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "event": "weapon_alert",
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "global_id": self.global_id,
            "weapon_type": self.weapon_type.value,
            "confidence": round(float(self.confidence), 4),
            "weapon_bbox": [round(float(v), 1) for v in self.weapon_bbox],
            "timestamp": round(self.timestamp, 3),
            "is_held": self.is_held,
            "details": self.details,
        }
