"""Data contracts and schemas for vehicle tracking and ANPR."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class VehicleType(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    BICYCLE = "bicycle"
    OTHER = "vehicle"


COCO_VEHICLE_MAP = {
    1: VehicleType.BICYCLE,
    2: VehicleType.CAR,
    3: VehicleType.MOTORCYCLE,
    5: VehicleType.BUS,
    7: VehicleType.TRUCK,
}


@dataclass
class PlateRecord:
    """Recognized license plate information."""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # relative or frame coordinates
    country_format: str = "GENERIC"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "country_format": self.country_format,
        }


@dataclass
class VehicleRecord:
    """One tracked vehicle in one frame."""
    camera_id: str
    frame_idx: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    vehicle_type: VehicleType
    conf: float
    plate: Optional[PlateRecord] = None
    speed_kmh: Optional[float] = None
    stationary_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "vehicle_type": self.vehicle_type.value,
            "conf": round(float(self.conf), 4),
            "plate": self.plate.to_dict() if self.plate else None,
            "speed_kmh": round(float(self.speed_kmh), 1) if self.speed_kmh is not None else None,
            "stationary_seconds": round(float(self.stationary_seconds), 1),
        }


@dataclass
class VehicleAlert:
    """Alert for stolen plate match, parking violation, or anomalous driving."""
    camera_id: str
    frame_idx: int
    track_id: int
    alert_type: str  # "watchlist_match", "parking_violation", "speeding"
    plate_text: Optional[str]
    confidence: float
    timestamp: float
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "event": "vehicle_alert",
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "alert_type": self.alert_type,
            "plate_text": self.plate_text,
            "confidence": round(float(self.confidence), 4),
            "timestamp": round(self.timestamp, 3),
            "details": self.details,
        }
