"""Unified Vehicle Tracking, ANPR and Watchlist Pipeline."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .anpr import LicensePlateReader
from .index import PlateIndex
from .schema import VehicleAlert, VehicleRecord
from .tracker import VehicleTracker


class VehicleANPRPipeline:
    """Unified pipeline connecting vehicle tracking, license plate reading, and hotlist alerts.

    Parameters
    ----------
    tracker : VehicleTracker
    plate_reader : Optional[LicensePlateReader]
    watchlist : Optional[PlateIndex]
    plate_read_interval : run OCR every N frames per vehicle to save GPU time (default: 5)
    """

    def __init__(self,
                 tracker: Optional[VehicleTracker] = None,
                 plate_reader: Optional[LicensePlateReader] = None,
                 watchlist: Optional[PlateIndex] = None,
                 plate_read_interval: int = 5):
        self.tracker = tracker or VehicleTracker()
        self.plate_reader = plate_reader or LicensePlateReader()
        self.watchlist = watchlist or PlateIndex()
        self.plate_read_interval = plate_read_interval

        # Cache plate per track: track_id -> PlateRecord
        self._cached_plates: Dict[int, dict] = {}
        # Track frames processed per vehicle
        self._track_frames: Dict[int, int] = {}

    def process(self, frame: np.ndarray, camera_id: str, frame_idx: int,
                timestamp: float) -> Tuple[List[VehicleRecord], List[VehicleAlert]]:
        """Process one camera frame for vehicles, plates, and watchlist alerts."""
        vehicles, alerts = self.tracker.update(frame, camera_id, frame_idx, timestamp)

        for v in vehicles:
            tid = v.track_id
            self._track_frames[tid] = self._track_frames.get(tid, 0) + 1

            # Check if we should attempt plate read
            should_read = (tid not in self._cached_plates or
                           (self._track_frames[tid] % self.plate_read_interval == 0))

            if should_read and self.plate_reader:
                plate = self.plate_reader.process_vehicle(frame, v.bbox)
                if plate:
                    self._cached_plates[tid] = plate

            if tid in self._cached_plates:
                v.plate = self._cached_plates[tid]

                # Check watchlist match
                if self.watchlist and self.watchlist.size > 0:
                    entry = self.watchlist.match(v.plate.text)
                    if entry:
                        alerts.append(VehicleAlert(
                            camera_id=camera_id,
                            frame_idx=frame_idx,
                            track_id=tid,
                            alert_type="watchlist_match",
                            plate_text=v.plate.text,
                            confidence=v.plate.confidence,
                            timestamp=timestamp,
                            details=f"WATCHLIST MATCH: Plate {entry.plate} ({entry.reason})",
                        ))

        return vehicles, alerts

    def draw(self, frame: np.ndarray, vehicles: List[VehicleRecord],
             alerts: Optional[List[VehicleAlert]] = None) -> np.ndarray:
        """Render vehicle bounding boxes, license plate tags, and alert banners."""
        vis = frame.copy()
        for v in vehicles:
            x1, y1, x2, y2 = (int(round(coord)) for coord in v.bbox)
            color = (255, 191, 0) # Deep sky blue / amber for vehicles

            label = f"#{v.track_id} {v.vehicle_type.value}"
            if v.plate:
                label += f" | [{v.plate.text}]"
                # Draw plate bounding box if available
                px1, py1, px2, py2 = (int(round(c)) for c in v.plate.bbox)
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 255), 2)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if alerts:
            h, w = vis.shape[:2]
            y_offset = 30
            for a in alerts:
                bg_color = (0, 0, 220) if a.alert_type == "watchlist_match" else (0, 140, 255)
                banner_text = f"ALERT: {a.details.upper()}"
                cv2.rectangle(vis, (0, y_offset - 22), (w, y_offset + 8), bg_color, -1)
                cv2.putText(vis, banner_text, (15, y_offset),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 38

        return vis
