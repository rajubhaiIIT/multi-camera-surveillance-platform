"""Unified Phase 4 Pose, Fall, and Violence processing pipeline."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import PoseDetector
from .fall import FallDetector
from .schema import FallAlert, FallState, FightAlert, PoseRecord
from .violence import ViolenceDetector


class PosePipeline:
    """Unified pipeline for pose estimation, fall detection, and violent interaction detection.

    Parameters
    ----------
    detector : PoseDetector
    fall_detector : Optional[FallDetector]
    violence_detector : Optional[ViolenceDetector]
    run_every : run pose estimation every N frames per camera to reduce GPU load (default: 1)
    """

    def __init__(self,
                 detector: Optional[PoseDetector] = None,
                 fall_detector: Optional[FallDetector] = None,
                 violence_detector: Optional[ViolenceDetector] = None,
                 run_every: int = 1):
        self.detector = detector or PoseDetector()
        self.fall_detector = fall_detector or FallDetector()
        self.violence_detector = violence_detector or ViolenceDetector()
        self.run_every = max(1, int(run_every))
        self._frame_count = 0

    def process(self, frame: np.ndarray, camera_id: str, frame_idx: int,
                timestamp: float, tracks: Optional[list] = None) -> Tuple[List[PoseRecord], List[FallAlert], List[FightAlert]]:
        """Process one camera frame.

        Parameters
        ----------
        frame : BGR image
        camera_id : camera identifier
        frame_idx : current frame number
        timestamp : seconds since video start or epoch
        tracks : optional list of TrackRecord objects from the person tracker

        Returns
        -------
        poses : list of PoseRecord
        fall_alerts : list of FallAlert emitted in this frame
        fight_alerts : list of FightAlert emitted in this frame
        """
        self._frame_count += 1
        if (self._frame_count - 1) % self.run_every != 0:
            return [], [], []

        # 1. Pose detection
        poses = self.detector.detect(frame, camera_id=camera_id, frame_idx=frame_idx)

        # 2. Association with tracking boxes
        if tracks is not None and len(tracks) > 0:
            person_tracks = [t for t in tracks if getattr(t, "cls", "person") == "person"]
            poses = self.detector.match_to_tracks(poses, person_tracks)

        # 3. Fall detection per pose
        fall_alerts = []
        for p in poses:
            fa = self.fall_detector.update(p, timestamp=timestamp)
            if fa is not None:
                fall_alerts.append(fa)

        # 4. Violence / fight detection across poses
        fight_alerts = self.violence_detector.update(poses, timestamp=timestamp)

        return poses, fall_alerts, fight_alerts

    def draw(self, frame: np.ndarray, poses: List[PoseRecord],
             fall_alerts: List[FallAlert], fight_alerts: List[FightAlert]) -> np.ndarray:
        """Render skeletons, bounding boxes, and alert banners onto the frame."""
        vis = self.detector.draw(frame, poses)
        h, w = vis.shape[:2]

        # Draw fall alert banner
        y_offset = 35
        for fa in fall_alerts:
            text = f"ALERT: FALL DETECTED! Track #{fa.track_id} (conf={fa.confidence:.2f})"
            cv2.rectangle(vis, (0, y_offset - 25), (w, y_offset + 10), (0, 0, 200), -1)
            cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 45

        # Draw fight alert banner
        for fta in fight_alerts:
            t1, t2 = fta.track_ids
            text = f"ALERT: VIOLENT STRUGGLE! Tracks #{t1} & #{t2} (conf={fta.confidence:.2f})"
            cv2.rectangle(vis, (0, y_offset - 25), (w, y_offset + 10), (0, 0, 220), -1)
            cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 45

        return vis
