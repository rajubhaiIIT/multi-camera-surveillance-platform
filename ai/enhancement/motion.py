"""Background subtraction and motion region extraction for surveillance streams."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class BackgroundMotionDetector:
    """MOG2-based background subtraction with shadow removal and morphological denoising.

    Parameters
    ----------
    history : number of frames for background model adaptation (default: 500)
    var_threshold : Mahalanobis distance threshold to classify pixel as foreground (default: 25)
    min_area : minimum contour area in pixels to treat as moving object (default: 400)
    """

    def __init__(self, history: int = 500, var_threshold: float = 25.0,
                 min_area: float = 400.0, detect_shadows: bool = False):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]], float]:
        """Process frame.
        
        Returns:
          fg_mask : binary foreground mask (255 = motion, 0 = background)
          boxes   : list of bounding boxes (x1, y1, x2, y2) around moving regions
          motion_pct : percentage of total frame area containing active motion
        """
        raw_mask = self.subtractor.apply(frame)

        # Threshold mask (values >= 127 are motion)
        _, binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)
        # Morphological opening to remove salt-and-pepper noise
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        dilated = cv2.dilate(denoised, self.kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = frame.shape[:2]
        total_pixels = h * w
        motion_pixels = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                boxes.append((float(x), float(y), float(x + cw), float(y + ch)))
                motion_pixels += area

        motion_pct = round(100.0 * float(motion_pixels) / max(1.0, float(total_pixels)), 2)
        return dilated, boxes, motion_pct

    def draw(self, frame: np.ndarray, boxes: List[Tuple[float, float, float, float]],
             motion_pct: float) -> np.ndarray:
        """Render motion contours and scene activity meter onto the frame."""
        vis = frame.copy()
        for b in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw motion activity badge
        badge_text = f"MOTION ACTIVITY: {motion_pct:.1f}% ({len(boxes)} regions)"
        color = (0, 200, 0) if motion_pct < 5.0 else ((0, 165, 255) if motion_pct < 25.0 else (0, 0, 255))
        cv2.rectangle(vis, (10, 10), (380, 42), (0, 0, 0), -1)
        cv2.putText(vis, badge_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return vis
