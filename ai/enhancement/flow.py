"""Dense optical flow computation and velocity field visualization."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class DenseOpticalFlow:
    """Compute Gunnar Farneback dense optical flow vectors and motion heatmaps.

    Parameters
    ----------
    pyr_scale : image scale (<1) to build pyramids for each frame (default: 0.5)
    levels : number of pyramid layers (default: 3)
    winsize : averaging window size (default: 15)
    iterations : number of iterations at each pyramid level (default: 3)
    """

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15,
                 iterations: int = 3):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.prev_gray: np.ndarray | None = None

    def compute(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Compute optical flow from previous frame.
        
        Returns:
          flow : (H, W, 2) array of (dx, dy) motion vectors
          mean_magnitude : average motion speed across pixels
          max_magnitude  : maximum motion speed detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros((h, w, 2), dtype=np.float32), 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        self.prev_gray = gray

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        max_mag = float(np.max(mag))
        return flow, mean_mag, max_mag

    def visualize_flow_hsv(self, flow: np.ndarray) -> np.ndarray:
        """Render dense optical flow as standard HSV color field.
        
        Hue represents motion direction; Value represents velocity.
        """
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255  # full saturation

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Angle in degrees / 2 -> 0..180 for OpenCV Hue
        hsv[..., 0] = ang * 180 / np.pi / 2
        # Normalize magnitude to 0..255 Value
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def draw_flow_arrows(self, frame: np.ndarray, flow: np.ndarray,
                         step: int = 16, min_mag: float = 1.0) -> np.ndarray:
        """Overlay motion vector arrow grid onto the original video frame."""
        vis = frame.copy()
        h, w = frame.shape[:2]
        y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        for (x1, y1), (x2, y2) in lines:
            if np.hypot(x2 - x1, y2 - y1) >= min_mag:
                cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 255), 1,
                                tipLength=0.3, line_type=cv2.LINE_AA)
                cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
        return vis
