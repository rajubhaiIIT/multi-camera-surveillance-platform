"""Surveillance camera stabilization to cancel pole vibration, wind shake, and sensor jitter."""
from __future__ import annotations

import cv2
import numpy as np


class VideoStabilizer:
    """Real-time and batch video stabilizer using Lucas-Kanade optical flow and trajectory smoothing.

    Parameters
    ----------
    smoothing_radius : window size for trajectory smoothing (default: 15 frames)
    max_corners : maximum feature points to track per frame (default: 200)
    """

    def __init__(self, smoothing_radius: int = 15, max_corners: int = 200):
        self.smoothing_radius = max(1, smoothing_radius)
        self.max_corners = max_corners
        self.prev_gray: np.ndarray | None = None
        self.transforms: list[np.ndarray] = []  # [dx, dy, da]
        self.trajectory: list[np.ndarray] = []  # cumulative [x, y, a]

    def estimate_motion(self, prev_img: np.ndarray, curr_img: np.ndarray) -> np.ndarray:
        """Estimate 2D rigid transform [dx, dy, d_angle] between two frames."""
        prev_pts = cv2.goodFeaturesToTrack(prev_img, maxCorners=self.max_corners,
                                          qualityLevel=0.01, minDistance=30, blockSize=3)
        if prev_pts is None or len(prev_pts) < 4:
            return np.zeros(3, dtype=np.float32)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None)
        valid = (status.flatten() == 1)
        prev_valid = prev_pts[valid]
        curr_valid = curr_pts[valid]

        if len(prev_valid) < 4:
            return np.zeros(3, dtype=np.float32)

        # Estimate partial affine (rigid: translation + rotation + uniform scale)
        matrix, inliers = cv2.estimateAffinePartial2D(prev_valid, curr_valid)
        if matrix is None:
            return np.zeros(3, dtype=np.float32)

        dx = matrix[0, 2]
        dy = matrix[1, 2]
        da = np.arctan2(matrix[1, 0], matrix[0, 0])
        return np.array([dx, dy, da], dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Online stabilization step. For live streams, applies exponential smoothing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.trajectory.append(np.zeros(3, dtype=np.float32))
            return frame

        motion = self.estimate_motion(self.prev_gray, gray)
        self.transforms.append(motion)
        cum_motion = self.trajectory[-1] + motion
        self.trajectory.append(cum_motion)
        self.prev_gray = gray

        # Compute smoothed trajectory from recent history
        window_size = min(len(self.trajectory), self.smoothing_radius)
        recent = np.array(self.trajectory[-window_size:])
        smoothed = np.mean(recent, axis=0)

        # Correction transform: difference between smoothed and actual
        diff = smoothed - cum_motion
        dx, dy, da = diff

        # Build affine matrix
        cos_a = np.cos(da)
        sin_a = np.sin(da)
        m = np.zeros((2, 3), dtype=np.float32)
        m[0, 0] = cos_a
        m[0, 1] = -sin_a
        m[1, 0] = sin_a
        m[1, 1] = cos_a
        m[0, 2] = dx
        m[1, 2] = dy

        h, w = frame.shape[:2]
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)
