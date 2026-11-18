"""Data contracts and schemas for Phase 4 Pose, Fall, and Violence analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

# Standard COCO 17 Keypoints index mapping:
# 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
# 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
# 9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
# 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections (pair of indices) for rendering
SKELETON_EDGES = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), # torso / spine
    (11, 12),         # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
]


class FallState(str, Enum):
    UPRIGHT = "upright"
    BENDING = "bending"
    FALLING = "falling"
    FALLEN = "fallen"
    RECOVERED = "recovered"


@dataclass
class Keypoints:
    """17 COCO keypoints formatted as (17, 3) array: [x, y, conf]."""
    data: np.ndarray  # shape (17, 3)

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray) or self.data.shape != (17, 3):
            raise ValueError(f"Keypoints data must be a numpy array of shape (17, 3), got {getattr(self.data, 'shape', None)}")

    @property
    def nose(self) -> np.ndarray: return self.data[0]
    @property
    def left_eye(self) -> np.ndarray: return self.data[1]
    @property
    def right_eye(self) -> np.ndarray: return self.data[2]
    @property
    def left_ear(self) -> np.ndarray: return self.data[3]
    @property
    def right_ear(self) -> np.ndarray: return self.data[4]
    @property
    def left_shoulder(self) -> np.ndarray: return self.data[5]
    @property
    def right_shoulder(self) -> np.ndarray: return self.data[6]
    @property
    def left_elbow(self) -> np.ndarray: return self.data[7]
    @property
    def right_elbow(self) -> np.ndarray: return self.data[8]
    @property
    def left_wrist(self) -> np.ndarray: return self.data[9]
    @property
    def right_wrist(self) -> np.ndarray: return self.data[10]
    @property
    def left_hip(self) -> np.ndarray: return self.data[11]
    @property
    def right_hip(self) -> np.ndarray: return self.data[12]
    @property
    def left_knee(self) -> np.ndarray: return self.data[13]
    @property
    def right_knee(self) -> np.ndarray: return self.data[14]
    @property
    def left_ankle(self) -> np.ndarray: return self.data[15]
    @property
    def right_ankle(self) -> np.ndarray: return self.data[16]

    @property
    def shoulders_center(self) -> Optional[np.ndarray]:
        """Midpoint of left and right shoulder if visible."""
        ls, rs = self.left_shoulder, self.right_shoulder
        if ls[2] > 0.1 and rs[2] > 0.1:
            return (ls[:2] + rs[:2]) / 2.0
        elif ls[2] > 0.1:
            return ls[:2]
        elif rs[2] > 0.1:
            return rs[:2]
        return None

    @property
    def hips_center(self) -> Optional[np.ndarray]:
        """Midpoint of left and right hip if visible."""
        lh, rh = self.left_hip, self.right_hip
        if lh[2] > 0.1 and rh[2] > 0.1:
            return (lh[:2] + rh[:2]) / 2.0
        elif lh[2] > 0.1:
            return lh[:2]
        elif rh[2] > 0.1:
            return rh[:2]
        return None

    @property
    def torso_center(self) -> Optional[np.ndarray]:
        """Center of the torso (midpoint between shoulder-center and hip-center)."""
        sc = self.shoulders_center
        hc = self.hips_center
        if sc is not None and hc is not None:
            return (sc + hc) / 2.0
        return sc if sc is not None else hc

    @property
    def torso_angle(self) -> float:
        """Angle of the torso vector (hips -> shoulders) relative to the vertical Y axis in degrees.
        
        0 deg   = upright standing / sitting straight
        45 deg  = bending forward/backward
        90 deg  = horizontal / lying down
        """
        sc = self.shoulders_center
        hc = self.hips_center
        if sc is None or hc is None:
            return 0.0
        # vector pointing from hips up to shoulders: (dx, dy)
        # Note: image coordinates has Y pointing downwards.
        dx = float(sc[0] - hc[0])
        dy = float(sc[1] - hc[1])  # negative when shoulders are above hips
        
        # angle with vertical (dy < 0 is vertical upward)
        # vertical unit vector is (0, -1)
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return 0.0
        # dot product with (0, -1): -dy / length
        cos_theta = max(-1.0, min(1.0, -dy / length))
        angle_rad = math.acos(cos_theta)
        return float(math.degrees(angle_rad))

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box (x1, y1, x2, y2) computed from valid keypoints."""
        valid = self.data[self.data[:, 2] > 0.1]
        if len(valid) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        x1 = float(np.min(valid[:, 0]))
        y1 = float(np.min(valid[:, 1]))
        x2 = float(np.max(valid[:, 0]))
        y2 = float(np.max(valid[:, 1]))
        return (x1, y1, x2, y2)

    @property
    def aspect_ratio(self) -> float:
        """Width / Height of the keypoint bounding box."""
        x1, y1, x2, y2 = self.bbox
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        return float(w / h)


@dataclass
class PoseRecord:
    """One person's pose in one frame."""
    camera_id: str
    frame_idx: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    keypoints: Keypoints
    conf: float
    global_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "global_id": self.global_id,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "conf": round(float(self.conf), 4),
            "torso_angle": round(self.keypoints.torso_angle, 1),
            "aspect_ratio": round(self.keypoints.aspect_ratio, 3),
        }


@dataclass
class FallAlert:
    """Alert emitted when a fall event is confirmed."""
    camera_id: str
    frame_idx: int
    track_id: int
    global_id: Optional[int]
    timestamp: float
    confidence: float
    state: FallState
    torso_angle: float
    aspect_ratio: float
    descent_velocity: float
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "event": "fall_alert",
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "global_id": self.global_id,
            "timestamp": round(self.timestamp, 3),
            "confidence": round(self.confidence, 4),
            "state": self.state.value,
            "torso_angle": round(self.torso_angle, 1),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "descent_velocity": round(self.descent_velocity, 2),
            "details": self.details,
        }


@dataclass
class FightAlert:
    """Alert emitted when a violent interaction between people is detected."""
    camera_id: str
    frame_idx: int
    track_ids: Tuple[int, int]
    global_ids: Tuple[Optional[int], Optional[int]]
    timestamp: float
    confidence: float
    motion_intensity: float
    proximity_ratio: float
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "event": "fight_alert",
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "track_ids": list(self.track_ids),
            "global_ids": list(self.global_ids),
            "timestamp": round(self.timestamp, 3),
            "confidence": round(self.confidence, 4),
            "motion_intensity": round(self.motion_intensity, 3),
            "proximity_ratio": round(self.proximity_ratio, 3),
            "details": self.details,
        }
