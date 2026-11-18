"""Physics-based temporal fall detection engine.

Analyzes keypoint geometry, torso inclination angle, aspect ratio flip,
and descent velocity over time with a state machine to robustly confirm falls
and filter out ordinary actions like sitting, bending, or tying shoes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import FallAlert, FallState, Keypoints, PoseRecord


@dataclass
class TrackState:
    """Historical buffer and state tracker for a single person track."""
    history: deque  # deque of (timestamp, frame_idx, PoseRecord)
    state: FallState = FallState.UPRIGHT
    fall_start_ts: Optional[float] = None
    fall_frame_idx: Optional[int] = None
    max_descent_vel: float = 0.0
    confirmed: bool = False


class FallDetector:
    """Multi-stage temporal fall detector.

    Parameters
    ----------
    angle_threshold : degrees (torso angle > this indicates horizontal/fallen posture, default: 60.0)
    aspect_threshold : w/h ratio (> this indicates horizontal body layout, default: 1.0)
    descent_velocity_threshold : fraction of body height dropped per second during fall (default: 1.2)
    inactivity_seconds : seconds the person must remain down before triggering a confirmed alarm (default: 1.0)
    buffer_seconds : duration of historical pose trajectory to keep (default: 3.0)
    """

    def __init__(self,
                 angle_threshold: float = 60.0,
                 aspect_threshold: float = 0.95,
                 descent_velocity_threshold: float = 1.0,
                 inactivity_seconds: float = 0.8,
                 buffer_seconds: float = 3.0):
        self.angle_threshold = angle_threshold
        self.aspect_threshold = aspect_threshold
        self.descent_velocity_threshold = descent_velocity_threshold
        self.inactivity_seconds = inactivity_seconds
        self.buffer_seconds = buffer_seconds
        self._tracks: Dict[Tuple[str, int], TrackState] = {}

    def update(self, pose: PoseRecord, timestamp: float) -> Optional[FallAlert]:
        """Update tracker with a new pose and check for fall condition."""
        key = (pose.camera_id, pose.track_id)
        if key not in self._tracks:
            self._tracks[key] = TrackState(history=deque(maxlen=60))
        tstate = self._tracks[key]
        tstate.history.append((timestamp, pose.frame_idx, pose))

        # Purge entries older than buffer_seconds
        while len(tstate.history) > 1 and (timestamp - tstate.history[0][0]) > self.buffer_seconds:
            tstate.history.popleft()

        torso_angle = pose.keypoints.torso_angle
        aspect = pose.keypoints.aspect_ratio
        is_horizontal = (torso_angle >= self.angle_threshold) or (aspect >= self.aspect_threshold)

        # 1. Check recovery: if person was falling/fallen and stands back up
        if tstate.state in (FallState.FALLING, FallState.FALLEN):
            if torso_angle < 35.0 and aspect < 0.6:
                alert = FallAlert(
                    camera_id=pose.camera_id,
                    frame_idx=pose.frame_idx,
                    track_id=pose.track_id,
                    global_id=pose.global_id,
                    timestamp=timestamp,
                    confidence=0.9,
                    state=FallState.RECOVERED,
                    torso_angle=torso_angle,
                    aspect_ratio=aspect,
                    descent_velocity=0.0,
                    details="Person recovered and stood back up.",
                )
                tstate.state = FallState.RECOVERED
                tstate.confirmed = False
                tstate.fall_start_ts = None
                return alert

        # 2. Check for transition to FALLING or FALLEN
        if is_horizontal:
            # Measure descent velocity from recent history (look back up to 1.0s)
            curr_center = pose.keypoints.torso_center
            h_curr = max(10.0, pose.bbox[3] - pose.bbox[1])
            max_vel = 0.0

            if curr_center is not None and len(tstate.history) >= 2:
                for past_ts, past_idx, past_pose in tstate.history:
                    dt = timestamp - past_ts
                    if 0.1 <= dt <= 1.0:
                        past_center = past_pose.keypoints.torso_center
                        if past_center is not None:
                            # In image coordinates, Y increases downwards
                            dy = float(curr_center[1] - past_center[1])
                            past_h = max(10.0, past_pose.bbox[3] - past_pose.bbox[1])
                            norm_vel = (dy / past_h) / dt  # body heights dropped per second
                            if norm_vel > max_vel:
                                max_vel = norm_vel

            if max_vel > tstate.max_descent_vel:
                tstate.max_descent_vel = max_vel

            if tstate.state in (FallState.UPRIGHT, FallState.RECOVERED, FallState.BENDING):
                # Transition to FALLING
                tstate.state = FallState.FALLING
                tstate.fall_start_ts = timestamp
                tstate.fall_frame_idx = pose.frame_idx

            elif tstate.state == FallState.FALLING:
                # If down for long enough, confirm as FALLEN
                time_down = timestamp - (tstate.fall_start_ts or timestamp)
                vel_ok = tstate.max_descent_vel >= self.descent_velocity_threshold

                if time_down >= self.inactivity_seconds and vel_ok and not tstate.confirmed:
                    # Verified fall alert
                    tstate.state = FallState.FALLEN
                    tstate.confirmed = True
                    conf = min(0.99, 0.7 + 0.15 * min(1.0, tstate.max_descent_vel) + 0.15 * min(1.0, torso_angle / 90.0))
                    return FallAlert(
                        camera_id=pose.camera_id,
                        frame_idx=pose.frame_idx,
                        track_id=pose.track_id,
                        global_id=pose.global_id,
                        timestamp=timestamp,
                        confidence=float(conf),
                        state=FallState.FALLEN,
                        torso_angle=torso_angle,
                        aspect_ratio=aspect,
                        descent_velocity=tstate.max_descent_vel,
                        details=f"Fall confirmed: horizontal for {time_down:.1f}s, descent vel {tstate.max_descent_vel:.1f} h/s",
                    )
        else:
            # Person is upright or slight bend
            if tstate.state == FallState.UPRIGHT:
                if torso_angle >= 35.0:
                    tstate.state = FallState.BENDING
            elif tstate.state == FallState.BENDING:
                if torso_angle < 30.0:
                    tstate.state = FallState.UPRIGHT

        return None

    def reset_track(self, camera_id: str, track_id: int):
        self._tracks.pop((camera_id, track_id), None)
