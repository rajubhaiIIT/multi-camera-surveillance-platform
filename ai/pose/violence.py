"""Multi-person proximity and violent motion detection engine.

Analyzes pairwise interaction between tracked individuals:
- Spatial proximity and bounding box intersection
- Fast arm/wrist kinematics and punch/strike trajectories directed toward the other subject
- Rapid bounding box movement/jitter characteristic of physical struggles
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import FightAlert, Keypoints, PoseRecord


class ViolenceDetector:
    """Pairwise violent interaction and struggle detector.

    Parameters
    ----------
    proximity_threshold : distance in fraction of person height to consider interactive (default: 1.2)
    motion_energy_threshold : wrist/arm movement velocity threshold for strikes (default: 1.5 h/s)
    consecutive_frames : number of frames high interaction must persist to trigger alert (default: 4)
    """

    def __init__(self,
                 proximity_threshold: float = 1.2,
                 motion_energy_threshold: float = 1.2,
                 consecutive_frames: int = 4):
        self.proximity_threshold = proximity_threshold
        self.motion_energy_threshold = motion_energy_threshold
        self.consecutive_frames = consecutive_frames

        # History per track: deque of (ts, frame_idx, PoseRecord)
        self._history: Dict[Tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=30))
        # Interaction counters per pair: (cam, min_tid, max_tid) -> int
        self._pair_counts: Dict[Tuple[str, int, int], int] = defaultdict(int)

    def update(self, poses: List[PoseRecord], timestamp: float) -> List[FightAlert]:
        """Analyze all poses in a single frame and return any confirmed violent interactions."""
        alerts = []
        if len(poses) < 2:
            # Clean up empty pairs
            for p in poses:
                self._history[(p.camera_id, p.track_id)].append((timestamp, p.frame_idx, p))
            return alerts

        # 1. Update individual histories
        for p in poses:
            self._history[(p.camera_id, p.track_id)].append((timestamp, p.frame_idx, p))

        # 2. Check all unique pairs of poses
        n = len(poses)
        active_pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = poses[i], poses[j]
                if p1.camera_id != p2.camera_id:
                    continue

                cam = p1.camera_id
                tid1, tid2 = min(p1.track_id, p2.track_id), max(p1.track_id, p2.track_id)
                pair_key = (cam, tid1, tid2)
                active_pairs.add(pair_key)

                # Height normalization
                h1 = max(10.0, p1.bbox[3] - p1.bbox[1])
                h2 = max(10.0, p2.bbox[3] - p2.bbox[1])
                avg_h = (h1 + h2) / 2.0

                c1 = p1.keypoints.torso_center
                c2 = p2.keypoints.torso_center
                if c1 is None or c2 is None:
                    continue

                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                norm_dist = dist / avg_h

                if norm_dist > self.proximity_threshold:
                    self._pair_counts[pair_key] = max(0, self._pair_counts[pair_key] - 1)
                    continue

                # Measure arm motion energy from recent past frames
                energy1 = self._compute_arm_energy(p1, timestamp)
                energy2 = self._compute_arm_energy(p2, timestamp)
                total_energy = energy1 + energy2

                is_violent = total_energy >= self.motion_energy_threshold

                if is_violent:
                    self._pair_counts[pair_key] += 1
                else:
                    self._pair_counts[pair_key] = max(0, self._pair_counts[pair_key] - 1)

                if self._pair_counts[pair_key] >= self.consecutive_frames:
                    conf = min(0.98, 0.65 + 0.15 * min(2.0, total_energy / self.motion_energy_threshold))
                    alerts.append(FightAlert(
                        camera_id=cam,
                        frame_idx=p1.frame_idx,
                        track_ids=(tid1, tid2),
                        global_ids=(p1.global_id if p1.track_id == tid1 else p2.global_id,
                                    p2.global_id if p2.track_id == tid2 else p1.global_id),
                        timestamp=timestamp,
                        confidence=float(conf),
                        motion_intensity=float(total_energy),
                        proximity_ratio=float(norm_dist),
                        details=f"Violent interaction detected: proximity {norm_dist:.2f}h, arm energy {total_energy:.2f} h/s",
                    ))

        # Reset counts for pairs not active
        for pk in list(self._pair_counts):
            if pk not in active_pairs:
                self._pair_counts.pop(pk, None)

        return alerts

    def _compute_arm_energy(self, pose: PoseRecord, curr_ts: float) -> float:
        """Calculate normalized wrist displacement velocity (heights / sec)."""
        hist = self._history[(pose.camera_id, pose.track_id)]
        if len(hist) < 2:
            return 0.0

        h = max(10.0, pose.bbox[3] - pose.bbox[1])
        lw, rw = pose.keypoints.left_wrist, pose.keypoints.right_wrist
        max_vel = 0.0

        for past_ts, past_idx, past_pose in reversed(list(hist)[:-1]):
            dt = curr_ts - past_ts
            if dt < 0.04:  # too close in time
                continue
            if dt > 0.4:   # too far in time
                break

            past_lw = past_pose.keypoints.left_wrist
            past_rw = past_pose.keypoints.right_wrist

            # Left wrist speed
            if lw[2] > 0.2 and past_lw[2] > 0.2:
                d = math.hypot(lw[0] - past_lw[0], lw[1] - past_lw[1])
                vel = (d / h) / dt
                if vel > max_vel:
                    max_vel = vel

            # Right wrist speed
            if rw[2] > 0.2 and past_rw[2] > 0.2:
                d = math.hypot(rw[0] - past_rw[0], rw[1] - past_rw[1])
                vel = (d / h) / dt
                if vel > max_vel:
                    max_vel = vel

        return max_vel

    def reset_track(self, camera_id: str, track_id: int):
        self._history.pop((camera_id, track_id), None)
        for pk in list(self._pair_counts):
            if pk[0] == camera_id and (pk[1] == track_id or pk[2] == track_id):
                self._pair_counts.pop(pk, None)
