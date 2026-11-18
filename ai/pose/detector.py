"""YOLO-Pose detector wrapper for multi-person pose estimation and skeleton visualization."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .schema import COCO_KEYPOINTS, SKELETON_EDGES, Keypoints, PoseRecord


DEFAULT_POSE_WEIGHTS = "yolo11n-pose.pt"


def _box_iou(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / (area1 + area2 - inter)


class PoseDetector:
    """Wrapper around YOLO-Pose (YOLO11n-pose) for keypoint extraction.

    Parameters
    ----------
    weights : path or model name (default: 'yolo11n-pose.pt')
    device : 'cuda:0', 'cpu', or 'auto'
    conf : detection confidence threshold
    """

    def __init__(self, weights: str = DEFAULT_POSE_WEIGHTS, device: str = "auto", conf: float = 0.25):
        import torch
        from ultralytics import YOLO

        self.conf = conf
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(weights)
        self.weights = weights

    def detect(self, frame: np.ndarray, camera_id: str = "cam", frame_idx: int = 0,
               imgsz: int = 640) -> List[PoseRecord]:
        """Detect human poses in a BGR frame."""
        res = self.model(frame, imgsz=imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        records = []
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return records

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kp_data = res.keypoints.data.cpu().numpy()  # shape (N, 17, 3)

        for i in range(len(boxes)):
            box = tuple(float(v) for v in boxes[i])
            c = float(confs[i])
            kpts = Keypoints(kp_data[i].astype(np.float32))
            records.append(PoseRecord(
                camera_id=camera_id,
                frame_idx=frame_idx,
                track_id=i + 1,  # temporary track_id if not matched to tracker
                bbox=box,
                keypoints=kpts,
                conf=c,
            ))
        return records

    def match_to_tracks(self, poses: List[PoseRecord], tracks: list, iou_thresh: float = 0.3) -> List[PoseRecord]:
        """Associate detected poses with existing TrackRecord objects by bounding box IoU."""
        if not tracks or not poses:
            return poses

        matched = []
        unassigned_tracks = list(tracks)
        for pose in poses:
            best_iou = 0.0
            best_track = None
            for trk in unassigned_tracks:
                iou = _box_iou(pose.bbox, trk.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = trk
            if best_track is not None and best_iou >= iou_thresh:
                matched.append(PoseRecord(
                    camera_id=best_track.camera_id,
                    frame_idx=best_track.frame_idx,
                    track_id=best_track.track_id,
                    bbox=best_track.bbox,
                    keypoints=pose.keypoints,
                    conf=pose.conf,
                    global_id=best_track.global_id,
                ))
                unassigned_tracks.remove(best_track)
            else:
                matched.append(pose)
        return matched

    def draw(self, frame: np.ndarray, poses: List[PoseRecord],
             kpt_conf_thresh: float = 0.3) -> np.ndarray:
        """Render skeletons and bounding boxes onto a copy of the frame."""
        vis = frame.copy()
        for p in poses:
            # Draw bounding box
            x1, y1, x2, y2 = (int(round(v)) for v in p.bbox)
            color = (0, 255, 0)
            label = f"#{p.track_id}" if p.global_id is None else f"P{p.global_id}"
            label += f" | {p.keypoints.torso_angle:.0f} deg"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            kpts = p.keypoints.data
            # Draw edges
            for idx1, idx2 in SKELETON_EDGES:
                pt1, pt2 = kpts[idx1], kpts[idx2]
                if pt1[2] >= kpt_conf_thresh and pt2[2] >= kpt_conf_thresh:
                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])), int(round(pt2[1])))
                    cv2.line(vis, p1, p2, (255, 128, 0), 2, cv2.LINE_AA)

            # Draw joints
            for j in range(17):
                pt = kpts[j]
                if pt[2] >= kpt_conf_thresh:
                    center = (int(round(pt[0])), int(round(pt[1])))
                    cv2.circle(vis, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
        return vis
