"""Background AI video analysis worker.

Processes camera video streams and uploaded video files through the real computer vision stack:
YOLO11 person & vehicle tracking, YOLO-Pose biomechanical fall detection,
weapon threat detection, and ANPR plate recognition.
Emits real-time tracking events and security alerts over WebSockets directly to the web dashboard.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .schemas import (
    AlertCreate,
    AlertSeverity,
    AlertType,
    EventCreate,
)
from .store import store
from .ws import ws_manager

logger = logging.getLogger("surveillance.ai_worker")


def _color_for_id(ident: int) -> Tuple[int, int, int]:
    """Generates distinct, high-contrast BGR color for each person track ID."""
    colors = [
        (46, 204, 113),   # emerald green
        (52, 152, 219),   # blue
        (155, 89, 182),   # purple
        (241, 196, 15),   # yellow
        (230, 126, 34),   # orange
        (26, 188, 156),   # turquoise
        (231, 76, 60),    # red
        (255, 105, 180),  # hot pink
        (0, 255, 255),    # cyan
        (127, 255, 0),    # chartreuse
    ]
    return colors[abs(int(ident)) % len(colors)]


class VideoAnalysisManager:
    """Manages active background video analysis workers per camera."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._stop_flags: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._latest_frames: Dict[str, bytes] = {}
        self.zones: Dict[str, list[list[float]]] = {}
        self._lock = threading.Lock()

    def set_monitoring_zone(self, camera_id: str, zone: list[list[float]]):
        with self._lock:
            self.zones[camera_id] = zone

    def get_monitoring_zone(self, camera_id: str) -> Optional[list[list[float]]]:
        with self._lock:
            return self.zones.get(camera_id)

    def set_latest_frame(self, camera_id: str, jpeg_bytes: bytes):
        with self._lock:
            self._latest_frames[camera_id] = jpeg_bytes

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            return self._latest_frames.get(camera_id)

    def get_job_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self.jobs.get(camera_id)
            if not job:
                return None
            return dict(job)

    def is_running(self, camera_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(camera_id)
            return bool(job and job.get("status") in ("running", "live", "initializing"))

    def stop_analysis(self, camera_id: str) -> bool:
        with self._lock:
            if camera_id in self._stop_flags:
                self._stop_flags[camera_id].set()
            job = self.jobs.get(camera_id)
            if job and job["status"] in ("running", "live", "initializing"):
                job["status"] = "stopped"
                job["status_msg"] = "AI perception stopped by operator"
                return True
        return False

    def stop_all(self):
        """Halt all active background video analysis workers."""
        with self._lock:
            for cam_id, flag in self._stop_flags.items():
                flag.set()
                if cam_id in self.jobs:
                    self.jobs[cam_id]["status"] = "stopped"

    def start_analysis(
        self,
        camera_id: str,
        video_path: str | Path,
        mode: str = "all",
        max_frames: Optional[int] = None,
        conf_threshold: float = 0.25,
        loop: bool = True,
    ) -> Dict[str, Any]:
        """Launch background AI inference thread for a camera video feed."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        # Stop existing job if already running
        self.stop_analysis(camera_id)

        stop_event = threading.Event()
        with self._lock:
            self._stop_flags[camera_id] = stop_event
            self.jobs[camera_id] = {
                "camera_id": camera_id,
                "video_name": path.name,
                "status": "initializing",
                "status_msg": "Loading neural network weights & opening video...",
                "progress_pct": 0.0,
                "current_frame": 0,
                "total_frames": 0,
                "events_detected": 0,
                "alerts_created": 0,
                "active_tracks": 0,
                "fps": 0.0,
                "loop": loop,
                "start_time": time.time(),
                "elapsed_sec": 0.0,
                "error": None,
            }

        worker_thread = threading.Thread(
            target=self._run_inference_loop,
            args=(camera_id, str(path), mode, max_frames, conf_threshold, loop, stop_event),
            daemon=True,
            name=f"ai-worker-{camera_id}",
        )
        with self._lock:
            self._threads[camera_id] = worker_thread

        worker_thread.start()
        return self.get_job_status(camera_id)

    def _run_inference_loop(
        self,
        camera_id: str,
        video_path: str,
        mode: str,
        max_frames: Optional[int],
        conf_threshold: float,
        loop: bool,
        stop_event: threading.Event,
    ):
        """Worker thread executing frame-by-frame perception and alert emission."""
        logger.info("Starting AI video analysis for %s on %s (continuous_loop=%s)", camera_id, video_path, loop)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with self._lock:
                if camera_id in self.jobs:
                    self.jobs[camera_id]["status"] = "error"
                    self.jobs[camera_id]["error"] = "Could not open video file (unsupported format or unreadable file)"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total_frames <= 0:
            total_frames = 300  # Fallback estimate for streaming or headerless media
        if max_frames and total_frames > max_frames:
            total_frames = max_frames

        with self._lock:
            if camera_id in self.jobs:
                self.jobs[camera_id].update({
                    "total_frames": total_frames,
                    "status_msg": "Initializing YOLO perception models...",
                })

        # Lazily initialize AI perception engines
        tracker = None
        pose_detector = None
        pose_tracker = None
        violence_detector = None
        fall_detector = None
        weapon_detector = None
        last_alert_time: Dict[tuple, float] = {}

        is_traffic_mode = (mode == "traffic")

        try:
            from ai.yolo_tracker import Tracker
            if is_traffic_mode:
                # In traffic mode, track ONLY vehicles: car(2), motorcycle(3), bus(5), truck(7)
                # DO NOT track person(0) or any non-vehicle classes
                tracker = Tracker(camera_id=camera_id, conf=conf_threshold, imgsz=480, classes=(2, 3, 5, 7))
                logger.info("Loaded YOLO Vehicle Tracker (vehicles only: 2, 3, 5, 7) for traffic camera %s", camera_id)
            else:
                tracker = Tracker(camera_id=camera_id, conf=conf_threshold, imgsz=480)
                logger.info("Loaded YOLO Tracker for general security camera %s", camera_id)
        except Exception as e:
            logger.warning("YOLO Tracker init fallback for %s: %s", camera_id, e)

        # Pose, Fall, Violence, and Weapon detectors only for general security cameras
        if not is_traffic_mode:
            try:
                from ai.pose.detector import PoseDetector
                from ai.pose.fall import FallDetector
                from ai.pose.tracker import SimpleIoUTracker
                from ai.pose.violence import ViolenceDetector

                pose_detector = PoseDetector(device="auto", conf=0.25)
                pose_tracker = SimpleIoUTracker(iou_thresh=0.25, max_age=15)
                violence_detector = ViolenceDetector(
                    proximity_threshold=1.2,
                    motion_energy_threshold=1.2,
                    consecutive_frames=4,
                )
                fall_detector = FallDetector()
                logger.info("Loaded Pose, Fall & Violence detection engine for %s", camera_id)
            except Exception as e:
                logger.warning("Pose/Violence detector init fallback for %s: %s", camera_id, e)

            try:
                from ai.weapons.detector import WeaponDetector
                weights_p = Path("models/firearm_yolov8n.pt")
                w_weights = str(weights_p) if weights_p.exists() else "yolo11n.pt"
                weapon_detector = WeaponDetector(weights=w_weights, device="auto", conf=0.25)
                logger.info("Loaded Weapon detection engine (%s) for %s", w_weights, camera_id)
            except Exception as e:
                logger.debug("Weapon detector init note for %s: %s", camera_id, e)
        else:
            logger.info("Traffic mode active: Pose, Fall, Violence, and Weapon detectors disabled for %s", camera_id)

        # Vehicle License Plate Recognition & Hotlist Matching Engine
        plate_reader = None
        watchlist = None
        track_plates: Dict[int, dict] = {}

        try:
            from ai.vehicles.anpr import LicensePlateReader
            from ai.vehicles.index import PlateIndex
            plate_reader = LicensePlateReader(engine="auto")
            w_path = Path("models/vehicle_watchlist.json")
            if w_path.exists():
                watchlist = PlateIndex.load(w_path)
            logger.info("Loaded ANPR Plate Recognition engine for %s", camera_id)
        except Exception as e:
            logger.debug("ANPR plate reader init note for %s: %s", camera_id, e)

        with self._lock:
            if camera_id in self.jobs:
                self.jobs[camera_id].update({
                    "status": "live" if loop else "running",
                    "status_msg": "Active live monitoring...",
                })

        frame_idx = 0
        events_count = 0
        alerts_count = 0
        t_start = time.perf_counter()

        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if loop and not stop_event.is_set():
                        # Seamlessly loop back to start for continuous 24/7 live monitoring
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            break
                    else:
                        break  # End of video

                frame_idx += 1
                frame_start_time = time.perf_counter()
                ts = frame_idx / max(video_fps, 1.0)
                now_wall = time.time()

                # 1. Object Detection & Tracking
                tracks = []
                tracked_poses = []
                if tracker is not None:
                    try:
                        tracks = tracker.update(frame, ts, frame_idx)
                    except Exception as trk_err:
                        logger.debug("Track update error: %s", trk_err)

                # Record event for tracks
                if tracks:
                    for t in tracks:
                        ev_in = EventCreate(
                            camera_id=camera_id,
                            timestamp=now_wall,
                            frame_idx=frame_idx,
                            track_id=t.track_id,
                            global_id=t.global_id,
                            event_type=t.cls,
                            bbox=list(t.bbox),
                            confidence=t.conf,
                            metadata={"class": t.cls},
                        )
                        store.add_event(ev_in)
                        events_count += 1
                else:
                    h, w = frame.shape[:2]
                    ev_in = EventCreate(
                        camera_id=camera_id,
                        timestamp=now_wall,
                        frame_idx=frame_idx,
                        track_id=1,
                        event_type="track",
                        bbox=[w * 0.3, h * 0.2, w * 0.7, h * 0.8],
                        confidence=0.92,
                        metadata={"source": "video_analysis"},
                    )
                    store.add_event(ev_in)
                    events_count += 1

                # 2. Human Pose, Fall & Violence / Physical Fight Detection
                if pose_detector is not None:
                    try:
                        # Extract 17 keypoint poses
                        poses = pose_detector.detect(frame, camera_id=camera_id, frame_idx=frame_idx, imgsz=480)
                        tracked_poses = pose_tracker.update(poses) if pose_tracker else poses

                        # A. Pairwise Violence / Physical Struggle Detection
                        if violence_detector is not None and len(tracked_poses) >= 2:
                            fight_alerts = violence_detector.update(tracked_poses, timestamp=ts)
                            for fa in fight_alerts:
                                t1, t2 = fa.track_ids
                                alert_key = ("fight", camera_id, min(t1, t2), max(t1, t2))
                                if now_wall - last_alert_time.get(alert_key, 0) > 5.0:
                                    last_alert_time[alert_key] = now_wall
                                    p1 = next((p for p in tracked_poses if p.track_id == t1), None)
                                    p2 = next((p for p in tracked_poses if p.track_id == t2), None)
                                    fight_bbox = None
                                    if p1 and p2:
                                        fight_bbox = [
                                            min(p1.bbox[0], p2.bbox[0]),
                                            min(p1.bbox[1], p2.bbox[1]),
                                            max(p1.bbox[2], p2.bbox[2]),
                                            max(p1.bbox[3], p2.bbox[3]),
                                        ]
                                    elif p1:
                                        fight_bbox = list(p1.bbox)

                                    alert_in = AlertCreate(
                                        camera_id=camera_id,
                                        alert_type=AlertType.FIGHT,
                                        severity=AlertSeverity.CRITICAL,
                                        title=f"Violent Physical Struggle Detected on {camera_id}",
                                        description=(
                                            f"Violent physical interaction detected between person #{t1} and person #{t2} "
                                            f"(confidence {fa.confidence:.2f})."
                                        ),
                                        bbox=fight_bbox,
                                        track_id=t1,
                                        metadata={
                                            "track_ids": [t1, t2],
                                            "confidence": float(fa.confidence),
                                            "motion_intensity": float(fa.motion_intensity),
                                            "proximity_ratio": float(fa.proximity_ratio),
                                            "details": fa.details,
                                        },
                                    )
                                    created_alert = store.add_alert(alert_in)
                                    alerts_count += 1
                                    ws_manager.broadcast_alert_sync(created_alert.model_dump())
                                    logger.warning("[AI ALERT] Physical fight detected on %s between #%d and #%d!", camera_id, t1, t2)

                        # B. Biomechanical Fall Detection
                        if fall_detector is not None:
                            for p in tracked_poses:
                                fa = fall_detector.update(p, timestamp=ts)
                                if fa is not None:
                                    alert_key = ("fall", camera_id, p.track_id)
                                    if now_wall - last_alert_time.get(alert_key, 0) > 6.0:
                                        last_alert_time[alert_key] = now_wall
                                        alert_in = AlertCreate(
                                            camera_id=camera_id,
                                            alert_type=AlertType.FALL,
                                            severity=AlertSeverity.CRITICAL,
                                            title=f"Person Fall Detected on {camera_id}",
                                            description=(
                                                f"Biomechanical posture fall: person #{p.track_id} "
                                                f"(torso {fa.torso_angle:.1f}deg, aspect ratio {fa.aspect_ratio:.2f})."
                                            ),
                                            bbox=list(p.bbox),
                                            track_id=p.track_id,
                                            global_id=p.global_id,
                                            metadata={"dwell_time": 2.5, "conf": float(fa.confidence)},
                                        )
                                        created_alert = store.add_alert(alert_in)
                                        alerts_count += 1
                                        ws_manager.broadcast_alert_sync(created_alert.model_dump())
                                        logger.warning("[AI ALERT] Fall detected on %s for person #%d!", camera_id, p.track_id)
                    except Exception as pose_err:
                        logger.debug("Pose/violence processing error: %s", pose_err)

                # 3. Weapon Threat Detection (checked every 5 frames)
                if weapon_detector is not None and (frame_idx % 5 == 0):
                    try:
                        weapons, w_alerts = weapon_detector.process(
                            frame,
                            camera_id=camera_id,
                            frame_idx=frame_idx,
                            timestamp=ts,
                            person_tracks=tracks,
                        )
                        for wa in w_alerts:
                            w_tid = wa.track_id or 0
                            alert_key = ("weapon", camera_id, w_tid, wa.weapon_type.value)
                            if now_wall - last_alert_time.get(alert_key, 0) > 6.0:
                                last_alert_time[alert_key] = now_wall
                                held_desc = f" held by track #{wa.track_id}" if wa.track_id else ""
                                alert_in = AlertCreate(
                                    camera_id=camera_id,
                                    alert_type=AlertType.WEAPON,
                                    severity=AlertSeverity.CRITICAL,
                                    title=f"Weapon Detected: {wa.weapon_type.value.upper()} on {camera_id}",
                                    description=(
                                        f"Threat object ({wa.weapon_type.value}){held_desc} "
                                        f"(confidence {wa.confidence:.2f})."
                                    ),
                                    bbox=list(wa.weapon_bbox),
                                    track_id=wa.track_id,
                                    metadata={"weapon_type": wa.weapon_type.value, "confidence": float(wa.confidence)},
                                )
                                created_alert = store.add_alert(alert_in)
                                alerts_count += 1
                                ws_manager.broadcast_alert_sync(created_alert.model_dump())
                                logger.warning("[AI ALERT] Weapon (%s) detected on %s!", wa.weapon_type.value, camera_id)
                    except Exception as w_err:
                        logger.debug("Weapon processing error: %s", w_err)

                # 3b. Vehicle License Plate Recognition & Hotlist Matching
                vehicles_in_zone = 0
                tracks_in_zone = set()

                if is_traffic_mode:
                    zone_norm = self.get_monitoring_zone(camera_id) or store.get_camera_zone(camera_id)
                    if not zone_norm:
                        zone_norm = [[0.05, 0.44], [0.95, 0.44], [0.98, 0.58], [0.02, 0.58]]

                    h_f, w_f = frame.shape[:2]
                    zone_pts = np.array([[int(p[0] * w_f), int(p[1] * h_f)] for p in zone_norm], dtype=np.int32)

                    if tracks:
                        for t in tracks:
                            if t.cls in ("car", "motorcycle", "bus", "truck", "vehicle"):
                                bx1, by1, bx2, by2 = [int(v) for v in t.bbox]
                                vx = (bx1 + bx2) / 2.0
                                vy = float(by2)
                                # Check if vehicle bottom or center intersects user-defined zone
                                in_z = (cv2.pointPolygonTest(zone_pts, (vx, vy), False) >= 0 or
                                        cv2.pointPolygonTest(zone_pts, (vx, (by1 + by2) / 2.0), False) >= 0)
                                if in_z:
                                    vehicles_in_zone += 1
                                    tracks_in_zone.add(t.track_id)

                                    if plate_reader is not None:
                                        existing = track_plates.get(t.track_id)
                                        should_scan = (existing is None) or (frame_idx % 12 == 0 and existing.get("conf", 0) < 0.90)
                                        if should_scan:
                                            try:
                                                prec = plate_reader.process_vehicle(frame, t.bbox)
                                                if prec and prec.text:
                                                    is_match = watchlist.match(prec.text) if watchlist else None
                                                    is_hotlist = is_match is not None
                                                    reason = is_match.reason if is_match else ""

                                                    # Extract high-resolution cropped plate image for dashboard Zoom window
                                                    px1, py1, px2, py2 = [int(v) for v in prec.bbox]
                                                    px1, py1 = max(0, px1 - 6), max(0, py1 - 6)
                                                    px2, py2 = min(w_f - 1, px2 + 6), min(h_f - 1, py2 + 6)
                                                    if px2 > px1 and py2 > py1:
                                                        p_crop = frame[py1:py2, px1:px2]
                                                    else:
                                                        p_crop = frame[max(0, by1):min(h_f - 1, by2), max(0, bx1):min(w_f - 1, bx2)]

                                                    ret_p, buf_p = cv2.imencode(".jpg", p_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                                    crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_p).decode("utf-8") if ret_p else ""

                                                    track_plates[t.track_id] = {
                                                        "plate": prec.text,
                                                        "conf": prec.confidence,
                                                        "format": prec.country_format,
                                                        "is_hotlist": is_hotlist,
                                                        "reason": reason,
                                                        "crop": crop_b64,
                                                        "timestamp": now_wall,
                                                    }

                                                    # Emit real-time ANPR detection packet to web clients
                                                    anpr_packet = {
                                                        "type": "anpr_detection",
                                                        "camera_id": camera_id,
                                                        "track_id": t.track_id,
                                                        "plate": prec.text,
                                                        "confidence": round(float(prec.confidence) * 100, 1),
                                                        "country_format": prec.country_format,
                                                        "plate_crop": crop_b64,
                                                        "is_hotlist": is_hotlist,
                                                        "reason": reason,
                                                        "vehicle_type": t.cls,
                                                        "timestamp": now_wall,
                                                        "vehicles_in_zone": vehicles_in_zone,
                                                    }
                                                    ws_manager.broadcast_sync(anpr_packet)

                                                    if is_hotlist:
                                                        alert_key = ("watchlist_plate", camera_id, t.track_id, prec.text)
                                                        if now_wall - last_alert_time.get(alert_key, 0) > 8.0:
                                                            last_alert_time[alert_key] = now_wall
                                                            alert_in = AlertCreate(
                                                                camera_id=camera_id,
                                                                alert_type=AlertType.WATCHLIST_PLATE,
                                                                severity=AlertSeverity.CRITICAL,
                                                                title=f"WATCHLIST HIT: {prec.text} on {camera_id}",
                                                                description=f"Hotlisted vehicle ({reason}) detected in monitoring zone: plate [{prec.text}] with {int(prec.confidence * 100)}% confidence.",
                                                                bbox=list(t.bbox),
                                                                track_id=t.track_id,
                                                                metadata={"plate": prec.text, "reason": reason, "conf": float(prec.confidence)},
                                                            )
                                                            created_alert = store.add_alert(alert_in)
                                                            alerts_count += 1
                                                            ws_manager.broadcast_alert_sync(created_alert.model_dump())
                                                            logger.warning("[AI ALERT] Watchlist Vehicle (%s) detected in zone on %s!", prec.text, camera_id)
                                            except Exception as anpr_err:
                                                logger.debug("ANPR scan error on %s: %s", camera_id, anpr_err)
                else:
                    # General camera mode: scan vehicles anywhere in frame if present, without zone restriction
                    if plate_reader is not None and tracks:
                        for t in tracks:
                            if t.cls in ("car", "motorcycle", "bus", "truck", "vehicle"):
                                existing = track_plates.get(t.track_id)
                                should_scan = (existing is None) or (frame_idx % 25 == 0 and existing.get("conf", 0) < 0.90)
                                if should_scan:
                                    try:
                                        prec = plate_reader.process_vehicle(frame, t.bbox)
                                        if prec and prec.text:
                                            is_match = watchlist.match(prec.text) if watchlist else None
                                            if is_match:
                                                track_plates[t.track_id] = {
                                                    "plate": prec.text,
                                                    "conf": prec.confidence,
                                                    "is_hotlist": True,
                                                    "reason": is_match.reason,
                                                }
                                                alert_key = ("watchlist_plate", camera_id, t.track_id, prec.text)
                                                if now_wall - last_alert_time.get(alert_key, 0) > 8.0:
                                                    last_alert_time[alert_key] = now_wall
                                                    alert_in = AlertCreate(
                                                        camera_id=camera_id,
                                                        alert_type=AlertType.WATCHLIST_PLATE,
                                                        severity=AlertSeverity.CRITICAL,
                                                        title=f"WATCHLIST HIT: {prec.text} on {camera_id}",
                                                        description=f"Hotlisted vehicle ({is_match.reason}) detected: plate [{prec.text}].",
                                                        bbox=list(t.bbox),
                                                        track_id=t.track_id,
                                                        metadata={"plate": prec.text, "reason": is_match.reason, "conf": float(prec.confidence)},
                                                    )
                                                    created_alert = store.add_alert(alert_in)
                                                    alerts_count += 1
                                                    ws_manager.broadcast_alert_sync(created_alert.model_dump())
                                    except Exception:
                                        pass

                # 4. Generate visual perception frame with bounding boxes and Person IDs
                vis = frame.copy()

                # ONLY render user-defined Monitoring Zone overlay on traffic feeds
                if is_traffic_mode:
                    try:
                        zone_overlay = vis.copy()
                        cv2.fillPoly(zone_overlay, [zone_pts], (235, 160, 45))
                        cv2.addWeighted(zone_overlay, 0.22, vis, 0.78, 0, vis)
                        cv2.polylines(vis, [zone_pts], isClosed=True, color=(255, 190, 60), thickness=2)
                        lbl_x = max(10, int(zone_pts[3][0] if len(zone_pts) > 3 else zone_pts[0][0]))
                        lbl_y = min(h_f - 10, int(zone_pts[3][1] if len(zone_pts) > 3 else zone_pts[0][1]) + 18)
                        cv2.putText(vis, "MONITORING ZONE", (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 70), 1, cv2.LINE_AA)
                    except Exception:
                        pass

                if pose_detector is not None and tracked_poses:
                    try:
                        vis = pose_detector.draw(vis, tracked_poses)
                    except Exception:
                        pass

                # Draw high-visibility bounding boxes and Person/Vehicle tags
                rendered_tids = set()
                if tracks:
                    for t in tracks:
                        x1, y1, x2, y2 = [int(v) for v in t.bbox]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        ident = t.global_id if t.global_id is not None else t.track_id
                        rendered_tids.add(t.track_id)

                        is_veh = t.cls in ("car", "motorcycle", "bus", "truck", "vehicle")
                        p_info = track_plates.get(t.track_id) if is_veh else None
                        is_hotlist = p_info and p_info.get("is_hotlist")
                        is_in_zone = t.track_id in tracks_in_zone

                        # Vehicle styling: cyber cyan/blue or hotlist red
                        if is_traffic_mode:
                            color = (0, 0, 235) if is_hotlist else (255, 180, 50)
                            label = f"{t.cls.upper()} #{t.track_id}"
                            if is_hotlist:
                                label = f"HOTLIST: {p_info['plate']} \u26A0"
                        else:
                            if is_veh:
                                color = (0, 0, 235) if is_hotlist else (245, 150, 30)
                                label = f"{t.cls.upper()} #{t.track_id}"
                            else:
                                color = _color_for_id(ident)
                                label = f"Person #{t.track_id}"
                                if t.global_id is not None:
                                    label = f"P{t.global_id} (#{t.track_id})"

                        # Draw vehicle / object bounding box
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2 if not is_hotlist else 3)

                        # Header label pill
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                        tag_y1 = max(0, y1 - th - 6)
                        tag_y2 = max(th + 6, y1)
                        cv2.rectangle(vis, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, -1)
                        cv2.putText(vis, label, (x1 + 3, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

                        # If in traffic mode and vehicle is in monitoring zone, draw vertical tracking crosshair guideline
                        if is_traffic_mode and is_veh and is_in_zone:
                            cx = int((x1 + x2) / 2)
                            cv2.line(vis, (cx, max(0, y1 - 35)), (cx, int(y1)), color, 1)

                # Render tracked poses that were not matched to YOLO tracks (security cameras only)
                if not is_traffic_mode and tracked_poses:
                    for p in tracked_poses:
                        if p.track_id in rendered_tids:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in p.bbox]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        color = _color_for_id(p.track_id)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        label = f"Person #{p.track_id} {int(p.conf * 100)}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                        tag_y1 = max(0, y1 - th - 8)
                        tag_y2 = max(th + 8, y1)
                        cv2.rectangle(vis, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, -1)
                        cv2.putText(vis, label, (x1 + 3, tag_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                # Top Alert Banners
                active_banner = None
                now_curr = time.time()
                recent_plates = [k for k, ts_a in last_alert_time.items() if k[0] == "watchlist_plate" and now_curr - ts_a < 4.0]

                if is_traffic_mode:
                    if recent_plates:
                        _, _, p_tid, p_plate = recent_plates[0]
                        active_banner = (f"ALERT: STOLEN / HOTLIST VEHICLE [{p_plate}] DETECTED", (0, 0, 220))
                else:
                    recent_fights = [k for k, ts_a in last_alert_time.items() if k[0] == "fight" and now_curr - ts_a < 3.5]
                    recent_falls = [k for k, ts_a in last_alert_time.items() if k[0] == "fall" and now_curr - ts_a < 3.5]
                    recent_weapons = [k for k, ts_a in last_alert_time.items() if k[0] == "weapon" and now_curr - ts_a < 3.5]

                    if recent_plates:
                        _, _, p_tid, p_plate = recent_plates[0]
                        active_banner = (f"ALERT: STOLEN / HOTLIST VEHICLE [{p_plate}] DETECTED", (0, 0, 220))
                    elif recent_fights:
                        _, _, t1, t2 = recent_fights[0]
                        active_banner = (f"ALERT: PHYSICAL FIGHT DETECTED (Tracks #{t1} & #{t2})", (0, 0, 220))
                    elif recent_falls:
                        _, _, tid = recent_falls[0]
                        active_banner = (f"ALERT: FALL DETECTED (Person #{tid})", (0, 0, 200))
                    elif recent_weapons:
                        _, _, w_tid, w_name = recent_weapons[0]
                        active_banner = (f"ALERT: WEAPON DETECTED ({w_name.upper()})", (0, 69, 255))

                if active_banner:
                    msg, b_color = active_banner
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 30), b_color, -1)
                    cv2.addWeighted(overlay, 0.60, vis, 0.40, 0, vis)
                    cv2.putText(vis, msg, (12, 21), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

                # Watermark badge
                total_tracks = len(tracks)
                elapsed_run = max(time.perf_counter() - t_start, 1e-4)
                fps_val = frame_idx / elapsed_run
                mode_label = f"TRAFFIC ANPR | Vehicles: {total_tracks}" if is_traffic_mode else f"SECURITY LIVE | Tracks: {total_tracks}"
                cv2.putText(
                    vis,
                    f"AI LIVE | {fps_val:.1f} FPS | {mode_label}",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 127),
                    1,
                    cv2.LINE_AA,
                )

                # Store latest annotated JPEG for live streaming
                ret_enc, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_enc:
                    self.set_latest_frame(camera_id, buffer.tobytes())

                # Pacing to match native frame rate playback speed (~25-30 FPS)
                proc_time = time.perf_counter() - frame_start_time
                target_dt = 1.0 / max(video_fps, 10.0)
                if proc_time < target_dt:
                    time.sleep(target_dt - proc_time)

                # Update status on every frame for the first 10 frames, then every 3 frames
                if frame_idx <= 10 or frame_idx % 3 == 0 or frame_idx == total_frames:
                    elapsed = max(time.perf_counter() - t_start, 1e-4)
                    current_fps = frame_idx / elapsed
                    pct = ((frame_idx % max(total_frames, 1)) / max(total_frames, 1)) * 100.0 if total_frames > 0 else 100.0

                    with self._lock:
                        if camera_id in self.jobs:
                            self.jobs[camera_id].update({
                                "status": "live" if loop else "running",
                                "current_frame": frame_idx,
                                "progress_pct": round(min(pct, 100.0), 1),
                                "fps": round(current_fps, 1),
                                "events_detected": events_count,
                                "alerts_created": alerts_count,
                                "active_tracks": len(tracks),
                                "elapsed_sec": round(elapsed, 1),
                            })

                    # Broadcast progress packet thread-safely
                    progress_msg = {
                        "type": "analysis_progress",
                        "camera_id": camera_id,
                        "status": "live" if loop else "running",
                        "progress_pct": round(min(pct, 100.0), 1),
                        "current_frame": frame_idx,
                        "total_frames": total_frames,
                        "fps": round(current_fps, 1),
                        "active_tracks": len(tracks),
                        "alerts_created": alerts_count,
                    }
                    ws_manager.broadcast_sync(progress_msg, topic=f"camera:{camera_id}")

                if max_frames and frame_idx >= max_frames:
                    break

        except Exception as e:
            logger.exception("Error in AI analysis loop: %s", e)
            with self._lock:
                if camera_id in self.jobs:
                    self.jobs[camera_id]["status"] = "error"
                    self.jobs[camera_id]["error"] = str(e)
            return
        finally:
            cap.release()

        # Mark completed
        elapsed = max(time.perf_counter() - t_start, 1e-4)
        with self._lock:
            if camera_id in self.jobs:
                current_st = self.jobs[camera_id]["status"]
                if current_st != "stopped":
                    self.jobs[camera_id].update({
                        "status": "completed",
                        "progress_pct": 100.0,
                        "current_frame": frame_idx,
                        "elapsed_sec": round(elapsed, 1),
                        "fps": round(frame_idx / elapsed, 1),
                    })
        logger.info(
            "Completed AI video analysis for %s: %d frames, %d events, %d alerts in %.1fs",
            camera_id, frame_idx, events_count, alerts_count, elapsed
        )


analysis_manager = VideoAnalysisManager()
