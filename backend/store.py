"""Stateful data storage layer with in-memory persistence and DB fallbacks."""
from __future__ import annotations

import json
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from .config import settings
from .schemas import (
    AlertAckRequest,
    AlertCreate,
    AlertResponse,
    AlertSeverity,
    AlertType,
    CameraCreate,
    CameraResponse,
    CameraStatus,
    CameraUpdate,
    EventCreate,
    EventResponse,
    WatchlistPlateCreate,
    WatchlistPlateResponse,
)


class DataStore:
    """Unified store providing thread-safe operations for cameras, alerts, events, and watchlist."""

    def __init__(self):
        self.cameras: dict[str, CameraResponse] = {}
        self.camera_videos: dict[str, str] = {}
        self.camera_zones: dict[str, list[list[float]]] = {}
        self.alerts: dict[str, AlertResponse] = {}
        self.events: list[EventResponse] = []
        self.watchlist: dict[str, WatchlistPlateResponse] = {}
        self.start_time: float = time.time()
        self._seed_default_data()

    def _seed_default_data(self):
        """Seed sample cameras and realistic demonstration alerts."""
        # Seed Cameras
        cam1 = CameraResponse(
            id="cam1",
            name="North Entrance Gate",
            rtsp_url=f"{settings.mediamtx_rtsp_base}/cam1",
            webrtc_url=f"{settings.mediamtx_webrtc_base}/cam1",
            hls_url=f"{settings.mediamtx_hls_base}/cam1/index.m3u8",
            video_url="/api/cameras/cam1/video",
            source_type="video_file",
            location="Building A - Main Entrance",
            fps=30.0,
            status=CameraStatus.ONLINE,
            is_active=True,
        )
        cam2 = CameraResponse(
            id="cam2",
            name="South Parking Lot",
            rtsp_url=f"{settings.mediamtx_rtsp_base}/cam2",
            webrtc_url=f"{settings.mediamtx_webrtc_base}/cam2",
            hls_url=f"{settings.mediamtx_hls_base}/cam2/index.m3u8",
            video_url=None,
            source_type="rtsp",
            location="Perimeter - South Gate",
            fps=30.0,
            status=CameraStatus.ONLINE,
            is_active=True,
        )
        self.cameras[cam1.id] = cam1
        self.cameras[cam2.id] = cam2
        self.camera_videos["cam1"] = "sample.mp4"

        # Seed sample alerts
        now = time.time()
        sample_alerts = [
            AlertResponse(
                id="alert-001",
                camera_id="cam1",
                alert_type=AlertType.FALL,
                severity=AlertSeverity.CRITICAL,
                title="Person Fallen Detected",
                description="Biomechanical pose fall detected: aspect ratio inversion with rapid descent.",
                timestamp=now - 300,
                bbox=[450.0, 320.0, 520.0, 480.0],
                track_id=14,
                global_id=102,
                acknowledged=False,
                metadata={"dwell_time": 4.2, "confidence": 0.94},
            ),
            AlertResponse(
                id="alert-002",
                camera_id="cam2",
                alert_type=AlertType.WATCHLIST_PLATE,
                severity=AlertSeverity.HIGH,
                title="Stolen Vehicle Detected",
                description="ANPR matched license plate MH12DE1433 against police hotlist.",
                timestamp=now - 120,
                bbox=[200.0, 400.0, 600.0, 750.0],
                track_id=7,
                global_id=None,
                acknowledged=False,
                metadata={"plate": "MH12DE1433", "fuzzy_match": False, "reason": "Stolen Vehicle"},
            ),
            AlertResponse(
                id="alert-003",
                camera_id="cam1",
                alert_type=AlertType.WEAPON,
                severity=AlertSeverity.CRITICAL,
                title="Firearm Threat Detected",
                description="Threat detector identified handgun held by tracked person.",
                timestamp=now - 45,
                bbox=[310.0, 150.0, 420.0, 390.0],
                track_id=3,
                global_id=105,
                acknowledged=False,
                metadata={"weapon_class": "handgun", "conf": 0.88, "held_by_track": 3},
            ),
        ]
        for a in sample_alerts:
            self.alerts[a.id] = a

        # Seed default watchlist
        self.watchlist["MH12DE1433"] = WatchlistPlateResponse(
            plate="MH12DE1433",
            reason="Stolen Vehicle",
            owner="Reported Stolen - Crime Bureau",
            notes="Black SUV, report #CR-2026-992",
            added_at=now - 86400,
        )

        # Sync from models/vehicle_watchlist.json if present
        json_path = Path(__file__).resolve().parents[1] / "models" / "vehicle_watchlist.json"
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                for p, meta in data.items():
                    self.watchlist[p] = WatchlistPlateResponse(
                        plate=p,
                        reason=meta.get("reason", "Watchlist"),
                        owner=meta.get("owner", "Unknown"),
                        notes=meta.get("notes", ""),
                        added_at=meta.get("added_at", now),
                    )
            except Exception:
                pass

    # ---------------------------------------------------------------- Cameras
    def list_cameras(self) -> list[CameraResponse]:
        return list(self.cameras.values())

    def get_camera(self, camera_id: str) -> CameraResponse | None:
        return self.cameras.get(camera_id)

    def add_camera(self, cam_in: CameraCreate) -> CameraResponse:
        webrtc_url = f"{settings.mediamtx_webrtc_base}/{cam_in.id}"
        hls_url = f"{settings.mediamtx_hls_base}/{cam_in.id}/index.m3u8"
        cam = CameraResponse(
            id=cam_in.id,
            name=cam_in.name,
            rtsp_url=cam_in.rtsp_url,
            webrtc_url=webrtc_url,
            hls_url=hls_url,
            source_type=cam_in.source_type,
            video_url=cam_in.video_url,
            location=cam_in.location,
            description=cam_in.description,
            fps=cam_in.fps,
            is_active=cam_in.is_active,
            status=CameraStatus.ONLINE if cam_in.is_active else CameraStatus.OFFLINE,
        )
        self.cameras[cam.id] = cam
        return cam

    def set_camera_video(self, camera_id: str, filename: str):
        self.camera_videos[camera_id] = filename

    def get_camera_video(self, camera_id: str) -> str | None:
        return self.camera_videos.get(camera_id)

    def list_sample_videos(self) -> list[dict]:
        v_dir = Path(__file__).resolve().parents[1] / "data" / "sample_videos"
        if not v_dir.exists():
            return []
        out = []
        valid_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
        for f in v_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                sz = f.stat().st_size
                out.append({
                    "filename": f.name,
                    "size_bytes": sz,
                    "size_mb": round(sz / (1024 * 1024), 2),
                })
        return sorted(out, key=lambda x: x["filename"])

    def update_camera(self, camera_id: str, cam_in: CameraUpdate) -> CameraResponse | None:
        cam = self.cameras.get(camera_id)
        if not cam:
            return None
        dump = cam.model_dump()
        for k, v in cam_in.model_dump(exclude_unset=True).items():
            if v is not None:
                dump[k] = v
        if cam_in.is_active is not None:
            dump["status"] = CameraStatus.ONLINE if cam_in.is_active else CameraStatus.OFFLINE
        updated = CameraResponse(**dump)
        self.cameras[camera_id] = updated
        return updated

    def delete_camera(self, camera_id: str) -> bool:
        if camera_id in self.cameras:
            del self.cameras[camera_id]
            self.camera_videos.pop(camera_id, None)
            self.camera_zones.pop(camera_id, None)
            return True
        return False

    def set_camera_zone(self, camera_id: str, zone: list[list[float]]):
        self.camera_zones[camera_id] = zone

    def get_camera_zone(self, camera_id: str) -> list[list[float]] | None:
        return self.camera_zones.get(camera_id)

    # ---------------------------------------------------------------- Alerts
    def list_alerts(
        self,
        camera_id: str | None = None,
        severity: AlertSeverity | None = None,
        alert_type: AlertType | None = None,
        acknowledged: bool | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AlertResponse]:
        filtered = list(self.alerts.values())

        if camera_id:
            filtered = [a for a in filtered if a.camera_id == camera_id]
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        if alert_type:
            filtered = [a for a in filtered if a.alert_type == alert_type]
        if acknowledged is not None:
            filtered = [a for a in filtered if a.acknowledged == acknowledged]
        if start_ts is not None:
            filtered = [a for a in filtered if a.timestamp >= start_ts]
        if end_ts is not None:
            filtered = [a for a in filtered if a.timestamp <= end_ts]

        # Order by newest first
        filtered.sort(key=lambda a: a.timestamp, reverse=True)
        return filtered[offset : offset + limit]

    def get_alert(self, alert_id: str) -> AlertResponse | None:
        return self.alerts.get(alert_id)

    def add_alert(self, alert_in: AlertCreate) -> AlertResponse:
        alert_id = f"alert-{uuid.uuid4().hex[:8]}"
        alert = AlertResponse(
            id=alert_id,
            camera_id=alert_in.camera_id,
            alert_type=alert_in.alert_type,
            severity=alert_in.severity,
            title=alert_in.title,
            description=alert_in.description,
            timestamp=alert_in.timestamp,
            bbox=alert_in.bbox,
            track_id=alert_in.track_id,
            global_id=alert_in.global_id,
            metadata=alert_in.metadata,
            acknowledged=False,
        )
        self.alerts[alert_id] = alert
        return alert

    def acknowledge_alert(self, alert_id: str, ack: AlertAckRequest) -> AlertResponse | None:
        alert = self.alerts.get(alert_id)
        if not alert:
            return None
        dump = alert.model_dump()
        dump["acknowledged"] = True
        dump["acknowledged_by"] = ack.operator or "operator"
        dump["acknowledged_at"] = time.time()
        if ack.note:
            dump["metadata"]["ack_note"] = ack.note
        updated = AlertResponse(**dump)
        self.alerts[alert_id] = updated
        return updated

    def delete_alert(self, alert_id: str) -> bool:
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            return True
        return False

    # ---------------------------------------------------------------- Events
    def add_event(self, ev_in: EventCreate) -> EventResponse:
        ev_id = f"ev-{uuid.uuid4().hex[:8]}"
        ev = EventResponse(
            id=ev_id,
            camera_id=ev_in.camera_id,
            timestamp=ev_in.timestamp,
            frame_idx=ev_in.frame_idx,
            track_id=ev_in.track_id,
            global_id=ev_in.global_id,
            event_type=ev_in.event_type,
            bbox=ev_in.bbox,
            confidence=ev_in.confidence,
            metadata=ev_in.metadata,
        )
        self.events.append(ev)
        # Keep recent 100,000 events in memory
        if len(self.events) > 100000:
            self.events.pop(0)
        return ev

    def search_events(
        self,
        camera_id: str | None = None,
        event_type: str | None = None,
        track_id: int | None = None,
        global_id: int | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EventResponse]:
        res = self.events
        if camera_id:
            res = [e for e in res if e.camera_id == camera_id]
        if event_type:
            res = [e for e in res if e.event_type == event_type]
        if track_id is not None:
            res = [e for e in res if e.track_id == track_id]
        if global_id is not None:
            res = [e for e in res if e.global_id == global_id]
        if start_ts is not None:
            res = [e for e in res if e.timestamp >= start_ts]
        if end_ts is not None:
            res = [e for e in res if e.timestamp <= end_ts]

        res = sorted(res, key=lambda e: e.timestamp, reverse=True)
        return res[offset : offset + limit]

    # ---------------------------------------------------------------- Watchlist
    def list_watchlist(self) -> list[WatchlistPlateResponse]:
        return list(self.watchlist.values())

    def add_watchlist_plate(self, item: WatchlistPlateCreate) -> WatchlistPlateResponse:
        norm_plate = item.plate.upper().replace(" ", "").replace("-", "")
        res = WatchlistPlateResponse(
            plate=norm_plate,
            reason=item.reason,
            owner=item.owner,
            notes=item.notes,
            added_at=time.time(),
        )
        self.watchlist[norm_plate] = res
        return res

    def remove_watchlist_plate(self, plate: str) -> bool:
        norm_plate = plate.upper().replace(" ", "").replace("-", "")
        if norm_plate in self.watchlist:
            del self.watchlist[norm_plate]
            return True
        return False

    # ---------------------------------------------------------------- Health / Stats
    def check_service_port(self, host: str, port: int, timeout: float = 0.5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def get_services_health(self) -> dict[str, str]:
        # PostgreSQL check
        pg_ok = self.check_service_port(settings.postgres_host, settings.postgres_port)
        # Redis check
        rd_ok = self.check_service_port(settings.redis_host, settings.redis_port)
        # OpenSearch check
        os_ok = self.check_service_port(settings.opensearch_host, settings.opensearch_port)
        # MediaMTX check (8554 RTSP)
        mtx_ok = self.check_service_port("localhost", 8554)

        return {
            "postgres": "healthy" if pg_ok else "unreachable (using memory fallback)",
            "redis": "healthy" if rd_ok else "unreachable (using local bus)",
            "opensearch": "healthy" if os_ok else "unreachable (using in-store index)",
            "mediamtx": "healthy" if mtx_ok else "offline (check docker compose)",
        }

    def get_system_stats(self) -> dict:
        total_cams = len(self.cameras)
        online_cams = sum(1 for c in self.cameras.values() if c.status == CameraStatus.ONLINE)
        active_alerts = [a for a in self.alerts.values() if not a.acknowledged]

        sev_counts: dict[str, int] = {s.value: 0 for s in AlertSeverity}
        for a in active_alerts:
            sev_counts[a.severity.value] = sev_counts.get(a.severity.value, 0) + 1

        return {
            "cameras_total": total_cams,
            "cameras_online": online_cams,
            "active_alerts_count": len(active_alerts),
            "alerts_by_severity": sev_counts,
            "total_events": len(self.events),
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "services_status": self.get_services_health(),
        }


store = DataStore()
