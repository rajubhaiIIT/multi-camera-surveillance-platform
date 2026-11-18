"""Pydantic data schemas and contracts for the FastAPI Backend."""
from __future__ import annotations

import time
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------- Auth Schemas
class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: UserRole
    username: str
    expires_in: int


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: UserRole = UserRole.OPERATOR
    full_name: str | None = None
    email: str | None = None


class UserResponse(BaseModel):
    username: str
    role: UserRole
    full_name: str | None = None
    email: str | None = None
    is_active: bool = True
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------- Camera Schemas
class CameraStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"


class CameraBase(BaseModel):
    name: str
    rtsp_url: str = ""
    location: str = "Main Campus"
    description: str | None = None
    fps: float = 30.0
    is_active: bool = True
    source_type: str = "rtsp"  # "rtsp" or "video_file"
    video_url: str | None = None


class CameraCreate(CameraBase):
    id: str = Field(..., description="Unique camera identifier, e.g. cam1, entrance_gate")


class CameraUpdate(BaseModel):
    name: str | None = None
    rtsp_url: str | None = None
    location: str | None = None
    description: str | None = None
    fps: float | None = None
    is_active: bool | None = None
    source_type: str | None = None
    video_url: str | None = None


class CameraResponse(CameraBase):
    id: str
    status: CameraStatus = CameraStatus.ONLINE
    webrtc_url: str | None = None
    hls_url: str | None = None
    video_url: str | None = None
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------- Alert Schemas
class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    FALL = "fall"
    FIGHT = "fight"
    WEAPON = "weapon"
    WATCHLIST_PLATE = "watchlist_plate"
    LOITERING = "loitering"
    PERIMETER = "perimeter"
    ANOMALY = "anomaly"


class AlertCreate(BaseModel):
    camera_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float = Field(default_factory=time.time)
    bbox: list[float] | None = None
    track_id: int | None = None
    global_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlertAckRequest(BaseModel):
    note: str | None = None
    operator: str | None = None


class AlertResponse(AlertCreate):
    id: str
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: float | None = None
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------- Event Schemas
class EventCreate(BaseModel):
    camera_id: str
    timestamp: float = Field(default_factory=time.time)
    frame_idx: int = 0
    track_id: int
    global_id: int | None = None
    event_type: str = "track"  # e.g., "track", "face", "pose", "weapon", "vehicle"
    bbox: list[float] = Field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventResponse(EventCreate):
    id: str
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------- Watchlist Schemas
class WatchlistPlateCreate(BaseModel):
    plate: str
    reason: str = "Flagged Vehicle"
    owner: str = "Unknown"
    notes: str = ""


class WatchlistPlateResponse(WatchlistPlateCreate):
    added_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------- Stats & Health
class SystemStatsResponse(BaseModel):
    cameras_total: int
    cameras_online: int
    active_alerts_count: int
    alerts_by_severity: dict[str, int]
    total_events: int
    uptime_seconds: float
    services_status: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    timestamp: float = Field(default_factory=time.time)
    version: str
    services: dict[str, str]
