"""Phase 7 unit tests: FastAPI Backend, JWT auth, RBAC, cameras, alerts, events, WebSocket, and OpenSearch fallback.

All tests run offline using Starlette/FastAPI TestClient with zero external infrastructure required.
"""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import AlertSeverity, AlertType, UserRole
from backend.store import store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def admin_token(client):
    res = client.post("/api/auth/token", data={"username": "admin", "password": "admin123"})
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


@pytest.fixture(scope="module")
def operator_token(client):
    res = client.post("/api/auth/token", data={"username": "operator", "password": "operator123"})
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


@pytest.fixture(scope="module")
def viewer_token(client):
    res = client.post("/api/auth/token", data={"username": "viewer", "password": "viewer123"})
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


# ---------------------------------------------------------------- Root & Health Tests
def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "docs_url" in data


def test_health_check_endpoints(client):
    for path in ("/healthz", "/api/health"):
        res = client.get(path)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "postgres" in data["services"]
        assert "opensearch" in data["services"]


# ---------------------------------------------------------------- Auth & RBAC Tests
def test_login_success_and_failure(client):
    # Success via JSON
    res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["role"] == UserRole.ADMIN

    # Failure
    res_bad = client.post("/api/auth/login", json={"username": "admin", "password": "wrongpassword"})
    assert res_bad.status_code == 401


def test_auth_me_endpoint(client, admin_token, viewer_token):
    # With admin token
    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {admin_token}"})
    assert res.status_code == 200
    assert res.json()["username"] == "admin"
    assert res.json()["role"] == "admin"

    # With viewer token
    res_v = client.get("/api/auth/me", headers={"Authorization": f"Bearer {viewer_token}"})
    assert res_v.status_code == 200
    assert res_v.json()["username"] == "viewer"
    assert res_v.json()["role"] == "viewer"

    # Unauthorized without token
    res_no = client.get("/api/auth/me")
    assert res_no.status_code == 401


def test_admin_rbac_user_management(client, admin_token, viewer_token):
    # Viewer cannot list users
    res_v = client.get("/api/auth/users", headers={"Authorization": f"Bearer {viewer_token}"})
    assert res_v.status_code == 403

    # Admin can list users
    res_a = client.get("/api/auth/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_a.status_code == 200
    users = res_a.json()
    assert any(u["username"] == "admin" for u in users)

    # Admin can create a new user
    new_user_payload = {
        "username": "patrol1",
        "password": "patrolpassword",
        "role": "operator",
        "full_name": "Patrol Officer 1",
    }
    res_create = client.post("/api/auth/users", json=new_user_payload, headers={"Authorization": f"Bearer {admin_token}"})
    assert res_create.status_code == 200
    assert res_create.json()["username"] == "patrol1"


# ---------------------------------------------------------------- Camera Registry Tests
def test_camera_crud_and_streams(client, admin_token, viewer_token):
    headers_admin = {"Authorization": f"Bearer {admin_token}"}
    headers_viewer = {"Authorization": f"Bearer {viewer_token}"}

    # List cameras
    res = client.get("/api/cameras", headers=headers_viewer)
    assert res.status_code == 200
    cameras = res.json()
    assert len(cameras) >= 2
    assert any(c["id"] == "cam1" for c in cameras)

    # Create camera (viewer denied, admin allowed)
    new_cam = {
        "id": "cam_east_gate",
        "name": "East Perimeter Gate",
        "rtsp_url": "rtsp://localhost:8554/cam_east_gate",
        "location": "Perimeter East",
        "fps": 25.0,
        "is_active": True,
    }
    res_deny = client.post("/api/cameras", json=new_cam, headers=headers_viewer)
    assert res_deny.status_code == 403

    res_ok = client.post("/api/cameras", json=new_cam, headers=headers_admin)
    assert res_ok.status_code == 200
    assert res_ok.json()["id"] == "cam_east_gate"

    # Get camera details
    res_get = client.get("/api/cameras/cam_east_gate", headers=headers_viewer)
    assert res_get.status_code == 200
    assert res_get.json()["location"] == "Perimeter East"

    # Streams URL
    res_stream = client.get("/api/cameras/cam_east_gate/streams", headers=headers_viewer)
    assert res_stream.status_code == 200
    streams = res_stream.json()
    assert "rtsp" in streams
    assert "webrtc" in streams
    assert "hls" in streams

    # Update camera
    res_up = client.put("/api/cameras/cam_east_gate", json={"location": "East Sector 4"}, headers=headers_admin)
    assert res_up.status_code == 200
    assert res_up.json()["location"] == "East Sector 4"

    # Delete camera
    res_del = client.delete("/api/cameras/cam_east_gate", headers=headers_admin)
    assert res_del.status_code == 204


# ---------------------------------------------------------------- Alerts Lifecycle Tests
def test_alerts_lifecycle(client, admin_token, operator_token):
    headers = {"Authorization": f"Bearer {operator_token}"}

    # Ingest new alert
    alert_payload = {
        "camera_id": "cam1",
        "alert_type": "fight",
        "severity": "high",
        "title": "Violent Struggle Detected",
        "description": "Pairwise kinetic violence classifier triggered: mutual high velocity interaction.",
        "bbox": [100.0, 150.0, 300.0, 450.0],
        "track_id": 9,
        "metadata": {"violence_score": 0.89},
    }
    res_post = client.post("/api/alerts", json=alert_payload, headers=headers)
    assert res_post.status_code == 200
    created = res_post.json()
    alert_id = created["id"]
    assert created["acknowledged"] is False

    # List alerts with filter
    res_filter = client.get("/api/alerts?alert_type=fight&severity=high", headers=headers)
    assert res_filter.status_code == 200
    results = res_filter.json()
    assert any(a["id"] == alert_id for a in results)

    # Acknowledge alert
    ack_payload = {"note": "Dispatched security officer to sector 1."}
    res_ack = client.post(f"/api/alerts/{alert_id}/ack", json=ack_payload, headers=headers)
    assert res_ack.status_code == 200
    ack_data = res_ack.json()
    assert ack_data["acknowledged"] is True
    assert ack_data["acknowledged_by"] == "operator"
    assert ack_data["metadata"]["ack_note"] == ack_payload["note"]


# ---------------------------------------------------------------- Events & Search Tests
def test_events_ingestion_and_search(client, operator_token):
    headers = {"Authorization": f"Bearer {operator_token}"}

    # Ingest event
    ev_payload = {
        "camera_id": "cam2",
        "frame_idx": 45,
        "track_id": 12,
        "global_id": 501,
        "event_type": "track",
        "bbox": [50.0, 60.0, 120.0, 200.0],
        "confidence": 0.95,
        "metadata": {"reid_dist": 0.12},
    }
    res = client.post("/api/events", json=ev_payload, headers=headers)
    assert res.status_code == 200
    ev_id = res.json()["id"]

    # Search event
    res_search = client.get("/api/events?camera_id=cam2&event_type=track", headers=headers)
    assert res_search.status_code == 200
    found = res_search.json()
    assert any(e["id"] == ev_id for e in found)


# ---------------------------------------------------------------- Vehicle Watchlist Tests
def test_vehicle_watchlist_crud(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Add plate
    plate_data = {
        "plate": "DL01AB9999",
        "reason": "Test Vehicle Hotlist",
        "owner": "Test Owner",
        "notes": "Silver sedan",
    }
    res_add = client.post("/api/vehicles/watchlist", json=plate_data, headers=headers)
    assert res_add.status_code == 200
    assert res_add.json()["plate"] == "DL01AB9999"

    # List watchlist
    res_list = client.get("/api/vehicles/watchlist", headers=headers)
    assert res_list.status_code == 200
    assert any(p["plate"] == "DL01AB9999" for p in res_list.json())

    # Delete plate
    res_del = client.delete("/api/vehicles/watchlist/DL01AB9999", headers=headers)
    assert res_del.status_code == 204


# ---------------------------------------------------------------- System Stats Tests
def test_system_stats(client, viewer_token):
    res = client.get("/api/stats", headers={"Authorization": f"Bearer {viewer_token}"})
    assert res.status_code == 200
    stats = res.json()
    assert stats["cameras_total"] >= 2
    assert "alerts_by_severity" in stats
    assert "services_status" in stats


# ---------------------------------------------------------------- WebSocket Streaming Tests
def test_websocket_live_connection_and_ping(client):
    with client.websocket_connect("/ws/live") as websocket:
        # Ping / pong test
        websocket.send_text(json.dumps({"action": "ping"}))
        resp = websocket.receive_text()
        data = json.loads(resp)
        assert data.get("type") == "pong"

        # Topic subscription
        websocket.send_text(json.dumps({"action": "subscribe", "topic": "camera:cam1"}))
        resp2 = websocket.receive_text()
        data2 = json.loads(resp2)
        assert data2.get("status") == "subscribed"
        assert data2.get("topic") == "camera:cam1"


def test_websocket_alerts_broadcast(client):
    with client.websocket_connect("/ws/alerts") as ws:
        # Ping
        ws.send_text(json.dumps({"action": "ping"}))
        resp = ws.receive_text()
        assert json.loads(resp).get("type") == "pong"
