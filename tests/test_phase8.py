"""Phase 8 unit tests: React Operator Dashboard structure, build assets, and backend API contracts.

Tests run offline with zero external network requirements.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def operator_token(client):
    res = client.post("/api/auth/token", data={"username": "operator", "password": "operator123"})
    assert res.status_code == 200
    return res.json()["access_token"]


# ---------------------------------------------------------------- Frontend Asset Tests
def test_frontend_package_configuration():
    pkg_file = FRONTEND_DIR / "package.json"
    assert pkg_file.exists(), "frontend/package.json must exist"
    data = json.loads(pkg_file.read_text(encoding="utf-8"))
    assert "dependencies" in data
    for dep in ("react", "react-dom", "react-router-dom", "lucide-react"):
        assert dep in data["dependencies"], f"Missing core dependency {dep}"
    assert "vite" in data["devDependencies"], "Missing vite in devDependencies"


def test_vite_proxy_configuration():
    vite_cfg = FRONTEND_DIR / "vite.config.js"
    assert vite_cfg.exists(), "frontend/vite.config.js must exist"
    content = vite_cfg.read_text(encoding="utf-8")
    assert "/api" in content, "Vite must proxy /api to backend"
    assert "/ws" in content, "Vite must proxy /ws to backend"
    assert "8000" in content, "Vite proxy must target backend port 8000"


def test_frontend_source_files_exist():
    src_dir = FRONTEND_DIR / "src"
    expected_files = [
        "main.jsx",
        "App.jsx",
        "Layout.jsx",
        "api.js",
        "useWebSocket.js",
        "index.css",
        "pages/LoginPage.jsx",
        "pages/DashboardPage.jsx",
        "pages/CamerasPage.jsx",
        "pages/AlertsPage.jsx",
        "pages/EventsPage.jsx",
        "pages/WatchlistPage.jsx",
    ]
    for rel_path in expected_files:
        p = src_dir / rel_path
        assert p.exists(), f"Frontend source file missing: {rel_path}"
        assert p.stat().st_size > 50, f"File {rel_path} appears empty"


def test_production_build_artifacts():
    dist_dir = FRONTEND_DIR / "dist"
    index_html = dist_dir / "index.html"
    assert index_html.exists(), "frontend/dist/index.html must exist from build"
    html_text = index_html.read_text(encoding="utf-8")
    assert "<div id=\"root\"></div>" in html_text
    assert "assets/index-" in html_text

    assets_dir = dist_dir / "assets"
    assert assets_dir.exists(), "frontend/dist/assets must exist"
    files = [f.name for f in assets_dir.iterdir()]
    assert any(f.endswith(".js") for f in files), "Built JavaScript bundle missing"
    assert any(f.endswith(".css") for f in files), "Built CSS bundle missing"


# ---------------------------------------------------------------- Frontend Contract Tests
def test_dashboard_api_contracts(client, operator_token):
    headers = {"Authorization": f"Bearer {operator_token}"}

    # Stats endpoint matches frontend Expectations
    res_stats = client.get("/api/stats", headers=headers)
    assert res_stats.status_code == 200
    st = res_stats.json()
    assert "cameras_total" in st
    assert "cameras_online" in st
    assert "active_alerts_count" in st
    assert "alerts_by_severity" in st
    assert "uptime_seconds" in st
    assert "services_status" in st

    # Cameras endpoint matches frontend Expectations
    res_cams = client.get("/api/cameras", headers=headers)
    assert res_cams.status_code == 200
    cams = res_cams.json()
    assert len(cams) >= 2
    assert "rtsp_url" in cams[0]
    assert "status" in cams[0]

    # Alerts endpoint matches frontend Expectations
    res_alerts = client.get("/api/alerts?limit=5&acknowledged=false", headers=headers)
    assert res_alerts.status_code == 200
    alerts = res_alerts.json()
    assert isinstance(alerts, list)
    if alerts:
        a = alerts[0]
        assert "camera_id" in a
        assert "severity" in a
        assert "title" in a
        assert "acknowledged" in a

    # Watchlist endpoint matches frontend Expectations
    res_wl = client.get("/api/vehicles/watchlist", headers=headers)
    assert res_wl.status_code == 200
    wl = res_wl.json()
    assert isinstance(wl, list)
    assert any(p["plate"] == "MH12DE1433" for p in wl)


def test_websocket_pubsub_contract(client):
    with client.websocket_connect("/ws/live") as ws:
        # Subscribe message
        ws.send_text(json.dumps({"action": "subscribe", "topic": "alerts"}))
        resp = ws.receive_text()
        data = json.loads(resp)
        assert data.get("status") == "subscribed"
        assert data.get("topic") == "alerts"


def test_video_upload_and_streaming(client, operator_token):
    headers = {"Authorization": f"Bearer {operator_token}"}

    # 1. Available videos on server
    res_av = client.get("/api/cameras/available_videos", headers=headers)
    assert res_av.status_code == 200
    vids = res_av.json()
    assert any(v["filename"] == "sample.mp4" for v in vids)

    # 2. Upload video sample directly as a camera feed
    dummy_video_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free"
    files = {"file": ("test_gate.mp4", dummy_video_bytes, "video/mp4")}
    data = {
        "camera_id": "cam_upload_test",
        "name": "Gate Upload Test",
        "location": "North Wing",
        "fps": 30.0,
    }
    res_up = client.post("/api/cameras/upload", files=files, data=data, headers=headers)
    assert res_up.status_code == 200
    cam_data = res_up.json()
    assert cam_data["id"] == "cam_upload_test"
    assert cam_data["source_type"] == "video_file"
    assert cam_data["video_url"] == "/api/cameras/cam_upload_test/video"

    # 3. Stream uploaded video
    res_stream = client.get("/api/cameras/cam_upload_test/video")
    assert res_stream.status_code in (200, 206)
    assert "video/mp4" in res_stream.headers["content-type"]
    assert res_stream.content == dummy_video_bytes

    # 4. Attach existing video to another camera
    res_attach = client.post(
        "/api/cameras/cam2/attach_video",
        json={"filename": "sample.mp4"},
        headers=headers,
    )
    assert res_attach.status_code == 200
    assert res_attach.json()["source_type"] == "video_file"
    assert res_attach.json()["video_url"] == "/api/cameras/cam2/video"

    # Clean up test camera
    client.delete("/api/cameras/cam_upload_test", headers={"Authorization": f"Bearer {operator_token}"})

