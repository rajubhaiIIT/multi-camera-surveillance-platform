"""Application configuration and environment settings."""
from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


def _load_env_file() -> dict[str, str]:
    """Parse .env file from project root if present."""
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    out: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


_FILE_ENV = _load_env_file()


def _get(key: str, default: str = "") -> str:
    """Read env with priority: OS environment > .env file > default."""
    return os.environ.get(key, _FILE_ENV.get(key, default))


class Settings(BaseModel):
    # App
    app_name: str = "Intelligent Multi-Camera Surveillance Platform API"
    app_version: str = "1.0.0"
    debug: bool = _get("DEBUG", "false").lower() in ("1", "true", "yes")

    # Host & Port
    api_host: str = _get("API_HOST", "0.0.0.0")
    api_port: int = int(_get("API_PORT", "8000"))

    # Security & JWT
    jwt_secret_key: str = _get("JWT_SECRET_KEY", "surveillance-secret-key-32-chars-minimum-token-key-2026")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = int(_get("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24)))

    # Initial admin credentials
    admin_user: str = _get("ADMIN_USER", "admin")
    admin_password: str = _get("ADMIN_PASSWORD", "admin123")

    # PostgreSQL + pgvector
    postgres_host: str = _get("POSTGRES_HOST", "localhost")
    postgres_port: int = int(_get("POSTGRES_PORT", "5432"))
    postgres_user: str = _get("POSTGRES_USER", "surveillance")
    postgres_password: str = _get("POSTGRES_PASSWORD", "")
    postgres_db: str = _get("POSTGRES_DB", "surveillance")

    # Redis
    redis_host: str = _get("REDIS_HOST", "localhost")
    redis_port: int = int(_get("REDIS_PORT", "6379"))

    # OpenSearch
    opensearch_host: str = _get("OPENSEARCH_HOST", "localhost")
    opensearch_port: int = int(_get("OPENSEARCH_PORT", "9200"))
    opensearch_index_events: str = "surveillance-events"
    opensearch_index_alerts: str = "surveillance-alerts"

    # MediaMTX Streaming
    mediamtx_rtsp_base: str = _get("MEDIAMTX_RTSP_BASE", "rtsp://localhost:8554")
    mediamtx_webrtc_base: str = _get("MEDIAMTX_WEBRTC_BASE", "http://localhost:8889")
    mediamtx_hls_base: str = _get("MEDIAMTX_HLS_BASE", "http://localhost:8888")

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "*"
    ]

    def postgres_conninfo(self) -> dict:
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "user": self.postgres_user,
            "password": self.postgres_password,
            "dbname": self.postgres_db,
        }


settings = Settings()
