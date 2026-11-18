"""FastAPI Application Entrypoint for the Intelligent Multi-Camera Surveillance Platform."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .ai_worker import analysis_manager
from .config import settings
from .routes import alerts, auth, cameras, events, stats, vehicles, ws
from .schemas import HealthResponse
from .store import store

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("surveillance.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup banner
    logger.info("=================================================================")
    logger.info("  %s v%s", settings.app_name, settings.app_version)
    logger.info("  Documentation: http://%s:%d/docs", settings.api_host, settings.api_port)
    logger.info("  Services Status: %s", store.get_services_health())
    logger.info("=================================================================")

    # Autonomous AI perception startup: automatically start live AI monitoring for active video feeds
    sample_videos_dir = Path(__file__).resolve().parents[1] / "data" / "sample_videos"
    for cam in store.list_cameras():
        v_name = store.get_camera_video(cam.id) or "sample.mp4"
        v_path = sample_videos_dir / v_name
        if v_path.exists():
            try:
                analysis_manager.start_analysis(cam.id, v_path, loop=True)
                logger.info("Autonomous AI live perception auto-started for camera %s (%s)", cam.id, v_name)
            except Exception as e:
                logger.warning("Could not auto-start AI for %s: %s", cam.id, e)

    yield

    # Shutdown: stop all active AI threads
    logger.info("Surveillance backend shutting down, halting AI workers...")
    analysis_manager.stop_all()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Production-grade REST and WebSocket API for real-time multi-camera tracking, "
        "biometric face recognition, pose fall/fight detection, weapon threats, "
        "vehicle tracking & ANPR plate recognition, and video enhancement."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for React / frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API and WebSocket routers
app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(alerts.router)
app.include_router(events.router)
app.include_router(vehicles.router)
app.include_router(stats.router)
app.include_router(ws.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "title": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "status": "online",
        "active_cameras": len(store.cameras),
        "alerts_count": len(store.alerts),
        "default_accounts": {
            "admin": "admin / admin123 (Full Control)",
            "operator": "operator / operator123 (Camera & Alert Ops)",
            "viewer": "viewer / viewer123 (Read Only)",
        },
    }


@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version=settings.app_version,
        services=store.get_services_health(),
    )


def main():
    """CLI runner: python -m backend.main"""
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
