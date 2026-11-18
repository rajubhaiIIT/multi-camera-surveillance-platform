"""Camera registry, stream management, and background AI video analysis routes."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated

import asyncio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..ai_worker import analysis_manager
from ..auth import get_current_user, require_role
from ..config import settings
from ..schemas import CameraCreate, CameraResponse, CameraUpdate, UserResponse, UserRole
from ..store import store

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


class AttachVideoRequest(BaseModel):
    filename: str


class ZoneUpdateRequest(BaseModel):
    zone: list[list[float]]


@router.get("", response_model=list[CameraResponse])
async def list_cameras(current_user: Annotated[UserResponse, Depends(get_current_user)]):
    return store.list_cameras()


@router.get("/available_videos")
async def list_available_videos(current_user: Annotated[UserResponse, Depends(get_current_user)]):
    """List sample videos existing in data/sample_videos."""
    return store.list_sample_videos()


@router.post("/upload", response_model=CameraResponse, dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))])
async def upload_camera_video(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    name: str = Form(...),
    location: str = Form("Main Campus"),
    fps: float = Form(30.0),
    auto_analyze: bool = Form(True),
):
    """Upload a video file from the browser and attach it to a new or existing camera."""
    upload_dir = Path(__file__).resolve().parents[2] / "data" / "sample_videos"
    upload_dir.mkdir(parents=True, exist_ok=True)

    clean_name = Path(file.filename or f"{camera_id}.mp4").name
    dest_path = upload_dir / clean_name
    with open(dest_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    store.set_camera_video(camera_id, clean_name)
    existing = store.get_camera(camera_id)
    if existing:
        updated = store.update_camera(camera_id, CameraUpdate(
            name=name,
            location=location,
            fps=fps,
            source_type="video_file",
            video_url=f"/api/cameras/{camera_id}/video",
            is_active=True,
        ))
        res_camera = updated
    else:
        cam_in = CameraCreate(
            id=camera_id,
            name=name,
            location=location,
            fps=fps,
            source_type="video_file",
            video_url=f"/api/cameras/{camera_id}/video",
            rtsp_url=f"{settings.mediamtx_rtsp_base}/{camera_id}",
            is_active=True,
        )
        res_camera = store.add_camera(cam_in)

    # Automatically trigger continuous background AI analysis on the uploaded video
    if auto_analyze:
        try:
            analysis_manager.start_analysis(camera_id, dest_path, loop=True)
        except Exception:
            pass

    return res_camera


@router.post("/{camera_id}/attach_video", response_model=CameraResponse, dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))])
async def attach_video(camera_id: str, payload: AttachVideoRequest):
    """Attach an existing video from data/sample_videos to a camera and start AI monitoring."""
    cam = store.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found")
    v_dir = Path(__file__).resolve().parents[2] / "data" / "sample_videos"
    file_path = v_dir / payload.filename
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video file '{payload.filename}' not found")

    store.set_camera_video(camera_id, payload.filename)
    updated = store.update_camera(camera_id, CameraUpdate(
        source_type="video_file",
        video_url=f"/api/cameras/{camera_id}/video",
        is_active=True,
    ))
    try:
        analysis_manager.start_analysis(camera_id, file_path, loop=True)
    except Exception:
        pass
    return updated


@router.get("/{camera_id}/zone")
async def get_camera_zone(camera_id: str):
    """Get the active user-defined plate detection monitoring zone for this camera."""
    zone = store.get_camera_zone(camera_id)
    if not zone:
        # Default road trapezoid: top-left, top-right, bottom-right, bottom-left (normalized 0.0-1.0)
        zone = [[0.05, 0.44], [0.95, 0.44], [0.98, 0.56], [0.02, 0.56]]
    return {"camera_id": camera_id, "zone": zone}


@router.post("/{camera_id}/zone")
async def set_camera_zone(camera_id: str, payload: ZoneUpdateRequest):
    """Update user-defined plate detection monitoring zone."""
    store.set_camera_zone(camera_id, payload.zone)
    analysis_manager.set_monitoring_zone(camera_id, payload.zone)
    return {"camera_id": camera_id, "zone": payload.zone, "status": "updated"}


@router.post("/{camera_id}/analyze", dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))])
async def trigger_ai_analysis(
    camera_id: str,
    max_frames: int | None = None,
    mode: str | None = None,
    current_user: Annotated[UserResponse, Depends(get_current_user)] = None,
):
    """Trigger background AI inference (YOLO tracking, fall/weapons or ANPR traffic) on camera feed."""
    cam = store.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found")

    resolved_mode = mode or getattr(cam, "camera_type", None) or ("traffic" if "traffic" in camera_id.lower() else "general")

    filename = store.get_camera_video(camera_id)
    v_dir = Path(__file__).resolve().parents[2] / "data" / "sample_videos"
    if not filename:
        fallback_vid = "final_indian_plates.mp4" if resolved_mode == "traffic" else "sample.mp4"
        if (v_dir / fallback_vid).exists():
            filename = fallback_vid
            store.set_camera_video(camera_id, filename)
        elif (v_dir / "sample.mp4").exists():
            filename = "sample.mp4"
            store.set_camera_video(camera_id, filename)
        else:
            raise HTTPException(status_code=400, detail="No video file attached to camera")

    file_path = v_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found")

    job = analysis_manager.start_analysis(camera_id, file_path, mode=resolved_mode, max_frames=max_frames)
    return job


@router.get("/{camera_id}/analysis_status")
async def get_ai_analysis_status(
    camera_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Get active AI analysis status, progress %, and metrics."""
    job = analysis_manager.get_job_status(camera_id)
    if not job:
        return {"camera_id": camera_id, "status": "idle"}
    return job


@router.post("/{camera_id}/stop_analysis", dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))])
async def stop_ai_analysis(
    camera_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_user)] = None,
):
    """Stop running AI analysis for a camera."""
    stopped = analysis_manager.stop_analysis(camera_id)
    return {"camera_id": camera_id, "stopped": stopped}


@router.get("/{camera_id}/video")
async def stream_camera_video(camera_id: str):
    """Stream video file for HTML5 in-browser playback with Range request support."""
    filename = store.get_camera_video(camera_id)
    v_dir = Path(__file__).resolve().parents[2] / "data" / "sample_videos"

    if not filename:
        if (v_dir / "sample.mp4").exists():
            filename = "sample.mp4"
            store.set_camera_video(camera_id, filename)
        else:
            raise HTTPException(status_code=404, detail="No video file associated with this camera")

    file_path = v_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found on server")

    ext = file_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    media_type = media_types.get(ext, "video/mp4")
    return FileResponse(file_path, media_type=media_type)


@router.get("/{camera_id}/stream")
async def stream_annotated_feed(camera_id: str):
    """Stream real-time MJPEG video with AI bounding boxes, Person IDs, and keypoint skeletons."""
    cam = store.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Auto-start AI analysis if not already running
    if not analysis_manager.is_running(camera_id):
        filename = store.get_camera_video(camera_id) or "sample.mp4"
        v_path = Path(__file__).resolve().parents[2] / "data" / "sample_videos" / filename
        if v_path.exists():
            try:
                analysis_manager.start_analysis(camera_id, v_path, loop=True)
            except Exception:
                pass

    async def frame_generator():
        last_frame_bytes = None
        # Wait up to 10 seconds for initial AI frame to warm up
        for _ in range(100):
            fb = analysis_manager.get_latest_frame(camera_id)
            if fb:
                break
            await asyncio.sleep(0.1)

        while True:
            fb = analysis_manager.get_latest_frame(camera_id)
            if fb and fb != last_frame_bytes:
                last_frame_bytes = fb
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(fb)).encode() + b"\r\n\r\n"
                    + fb + b"\r\n"
                )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, current_user: Annotated[UserResponse, Depends(get_current_user)]):
    cam = store.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found")
    return cam


@router.post("", response_model=CameraResponse, dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))])
async def create_camera(cam_in: CameraCreate):
    if store.get_camera(cam_in.id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Camera with id '{cam_in.id}' already exists")
    return store.add_camera(cam_in)


@router.put("/{camera_id}", response_model=CameraResponse, dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))])
async def update_camera(camera_id: str, cam_in: CameraUpdate):
    updated = store.update_camera(camera_id, cam_in)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found")
    return updated


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_role([UserRole.ADMIN]))])
async def delete_camera(camera_id: str):
    if not store.delete_camera(camera_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found")
    return None


@router.get("/{camera_id}/streams")
async def get_camera_streams(camera_id: str, current_user: Annotated[UserResponse, Depends(get_current_user)]):
    cam = store.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found")
    return {
        "camera_id": cam.id,
        "rtsp": cam.rtsp_url,
        "webrtc": cam.webrtc_url or f"{settings.mediamtx_webrtc_base}/{cam.id}",
        "hls": cam.hls_url or f"{settings.mediamtx_hls_base}/{cam.id}/index.m3u8",
        "video_url": cam.video_url,
    }
