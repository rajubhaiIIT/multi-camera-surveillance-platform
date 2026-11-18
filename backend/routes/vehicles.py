"""Vehicle and ANPR Watchlist API routes."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth import get_current_user, require_role
from ..schemas import (
    EventResponse,
    UserResponse,
    UserRole,
    WatchlistPlateCreate,
    WatchlistPlateResponse,
)
from ..store import store

router = APIRouter(prefix="/api/vehicles", tags=["Vehicles & ANPR"])


@router.get("/watchlist", response_model=list[WatchlistPlateResponse])
async def list_watchlist(current_user: Annotated[UserResponse, Depends(get_current_user)]):
    return store.list_watchlist()


@router.post(
    "/watchlist",
    response_model=WatchlistPlateResponse,
    dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))],
)
async def add_watchlist_plate(item: WatchlistPlateCreate):
    return store.add_watchlist_plate(item)


@router.delete(
    "/watchlist/{plate}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))],
)
async def remove_watchlist_plate(plate: str):
    if not store.remove_watchlist_plate(plate):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plate '{plate}' not found in watchlist")
    return None


@router.get("/sightings", response_model=list[EventResponse])
async def search_plate_sightings(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    plate: str | None = None,
    camera_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
):
    events = store.search_events(camera_id=camera_id, event_type="vehicle", limit=1000)
    if plate:
        clean_target = plate.upper().replace(" ", "").replace("-", "")
        events = [e for e in events if clean_target in str(e.metadata.get("plate", "")).upper()]
    return events[:limit]


@router.get("/anpr/engine")
async def get_anpr_engine(current_user: Annotated[UserResponse, Depends(get_current_user)]):
    """Retrieve active ANPR OCR engine and available alternative models."""
    import os
    return {
        "active_engine": os.environ.get("ANPR_ENGINE", "cct_s"),
        "available_engines": [
            {
                "id": "cct_s",
                "name": "CCT-S v2 (High-Accuracy Neural Model - Active)",
                "description": "Upgraded Compact Convolutional Transformer v2 Global model. Resolves character misreadings.",
            },
            {
                "id": "cct_xs",
                "name": "CCT-XS v2 (Legacy Model)",
                "description": "Original Fast-ALPR model preserved intact for instant comparison.",
            },
            {
                "id": "easyocr",
                "name": "EasyOCR (PyTorch GPU CRAFT)",
                "description": "Deep-learning text recognition engine powered by PyTorch.",
            },
            {
                "id": "mobile_vit",
                "name": "MobileViT-v2 Global",
                "description": "Vision Transformer model for international license plates.",
            },
        ],
    }


@router.post("/anpr/engine")
async def set_anpr_engine(
    engine: str = Query(..., description="Engine ID to activate: cct_s, cct_xs, easyocr, or mobile_vit"),
    current_user: Annotated[UserResponse, Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))] = None,
):
    """Dynamically switch active ANPR OCR model engine."""
    import os
    target = engine.lower().strip()
    valid = {"cct_s", "cct_xs", "easyocr", "mobile_vit"}
    if target not in valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid engine '{engine}'. Valid choices are: {sorted(valid)}",
        )
    os.environ["ANPR_ENGINE"] = target
    return {"status": "ok", "active_engine": target}

