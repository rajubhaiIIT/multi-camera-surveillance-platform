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
