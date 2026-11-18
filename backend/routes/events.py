"""Perception events and tracking stream ingestion/search routes."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ..auth import get_current_user
from ..opensearch_service import opensearch_service
from ..schemas import EventCreate, EventResponse, UserResponse
from ..store import store
from ..ws import ws_manager

router = APIRouter(prefix="/api/events", tags=["Events"])


@router.get("", response_model=list[EventResponse])
async def search_events(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    camera_id: str | None = None,
    event_type: str | None = None,
    track_id: int | None = None,
    global_id: int | None = None,
    query: str | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    # Try OpenSearch first if full-text query is supplied
    if query:
        os_results = opensearch_service.search_events(query=query, camera_id=camera_id, event_type=event_type, limit=limit)
        if os_results is not None:
            return [EventResponse(**r) for r in os_results]

    return store.search_events(
        camera_id=camera_id,
        event_type=event_type,
        track_id=track_id,
        global_id=global_id,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=EventResponse)
async def ingest_event(
    event_in: EventCreate,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    event = store.add_event(event_in)

    # Real-time WebSocket emission to camera channel
    await ws_manager.broadcast_event(event.model_dump())

    # OpenSearch index
    opensearch_service.index_event(event)

    return event
