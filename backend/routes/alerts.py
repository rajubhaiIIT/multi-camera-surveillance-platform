"""Alert ingestion, listing, filtering, and acknowledgment routes."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth import get_current_user, require_role
from ..opensearch_service import opensearch_service
from ..schemas import (
    AlertAckRequest,
    AlertCreate,
    AlertResponse,
    AlertSeverity,
    AlertType,
    UserResponse,
    UserRole,
)
from ..store import store
from ..ws import ws_manager

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("", response_model=list[AlertResponse])
async def list_alerts(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    camera_id: str | None = None,
    severity: AlertSeverity | None = None,
    alert_type: AlertType | None = None,
    acknowledged: bool | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return store.list_alerts(
        camera_id=camera_id,
        severity=severity,
        alert_type=alert_type,
        acknowledged=acknowledged,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=limit,
        offset=offset,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str, current_user: Annotated[UserResponse, Depends(get_current_user)]):
    alert = store.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert '{alert_id}' not found")
    return alert


@router.post("", response_model=AlertResponse)
async def create_alert(
    alert_in: AlertCreate,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    # Store alert
    alert = store.add_alert(alert_in)

    # Broadcast via WebSocket
    await ws_manager.broadcast_alert(alert.model_dump())

    # Index in OpenSearch
    opensearch_service.index_alert(alert)

    return alert


@router.post("/{alert_id}/ack", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: str,
    ack: AlertAckRequest,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    if not ack.operator:
        ack.operator = current_user.username
    updated = store.acknowledge_alert(alert_id, ack)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert '{alert_id}' not found")

    # Broadcast update
    await ws_manager.broadcast_json({"type": "alert_ack", "payload": updated.model_dump()}, topic="alerts")
    return updated


@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_role([UserRole.ADMIN]))])
async def delete_alert(alert_id: str):
    if not store.delete_alert(alert_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert '{alert_id}' not found")
    return None
