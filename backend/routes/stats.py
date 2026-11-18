"""System health, metrics, and operations dashboard statistics routes."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..auth import get_current_user
from ..schemas import SystemStatsResponse, UserResponse
from ..store import store

router = APIRouter(prefix="/api/stats", tags=["System & Statistics"])


@router.get("", response_model=SystemStatsResponse)
async def get_system_stats(current_user: Annotated[UserResponse, Depends(get_current_user)]):
    stats = store.get_system_stats()
    return SystemStatsResponse(**stats)
