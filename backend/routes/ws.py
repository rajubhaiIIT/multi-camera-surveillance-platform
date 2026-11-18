"""WebSocket real-time streaming routes."""
from __future__ import annotations

import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..ws import ws_manager

logger = logging.getLogger("surveillance.routes.ws")
router = APIRouter(prefix="/ws", tags=["WebSocket Streams"])


@router.websocket("/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """General real-time WebSocket for live tracking frames, alerts, and system notifications."""
    await ws_manager.connect(websocket, initial_topics=["all", "alerts"])
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                action = data.get("action")
                if action == "subscribe":
                    topic = data.get("topic")
                    if topic:
                        await ws_manager.subscribe(websocket, topic)
                        await websocket.send_text(json.dumps({"status": "subscribed", "topic": topic}))
                elif action == "unsubscribe":
                    topic = data.get("topic")
                    if topic:
                        await ws_manager.unsubscribe(websocket, topic)
                        await websocket.send_text(json.dumps({"status": "unsubscribed", "topic": topic}))
                elif action == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.debug("WebSocket error: %s", e)
        await ws_manager.disconnect(websocket)


@router.websocket("/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """Dedicated low-latency WebSocket channel exclusively for alerts."""
    await ws_manager.connect(websocket, initial_topics=["alerts"])
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                if data.get("action") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.debug("Alerts WebSocket error: %s", e)
        await ws_manager.disconnect(websocket)
