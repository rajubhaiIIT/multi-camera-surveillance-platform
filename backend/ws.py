"""WebSocket connection manager for real-time alerting and stream event broadcasting."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("surveillance.ws")


class ConnectionManager:
    """Manages active WebSockets and dispatches topic-filtered updates to connected clients."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.subscriptions: dict[WebSocket, set[str]] = {}
        self._thread_lock = threading.Lock()
        self.main_loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket, initial_topics: list[str] | None = None):
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        await websocket.accept()
        with self._thread_lock:
            self.active_connections.append(websocket)
            topics = set(initial_topics or ["all", "alerts"])
            self.subscriptions[websocket] = topics
        logger.info("WebSocket connected (%d total clients)", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket):
        with self._thread_lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            self.subscriptions.pop(websocket, None)
        logger.info("WebSocket disconnected (%d remaining)", len(self.active_connections))

    async def subscribe(self, websocket: WebSocket, topic: str):
        with self._thread_lock:
            if websocket in self.subscriptions:
                self.subscriptions[websocket].add(topic)

    async def unsubscribe(self, websocket: WebSocket, topic: str):
        with self._thread_lock:
            if websocket in self.subscriptions:
                self.subscriptions[websocket].discard(topic)

    def broadcast_sync(self, data: dict[str, Any], topic: str = "all"):
        """Thread-safe dispatch for background worker threads to broadcast on the main event loop."""
        if self.main_loop and self.main_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.broadcast_json(data, topic=topic), self.main_loop)
            except Exception as e:
                logger.debug("Failed threadsafe broadcast: %s", e)

    def broadcast_alert_sync(self, alert_data: dict[str, Any]):
        payload = {
            "type": "alert",
            "topic": "alerts",
            "payload": alert_data,
        }
        self.broadcast_sync(payload, topic="alerts")

    async def broadcast_json(self, data: dict[str, Any], topic: str = "all"):
        message = json.dumps(data)
        disconnected: list[WebSocket] = []

        with self._thread_lock:
            conns = list(self.active_connections)

        for ws in conns:
            user_topics = self.subscriptions.get(ws, set())
            if topic == "all" or "all" in user_topics or topic in user_topics:
                try:
                    await ws.send_text(message)
                except Exception:
                    disconnected.append(ws)

        for dead_ws in disconnected:
            await self.disconnect(dead_ws)

    async def broadcast_alert(self, alert_data: dict[str, Any]):
        payload = {
            "type": "alert",
            "topic": "alerts",
            "payload": alert_data,
        }
        await self.broadcast_json(payload, topic="alerts")

    async def broadcast_event(self, event_data: dict[str, Any]):
        cam_id = event_data.get("camera_id")
        topic = f"camera:{cam_id}" if cam_id else "events"
        payload = {
            "type": "event",
            "topic": topic,
            "payload": event_data,
        }
        await self.broadcast_json(payload, topic=topic)


ws_manager = ConnectionManager()
