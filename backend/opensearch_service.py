"""OpenSearch integration for distributed event indexing and full-text search with fallback."""
from __future__ import annotations

import logging
from typing import Any

from .config import settings
from .schemas import AlertResponse, EventResponse

logger = logging.getLogger("surveillance.opensearch")


class OpenSearchService:
    def __init__(self):
        self.client = None
        self._connected = False
        self._try_connect()

    def _try_connect(self):
        try:
            from opensearchpy import OpenSearch
            self.client = OpenSearch(
                hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
                timeout=2,
            )
            # Ping test
            if self.client.ping():
                self._connected = True
                self._ensure_indices()
                logger.info("Connected to OpenSearch cluster")
            else:
                self._connected = False
        except Exception as e:
            self._connected = False
            self.client = None
            logger.debug("OpenSearch offline, fallback mode enabled: %s", e)

    def is_connected(self) -> bool:
        return self._connected

    def _ensure_indices(self):
        if not self._connected or not self.client:
            return
        try:
            for idx in [settings.opensearch_index_events, settings.opensearch_index_alerts]:
                if not self.client.indices.exists(index=idx):
                    self.client.indices.create(
                        index=idx,
                        body={
                            "settings": {
                                "number_of_shards": 1,
                                "number_of_replicas": 0,
                            }
                        },
                    )
        except Exception as e:
            logger.warning("Could not ensure OpenSearch indices: %s", e)

    def index_event(self, ev: EventResponse):
        if not self._connected or not self.client:
            return
        try:
            self.client.index(
                index=settings.opensearch_index_events,
                id=ev.id,
                body=ev.model_dump(),
                refresh=False,
            )
        except Exception as e:
            logger.debug("OpenSearch index_event error: %s", e)

    def index_alert(self, alert: AlertResponse):
        if not self._connected or not self.client:
            return
        try:
            self.client.index(
                index=settings.opensearch_index_alerts,
                id=alert.id,
                body=alert.model_dump(),
                refresh=False,
            )
        except Exception as e:
            logger.debug("OpenSearch index_alert error: %s", e)

    def search_events(
        self,
        query: str | None = None,
        camera_id: str | None = None,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[dict] | None:
        if not self._connected or not self.client:
            return None  # Triggers fallback to store
        try:
            must_clauses: list[dict] = []
            if camera_id:
                must_clauses.append({"term": {"camera_id": camera_id}})
            if event_type:
                must_clauses.append({"term": {"event_type": event_type}})
            if query:
                must_clauses.append({"multi_match": {"query": query, "fields": ["metadata.*", "camera_id"]}})

            body = {
                "size": limit,
                "sort": [{"timestamp": {"order": "desc"}}],
                "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
            }
            res = self.client.search(index=settings.opensearch_index_events, body=body)
            return [hit["_source"] for hit in res["hits"]["hits"]]
        except Exception as e:
            logger.warning("OpenSearch search failed, using fallback: %s", e)
            return None


opensearch_service = OpenSearchService()
