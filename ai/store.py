"""PostgreSQL + pgvector storage for global identities and tracklet embeddings."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .tracklets import TrackletState


def _read_dotenv() -> dict:
    """Settings from the project's .env file (real environment variables still take priority)."""
    f = Path(__file__).resolve().parents[1] / ".env"
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def conninfo_from_env(env: dict | None = None) -> dict:
    e = {**_read_dotenv(), **os.environ, **(env or {})}
    return dict(host=e.get("POSTGRES_HOST", "localhost"), port=int(e.get("POSTGRES_PORT", 5432)),
                user=e.get("POSTGRES_USER", "surveillance"), password=e.get("POSTGRES_PASSWORD", ""),
                dbname=e.get("POSTGRES_DB", "surveillance"))


def vec(e) -> str:
    """numpy vector -> pgvector text literal."""
    return "[" + ",".join(f"{float(x):.6f}" for x in np.asarray(e).ravel()) + "]"


class IdentityStore:
    def __init__(self, **conn):
        import psycopg
        self.conn = psycopg.connect(autocommit=True, **(conn or conninfo_from_env()))

    def close(self):
        self.conn.close()

    def save_tracklet(self, s: TrackletState):
        """Store a finished tracklet (its mean embedding) under its global ID. Safe to call twice."""
        if s.global_id is None or not s.embs:
            return
        with self.conn.transaction():
            self.conn.execute(
                """INSERT INTO global_identities (global_id, first_ts, last_ts, last_camera_id)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (global_id) DO UPDATE SET
                     first_ts = LEAST(global_identities.first_ts, EXCLUDED.first_ts),
                     last_ts = GREATEST(global_identities.last_ts, EXCLUDED.last_ts),
                     last_camera_id = CASE WHEN EXCLUDED.last_ts >= global_identities.last_ts
                                           THEN EXCLUDED.last_camera_id ELSE global_identities.last_camera_id END""",
                (s.global_id, s.first_ts, s.last_ts, s.camera_id))
            self.conn.execute(
                """INSERT INTO tracklet_embeddings
                   (global_id, camera_id, track_id, first_ts, last_ts, n_frames, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT (camera_id, track_id, first_ts) DO UPDATE SET
                     last_ts = EXCLUDED.last_ts, n_frames = EXCLUDED.n_frames, embedding = EXCLUDED.embedding""",
                (s.global_id, s.camera_id, s.track_id, s.first_ts, s.last_ts, s.n_frames, vec(s.mean_embedding())))

    def nearest(self, embedding, k: int = 5) -> list[dict]:
        """'Find this person': the k most similar stored tracklets (cosine similarity, higher = more alike)."""
        rows = self.conn.execute(
            """SELECT global_id, camera_id, track_id, first_ts, last_ts, 1 - (embedding <=> %s::vector) AS sim
               FROM tracklet_embeddings ORDER BY embedding <=> %s::vector LIMIT %s""",
            (vec(embedding), vec(embedding), k)).fetchall()
        return [dict(global_id=r[0], camera_id=r[1], track_id=r[2], first_ts=r[3], last_ts=r[4], similarity=float(r[5]))
                for r in rows]

    def trajectory(self, global_id: int) -> list[dict]:
        """Where and when an identity was seen, in time order."""
        rows = self.conn.execute(
            "SELECT camera_id, track_id, first_ts, last_ts FROM tracklet_embeddings WHERE global_id = %s ORDER BY first_ts",
            (global_id,)).fetchall()
        return [dict(camera_id=r[0], track_id=r[1], first_ts=r[2], last_ts=r[3]) for r in rows]
