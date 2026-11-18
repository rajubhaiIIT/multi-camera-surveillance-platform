"""Per-camera tracklet bookkeeping: collect appearance embeddings for every (camera, track_id)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackletState:
    camera_id: str
    track_id: int
    first_ts: float
    last_ts: float
    n_frames: int = 0
    embs: deque = field(default_factory=lambda: deque(maxlen=30))
    global_id: int | None = None
    since_observe: int = 0

    @property
    def key(self) -> tuple:
        return (self.camera_id, self.track_id)

    def mean_embedding(self) -> np.ndarray:
        """Average of the stored embeddings, re-normalised. Averaging many frames beats using one frame."""
        m = np.mean(np.stack(self.embs), axis=0)
        return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)


class TrackletBook:
    def __init__(self, max_embeddings: int = 30, stale_after_s: float = 3.0):
        self.max_embeddings, self.stale_after_s = max_embeddings, stale_after_s
        self.states: dict[tuple, TrackletState] = {}

    def touch(self, camera_id: str, track_id: int, ts: float) -> TrackletState:
        st = self.states.get((camera_id, track_id))
        if st is None:
            st = TrackletState(camera_id, track_id, ts, ts, embs=deque(maxlen=self.max_embeddings))
            self.states[st.key] = st
        st.last_ts = max(st.last_ts, ts)
        st.n_frames += 1
        st.since_observe += 1
        return st

    def expire(self, camera_id: str, now_ts: float) -> list[TrackletState]:
        """Remove and return this camera's tracklets not seen for `stale_after_s`."""
        gone = [s for s in self.states.values()
                if s.camera_id == camera_id and now_ts - s.last_ts > self.stale_after_s]
        for s in gone:
            del self.states[s.key]
        return gone

    def drain(self, camera_id: str | None = None) -> list[TrackletState]:
        """Remove and return everything (end of stream)."""
        gone = [s for s in self.states.values() if camera_id is None or s.camera_id == camera_id]
        for s in gone:
            del self.states[s.key]
        return gone
