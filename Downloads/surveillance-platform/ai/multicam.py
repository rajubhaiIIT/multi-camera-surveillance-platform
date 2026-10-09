"""Multi-camera pipeline: tracker -> person crops -> embeddings (FrameEmbedder), then tracklet book -> global IDs."""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Callable

import numpy as np

from .crosscam import GlobalMatcher, TrackletInfo
from .reid.embedder import crop_person
from .schema import TrackRecord
from .tracklets import TrackletBook, TrackletState


class FrameEmbedder:
    """Per-frame perception: detect + track + describe each person. Holds NO identity-matching state,
    so its output can be cached and re-matched many times with different settings (see ai/replay.py).

    trackers: {camera_id: object with .update(image, ts, idx) -> list[TrackRecord]}   (one per camera!)
    embedder: object with .embed(list_of_BGR_crops) -> ndarray [N, D], L2-normalised
    """

    def __init__(self, trackers: dict, embedder, embed_every: int = 3, warmup_embeddings: int = 3,
                 min_crop_h: int = 48, person_class: str = "person", crop_sink: Callable | None = None):
        self.trackers, self.embedder = trackers, embedder
        self.embed_every, self.warmup, self.min_crop_h = embed_every, warmup_embeddings, min_crop_h
        self.person_class, self.crop_sink = person_class, crop_sink      # crop_sink(camera, track_id, n_emb, crop)
        self.seen: dict = defaultdict(int)
        self.n_emb: dict = defaultdict(int)

    def run(self, camera_id: str, image, ts: float, idx: int) -> tuple[list[TrackRecord], dict]:
        recs = self.trackers[camera_id].update(image, ts, idx)
        crops, tids = [], []
        for r in recs:
            if r.cls != self.person_class:
                continue
            k = (camera_id, r.track_id)
            self.seen[k] += 1
            if self.n_emb[k] < self.warmup or self.seen[k] % self.embed_every == 0:
                c = crop_person(image, r.bbox, self.min_crop_h)
                if c is not None:
                    crops.append(c)
                    tids.append(r.track_id)
        embs = {}
        if crops:
            for tid, crop, e in zip(tids, crops, self.embedder.embed(crops)):
                embs[tid] = e
                self.n_emb[(camera_id, tid)] += 1
                if self.crop_sink:
                    self.crop_sink(camera_id, tid, self.n_emb[(camera_id, tid)], crop)
        return recs, embs


class MultiCamPipeline:
    """process() for every frame of every camera (any order; ALL timestamps must share one clock).

    on_tracklet_end: optional callback(TrackletState) when a track disappears (e.g. save to the database)
    Leave trackers/embedder as None to use only ingest() with cached embeddings.
    """

    def __init__(self, trackers: dict | None, embedder, matcher: GlobalMatcher, book: TrackletBook | None = None,
                 embed_every: int = 3, min_crop_h: int = 48, observe_every: int = 15,
                 person_class: str = "person", on_tracklet_end: Callable[[TrackletState], None] | None = None):
        self.matcher = matcher
        self.book = book or TrackletBook()
        self.observe_every, self.person_class, self.on_end = observe_every, person_class, on_tracklet_end
        self.fe = (FrameEmbedder(trackers, embedder, embed_every, matcher.cfg.min_embeddings, min_crop_h, person_class)
                   if trackers is not None else None)

    def process(self, camera_id: str, image, ts: float, idx: int) -> list[TrackRecord]:
        recs, embs = self.fe.run(camera_id, image, ts, idx)
        return self.ingest(camera_id, ts, recs, embs)

    def ingest(self, camera_id: str, ts: float, recs: list[TrackRecord], embs: dict) -> list[TrackRecord]:
        """Identity matching for one camera frame. embs: {track_id: embedding computed in this frame}."""
        persons = [r for r in recs if r.cls == self.person_class]
        states = {}
        for r in persons:
            st = self.book.touch(camera_id, r.track_id, ts)
            states[r.track_id] = st
            if r.track_id in embs:
                st.embs.append(embs[r.track_id])

        ready = [s for s in states.values() if s.global_id is None and len(s.embs) >= self.matcher.cfg.min_embeddings]
        if ready:
            res = self.matcher.match([TrackletInfo(s.camera_id, s.track_id, s.first_ts, s.last_ts,
                                                   s.mean_embedding()) for s in ready])
            for s in ready:
                s.global_id = res.get(s.key)

        for tid, s in states.items():                        # keep identities fresh while the person stays visible
            if s.global_id is not None:
                refresh = s.since_observe >= self.observe_every
                self.matcher.observe(camera_id, tid, ts, s.mean_embedding() if refresh else None)
                if refresh:
                    s.since_observe = 0

        for cam in {s.camera_id for s in self.book.states.values()} | {camera_id}:
            for s in self.book.expire(cam, ts):               # tracks nobody has seen lately: close them
                self._finish(s)

        return [dataclasses.replace(r, global_id=states[r.track_id].global_id) if r.cls == self.person_class else r
                for r in recs]

    def flush(self, camera_id: str | None = None):
        """End of stream: close every open tracklet."""
        for s in self.book.drain(camera_id):
            self._finish(s)

    def _finish(self, s: TrackletState):
        if s.global_id is not None:
            self.matcher.end_track(s.camera_id, s.track_id, s.last_ts)
        if self.on_end:
            self.on_end(s)
