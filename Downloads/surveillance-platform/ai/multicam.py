"""Multi-camera pipeline step: tracker -> person crops -> embeddings -> tracklet book -> global IDs."""
from __future__ import annotations

import dataclasses
from typing import Callable

from .crosscam import GlobalMatcher, TrackletInfo
from .reid.embedder import crop_person
from .schema import TrackRecord
from .tracklets import TrackletBook, TrackletState


class MultiCamPipeline:
    """Call process() for every frame of every camera (any order, timestamps must share one clock).

    trackers: {camera_id: object with .update(image, ts, idx) -> list[TrackRecord]}  (one per camera!)
    embedder: object with .embed(list_of_BGR_crops) -> ndarray [N, D], L2-normalised
    on_tracklet_end: optional callback(TrackletState) when a track disappears (e.g. save to the database)
    """

    def __init__(self, trackers: dict, embedder, matcher: GlobalMatcher, book: TrackletBook | None = None,
                 embed_every: int = 3, min_crop_h: int = 48, observe_every: int = 15,
                 person_class: str = "person", on_tracklet_end: Callable[[TrackletState], None] | None = None):
        self.trackers, self.embedder, self.matcher = trackers, embedder, matcher
        self.book = book or TrackletBook()
        self.embed_every, self.min_crop_h, self.observe_every = embed_every, min_crop_h, observe_every
        self.person_class, self.on_end = person_class, on_tracklet_end

    def process(self, camera_id: str, image, ts: float, idx: int) -> list[TrackRecord]:
        recs = self.trackers[camera_id].update(image, ts, idx)
        persons = [r for r in recs if r.cls == self.person_class]

        crops, owners = [], []
        for r in persons:
            st = self.book.touch(camera_id, r.track_id, ts)
            wants = len(st.embs) < self.matcher.cfg.min_embeddings or st.n_frames % self.embed_every == 0
            crop = crop_person(image, r.bbox, self.min_crop_h) if wants else None
            if crop is not None:
                crops.append(crop)
                owners.append(st)
        if crops:
            for st, e in zip(owners, self.embedder.embed(crops)):
                st.embs.append(e)

        ready = [s for s in (self.book.states[(camera_id, r.track_id)] for r in persons)
                 if s.global_id is None and len(s.embs) >= self.matcher.cfg.min_embeddings]
        if ready:
            res = self.matcher.match([TrackletInfo(s.camera_id, s.track_id, s.first_ts, s.last_ts,
                                                   s.mean_embedding()) for s in ready])
            for s in ready:
                s.global_id = res.get(s.key)

        for r in persons:                                   # keep identities fresh while the person stays visible
            s = self.book.states[(camera_id, r.track_id)]
            if s.global_id is not None:
                refresh = s.since_observe >= self.observe_every
                self.matcher.observe(camera_id, r.track_id, ts, s.mean_embedding() if refresh else None)
                if refresh:
                    s.since_observe = 0

        for s in self.book.expire(camera_id, ts):
            self._finish(s)

        return [dataclasses.replace(r, global_id=self.book.states[(camera_id, r.track_id)].global_id)
                if r.cls == self.person_class else r for r in recs]

    def flush(self, camera_id: str | None = None):
        """End of stream: close every open tracklet."""
        for s in self.book.drain(camera_id):
            self._finish(s)

    def _finish(self, s: TrackletState):
        if s.global_id is not None:
            self.matcher.end_track(s.camera_id, s.track_id, s.last_ts)
        if self.on_end:
            self.on_end(s)
