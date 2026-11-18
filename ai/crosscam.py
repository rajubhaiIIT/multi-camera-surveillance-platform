"""Cross-camera identity matching: link tracklets from different cameras into one global ID.

How a tracklet gets a global ID
  1. Compare its appearance embedding with every known identity (cosine similarity, best match in the
     identity's gallery).
  2. Throw out impossible matches (see _allowed): the identity is already visible in this camera, is visible
     in a non-overlapping camera right now, or last appeared too recently / too long ago for the walk
     between the two cameras.
  3. Solve all waiting tracklets together with the Hungarian algorithm, so two people can never grab the
     same identity. A tracklet with no good-enough match starts a new identity.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

INVALID = 1e6


@dataclass
class MatcherConfig:
    sim_threshold: float = 0.5     # cosine similarity needed to join an existing identity. TUNE on your data.
    max_gap_s: float = 600.0       # an identity unseen for longer than this is not matched again
    gallery_size: int = 20         # embeddings (one per tracklet) kept per identity
    min_embeddings: int = 3        # a tracklet needs this many embeddings before it may be matched
    use_constraints: bool = True   # False = appearance only (for the ablation table)


@dataclass
class Topology:
    """Camera layout rules. Default: cameras do not overlap and travel between them takes >= 0 s."""
    min_travel_s: dict = field(default_factory=dict)   # {("camA","camB"): seconds}, symmetric
    overlapping: set = field(default_factory=set)      # {frozenset({"camA","camB"})}: can see the same person at once
    default_min_travel_s: float = 0.0
    all_overlapping: bool = False                      # every pair of cameras sees the same area (e.g. WILDTRACK)

    def travel(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return self.min_travel_s.get((a, b), self.min_travel_s.get((b, a), self.default_min_travel_s))

    def overlap(self, a: str, b: str) -> bool:
        return a != b and (self.all_overlapping or frozenset((a, b)) in self.overlapping)


@dataclass
class TrackletInfo:
    camera_id: str
    track_id: int
    first_ts: float
    last_ts: float
    emb: np.ndarray

    @property
    def key(self):
        return (self.camera_id, self.track_id)


@dataclass
class Identity:
    gid: int
    first_ts: float
    last_ts: float
    last_cam: str
    gallery: OrderedDict = field(default_factory=OrderedDict)   # (cam, track_id) -> embedding
    active: dict = field(default_factory=dict)                  # camera -> set(track_ids) currently visible


class GlobalMatcher:
    def __init__(self, cfg: MatcherConfig | None = None, topology: Topology | None = None):
        self.cfg, self.topo = cfg or MatcherConfig(), topology or Topology()
        self.identities: dict[int, Identity] = {}
        self.bound: dict[tuple, int] = {}
        self._next = 1

    # ---- constraints -------------------------------------------------------
    def _allowed(self, t: TrackletInfo, ident: Identity) -> bool:
        if not self.cfg.use_constraints:
            return True
        cam = t.camera_id
        if ident.active.get(cam):                       # already visible in this very camera: two people, not one
            return False
        for other_cam, tids in ident.active.items():    # visible elsewhere right now
            if tids and other_cam != cam and not self.topo.overlap(cam, other_cam):
                return False
        gap = t.first_ts - ident.last_ts
        if gap > self.cfg.max_gap_s:
            return False
        if gap < self.topo.travel(ident.last_cam, cam):
            # not enough time to walk between the cameras (negative gap = overlap in time)
            return self.topo.overlap(cam, ident.last_cam)
        return True

    @staticmethod
    def _similarity(emb: np.ndarray, ident: Identity) -> float:
        g = np.stack(list(ident.gallery.values()))
        return float((g @ emb).max())

    # ---- main API ------------------------------------------------------------
    def match(self, tracklets: list[TrackletInfo]) -> dict[tuple, int]:
        """Assign a global ID to each tracklet in the list. Returns {(camera, track_id): global_id}."""
        todo = [t for t in tracklets if t.key not in self.bound]
        if not todo:
            return {}
        ids = list(self.identities.values())
        n, m = len(todo), len(ids)
        new_cost = 1.0 - self.cfg.sim_threshold
        cost = np.full((n, m + n), INVALID)
        cost[np.arange(n), m + np.arange(n)] = new_cost          # "start a new identity" options
        for i, t in enumerate(todo):
            for j, ident in enumerate(ids):
                if self._allowed(t, ident):
                    s = self._similarity(t.emb, ident)
                    if s >= self.cfg.sim_threshold:
                        cost[i, j] = 1.0 - s
        rows, cols = linear_sum_assignment(cost)
        out = {}
        for i, j in zip(rows, cols):
            t = todo[i]
            ident = ids[j] if j < m else self._new_identity(t)
            self._bind(ident, t)
            out[t.key] = ident.gid
        return out

    def observe(self, camera_id: str, track_id: int, ts: float, emb: np.ndarray | None = None):
        """Heartbeat for a bound, still-visible track: keeps 'last seen' fresh, refreshes its embedding."""
        gid = self.bound.get((camera_id, track_id))
        if gid is None:
            return
        ident = self.identities[gid]
        ident.last_ts, ident.last_cam = max(ident.last_ts, ts), camera_id
        if emb is not None:
            ident.gallery[(camera_id, track_id)] = emb
            ident.gallery.move_to_end((camera_id, track_id))

    def end_track(self, camera_id: str, track_id: int, ts: float):
        gid = self.bound.get((camera_id, track_id))
        if gid is None:
            return
        ident = self.identities[gid]
        ident.active.get(camera_id, set()).discard(track_id)
        ident.last_ts, ident.last_cam = max(ident.last_ts, ts), camera_id

    def identity_for(self, camera_id: str, track_id: int) -> int | None:
        return self.bound.get((camera_id, track_id))

    # ---- internals -----------------------------------------------------------
    def _new_identity(self, t: TrackletInfo) -> Identity:
        ident = Identity(self._next, t.first_ts, t.last_ts, t.camera_id)
        self.identities[ident.gid] = ident
        self._next += 1
        return ident

    def _bind(self, ident: Identity, t: TrackletInfo):
        ident.active.setdefault(t.camera_id, set()).add(t.track_id)
        ident.gallery[t.key] = t.emb
        while len(ident.gallery) > self.cfg.gallery_size:
            ident.gallery.popitem(last=False)
        ident.last_ts, ident.last_cam = max(ident.last_ts, t.last_ts), t.camera_id
        self.bound[t.key] = ident.gid


def load_config(path) -> tuple[MatcherConfig, Topology]:
    """Read matcher settings and camera layout from a JSON file, e.g.
    {"matcher": {"sim_threshold": 0.55, "max_gap_s": 300},
     "min_travel_s": {"cam1,cam2": 8}, "overlapping": [["cam2", "cam3"]], "default_min_travel_s": 0}"""
    import json
    d = json.loads(open(path).read())
    cfg = MatcherConfig(**d.get("matcher", {}))
    topo = Topology(
        min_travel_s={tuple(k.split(",")): float(v) for k, v in d.get("min_travel_s", {}).items()},
        overlapping={frozenset(p) for p in d.get("overlapping", [])},
        default_min_travel_s=float(d.get("default_min_travel_s", 0.0)),
        all_overlapping=bool(d.get("all_overlapping", False)))
    return cfg, topo
