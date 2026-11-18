"""Save what the perception stage saw (tracks + embeddings) so identity matching can be re-run cheaply.

A run folder contains: meta.json, events.jsonl (one line per processed camera frame),
embeddings.npy (float16, in the order the events mention them), crops/<camera>/<track>_<k>.jpg
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import TrackRecord


@dataclass
class Event:
    cam: str
    ts: float
    idx: int
    recs: list
    embs: dict


@dataclass
class Run:
    dir: Path
    meta: dict
    events: list


class RunWriter:
    def __init__(self, out_dir: str | Path, meta: dict):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self._f = open(self.dir / "events.jsonl", "w")
        self._embs: list = []

    def add(self, cam: str, ts: float, idx: int, recs, embs: dict):
        self._f.write(json.dumps({"cam": cam, "ts": round(ts, 4), "idx": idx,
                                  "recs": [r.to_dict() for r in recs], "emb_ids": list(embs)}) + "\n")
        self._embs.extend(embs.values())

    def close(self):
        self._f.close()
        arr = np.stack(self._embs).astype(np.float16) if self._embs else np.zeros((0, 1), np.float16)
        np.save(self.dir / "embeddings.npy", arr)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))


def load_run(path: str | Path) -> Run:
    d = Path(path)
    if not (d / "events.jsonl").exists():
        raise FileNotFoundError(f"{d} is not a run folder (no events.jsonl). Run the 'extract' step first.")
    meta = json.loads((d / "meta.json").read_text())
    arr = np.load(d / "embeddings.npy").astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    events, k = [], 0
    for line in (d / "events.jsonl").read_text().splitlines():
        o = json.loads(line)
        embs = {}
        for tid in o["emb_ids"]:
            embs[tid] = arr[k]
            k += 1
        events.append(Event(o["cam"], o["ts"], o["idx"], [TrackRecord.from_dict(r) for r in o["recs"]], embs))
    return Run(d, meta, events)
