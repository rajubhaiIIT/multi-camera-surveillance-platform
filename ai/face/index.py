"""Enrollment and matching: store face embeddings, search for a person.

Backed by either an in-memory numpy index (default, zero dependencies) or
pgvector (pass store= for production use, same interface as ai.store.IdentityStore).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Person:
    name: str
    person_id: str                                      # unique key
    embeddings: list = field(default_factory=list)      # list of (512,) arrays
    notes: str = ""

    @property
    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        m = np.mean(np.stack(self.embeddings), axis=0)
        return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)


@dataclass
class MatchResult:
    person_id: str
    name: str
    similarity: float         # cosine similarity, 0..1
    matched: bool             # True if similarity >= threshold


class FaceIndex:
    """In-memory enrollment and cosine-similarity search.

    Usage
    -----
    idx = FaceIndex(threshold=0.4)
    idx.enroll("person_001", "Aryan", [emb1, emb2, emb3])
    results = idx.search(query_embedding, top_k=3)
    idx.save("models/face_index.json")
    idx = FaceIndex.load("models/face_index.json")
    """

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self._persons: dict[str, Person] = {}

    # ---- enrollment --------------------------------------------------------
    def enroll(self, person_id: str, name: str,
               embeddings: list[np.ndarray], notes: str = "") -> Person:
        """Add or update a person. Merges embeddings if the person already exists."""
        if person_id in self._persons:
            p = self._persons[person_id]
            p.name = name
            p.embeddings.extend(embeddings)
            p.notes = notes or p.notes
        else:
            p = Person(name, person_id, list(embeddings), notes)
            self._persons[person_id] = p
        return p

    def remove(self, person_id: str) -> bool:
        return bool(self._persons.pop(person_id, None))

    def clear(self):
        self._persons.clear()

    @property
    def size(self) -> int:
        return len(self._persons)

    # ---- search ------------------------------------------------------------
    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[MatchResult]:
        """Return the top-k most similar enrolled people, sorted by similarity."""
        if not self._persons:
            return []
        emb = np.asarray(embedding, np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        # build matrix of mean embeddings on the fly (fast for hundreds of people)
        ids, means = [], []
        for pid, p in self._persons.items():
            m = p.mean_embedding
            if m is not None:
                ids.append(pid)
                means.append(m)
        if not means:
            return []
        mat = np.stack(means)           # (N, 512)
        sims = (mat @ emb).tolist()     # cosine similarity (embeddings are L2-normalised)
        order = sorted(range(len(ids)), key=lambda i: -sims[i])[:top_k]
        return [MatchResult(ids[i], self._persons[ids[i]].name,
                            float(sims[i]), sims[i] >= self.threshold)
                for i in order]

    def identify(self, embedding: np.ndarray) -> MatchResult | None:
        """Return the best match if it meets the threshold, else None (unknown)."""
        results = self.search(embedding, top_k=1)
        if results and results[0].matched:
            return results[0]
        return None

    # ---- persistence -------------------------------------------------------
    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"threshold": self.threshold, "persons": {
            pid: {"name": p.name, "notes": p.notes,
                  "embeddings": [e.tolist() for e in p.embeddings]}
            for pid, p in self._persons.items()}}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "FaceIndex":
        data = json.loads(Path(path).read_text())
        idx = cls(threshold=data.get("threshold", 0.4))
        for pid, p in data.get("persons", {}).items():
            embs = [np.array(e, np.float32) for e in p.get("embeddings", [])]
            idx.enroll(pid, p["name"], embs, p.get("notes", ""))
        return idx
