"""Hotlist and watchlist index for license plates with exact and fuzzy matching."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def normalize_plate(text: str) -> str:
    """Normalize plate text: uppercase, remove non-alphanumeric characters."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


@dataclass
class WatchlistEntry:
    plate: str
    reason: str
    owner: str = ""
    notes: str = ""


class PlateIndex:
    """Watchlist for stolen, wanted, or VIP license plates.

    Parameters
    ----------
    max_edit_distance : maximum character substitution/insertion/deletion to count as match (default: 1)
    """

    def __init__(self, max_edit_distance: int = 1):
        self.max_edit_distance = max_edit_distance
        self._entries: Dict[str, WatchlistEntry] = {}

    def enroll(self, plate: str, reason: str, owner: str = "", notes: str = "") -> WatchlistEntry:
        clean = normalize_plate(plate)
        if not clean:
            raise ValueError("License plate cannot be empty")
        entry = WatchlistEntry(plate=clean, reason=reason, owner=owner, notes=notes)
        self._entries[clean] = entry
        return entry

    def remove(self, plate: str) -> bool:
        clean = normalize_plate(plate)
        return bool(self._entries.pop(clean, None))

    def clear(self):
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    def match(self, plate: str) -> Optional[WatchlistEntry]:
        """Check if plate is in the watchlist via exact match or fuzzy edit distance."""
        clean = normalize_plate(plate)
        if not clean:
            return None

        # 1. Exact match
        if clean in self._entries:
            return self._entries[clean]

        # 2. Fuzzy match within max_edit_distance
        if self.max_edit_distance > 0 and len(clean) >= 4:
            for hot_plate, entry in self._entries.items():
                if abs(len(hot_plate) - len(clean)) <= self.max_edit_distance:
                    dist = _levenshtein(hot_plate, clean)
                    if dist <= self.max_edit_distance:
                        return entry
        return None

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_edit_distance": self.max_edit_distance,
            "entries": {
                k: {"reason": v.reason, "owner": v.owner, "notes": v.notes}
                for k, v in self._entries.items()
            }
        }
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PlateIndex":
        p = Path(path)
        data = json.loads(p.read_text())
        idx = cls(max_edit_distance=data.get("max_edit_distance", 1))
        for k, v in data.get("entries", {}).items():
            idx.enroll(k, reason=v.get("reason", ""), owner=v.get("owner", ""), notes=v.get("notes", ""))
        return idx
