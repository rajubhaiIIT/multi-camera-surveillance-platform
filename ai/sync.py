"""Line up several phone recordings in time using a hand clap at the start of every recording.

    python -m ai.sync cam1=videos/a.mp4 cam2=videos/b.mp4 cam3=videos/c.mp4 --out offsets.json

Needs ffmpeg. The offsets (seconds, all >= 0) are added to each camera's video time so the claps coincide.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

SR = 16000


def extract_audio(video: str, sr: int = SR, max_seconds: float = 60.0) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-t", str(max_seconds), "-i", video, "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video}: {out.stderr.decode(errors='ignore')[:200]}")
    return np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32)


def detect_clap(audio: np.ndarray, sr: int = SR, search_s: float = 30.0, min_ratio: float = 8.0) -> float | None:
    """Time (seconds) where the loudest sharp sound in the first `search_s` seconds begins; None if no clear spike."""
    x = np.abs(audio[: int(sr * search_s)])
    if len(x) < sr // 10:
        return None
    hop = sr // 200                                          # 5 ms steps
    n = len(x) // hop
    env = x[: n * hop].reshape(n, hop).max(axis=1)
    peak = env.max()
    if peak <= 0 or peak < min_ratio * (np.median(env) + 1.0):
        return None
    first = int(np.argmax(env >= 0.5 * peak))                # onset of the loudest event
    return first * hop / sr


def compute_offsets(claps: dict[str, float]) -> dict[str, float]:
    """offset = (latest clap time) - (this camera's clap time), so every offset is >= 0."""
    latest = max(claps.values())
    return {cam: round(latest - t, 3) for cam, t in claps.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="cam=path pairs")
    ap.add_argument("--out", default="offsets.json")
    a = ap.parse_args(argv)
    claps = {}
    for kv in a.videos:
        cam, path = kv.split("=", 1)
        t = detect_clap(extract_audio(path))
        if t is None:
            sys.exit(f"no clear clap found in the first 30 s of {path}. Clap louder and closer to the phone, "
                     "or write the offsets by hand.")
        claps[cam] = t
        print(f"{cam}: clap at {t:.3f} s")
    off = compute_offsets(claps)
    json.dump(off, open(a.out, "w"), indent=2)
    print("offsets:", off, "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
