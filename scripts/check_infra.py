#!/usr/bin/env python3
"""Check that every Phase 0 service is up. Exit code 0 only if all pass.

    python scripts/check_infra.py               # services
    python scripts/check_infra.py --stream cam1 # also probe rtsp://localhost:8554/cam1 (needs ffprobe)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path


def load_env() -> dict:
    env = {}
    f = Path(__file__).resolve().parents[1] / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k in env or k.startswith(("POSTGRES_", "REDIS_", "OPENSEARCH_", "RTSP_"))})
    return env


ENV = load_env()
HOST = "localhost"


def port(key: str, default: int) -> int:
    return int(ENV.get(key, default))


def check_postgres() -> str:
    try:
        import psycopg
    except ImportError:
        raise RuntimeError("psycopg not installed (pip install -r requirements.txt)")
    with psycopg.connect(
        host=HOST, port=port("POSTGRES_PORT", 5432),
        user=ENV.get("POSTGRES_USER", "surveillance"),
        password=ENV.get("POSTGRES_PASSWORD", ""),
        dbname=ENV.get("POSTGRES_DB", "surveillance"), connect_timeout=5,
    ) as conn:
        row = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        if not row:
            raise RuntimeError("connected, but pgvector extension is missing")
        # prove vector search works, not just that the extension exists
        conn.execute("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector").fetchone()
        return f"pgvector {row[0]}"


def check_redis() -> str:
    with socket.create_connection((HOST, port("REDIS_PORT", 6379)), timeout=5) as s:
        s.sendall(b"PING\r\n")
        reply = s.recv(64)
    if not reply.startswith(b"+PONG"):
        raise RuntimeError(f"unexpected reply {reply!r}")
    return "PONG"


def check_opensearch() -> str:
    url = f"http://{HOST}:{port('OPENSEARCH_PORT', 9200)}/_cluster/health"
    with urllib.request.urlopen(url, timeout=5) as r:
        data = json.load(r)
    if data.get("status") not in ("green", "yellow"):
        raise RuntimeError(f"cluster status {data.get('status')}")
    return f"cluster {data['status']}"


def check_mediamtx() -> str:
    p = port("RTSP_PORT", 8554)
    with socket.create_connection((HOST, p), timeout=5):
        return f"RTSP port {p} open"


def check_stream(name: str) -> str:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found (install ffmpeg)")
    url = f"rtsp://{HOST}:{port('RTSP_PORT', 8554)}/{name}"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", url],
        capture_output=True, text=True, timeout=20,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(out.stderr.strip() or "no video stream (is the fake camera running?)")
    return f"{url} -> {out.stdout.strip()}"


def run(checks: list[tuple[str, callable]]) -> bool:
    ok = True
    for name, fn in checks:
        try:
            print(f"[ OK ] {name:11s} {fn()}")
        except Exception as e:  # noqa: BLE001 - report every failure
            ok = False
            print(f"[FAIL] {name:11s} {type(e).__name__}: {e}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", help="also probe this RTSP path, e.g. cam1")
    a = ap.parse_args()
    checks = [("postgres", check_postgres), ("redis", check_redis),
              ("opensearch", check_opensearch), ("mediamtx", check_mediamtx)]
    if a.stream:
        checks.append((f"stream:{a.stream}", lambda: check_stream(a.stream)))
    good = run(checks)
    print("\nAll checks passed." if good else "\nSome checks failed. See `make logs`.")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
