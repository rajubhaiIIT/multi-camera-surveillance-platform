#!/usr/bin/env python3
"""Create the Phase 2 tables (safe to run repeatedly).   python scripts/init_db.py [--dim 512]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_infra import ENV  # noqa: E402  (reads .env)
from ai.store import conninfo_from_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=512, help="embedding size of your Re-ID model")
a = ap.parse_args()

import psycopg  # noqa: E402

sql = (ROOT / "infra" / "postgres" / "schema_phase2.sql").read_text().replace("__DIM__", str(a.dim))
with psycopg.connect(autocommit=True, **conninfo_from_env(ENV)) as conn:
    conn.execute(sql)
    n = conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name IN "
                     "('global_identities','tracklet_embeddings')").fetchone()[0]
print(f"OK: {n}/2 tables present, embedding size {a.dim}")
