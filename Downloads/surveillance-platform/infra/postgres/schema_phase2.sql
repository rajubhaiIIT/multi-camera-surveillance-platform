-- Phase 2 tables: global identities and per-tracklet appearance embeddings.
-- Applied (safely, repeatedly) by:  python scripts/init_db.py
-- __DIM__ is replaced with the embedding size of your Re-ID model (default 512).
-- Phase 7 moves these into Alembic migrations together with the rest of the schema.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS global_identities (
    global_id        BIGINT PRIMARY KEY,
    first_ts         DOUBLE PRECISION NOT NULL,
    last_ts          DOUBLE PRECISION NOT NULL,
    last_camera_id   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracklet_embeddings (
    id           BIGSERIAL PRIMARY KEY,
    global_id    BIGINT NOT NULL REFERENCES global_identities(global_id) ON DELETE CASCADE,
    camera_id    TEXT NOT NULL,
    track_id     INTEGER NOT NULL,
    first_ts     DOUBLE PRECISION NOT NULL,
    last_ts      DOUBLE PRECISION NOT NULL,
    n_frames     INTEGER NOT NULL,
    embedding    vector(__DIM__) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (camera_id, track_id, first_ts)
);

CREATE INDEX IF NOT EXISTS tracklet_embeddings_global_idx ON tracklet_embeddings (global_id);
CREATE INDEX IF NOT EXISTS tracklet_embeddings_vec_idx
    ON tracklet_embeddings USING hnsw (embedding vector_cosine_ops);
