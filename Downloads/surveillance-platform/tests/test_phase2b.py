"""Own-recording tools: clap sync, extract, cache, labeling, replay, scoring, ablations (stub tracker + embedder)."""
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from ai.crosscam import MatcherConfig, Topology, load_config  # noqa: E402
from ai.schema import TrackRecord  # noqa: E402

RED, BLUE, GREEN = (0, 0, 255), (255, 0, 0), (0, 255, 0)
A_BOX, B_BOX, C_BOX = (50, 50, 150, 250), (300, 50, 400, 250), (500, 50, 600, 250)
FPS, N_FRAMES = 10, 100


class ScriptTracker:
    def __init__(self, cam, script):
        self.cam, self.script = cam, script

    def update(self, image, ts, idx):
        return [TrackRecord(self.cam, ts, idx, tid, bb, "person", 0.9) for tid, bb, _ in self.script]


class ColourEmbedder:
    def embed(self, crops):
        v = np.stack([c.reshape(-1, 3).mean(axis=0) for c in crops]).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def write_video(path: Path, people):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
    for _ in range(N_FRAMES):
        img = np.zeros((480, 640, 3), np.uint8)
        for _, (x1, y1, x2, y2), col in people:
            img[y1:y2, x1:x2] = col
        vw.write(img)
    vw.release()


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """Cam1 (t = 0-10 s): A in red, B in blue.  Cam2 (t = 30-40 s): B, A (different track ids) and a stranger in green."""
    d = tmp_path_factory.mktemp("rec")
    cam1 = [(1, A_BOX, RED), (2, B_BOX, BLUE)]
    cam2 = [(7, B_BOX, BLUE), (8, A_BOX, RED), (9, C_BOX, GREEN)]
    write_video(d / "cam1.mp4", cam1)
    write_video(d / "cam2.mp4", cam2)
    from ai.multicam import FrameEmbedder
    from ai.multicam_extract import extract_run
    fe = FrameEmbedder({"cam1": ScriptTracker("cam1", cam1), "cam2": ScriptTracker("cam2", cam2)},
                       ColourEmbedder(), embed_every=3)
    out = extract_run({"cam1": d / "cam1.mp4", "cam2": d / "cam2.mp4"}, {"cam1": 0.0, "cam2": 30.0},
                      fe, d / "run", stride=2, log=None)
    return out


def write_labels(run_dir: Path, mapping: dict):
    import csv
    rows = list(csv.DictReader(open(run_dir / "labels.csv")))
    with open(run_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            r["person"] = mapping.get((r["camera"], int(r["track_id"])), "")
            w.writerow(r)


@pytest.fixture(scope="module")
def labeled(recording):
    from ai.labeling import make_contact_sheets, write_label_template
    from ai.runcache import load_run
    run = load_run(recording)
    make_contact_sheets(run)
    write_label_template(run)
    write_labels(run.dir, {("cam1", 1): "anna", ("cam1", 2): "ben", ("cam2", 7): "ben", ("cam2", 8): "anna"})
    return run


# ---------------------------------------------------------------- extract + cache
def test_extract_saves_events_embeddings_crops_and_meta(recording):
    from ai.runcache import load_run
    run = load_run(recording)
    assert len([e for e in run.events if e.cam == "cam1"]) == N_FRAMES // 2        # stride 2
    cam2_ts = [e.ts for e in run.events if e.cam == "cam2"]
    assert min(cam2_ts) == pytest.approx(30.0) and max(cam2_ts) > 39.0             # offset applied
    assert sum(len(e.embs) for e in run.events) > 0
    e = next(e for e in run.events if e.embs)
    v = next(iter(e.embs.values()))
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-3)
    assert (recording / "crops" / "cam1" / "1_1.jpg").exists()
    m = run.meta["cameras"]["cam2"]
    assert m["offset"] == 30.0 and m["stride"] == 2 and m["last_idx"] == 98


def test_negative_offset_is_rejected(tmp_path):
    from ai.multicam import FrameEmbedder
    from ai.multicam_extract import extract_run
    write_video(tmp_path / "v.mp4", [(1, A_BOX, RED)])
    fe = FrameEmbedder({"c": ScriptTracker("c", [])}, ColourEmbedder())
    with pytest.raises(ValueError):
        extract_run({"c": tmp_path / "v.mp4"}, {"c": -1.0}, fe, tmp_path / "r", log=None)


def test_load_run_rejects_a_folder_that_is_not_a_run(tmp_path):
    from ai.runcache import load_run
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)


# ---------------------------------------------------------------- labeling
def test_label_template_and_sheets(labeled):
    import csv
    rows = list(csv.DictReader(open(labeled.dir / "labels.csv")))
    assert {(r["camera"], int(r["track_id"])) for r in rows} == {("cam1", 1), ("cam1", 2), ("cam2", 7), ("cam2", 8), ("cam2", 9)}
    assert any((labeled.dir / "sheets").glob("sheet_cam1_*.jpg")) and any((labeled.dir / "sheets").glob("sheet_cam2_*.jpg"))


def test_template_never_overwrites_existing_labels(labeled):
    from ai.labeling import read_labels, write_label_template
    write_label_template(labeled)                      # must not erase what write_labels() filled in
    assert read_labels(labeled.dir / "labels.csv")[("cam2", 8)] == "anna"


def test_blank_and_ignore_labels_are_skipped(tmp_path):
    from ai.labeling import read_labels
    p = tmp_path / "l.csv"
    p.write_text("camera,track_id,frames,first_ts,last_ts,person\nc1,1,20,0,1,\nc1,2,20,0,1,ignore\nc1,3,20,0,1,Bob\n")
    assert read_labels(p) == {("c1", 3): "Bob"}


# ---------------------------------------------------------------- replay + scoring
def test_same_clothes_in_second_camera_scores_perfectly(labeled):
    from ai.labeling import read_labels, score_run
    from ai.replay import replay
    res = score_run(labeled, replay(labeled), read_labels(labeled.dir / "labels.csv"))
    assert res["mtmc"]["IDF1"] == pytest.approx(100, abs=0.5)
    assert res["links"]["cross_camera"]["f1"] == pytest.approx(100)
    assert res["true_people"] == 2 and res["predicted_identities"] == 2 and res["unmatched_tracklets"] == 0


def test_backfilled_ids_cover_every_frame_of_a_matched_track(labeled):
    from ai.replay import replay
    recs = replay(labeled)
    t1 = [r for r in recs if (r.camera_id, r.track_id) == ("cam1", 1)]
    assert len(t1) == N_FRAMES // 2 and len({r.global_id for r in t1}) == 1 and t1[0].global_id is not None


def test_impossible_travel_time_splits_identities_and_lowers_the_score(labeled):
    from ai.labeling import read_labels, score_run
    from ai.replay import replay
    topo = Topology(min_travel_s={("cam1", "cam2"): 500.0})            # cameras 500 s apart, but only 20 s passed
    res = score_run(labeled, replay(labeled, topology=topo), read_labels(labeled.dir / "labels.csv"))
    assert res["predicted_identities"] == 4 and res["links"]["cross_camera"]["recall"] == 0
    assert res["mtmc"]["IDF1"] < 60


def test_rules_only_ablation_runs_and_is_not_better_than_full(labeled):
    from ai.labeling import read_labels, score_run
    from ai.replay import replay
    labels = read_labels(labeled.dir / "labels.csv")
    full = score_run(labeled, replay(labeled), labels)["mtmc"]["IDF1"]
    no_reid = score_run(labeled, replay(labeled, no_reid=True), labels)["mtmc"]["IDF1"]
    assert no_reid <= full + 1e-6


def test_appearance_only_lets_two_lookalikes_in_one_camera_share_an_id():
    from ai.crosscam import GlobalMatcher, TrackletInfo
    e = np.array([1.0, 0.0], np.float32)
    on, off = GlobalMatcher(MatcherConfig()), GlobalMatcher(MatcherConfig(use_constraints=False))
    for m, expect_same in ((on, False), (off, True)):
        a = m.match([TrackletInfo("c1", 1, 0, 5, e)])[("c1", 1)]
        b = m.match([TrackletInfo("c1", 2, 1, 6, e)])[("c1", 2)]
        assert (a == b) is expect_same


def test_scoring_without_labels_gives_a_clear_error(labeled, tmp_path):
    from ai.labeling import score_run
    from ai.replay import replay
    with pytest.raises(ValueError, match="no labeled tracks"):
        score_run(labeled, replay(labeled), {})


def test_render_writes_one_video_per_camera(labeled, tmp_path):
    from ai.replay import render_run, replay
    paths = render_run(labeled, replay(labeled), tmp_path / "render")
    assert sorted(p.name for p in paths) == ["cam1.mp4", "cam2.mp4"] and all(p.stat().st_size > 0 for p in paths)


def test_link_scores_hand_example():
    from ai.mtmc_eval import tracklet_link_scores
    items = [("c1", "a", 1), ("c2", "a", 1), ("c1", "b", 2), ("c2", "b", 1)]    # last one wrongly joined to a's id
    cross = tracklet_link_scores(items)["cross_camera"]
    # cross-camera pairs: (c1a,c2a) tp, (c1a,c2b) fp, (c1b,c2a) tn, (c1b,c2b) fn
    assert cross["precision"] == pytest.approx(50.0) and cross["recall"] == pytest.approx(50.0)


# ---------------------------------------------------------------- config file
def test_load_config(tmp_path):
    p = tmp_path / "topo.json"
    p.write_text(json.dumps({"matcher": {"sim_threshold": 0.7, "max_gap_s": 90},
                             "min_travel_s": {"cam1,cam2": 8}, "overlapping": [["cam2", "cam3"]]}))
    cfg, topo = load_config(p)
    assert cfg.sim_threshold == 0.7 and cfg.max_gap_s == 90
    assert topo.travel("cam2", "cam1") == 8 and topo.overlap("cam3", "cam2") and topo.travel("cam1", "cam3") == 0


# ---------------------------------------------------------------- clap sync (needs ffmpeg)
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_clap_video(path: Path, clap_at: float, seconds: float = 6.0):
    sr = 16000
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 300, int(sr * seconds))                          # quiet room noise
    s = int(clap_at * sr)
    audio[s:s + 400] += rng.normal(0, 12000, 400)                          # a short, loud clap
    wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes(np.clip(audio, -32000, 32000).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}:r=10",
                    "-i", str(wav), "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(path)], check=True)


@needs_ffmpeg
def test_clap_detection_and_offsets(tmp_path):
    from ai.sync import compute_offsets, detect_clap, extract_audio
    make_clap_video(tmp_path / "a.mp4", 1.2)
    make_clap_video(tmp_path / "b.mp4", 3.5)
    ta, tb = (detect_clap(extract_audio(str(tmp_path / f"{n}.mp4"))) for n in "ab")
    assert ta == pytest.approx(1.2, abs=0.05) and tb == pytest.approx(3.5, abs=0.05)
    off = compute_offsets({"a": ta, "b": tb})
    assert off["b"] == 0 and off["a"] == pytest.approx(2.3, abs=0.1)


def test_no_clap_in_quiet_noise_returns_none():
    from ai.sync import detect_clap
    noise = np.random.default_rng(1).normal(0, 300, 16000 * 10).astype(np.float32)
    assert detect_clap(noise) is None
