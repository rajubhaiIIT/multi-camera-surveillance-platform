"""Phase 1 tests.

Offline (default):          pytest
TrackEval needed:           runs automatically if installed, otherwise skipped
Model tests (downloads yolo11n.pt, uses GPU if present):   pytest -m model
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.mot_io import read_mot, write_mot  # noqa: E402
from ai.schema import TrackRecord  # noqa: E402


def rec(**kw):
    base = dict(camera_id="cam1", frame_ts=1.0, frame_idx=0, track_id=1,
                bbox=(10.0, 20.0, 50.0, 90.0), cls="person", conf=0.9)
    base.update(kw)
    return TrackRecord(**base)


# ---------- contract ----------
def test_record_dict_has_the_agreed_keys():
    d = rec().to_dict()
    assert set(d) == {"camera_id", "frame_ts", "frame_idx", "track_id", "bbox", "class", "conf"}
    assert d["bbox"] == [10.0, 20.0, 50.0, 90.0] and d["class"] == "person"


def test_record_roundtrip_through_json():
    r = rec()
    assert TrackRecord.from_dict(json.loads(json.dumps(r.to_dict()))) == r


@pytest.mark.parametrize("bad", [
    dict(camera_id=""), dict(track_id=-1), dict(bbox=(5, 5, 5, 9)), dict(bbox=(1, 2, 3)),
    dict(bbox=(0, 0, float("nan"), 5)), dict(conf=1.5), dict(cls=""), dict(frame_ts=float("inf")),
])
def test_record_rejects_malformed_values(bad):
    with pytest.raises(ValueError):
        rec(**bad)


# ---------- MOT file format ----------
def test_mot_roundtrip(tmp_path):
    rows = [(1, 3, 10.0, 20.0, 50.0, 80.0, 0.8), (2, 3, 12.0, 22.0, 52.0, 82.0, 0.7)]
    p = tmp_path / "seq.txt"
    write_mot(p, rows)
    assert p.read_text().splitlines()[0].startswith("1,3,10.00,20.00,40.00,60.00,0.8000")
    back = read_mot(p)
    assert [r[:2] for r in back] == [(1, 3), (2, 3)]
    assert back[0][4] == pytest.approx(50.0) and back[0][5] == pytest.approx(80.0)


# ---------- drawing ----------
def test_draw_changes_pixels_and_keeps_shape():
    from ai.viz import draw
    img = np.zeros((120, 160, 3), np.uint8)
    out = draw(img.copy(), [rec(bbox=(20, 30, 100, 100))], fps=12.0)
    assert out.shape == img.shape and out.any()


# ---------- sources + worker (no model needed) ----------
def make_video(path, n=20, w=160, h=120, fps=10):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), i * 5 % 255, np.uint8)
        vw.write(f)
    vw.release()


def test_file_source_yields_ordered_frames(tmp_path):
    from ai.sources import FileSource
    p = tmp_path / "v.mp4"
    make_video(p, n=20)
    frames = list(FileSource(str(p)))
    assert len(frames) == 20
    assert [f.idx for f in frames] == list(range(20))
    ts = [f.ts for f in frames]
    assert ts == sorted(ts) and ts[1] == pytest.approx(0.1, abs=1e-3)


def test_file_source_missing_file_raises(tmp_path):
    from ai.sources import FileSource
    with pytest.raises(IOError):
        FileSource(str(tmp_path / "nope.mp4"))


class StubTracker:
    """One fixed box per frame; lets us test the worker without torch."""
    def update(self, image, ts, idx):
        return [TrackRecord("camX", ts, idx, 7, (10.0, 10.0, 60.0, 90.0), "person", 0.9),
                TrackRecord("camX", ts, idx, 8, (70.0, 10.0, 90.0, 40.0), "car", 0.1)]  # below min_conf


def test_worker_writes_valid_jsonl_and_filters_low_conf(tmp_path):
    from ai.sources import FileSource
    from ai.worker import run
    p = tmp_path / "v.mp4"
    make_video(p, n=15)
    out = tmp_path / "out.jsonl"
    stats = run(FileSource(str(p)), StubTracker(), min_conf=0.3, jsonl_path=str(out), log_every=0)
    assert stats["frames"] == 15 and stats["records"] == 15
    lines = out.read_text().splitlines()
    assert len(lines) == 15
    for ln in lines:
        r = TrackRecord.from_dict(json.loads(ln))      # contract: must validate
        assert r.track_id == 7 and r.cls == "person"


def test_worker_max_frames_and_video_output(tmp_path):
    from ai.sources import FileSource
    from ai.worker import run
    p = tmp_path / "v.mp4"
    make_video(p, n=30)
    vid = tmp_path / "ann.mp4"
    stats = run(FileSource(str(p)), StubTracker(), save_video=str(vid), max_frames=10, log_every=0)
    assert stats["frames"] == 10 and vid.exists() and vid.stat().st_size > 0


# ---------- evaluation pipeline (TrackEval) ----------
def make_synthetic_mot(root: Path, n_frames=60):
    """Two pedestrians crossing, with a real MOT17-style gt.txt and seqinfo.ini."""
    seq = root / "SYN-01-SDP"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=SYN-01-SDP\nimDir=img1\nframeRate=30\n"
        f"seqLength={n_frames}\nimWidth=640\nimHeight=480\nimExt=.jpg\n")
    gt, pred = [], []
    for f in range(1, n_frames + 1):
        for tid, (x0, dx) in {1: (50, 4), 2: (400, -4)}.items():
            x, y, w, h = x0 + dx * f, 100 + 50 * tid, 40, 100
            gt.append(f"{f},{tid},{x},{y},{w},{h},1,1,1.0")
            pred.append((f, tid, float(x), float(y), float(x + w), float(y + h), 0.9))
    (seq / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
    return seq, pred


def test_trackeval_scores_perfect_tracking_as_100(tmp_path):
    pytest.importorskip("trackeval")
    from ai.mot_eval import evaluate
    seq, pred = make_synthetic_mot(tmp_path / "data")
    pf = tmp_path / "pred" / "SYN-01-SDP.txt"
    write_mot(pf, pred)
    res = evaluate(tmp_path / "te", {"SYN-01-SDP": seq}, {"SYN-01-SDP": pf}, "perfect", print_results=False)
    c = res["combined"]
    assert c["HOTA"] == pytest.approx(100, abs=0.5)
    assert c["IDF1"] == pytest.approx(100, abs=0.5) and c["MOTA"] == pytest.approx(100, abs=0.5)
    assert c["IDSW"] == 0


def test_trackeval_penalises_identity_switches(tmp_path):
    pytest.importorskip("trackeval")
    from ai.mot_eval import evaluate
    seq, pred = make_synthetic_mot(tmp_path / "data")
    # swap the two IDs half way: perfect boxes, wrong identities
    bad = [(f, ({1: 2, 2: 1}[t] if f > 30 else t), *rest) for (f, t, *rest) in pred]
    pf = tmp_path / "pred" / "SYN-01-SDP.txt"
    write_mot(pf, bad)
    res = evaluate(tmp_path / "te", {"SYN-01-SDP": seq}, {"SYN-01-SDP": pf}, "swapped", print_results=False)
    c = res["combined"]
    assert c["IDSW"] >= 2
    assert c["IDF1"] < 80 and c["HOTA"] < 95
    assert c["DetA"] == pytest.approx(100, abs=0.5)   # boxes are right; only identity is wrong


# ---------- real model (pytest -m model) ----------
@pytest.mark.model
def test_tracker_returns_valid_records_on_real_frames():
    pytest.importorskip("ultralytics")
    import cv2
    from ai.yolo_tracker import Tracker
    t = Tracker("cam1", imgsz=640, device="auto")
    assert t.update(np.zeros((480, 640, 3), np.uint8), 0.0, 0) == []   # blank frame: no crash, no tracks
    base = ROOT / "data" / "raw" / "mot17"
    imgs = sorted(base.rglob("MOT17-04-SDP/img1/*.jpg"))[:30]
    if not imgs:
        pytest.skip("MOT17 not downloaded")
    seen = 0
    for i, p in enumerate(imgs):
        for r in t.update(cv2.imread(str(p)), i / 30, i):
            TrackRecord.from_dict(r.to_dict())      # contract holds on real output
            seen += 1
    assert seen > 0
