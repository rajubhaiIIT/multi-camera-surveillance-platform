"""Phase 3 tests: face detector wrapper, enrollment index, LFW loader, pipeline.

All offline (no real InsightFace model needed): a stub detector is used.
The real model test runs with: pytest -m model
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.face.index import FaceIndex, MatchResult, Person
from ai.face.detector import FaceRecord  # noqa: E402
from ai.face.eval_lfw import eval_lfw, load_pairs  # noqa: E402


# ---------------------------------------------------------------- helpers
# Two orthogonal base directions in 512-d space.
# Tests use these + small noise so similarity scores are predictable.
_rng0 = np.random.default_rng(42)
RED  = _rng0.normal(size=512).astype(np.float32); RED  /= np.linalg.norm(RED)
BLUE = _rng0.normal(size=512).astype(np.float32)
BLUE -= BLUE.dot(RED) * RED; BLUE /= np.linalg.norm(BLUE)   # orthogonalised


def make_emb(direction: np.ndarray, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """Unit embedding close to `direction` with a little noise."""
    rng = np.random.default_rng(seed)
    v = direction + rng.normal(0, noise, direction.shape).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def unit(*v):
    v = np.array(v, np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- FaceRecord
def test_face_record_to_dict_has_expected_keys():
    r = FaceRecord((10.0, 20.0, 60.0, 100.0), 0.92, make_emb(RED), None)
    d = r.to_dict()
    assert set(d) == {"bbox", "det_score", "embedding"}
    assert d["bbox"] == [10.0, 20.0, 60.0, 100.0]
    assert len(d["embedding"]) == 512


def test_face_record_area_and_face_height():
    r = FaceRecord((10.0, 20.0, 60.0, 120.0), 0.8, make_emb(RED), None)
    assert r.area == pytest.approx(50.0 * 100.0)
    assert r.face_height == pytest.approx(100.0)


# ---------------------------------------------------------------- FaceIndex
def test_enroll_and_search_returns_correct_person():
    idx = FaceIndex(threshold=0.5)
    idx.enroll("p1", "Alice", [make_emb(RED,0.02,1), make_emb(RED,0.02,2)])
    idx.enroll("p2", "Bob",   [make_emb(BLUE,0.02,3), make_emb(BLUE,0.02,4)])
    results = idx.search(make_emb(RED,0.01,5), top_k=2)
    assert results[0].person_id == "p1" and results[0].matched
    assert results[1].person_id == "p2" and not results[1].matched


def test_identify_returns_none_below_threshold():
    idx = FaceIndex(threshold=0.8)        # same-direction embeddings score about 0.95, orthogonal ones about 0
    idx.enroll("p1", "Alice", [make_emb(RED,0.01,1)])
    assert idx.identify(make_emb(RED,0.01,2)) is not None  # same direction: passes
    assert idx.identify(make_emb(BLUE,0.01,3)) is None      # orthogonal: below threshold


def test_enroll_same_person_twice_merges_embeddings():
    idx = FaceIndex()
    idx.enroll("p1", "Alice", [make_emb(RED,0.02,1)])
    idx.enroll("p1", "Alice Updated", [make_emb(RED,0.02,2), make_emb(RED,0.02,3)])
    assert len(idx._persons["p1"].embeddings) == 3
    assert idx._persons["p1"].name == "Alice Updated"


def test_remove_person():
    idx = FaceIndex()
    idx.enroll("p1", "Alice", [make_emb(RED,0.01,1)])
    assert idx.remove("p1") is True and idx.size == 0
    assert idx.remove("p1") is False


def test_search_on_empty_index_returns_empty_list():
    assert FaceIndex().search(make_emb(RED)) == []


def test_mean_embedding_is_unit_length():
    idx = FaceIndex()
    idx.enroll("p1", "Alice", [make_emb(RED,0.02,i) for i in range(3)])
    m = idx._persons["p1"].mean_embedding
    assert m is not None
    assert np.linalg.norm(m) == pytest.approx(1.0, abs=1e-5)


def test_person_with_no_embeddings_has_none_mean():
    p = Person("Alice", "p1")
    assert p.mean_embedding is None


def test_save_and_load_roundtrip(tmp_path):
    idx = FaceIndex(threshold=0.35)
    idx.enroll("p1", "Alice", [make_emb(RED,0.02,1), make_emb(RED,0.02,2)])
    idx.enroll("p2", "Bob",   [make_emb(BLUE,0.02,3)])
    idx.save(tmp_path / "index.json")
    idx2 = FaceIndex.load(tmp_path / "index.json")
    assert idx2.threshold == 0.35 and idx2.size == 2
    assert idx2._persons["p1"].name == "Alice"
    assert len(idx2._persons["p1"].embeddings) == 2
    r = idx2.search(make_emb(RED,0.01,99), top_k=1)
    assert r[0].person_id == "p1"


def test_load_from_nonexistent_path_raises():
    with pytest.raises(FileNotFoundError):
        FaceIndex.load("/no/such/file.json")


def test_index_handles_many_people(tmp_path):
    idx = FaceIndex(threshold=0.5)
    dirs = [make_emb(RED if i%2==0 else BLUE, noise=0.001, seed=i) for i in range(200)]
    for i in range(200):
        idx.enroll(f"p{i}", f"Person {i}", [dirs[i]])
    assert idx.size == 200
    r = idx.search(make_emb(RED,0.001,9999), top_k=5)
    assert r[0].person_id in {f"p{i}" for i in range(0,200,2)} and len(r) == 5


# ---------------------------------------------------------------- LFW evaluation
def make_lfw(root: Path, n_pos: int = 20, n_neg: int = 20) -> tuple[Path, np.ndarray, list]:
    """Build a minimal LFW folder with a pairs.txt and stub images."""
    root.mkdir(parents=True)
    pairs_lines = [f"{n_pos}\t{n_neg}"]
    sims, labels = [], []
    img = np.zeros((250, 250, 3), np.uint8)
    for i in range(n_pos):
        name = f"Person_{i:04d}"
        (root / name).mkdir(exist_ok=True)
        cv2.imwrite(str(root / name / f"{name}_0001.jpg"), img)
        cv2.imwrite(str(root / name / f"{name}_0002.jpg"), img)
        pairs_lines.append(f"{name}\t1\t2")
        sims.append(0.8); labels.append(1)
    for i in range(n_neg):
        a, b = f"Person_{i:04d}", f"Person_{(i + 1) % n_pos:04d}"
        pairs_lines.append(f"{a}\t1\t{b}\t1")
        sims.append(0.1); labels.append(0)
    (root / "pairs.txt").write_text("\n".join(pairs_lines))
    return root, np.array(sims, np.float32), labels


def test_load_pairs_reads_positive_and_negative(tmp_path):
    root, _, _ = make_lfw(tmp_path / "lfw", 10, 10)
    pairs, labels = load_pairs(root)
    assert len(pairs) == 20 and labels.count(1) == 10 and labels.count(0) == 10
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


def test_eval_lfw_perfect_sims_give_100_accuracy():
    sims = np.array([0.9] * 100 + [0.1] * 100, np.float32)
    labels = [1] * 100 + [0] * 100
    valid = [True] * 200
    r = eval_lfw(sims, labels, valid)
    assert r["accuracy"] == pytest.approx(100.0, abs=0.01)
    assert r["valid_pairs"] == 200


def test_eval_lfw_random_sims_give_around_50_accuracy():
    rng = np.random.default_rng(42)
    sims = np.concatenate([rng.uniform(0.3, 0.9, 100), rng.uniform(-0.4, 0.1, 100)]).astype(np.float32)
    labels = [1] * 100 + [0] * 100
    valid = [True] * 200
    r = eval_lfw(sims, labels, valid)
    assert r["accuracy"] > 80


def test_eval_lfw_skips_invalid_pairs():
    sims = np.array([0.9, 0.5, 0.1], np.float32)
    labels = [1, 1, 0]
    valid = [True, False, True]
    r = eval_lfw(sims, labels, valid)
    assert r["valid_pairs"] == 2


# ---------------------------------------------------------------- FacePipeline
def test_pipeline_identifies_after_min_hits_and_uses_majority_vote():
    from ai.face.pipeline import FacePipeline
    from ai.schema import TrackRecord

    class StubDetector:
        def detect(self, img):
            return [FaceRecord((5.0, 5.0, 55.0, 105.0), 0.9, make_emb(RED,0.01,99), None)]

    idx = FaceIndex(threshold=0.5)
    idx.enroll("alice", "Alice", [make_emb(RED,0.01,1), make_emb(RED,0.01,2)])
    idx.enroll("bob",   "Bob",   [make_emb(BLUE,0.01,3)])

    pipe = FacePipeline(StubDetector(), idx, run_every=1, min_hits=2)
    frame = np.zeros((480, 640, 3), np.uint8)
    rec = TrackRecord("cam1", 0.0, 0, 1, (10.0, 10.0, 100.0, 200.0), "person", 0.9)

    hits1 = pipe.process(frame, [rec])
    assert hits1 == []                            # only 1 hit so far, need 2

    rec2 = TrackRecord("cam1", 1.0, 1, 1, (10.0, 10.0, 100.0, 200.0), "person", 0.9)
    hits2 = pipe.process(frame, [rec2])
    assert len(hits2) == 1
    assert hits2[0].name == "Alice" and hits2[0].person_id == "alice"
    assert hits2[0].similarity > 0.5


def test_pipeline_skips_non_person_tracks():
    from ai.face.pipeline import FacePipeline
    from ai.schema import TrackRecord

    class StubDetector:
        def detect(self, img):
            return [FaceRecord((5.0, 5.0, 55.0, 105.0), 0.9, make_emb(RED,0.01,1), None)]

    idx = FaceIndex(threshold=0.5)
    idx.enroll("p1", "Alice", [make_emb(RED,0.01,2)])
    pipe = FacePipeline(StubDetector(), idx, run_every=1, min_hits=1)
    frame = np.zeros((480, 640, 3), np.uint8)
    car = TrackRecord("cam1", 0.0, 0, 2, (10.0, 10.0, 100.0, 200.0), "car", 0.9)
    assert pipe.process(frame, [car]) == []


def test_pipeline_resets_track_state():
    from ai.face.pipeline import FacePipeline
    from ai.schema import TrackRecord

    class StubDetector:
        def detect(self, img):
            return [FaceRecord((5.0, 5.0, 55.0, 105.0), 0.9, make_emb(RED,0.01,1), None)]

    idx = FaceIndex(threshold=0.5)
    idx.enroll("p1", "Alice", [make_emb(RED,0.01,2)])
    pipe = FacePipeline(StubDetector(), idx, run_every=1, min_hits=2)
    frame = np.zeros((480, 640, 3), np.uint8)
    rec = TrackRecord("cam1", 0.0, 0, 7, (10.0, 10.0, 100.0, 200.0), "person", 0.9)
    pipe.process(frame, [rec])                   # 1 hit
    pipe.reset_track("cam1", 7)
    pipe.process(frame, [rec])                   # reset: 1 hit again, not 2
    assert ("cam1", 7) in pipe._frame_count
    assert len(pipe._hits.get(("cam1", 7), [])) == 1


# ---------------------------------------------------------------- real model (pytest -m model)
@pytest.mark.model
def test_insightface_detects_faces_on_real_image():
    insightface = pytest.importorskip("insightface")
    import os
    from ai.face.detector import FaceDetector
    det = FaceDetector(det_size=(320, 320), min_det_score=0.3, min_face_height=30.0)
    img_path = ROOT / "data" / "raw" / "lfw"
    imgs = list(img_path.rglob("*.jpg"))[:5] if img_path.exists() else []
    if not imgs:
        pytest.skip("LFW not downloaded")
    found = 0
    for p in imgs:
        faces = det.detect(cv2.imread(str(p)))
        found += bool(faces)
    assert found > 0, "no faces detected on real LFW images"


# ---------------------------------------------------------------- LFW folder layouts and end-to-end verification (stub detector)
def test_find_lfw_root_handles_the_nested_layout_of_the_official_archive(tmp_path):
    from ai.face.eval_lfw import find_lfw_root, load_pairs
    base = tmp_path / "lfw_data"
    root, _, _ = make_lfw(base / "lfw", 5, 5)                 # images in data/lfw/lfw/<person>/...
    (base / "pairs.txt").write_text((root / "pairs.txt").read_text())
    (root / "pairs.txt").unlink()                              # pairs.txt one level above, as the downloader saves it
    assert find_lfw_root(base) == root
    pairs, labels = load_pairs(find_lfw_root(base))
    assert len(pairs) == 10 and all(p.exists() for pair in pairs for p in pair)


def test_find_lfw_root_gives_a_clear_error(tmp_path):
    from ai.face.eval_lfw import find_lfw_root
    with pytest.raises(FileNotFoundError, match="download_data.py fetch lfw"):
        find_lfw_root(tmp_path)


def test_lfw_pipeline_end_to_end_with_a_colour_based_stub(tmp_path):
    """Every person gets a unique colour: same-person pairs embed identically, different-person pairs do not."""
    from ai.face.eval_lfw import eval_lfw, load_pairs, pairs_to_embeddings
    root = tmp_path / "lfw"
    root.mkdir()
    n = 30
    lines = [f"1\t{n}"]
    rng = np.random.default_rng(3)
    for i in range(n):
        name = f"P{i:03d}"
        (root / name).mkdir()
        colour = rng.integers(20, 235, 3)
        for k in (1, 2):
            cv2.imwrite(str(root / name / f"{name}_{k:04d}.jpg"), np.full((60, 60, 3), colour, np.uint8))
        lines.append(f"{name}\t1\t2")
    for i in range(n):
        lines.append(f"P{i:03d}\t1\tP{(i + 1) % n:03d}\t1")
    (root / "pairs.txt").write_text("\n".join(lines))

    class ColourDetector:
        def detect(self, img):
            v = np.zeros(512, np.float32)
            v[:3] = img.reshape(-1, 3).mean(axis=0) - 127.0          # colour as a direction
            v[3:] = 1e-3
            return [FaceRecord((0.0, 0.0, 60.0, 60.0), 0.9, v / np.linalg.norm(v), None)]

    pairs, labels = load_pairs(root)
    sims, valid = pairs_to_embeddings(pairs, ColourDetector())
    assert all(valid) and sims[:n].min() > 0.99
    assert eval_lfw(sims, labels, valid, n_folds=10)["accuracy"] > 90


def test_pick_center_avoids_a_bystander_with_a_higher_score():
    from ai.face.eval_lfw import pick_face
    target = FaceRecord((90.0, 90.0, 160.0, 170.0), 0.60, make_emb(RED), None)      # centred, lower score
    bystander = FaceRecord((5.0, 5.0, 45.0, 60.0), 0.95, make_emb(BLUE), None)     # corner, higher score
    shape = (250, 250, 3)
    assert pick_face([target, bystander], shape, "center") is target
    assert pick_face([target, bystander], shape, "score") is bystander


def test_pairs_to_embeddings_reports_multi_face_statistics(tmp_path):
    from ai.face.eval_lfw import load_pairs, pairs_to_embeddings
    root, _, _ = make_lfw(tmp_path / "lfw", 4, 4)

    class TwoFaces:
        def detect(self, img):
            return [FaceRecord((5.0, 5.0, 45.0, 60.0), 0.95, make_emb(BLUE), None),
                    FaceRecord((100.0, 90.0, 150.0, 160.0), 0.6, make_emb(RED), None)]
    pairs, _ = load_pairs(root)
    st: dict = {}
    sims, valid = pairs_to_embeddings(pairs, TwoFaces(), pick="center", stats=st)
    assert st["multi_face"] == st["images"] == st["with_face"] and st["pick_differs"] == st["images"]
    assert all(valid) and sims[0] > 0.99                     # both images of a pair use the centred (RED) face
    sims2, _ = pairs_to_embeddings(pairs, TwoFaces(), pick="score")
    assert sims2[0] > 0.99                                    # also identical here: both pick BLUE, so no information lost


def test_worst_pairs_lists_the_most_wrong_pairs():
    from ai.face.eval_lfw import worst_pairs
    pairs = [(f"a{i}", f"b{i}") for i in range(6)]
    labels, sims, valid = [1, 1, 1, 0, 0, 0], np.array([0.9, 0.2, 0.5, 0.1, 0.8, 0.4], np.float32), [True] * 6
    w = worst_pairs(pairs, labels, sims, valid, n=2)
    assert [x["a"] for x in w["same_person_lowest"]] == ["a1", "a2"]
    assert [x["a"] for x in w["different_person_highest"]] == ["a4", "a5"]


# ---------------------------------------------------------------- degrade_image
def test_degrade_scale_reduces_information():
    from ai.face.eval_lfw import degrade_image
    img = np.random.default_rng(0).integers(0, 255, (250, 250, 3), dtype=np.uint8)
    degraded = degrade_image(img.copy(), scale=0.1)
    # After drastic downscale + upscale the image is very blurry → pixel-level differences from original
    assert degraded.shape == img.shape
    assert not np.array_equal(degraded, img)


def test_degrade_jpeg_changes_image():
    from ai.face.eval_lfw import degrade_image
    img = np.random.default_rng(1).integers(0, 255, (250, 250, 3), dtype=np.uint8)
    degraded = degrade_image(img.copy(), jpeg_quality=5)
    assert degraded.shape == img.shape
    assert not np.array_equal(degraded, img)


def test_degrade_identity_when_no_params():
    from ai.face.eval_lfw import degrade_image
    img = np.zeros((100, 100, 3), np.uint8)
    assert np.array_equal(degrade_image(img.copy(), scale=None, jpeg_quality=None), img)


# ---------------------------------------------------------------- tar_at_far
def test_tar_at_far_perfect_model():
    from ai.face.eval_lfw import tar_at_far
    # Same-person pairs all have sim 0.9, different-person all have 0.1 → perfect separation
    sims = np.array([0.9] * 100 + [0.1] * 100, np.float32)
    labels = [1] * 100 + [0] * 100
    valid = [True] * 200
    res = tar_at_far(sims, labels, valid, far_targets=[0.01, 0.001])
    assert res["tar@far=0.01"] == pytest.approx(1.0, abs=0.01)
    assert res["tar@far=0.001"] == pytest.approx(1.0, abs=0.01)
    assert res["auc"] > 0.99


def test_tar_at_far_random_model_gives_auc_near_half():
    from ai.face.eval_lfw import tar_at_far
    rng = np.random.default_rng(7)
    sims = rng.uniform(-0.5, 0.5, 200).astype(np.float32)
    labels = [1] * 100 + [0] * 100
    valid = [True] * 200
    res = tar_at_far(sims, labels, valid, far_targets=[0.1])
    # A random model should have AUC close to 0.5
    assert 0.3 < res["auc"] < 0.7


def test_tar_at_far_reports_genuine_impostor_counts():
    from ai.face.eval_lfw import tar_at_far
    sims = np.array([0.8] * 50 + [0.2] * 50, np.float32)
    labels = [1] * 50 + [0] * 50
    valid = [True] * 100
    res = tar_at_far(sims, labels, valid, far_targets=[0.01])
    assert res["genuine_pairs"] == 50
    assert res["impostor_pairs"] == 50


def test_tar_at_far_skips_invalid_pairs():
    from ai.face.eval_lfw import tar_at_far
    sims = np.array([0.9, 0.5, 0.1], np.float32)
    labels = [1, 1, 0]
    valid = [True, False, True]   # pair 1 is invalid
    res = tar_at_far(sims, labels, valid, far_targets=[0.1])
    # Only 1 genuine (pair 0) and 1 impostor (pair 2) valid
    assert res["genuine_pairs"] == 1
    assert res["impostor_pairs"] == 1


# ---------------------------------------------------------------- cmd_face_run integration test
def test_cmd_face_run_offline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ai.recording import cmd_face_run
    from ai.runcache import RunWriter
    from ai.schema import TrackRecord

    # Create synthetic run folder
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir()
    meta = {
        "created": "2026-10-19T00:00:00",
        "stride": 1,
        "cameras": {
            "c0": {
                "video": str(run_dir / "c0_dummy.avi"),
                "offset": 0.0,
                "fps": 25.0,
                "stride": 1,
                "frames_total": 5,
                "last_idx": 4,
                "frames_processed": 5,
            }
        },
    }
    writer = RunWriter(run_dir, meta)
    # Write synthetic event
    rec = TrackRecord("c0", 0.0, 0, 1, (10.0, 10.0, 50.0, 100.0), "person", 0.9)
    writer.add("c0", 0.0, 0, [rec], {1: np.zeros(512, np.float32)})
    writer.close()

    # Create dummy video frame
    frame = np.zeros((200, 200, 3), np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vid_out = cv2.VideoWriter(str(run_dir / "c0_dummy.avi"), fourcc, 25.0, (200, 200))
    for _ in range(5):
        vid_out.write(frame)
    vid_out.release()

    # Create mock face index
    idx = FaceIndex(threshold=0.3)
    idx.enroll("p1", "Alice", [make_emb(RED)])
    idx_path = tmp_path / "face_index.json"
    idx.save(idx_path)

    # Mock FaceDetector inside cmd_face_run
    class DummyDetector:
        def __init__(self, *args, **kwargs):
            pass
        def detect(self, img):
            return [FaceRecord((5.0, 5.0, 35.0, 45.0), 0.95, make_emb(RED, 0.01, 1), None)]

    all_recs = [TrackRecord("c0", i * 0.04, i, 1, (10.0, 10.0, 50.0, 100.0), "person", 0.9) for i in range(5)]

    monkeypatch.setattr("ai.face.detector.FaceDetector", DummyDetector)
    monkeypatch.setattr("ai.recording.load_run", lambda p: SimpleNamespace(dir=Path(p), events=[], meta=meta))
    monkeypatch.setattr("ai.recording.replay", lambda run, cfg, topo, stale_after_s: all_recs)

    class MockMediaReader:
        def __init__(self, media, stride=1, folder_fps=25.0):
            self.fps = folder_fps
            self.stride = stride
        def frames(self):
            for i in range(5):
                yield i, np.zeros((200, 200, 3), np.uint8)

    monkeypatch.setattr("ai.sources.MediaReader", MockMediaReader)





    args = SimpleNamespace(
        run=str(run_dir),
        index=str(idx_path),
        threshold=0.3,
        model_root="dummy",
        det_size=320,
        min_det_score=0.4,
        min_face_height=30.0,
        run_every=1,
        min_hits=1,
        stale_after=3.0,
        config=None,
        render=True,
    )

    ret = cmd_face_run(args)
    assert ret == 0
    assert (run_dir / "face_events.jsonl").exists()
    assert (run_dir / "face_summary.json").exists()
    assert (run_dir / "face_render" / "c0.mp4").exists()
    
    summary = json.loads((run_dir / "face_summary.json").read_text())
    assert "p1" in summary
    assert summary["p1"]["name"] == "Alice"
    assert summary["p1"]["hits"] > 0


