"""Phase 2 tests: Re-ID data/model/metrics, cross-camera matching, multi-camera pipeline, MTMC scoring, storage.

Default (pytest):  everything except the database tests.
Database tests:    make sure Postgres is up and `python scripts/init_db.py` has run, then  pytest -m infra -k store
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from ai.crosscam import GlobalMatcher, MatcherConfig, Topology, TrackletInfo  # noqa: E402
from ai.mot_io import write_mot  # noqa: E402
from ai.reid.data import PKSampler, find_market_root, parse_market_name  # noqa: E402
from ai.reid.losses import LabelSmoothCE, batch_hard_triplet  # noqa: E402
from ai.reid.metrics import evaluate_market  # noqa: E402
from ai.schema import TrackRecord  # noqa: E402
from ai.tracklets import TrackletBook  # noqa: E402


def unit(*v):
    v = np.array(v, np.float32)
    return v / np.linalg.norm(v)


def angle_vec(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)], np.float32)


# ---------------------------------------------------------------- data
def test_parse_market_name():
    assert parse_market_name("0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_market_name("/x/-1_c3s2_000551_00.jpg") == (-1, 3)
    assert parse_market_name("0000_c6s3_001_00.jpg") == (0, 6)
    with pytest.raises(ValueError):
        parse_market_name("random.jpg")


def test_find_market_root_searches_below(tmp_path):
    (tmp_path / "a" / "Market-1501-v15.09.15" / "bounding_box_train").mkdir(parents=True)
    assert find_market_root(tmp_path).name == "Market-1501-v15.09.15"
    with pytest.raises(FileNotFoundError):
        find_market_root(tmp_path / "nothing")


def test_pk_sampler_gives_p_identities_with_k_images_each():
    labels = [i // 5 for i in range(200)]                 # 40 people x 5 images
    s = PKSampler(labels, p=8, k=4, seed=1)
    batches = list(s)
    assert len(batches) == len(s) == 200 // 32
    for b in batches:
        got = [labels[i] for i in b]
        assert len(b) == 32 and len(set(got)) == 8
        assert all(got.count(x) == 4 for x in set(got))


def test_pk_sampler_handles_people_with_few_images():
    s = PKSampler([0, 0, 1, 2, 2, 2, 3, 3], p=4, k=3)
    assert len(next(iter(s))) == 12


# ---------------------------------------------------------------- metrics
def test_rank_and_map_match_hand_calculation():
    # query person 1 (cam 1). Gallery, nearest first: junk, same-person-same-cam (ignored), person 2, person 1 (cam 2), person 1 (cam 3)
    q = angle_vec(0)[None]
    g = np.stack([angle_vec(2), angle_vec(4), angle_vec(10), angle_vec(20), angle_vec(30)])
    gp, gc = np.array([-1, 1, 2, 1, 1]), np.array([2, 1, 2, 2, 3])
    r = evaluate_market(q, np.array([1]), np.array([1]), g, gp, gc)
    assert r["rank1"] == 0.0 and r["rank5"] == 100.0
    assert r["mAP"] == pytest.approx((1 / 2 + 2 / 3) / 2 * 100, abs=0.01)   # matches at ranks 2 and 3


def test_perfect_ranking_scores_100_and_skips_queries_without_match():
    qf = np.stack([angle_vec(0), angle_vec(90)])
    gf = np.stack([angle_vec(1), angle_vec(91), angle_vec(180)])
    r = evaluate_market(qf, np.array([1, 9]), np.array([1, 1]), gf, np.array([1, 9, 5]), np.array([2, 2, 2]))
    assert r["rank1"] == 100 and r["mAP"] == pytest.approx(100)
    r2 = evaluate_market(qf, np.array([1, 77]), np.array([1, 1]), gf, np.array([1, 9, 5]), np.array([2, 2, 2]))
    assert r2["valid_queries"] == 1 and r2["skipped_queries"] == 1


# ---------------------------------------------------------------- losses and model
def test_triplet_loss_zero_when_separated_and_positive_when_mixed():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[1., 0], [1, 0.01], [0, 1], [0.01, 1]])
    bad = torch.tensor([[1., 0], [0, 1], [1, 0.01], [0.01, 1]])
    assert batch_hard_triplet(good, labels).item() == pytest.approx(0.0, abs=1e-6)
    assert batch_hard_triplet(bad, labels).item() > 0.3


def test_label_smooth_ce_is_finite_and_lower_for_correct_logits():
    ce = LabelSmoothCE()
    t = torch.tensor([0, 1])
    good, bad = torch.tensor([[5., 0], [0, 5]]), torch.tensor([[0., 5], [5, 0]])
    assert torch.isfinite(ce(good, t)) and ce(good, t) < ce(bad, t)


def test_model_shapes_and_last_stride():
    from ai.reid.model import ReIDNet
    m = ReIDNet("resnet18", num_classes=10, feat_dim=64, pretrained=False)
    x = torch.randn(4, 3, 256, 128)
    assert m.body(x).shape[-2:] == (16, 8)            # last stride 1 keeps a bigger feature map
    m.train()
    logits, feat = m(x)
    assert logits.shape == (4, 10) and feat.shape == (4, 64)
    m.eval()
    assert m(x).shape == (4, 64)


def test_preprocess_and_crop_person():
    from ai.reid.embedder import crop_person, preprocess
    batch = preprocess([np.zeros((100, 40, 3), np.uint8), np.full((300, 90, 3), 255, np.uint8)])
    assert batch.shape == (2, 3, 256, 128) and batch.dtype == np.float32
    frame = np.zeros((480, 640, 3), np.uint8)
    assert crop_person(frame, (10, 10, 30, 40)) is None                       # too small
    assert crop_person(frame, (-50, -50, 70, 150)).shape[:2] == (150, 70)     # clamped to the frame


# ---------------------------------------------------------------- tracklets
def test_tracklet_mean_embedding_is_unit_length_and_expiry_works():
    book = TrackletBook(stale_after_s=2.0)
    for t in range(4):
        st = book.touch("c1", 5, float(t))
        st.embs.append(unit(1, 0.1 * t, 0))
    assert np.linalg.norm(book.states[("c1", 5)].mean_embedding()) == pytest.approx(1.0, abs=1e-5)
    assert book.expire("c1", 4.0) == []                 # last seen at 3.0 -> not stale yet
    assert len(book.expire("c1", 6.0)) == 1 and not book.states
    assert book.expire("c2", 99.0) == []


# ---------------------------------------------------------------- cross-camera matcher
def T(cam, tid, first, last, emb):
    return TrackletInfo(cam, tid, first, last, emb)


RED, BLUE = unit(1, 0, 0), unit(0, 1, 0)


def test_same_person_in_another_camera_gets_same_global_id_and_different_person_does_not():
    m = GlobalMatcher()
    a = m.match([T("c1", 1, 0, 10, RED), T("c1", 2, 0, 10, BLUE)])
    m.end_track("c1", 1, 10)
    m.end_track("c1", 2, 10)
    b = m.match([T("c2", 7, 20, 30, unit(1, 0.05, 0)), T("c2", 8, 20, 30, unit(0.05, 1, 0))])
    assert a[("c1", 1)] != a[("c1", 2)]
    assert b[("c2", 7)] == a[("c1", 1)] and b[("c2", 8)] == a[("c1", 2)]


def test_two_look_alikes_in_one_camera_never_share_an_id():
    m = GlobalMatcher()
    first = m.match([T("c1", 1, 0, 5, RED)])
    second = m.match([T("c1", 2, 3, 8, RED)])          # same clothes, same camera, overlapping in time
    assert first[("c1", 1)] != second[("c1", 2)]


def test_joint_assignment_gives_one_identity_to_only_one_of_two_candidates():
    m = GlobalMatcher()
    g = m.match([T("c1", 1, 0, 5, RED)])[("c1", 1)]
    m.end_track("c1", 1, 5)
    res = m.match([T("c2", 3, 10, 12, RED), T("c3", 4, 10, 12, unit(1, 0.02, 0))])
    assert sorted(res.values()).count(g) == 1 and len(set(res.values())) == 2


def test_identity_not_matched_after_max_gap():
    m = GlobalMatcher(MatcherConfig(max_gap_s=100))
    g = m.match([T("c1", 1, 0, 5, RED)])[("c1", 1)]
    m.end_track("c1", 1, 5)
    assert m.match([T("c2", 2, 500, 510, RED)])[("c2", 2)] != g


def test_travel_time_between_cameras_is_enforced():
    topo = Topology(min_travel_s={("c1", "c2"): 30.0})
    m = GlobalMatcher(topology=topo)
    g = m.match([T("c1", 1, 0, 10, RED)])[("c1", 1)]
    m.end_track("c1", 1, 10)
    assert m.match([T("c2", 2, 20, 25, RED)])[("c2", 2)] != g     # only 10 s later: impossible
    m2 = GlobalMatcher(topology=topo)
    g2 = m2.match([T("c1", 1, 0, 10, RED)])[("c1", 1)]
    m2.end_track("c1", 1, 10)
    assert m2.match([T("c2", 2, 45, 50, RED)])[("c2", 2)] == g2   # 35 s later: fine


def test_cannot_be_in_two_non_overlapping_cameras_at_once_unless_they_overlap():
    m = GlobalMatcher()
    g = m.match([T("c1", 1, 0, 50, RED)])[("c1", 1)]               # still visible in c1
    assert m.match([T("c2", 2, 10, 20, RED)])[("c2", 2)] != g
    m2 = GlobalMatcher(topology=Topology(overlapping={frozenset({"c1", "c2"})}))
    g2 = m2.match([T("c1", 1, 0, 50, RED)])[("c1", 1)]
    assert m2.match([T("c2", 2, 10, 20, RED)])[("c2", 2)] == g2    # overlapping cameras may share a view


def test_track_ending_in_same_camera_allows_rematch_after_fragmentation():
    m = GlobalMatcher()
    g = m.match([T("c1", 1, 0, 10, RED)])[("c1", 1)]
    m.end_track("c1", 1, 10)
    assert m.match([T("c1", 9, 14, 20, RED)])[("c1", 9)] == g


def test_already_bound_tracklets_are_not_reassigned():
    m = GlobalMatcher()
    t = T("c1", 1, 0, 5, RED)
    first = m.match([t])
    assert m.match([t]) == {} and m.identity_for("c1", 1) == first[("c1", 1)]


# ---------------------------------------------------------------- multi-camera pipeline (stub tracker + embedder)
class StubTracker:
    def __init__(self, camera_id, script):
        self.camera_id, self.script = camera_id, script      # script: {idx: [(track_id, bbox, colour), ...]}

    def update(self, image, ts, idx):
        return [TrackRecord(self.camera_id, ts, idx, tid, bb, "person", 0.9) for tid, bb, _ in self.script.get(idx, [])]


class ColourEmbedder:
    """'Appearance' = mean colour of the crop, so people wearing the same colour look identical."""
    def embed(self, crops):
        v = np.stack([c.reshape(-1, 3).mean(axis=0) for c in crops]).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def frame_with(people):
    img = np.zeros((480, 640, 3), np.uint8)
    for _, (x1, y1, x2, y2), colour in people:
        img[y1:y2, x1:x2] = colour
    return img


def test_pipeline_links_same_clothes_across_cameras_and_reports_finished_tracklets():
    from ai.multicam import MultiCamPipeline
    RED_BGR, BLUE_BGR = (0, 0, 255), (255, 0, 0)
    a_box, b_box = (50, 50, 150, 250), (300, 50, 400, 250)
    cam1 = {i: [(1, a_box, RED_BGR), (2, b_box, BLUE_BGR)] for i in range(10)}
    cam2 = {i: [(7, b_box, BLUE_BGR), (8, a_box, RED_BGR)] for i in range(10)}      # different ids, order, positions
    trackers = {"c1": StubTracker("c1", cam1), "c2": StubTracker("c2", cam2)}
    ended = []
    pipe = MultiCamPipeline(trackers, ColourEmbedder(), GlobalMatcher(), on_tracklet_end=ended.append)

    out1 = [pipe.process("c1", frame_with(cam1[i]), float(i), i) for i in range(10)]
    ids1 = {r.track_id: r.global_id for r in out1[-1]}
    assert out1[0][0].global_id is None                       # not enough evidence in the very first frame
    assert ids1[1] is not None and ids1[2] is not None and ids1[1] != ids1[2]

    # c1 goes quiet; much later c2 sees the same two people
    pipe.process("c1", np.zeros((480, 640, 3), np.uint8), 20.0, 10)
    out2 = [pipe.process("c2", frame_with(cam2[i]), 30.0 + i, i) for i in range(10)]
    ids2 = {r.track_id: r.global_id for r in out2[-1]}
    assert ids2[8] == ids1[1] and ids2[7] == ids1[2]          # red <-> red, blue <-> blue

    pipe.flush()
    assert {(s.camera_id, s.track_id) for s in ended} == {("c1", 1), ("c1", 2), ("c2", 7), ("c2", 8)}
    assert all(s.global_id is not None and len(s.embs) > 0 for s in ended)


def test_pipeline_leaves_vehicles_without_global_id():
    from ai.multicam import MultiCamPipeline

    class CarTracker:
        def update(self, image, ts, idx):
            return [TrackRecord("c1", ts, idx, 3, (10, 10, 200, 100), "car", 0.9)]
    pipe = MultiCamPipeline({"c1": CarTracker()}, ColourEmbedder(), GlobalMatcher())
    out = pipe.process("c1", np.zeros((480, 640, 3), np.uint8), 0.0, 0)
    assert out[0].cls == "car" and out[0].global_id is None


# ---------------------------------------------------------------- MTMC scoring (needs TrackEval)
def mtmc_cams(swap_in_b: bool):
    gt_a, gt_b, pred_a, pred_b = [], [], [], []
    for f in range(1, 41):
        for gid, x0 in ((1, 50), (2, 400)):
            box = (x0 + f, 100, x0 + f + 40, 200)
            gt_a.append((f, gid, *box))
            gt_b.append((f, gid, *box))
            pred_a.append((f, gid, *box, 0.9))
            pid = ({1: 2, 2: 1}[gid] if swap_in_b else gid)
            pred_b.append((f, pid, *box, 0.9))
    return {"A": {"gt": gt_a, "pred": pred_a, "n_frames": 40}, "B": {"gt": gt_b, "pred": pred_b, "n_frames": 40}}


def test_mtmc_scores_consistent_global_ids_as_perfect(tmp_path):
    pytest.importorskip("trackeval")
    from ai.mtmc_eval import evaluate_mtmc
    c = evaluate_mtmc(tmp_path, mtmc_cams(False), print_results=False)["combined"]
    assert c["IDF1"] == pytest.approx(100, abs=0.5) and c["HOTA"] == pytest.approx(100, abs=0.5)


def test_mtmc_punishes_a_different_global_id_in_the_second_camera(tmp_path):
    pytest.importorskip("trackeval")
    from ai.mtmc_eval import evaluate_mtmc
    c = evaluate_mtmc(tmp_path, mtmc_cams(True), print_results=False)["combined"]
    assert c["IDF1"] < 60 and c["DetA"] == pytest.approx(100, abs=0.5)    # boxes right, cross-camera identity wrong


# ---------------------------------------------------------------- end-to-end training smoke test (tiny synthetic data)
def make_fake_market(root: Path):
    import cv2
    rng = np.random.default_rng(0)
    for d in ("bounding_box_train", "bounding_box_test", "query"):
        (root / d).mkdir(parents=True)

    def person(pid, cam):
        r = np.random.default_rng(pid * 7 + 1)
        img = np.zeros((128, 64, 3), np.uint8)
        img[:64], img[64:] = r.integers(30, 255, 3), r.integers(30, 255, 3)
        return np.clip(img.astype(int) + rng.integers(-20, 20, img.shape), 0, 255).astype(np.uint8)
    n = 0
    for pid in range(1, 13):
        for cam in range(1, 4):
            for _ in range(2):
                cv2.imwrite(str(root / "bounding_box_train" / f"{pid:04d}_c{cam}s1_{n:06d}_00.jpg"), person(pid, cam)); n += 1
    for pid in range(101, 109):
        cv2.imwrite(str(root / "query" / f"{pid:04d}_c1s1_{n:06d}_00.jpg"), person(pid, 1)); n += 1
        for cam in (2, 3):
            cv2.imwrite(str(root / "bounding_box_test" / f"{pid:04d}_c{cam}s1_{n:06d}_00.jpg"), person(pid, cam)); n += 1


def test_training_runs_end_to_end_and_writes_loadable_checkpoint(tmp_path):
    from ai.reid.embedder import ReIDEmbedder
    from ai.reid.train import main as train_main
    make_fake_market(tmp_path / "m")
    out = tmp_path / "out"
    rc = train_main(["--data", str(tmp_path / "m"), "--out", str(out), "--backbone", "resnet18", "--feat-dim", "32",
                     "--epochs", "1", "--p", "4", "--k", "2", "--workers", "0", "--no-pretrained",
                     "--device", "cpu", "--max-batches", "2", "--eval-every", "1"])
    assert rc == 0 and (out / "best.pt").exists() and (out / "metrics.json").exists()
    import json
    m = json.loads((out / "metrics.json").read_text())
    assert {"rank1", "rank5", "mAP", "threshold_suggestion"} <= set(m)
    e = ReIDEmbedder(str(out / "best.pt"), device="cpu").embed([np.zeros((100, 40, 3), np.uint8)])
    assert e.shape == (1, 32) and np.linalg.norm(e) == pytest.approx(1.0, abs=1e-4)


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    from ai.reid.embedder import ReIDEmbedder
    from ai.reid.export import OnnxEmbedder, export_onnx
    from ai.reid.model import ReIDNet, save_checkpoint
    ck = tmp_path / "ck.pt"
    save_checkpoint(ck, ReIDNet("resnet18", 5, 32, pretrained=False).eval(), "resnet18", 5, 0, None)
    onnx = export_onnx(str(ck), str(tmp_path / "m.onnx"))
    crops = [np.random.default_rng(i).integers(0, 255, (120, 50, 3), dtype=np.uint8) for i in range(3)]
    a, b = ReIDEmbedder(str(ck), device="cpu").embed(crops), OnnxEmbedder(onnx).embed(crops)
    assert np.abs(a - b).max() < 1e-3


# ---------------------------------------------------------------- database (needs Postgres + scripts/init_db.py)
@pytest.fixture
def store():
    pytest.importorskip("psycopg")
    from ai.store import IdentityStore
    s = IdentityStore()
    yield s
    s.conn.execute("DELETE FROM global_identities WHERE global_id >= 9000000")
    s.close()


def fake_state(gid, cam, tid, t0, t1, emb):
    from ai.tracklets import TrackletState
    st = TrackletState(cam, tid, t0, t1, n_frames=10, global_id=gid)
    st.embs.append(emb)
    return st


@pytest.mark.infra
def test_store_saves_searches_and_orders_trajectory(store):
    rng = np.random.default_rng(1)
    embs = [(v / np.linalg.norm(v)).astype(np.float32) for v in rng.normal(size=(3, 512))]
    store.save_tracklet(fake_state(9000001, "c2", 5, 200.0, 210.0, embs[0]))
    store.save_tracklet(fake_state(9000001, "c1", 3, 100.0, 110.0, embs[1]))
    store.save_tracklet(fake_state(9000002, "c1", 4, 100.0, 110.0, embs[2]))
    store.save_tracklet(fake_state(9000001, "c1", 3, 100.0, 112.0, embs[1]))      # duplicate: must not add a row
    hits = [h for h in store.nearest(embs[0], k=50) if h["global_id"] >= 9000000]
    assert hits[0]["global_id"] == 9000001 and hits[0]["camera_id"] == "c2"
    assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-3)
    traj = store.trajectory(9000001)
    assert [t["camera_id"] for t in traj] == ["c1", "c2"] and len(traj) == 2
    assert traj[0]["last_ts"] == 112.0
