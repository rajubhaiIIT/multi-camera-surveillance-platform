"""WILDTRACK loader + ground-truth scoring, on a small mock copy of the official folder layout."""
import json
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from ai.crosscam import Topology  # noqa: E402
from ai.schema import TrackRecord  # noqa: E402

N, RED, BLUE = 10, (0, 0, 255), (255, 0, 0)
# person 1 (red) is visible in cameras 0 and 1, person 2 (blue) in cameras 1 and 2
VISIBLE = {1: {0, 1}, 2: {1, 2}}
COLOUR = {1: RED, 2: BLUE}


def box(pid, f):
    x = 60 + 120 * pid + 4 * f
    return (x, 60 + 3 * f, x + 50, 220 + 3 * f)


def make_mock(root: Path, view_base: int = 0) -> Path:
    base = root / "Wildtrack_dataset"
    (base / "annotations_positions").mkdir(parents=True)
    for cam in range(3):
        (base / "Image_subsets" / f"C{cam + 1}").mkdir(parents=True)
    for f in range(N):
        stem = f"{5 * f:08d}"
        frame = []
        for pid in (1, 2):
            views = []
            for cam in range(3):
                if cam in VISIBLE[pid]:
                    x1, y1, x2, y2 = box(pid, f)
                    views.append({"viewNum": cam + view_base, "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2})
                else:
                    views.append({"viewNum": cam + view_base, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1})
            frame.append({"personID": pid, "positionID": 1000 + pid, "views": views})
        (base / "annotations_positions" / f"{stem}.json").write_text(json.dumps(frame))
        for cam in range(3):
            img = np.zeros((300, 640, 3), np.uint8)
            for pid in (1, 2):
                if cam in VISIBLE[pid]:
                    x1, y1, x2, y2 = box(pid, f)
                    img[y1:y2, x1:x2] = COLOUR[pid]
            cv2.imwrite(str(base / "Image_subsets" / f"C{cam + 1}" / f"{stem}.png"), img)
    return root


class GtTracker:
    """Perfect detector: returns the ground-truth boxes of this camera (track id = person id x 10 + camera)."""
    def __init__(self, cam_idx):
        self.cam = cam_idx

    def update(self, image, ts, idx):
        return [TrackRecord(f"C{self.cam + 1}", ts, idx, pid * 10 + self.cam, tuple(map(float, box(pid, idx))), "person", 0.9)
                for pid in (1, 2) if self.cam in VISIBLE[pid]]


class ColourEmbedder:
    def embed(self, crops):
        v = np.stack([c.reshape(-1, 3).mean(axis=0) for c in crops]).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_loader_reads_frames_and_only_visible_boxes(tmp_path):
    from ai.wildtrack import load_wildtrack
    images, gt = load_wildtrack(make_mock(tmp_path))
    assert sorted(images) == ["C1", "C2", "C3"] and all(len(v) == N for v in images.values())
    assert images["C1"][1].endswith("00000005.png")
    assert (len(gt["C1"]), len(gt["C2"]), len(gt["C3"])) == (N, 2 * N, N)          # invisible views (-1) dropped
    assert {row[1] for row in gt["C2"]} == {1, 2} and gt["C1"][0][:2] == [0, 1]


def test_loader_handles_one_based_view_numbers(tmp_path):
    from ai.wildtrack import load_wildtrack
    _, gt = load_wildtrack(make_mock(tmp_path, view_base=1))
    assert (len(gt["C1"]), len(gt["C2"]), len(gt["C3"])) == (N, 2 * N, N)


def test_loader_camera_and_frame_filters_and_errors(tmp_path):
    from ai.wildtrack import find_root, load_wildtrack
    root = make_mock(tmp_path)
    images, gt = load_wildtrack(root, cams=["C2", "C3"], max_frames=4)
    assert sorted(images) == ["C2", "C3"] and len(images["C2"]) == 4 and max(r[0] for r in gt["C2"]) == 3
    with pytest.raises(ValueError, match="unknown camera"):
        load_wildtrack(root, cams=["C9"])
    with pytest.raises(FileNotFoundError):
        find_root(tmp_path / "nowhere")


def run_mock(tmp_path):
    from ai.multicam import FrameEmbedder
    from ai.multicam_extract import extract_run
    from ai.wildtrack import load_wildtrack
    images, gt = load_wildtrack(make_mock(tmp_path))
    fe = FrameEmbedder({f"C{i + 1}": GtTracker(i) for i in range(3)}, ColourEmbedder(), embed_every=2)
    from ai.runcache import load_run
    return load_run(extract_run(images, {}, fe, tmp_path / "run", stride=1, folder_fps=2.0, log=None)), gt


def test_extract_from_image_list_uses_2fps_timestamps(tmp_path):
    run, _ = run_mock(tmp_path)
    ts = sorted({e.ts for e in run.events if e.cam == "C1"})
    assert ts[:3] == [0.0, 0.5, 1.0] and run.meta["cameras"]["C1"]["fps"] == 2.0
    assert run.meta["cameras"]["C1"]["last_idx"] == N - 1 and isinstance(run.meta["cameras"]["C1"]["video"], list)


def test_overlapping_cameras_give_the_same_person_one_id_and_scores_100(tmp_path):
    from ai.labeling import score_run_gt
    from ai.replay import replay
    run, gt = run_mock(tmp_path)
    res = score_run_gt(run, replay(run, topology=Topology(all_overlapping=True)), gt)
    assert res["true_people"] == 2 and res["predicted_identities"] == 2
    assert res["mtmc"]["IDF1"] == pytest.approx(100, abs=0.5) and res["mtmc"]["DetA"] == pytest.approx(100, abs=0.5)


def test_without_the_overlap_rule_a_person_seen_by_two_cameras_at_once_is_split(tmp_path):
    from ai.labeling import score_run_gt
    from ai.replay import replay
    run, gt = run_mock(tmp_path)
    res = score_run_gt(run, replay(run, topology=Topology()), gt)       # default: cameras cannot overlap
    assert res["predicted_identities"] == 4 and res["mtmc"]["IDF1"] < 90


def test_scoring_penalises_missed_people(tmp_path):
    from ai.labeling import score_run_gt
    from ai.replay import replay
    run, gt = run_mock(tmp_path)
    gt2 = {c: rows + [[r[0], 100 + r[1], r[2] + 200, r[3], r[4] + 200, r[5]] for r in rows] for c, rows in gt.items()}  # extra unseen people
    res = score_run_gt(run, replay(run, topology=Topology(all_overlapping=True)), gt2)
    assert res["mtmc"]["DetA"] < 70 and res["mtmc"]["FN"] > 0


def test_render_works_from_image_lists(tmp_path):
    from ai.replay import render_run, replay
    run, _ = run_mock(tmp_path)
    paths = render_run(run, replay(run, topology=Topology(all_overlapping=True)), tmp_path / "render")
    assert len(paths) == 3 and all(p.stat().st_size > 0 for p in paths)


def test_cli_prints_a_clear_error_instead_of_a_traceback(tmp_path, capsys):
    from ai import recording
    assert recording.main(["wildtrack", "--root", str(tmp_path / "missing"), "--out", str(tmp_path / "o")]) == 1
    assert "error:" in capsys.readouterr().out


# ---------------------------------------------------------------- diagnose + oracle
def run_with(tmp_path, tracker_factory):
    from ai.multicam import FrameEmbedder
    from ai.multicam_extract import extract_run
    from ai.runcache import load_run
    from ai.wildtrack import load_wildtrack
    images, gt = load_wildtrack(make_mock(tmp_path))
    fe = FrameEmbedder({f"C{i + 1}": tracker_factory(i) for i in range(3)}, ColourEmbedder(), embed_every=2)
    return load_run(extract_run(images, {}, fe, tmp_path / "run", stride=1, folder_fps=2.0, log=None)), gt


def test_diagnose_perfect_detector(tmp_path):
    from ai.diagnose import detection_report
    run, gt = run_with(tmp_path, GtTracker)
    t = detection_report(run, gt)["total"]
    assert t["recall"] == 100.0 and t["precision_all"] == 100.0 and t["fp_inside"] + t["fp_outside"] + t["fp_unknown"] == 0


class ExtraBoxTracker(GtTracker):
    """Perfect, plus one box per frame far from where anyone is annotated."""
    def update(self, image, ts, idx):
        recs = super().update(image, ts, idx)
        return recs + [TrackRecord(f"C{self.cam + 1}", ts, idx, 900 + self.cam, (570.0, 250.0, 620.0, 295.0), "person", 0.6)]


def test_diagnose_separates_false_positives_outside_the_annotated_area(tmp_path):
    from ai.diagnose import detection_report
    run, gt = run_with(tmp_path, ExtraBoxTracker)
    t = detection_report(run, gt)["total"]
    # only camera C2 sees two people, so only C2 has an estimable area; C1 and C3 extra boxes are "unknown"
    assert t["recall"] == 100.0 and t["fp_outside"] == N and t["fp_inside"] == 0 and t["fp_unknown"] == 2 * N
    assert t["precision_all"] < 100 and t["precision_inside_area"] == 100.0 and t["fp_outside_share"] == 100.0


class OnlyFirstPerson(GtTracker):
    def update(self, image, ts, idx):
        return [r for r in super().update(image, ts, idx) if r.track_id // 10 == 1]


def test_diagnose_counts_missed_people(tmp_path):
    from ai.diagnose import detection_report
    run, gt = run_with(tmp_path, OnlyFirstPerson)
    t = detection_report(run, gt)["total"]
    assert t["fn"] == 2 * N and t["recall"] == 50.0      # person 2 (2 cameras x N frames) is never found


def test_oracle_tracker_returns_the_ground_truth(tmp_path):
    from ai.wildtrack import OracleTracker, load_wildtrack
    _, gt = load_wildtrack(make_mock(tmp_path))
    recs = OracleTracker("C2", gt["C2"]).update(None, 1.0, 2)
    assert sorted(r.track_id for r in recs) == [1, 2] and all(r.conf == 1.0 and r.cls == "person" for r in recs)
    assert OracleTracker("C2", gt["C2"]).update(None, 0, 999) == []


def test_cli_oracle_diagnose_and_score_run_end_to_end(tmp_path, capsys):
    from ai import recording
    from ai.reid.model import ReIDNet, save_checkpoint
    ck = tmp_path / "ck.pt"
    save_checkpoint(ck, ReIDNet("resnet18", 5, 32, pretrained=False).eval(), "resnet18", 5, 0, None)
    root, out = make_mock(tmp_path / "data"), tmp_path / "out"
    assert recording.main(["wildtrack", "--root", str(root), "--out", str(out), "--oracle",
                           "--ckpt", str(ck), "--device", "cpu"]) == 0
    assert "ORACLE" in capsys.readouterr().out
    assert recording.main(["diagnose", str(out)]) == 0
    assert "found 100.0% of ground-truth people" in capsys.readouterr().out
    assert recording.main(["score", str(out), "--stale-after", "6"]) == 0
    assert recording.main(["diagnose", str(tmp_path)]) == 1          # not a run folder: clear error, not a crash


# ---------------------------------------------------------------- Re-ID crop export + fine-tuning
def make_mock_with_late_people(root: Path) -> Path:
    """The mock plus two people (3, 4) who only appear from frame 5 on, so a person-disjoint split exists."""
    make_mock(root)
    ann = root / "Wildtrack_dataset" / "annotations_positions"
    for f in range(5, N):
        p = ann / f"{5 * f:08d}.json"
        frame = json.loads(p.read_text())
        for pid in (3, 4):
            x = 20 + 100 * pid + 4 * f
            views = [{"viewNum": c, "xmin": x, "ymin": 40, "xmax": x + 50, "ymax": 200} if c == 0 else
                     {"viewNum": c, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1} for c in range(3)]
            frame.append({"personID": pid, "positionID": 2000 + pid, "views": views})
        p.write_text(json.dumps(frame))
    return root


def export(tmp_path, **kw):
    from ai.wildtrack import export_reid_dataset, load_wildtrack
    images, gt = load_wildtrack(make_mock_with_late_people(tmp_path))
    return export_reid_dataset(images, gt, tmp_path / "wt_reid", test_frames=5, train_stride=1, **kw), tmp_path / "wt_reid"


def test_export_makes_a_person_disjoint_market_style_dataset(tmp_path):
    from ai.reid.data import load_split, parse_market_name
    info, out = export(tmp_path)
    assert info["test_people"] == 2 and info["train_people"] == 2
    train = {pid for _, pid, _ in load_split(out, "train")}
    query = [(pid, cam) for _, pid, cam in load_split(out, "query")]
    gallery = {pid for _, pid, _ in load_split(out, "gallery")}
    assert train == {3, 4} and gallery == {1, 2} and {p for p, _ in query} == {1, 2}     # nobody is in both
    assert sorted(query) == [(1, 1), (1, 2), (2, 2), (2, 3)]                              # one query per (person, camera)
    assert all(parse_market_name(f.name)[1] >= 1 for f in (out / "query").glob("*.jpg"))


def test_export_needs_people_outside_the_test_frames(tmp_path):
    from ai.wildtrack import export_reid_dataset, load_wildtrack
    images, gt = load_wildtrack(make_mock(tmp_path))                  # only persons 1, 2, who are present from frame 0
    with pytest.raises(ValueError, match="no training people"):
        export_reid_dataset(images, gt, tmp_path / "o", test_frames=5)


def test_exported_dataset_can_be_scored_with_the_market_tools(tmp_path):
    import torch
    from ai.reid.eval_market import evaluate_folder
    from ai.reid.model import ReIDNet
    _, out = export(tmp_path)
    res = evaluate_folder(ReIDNet("resnet18", 4, 32, pretrained=False).eval(), str(out), "cpu", workers=0, with_threshold=False)
    assert res["queries"] == 4 and 0 <= res["rank1"] <= 100 and res["valid_queries"] >= 1


def test_load_init_copies_weights_but_not_the_classifier(tmp_path):
    from ai.reid.model import ReIDNet, save_checkpoint
    from ai.reid.train import load_init
    src = ReIDNet("resnet18", 5, 32, pretrained=False).eval()
    ck = tmp_path / "ck.pt"
    save_checkpoint(ck, src, "resnet18", 5, 0, None)
    dst = ReIDNet("resnet18", 9, 32, pretrained=False)                # different number of training people
    n, missing = load_init(dst, str(ck))
    assert n > 0 and missing == ["classifier.weight"]
    assert torch.equal(dst.body[0].weight, src.body[0].weight)
    with pytest.raises(ValueError, match="embedding size"):
        load_init(ReIDNet("resnet18", 9, 64, pretrained=False), str(ck))


def test_finetune_command_runs_on_the_exported_dataset(tmp_path):
    from ai.reid.model import ReIDNet, save_checkpoint
    from ai.reid.train import main as train_main
    _, out = export(tmp_path)
    ck = tmp_path / "init.pt"
    save_checkpoint(ck, ReIDNet("resnet18", 751, 32, pretrained=False).eval(), "resnet18", 751, 0, None)
    rc = train_main(["--data", str(out), "--out", str(tmp_path / "ft"), "--backbone", "resnet18", "--feat-dim", "32",
                     "--epochs", "1", "--p", "2", "--k", "2", "--workers", "0", "--no-pretrained", "--device", "cpu",
                     "--max-batches", "2", "--eval-every", "1", "--init-ckpt", str(ck)])
    assert rc == 0 and (tmp_path / "ft" / "best.pt").exists()


def test_export_cli_prints_the_next_step(tmp_path, capsys):
    from ai import recording
    root = make_mock_with_late_people(tmp_path / "d")
    assert recording.main(["export-reid", "--root", str(root), "--out", str(tmp_path / "o"),
                           "--test-frames", "5", "--train-stride", "1"]) == 0
    assert "eval_market" in capsys.readouterr().out


# ---------------------------------------------------------------- detection-only evaluation
def _box(x, y=0, w=50, h=100):
    return (float(x), float(y), float(x + w), float(y + h))


def test_average_precision_matches_hand_calculations():
    from ai.detect_eval import average_precision
    gts = {"f": np.array([_box(0), _box(200)], float)}
    # TP (0.9), FP (0.8), TP (0.7): precision 1 at recall 0.5, then 2/3 at recall 1  ->  0.5*1 + 0.5*2/3
    dets = [("f", 0.9, _box(0)), ("f", 0.8, _box(500)), ("f", 0.7, _box(200))]
    assert average_precision(dets, gts) == pytest.approx(0.5 + 0.5 * 2 / 3, abs=1e-6)
    one = {"f": np.array([_box(0)], float)}
    assert average_precision([("f", 0.9, _box(500)), ("f", 0.8, _box(0))], one) == pytest.approx(0.5)   # FP ranked first
    assert average_precision([("f", 0.9, _box(0))], one) == pytest.approx(1.0)
    assert average_precision([], one) == 0.0 and average_precision([("f", 0.9, _box(0))], {"f": np.zeros((0, 4))}) is None


def test_each_ground_truth_box_can_only_be_found_once():
    from ai.detect_eval import average_precision
    one = {"f": np.array([_box(0)], float)}
    two_on_one = [("f", 0.9, _box(0)), ("f", 0.8, _box(1))]            # the second is a duplicate: a false positive
    assert average_precision(two_on_one, one) == pytest.approx(1.0)      # still full recall at precision 1 first
    assert average_precision([("f", 0.9, _box(500)), ("f", 0.8, _box(0)), ("f", 0.7, _box(1))], one) == pytest.approx(0.5)


def test_boxes_outside_the_annotated_area_can_be_ignored():
    from ai.detect_eval import average_precision
    one = {"f": np.array([_box(0)], float)}
    dets = [("f", 0.9, _box(600)), ("f", 0.8, _box(0))]                  # a confident detection of an unlabeled person
    assert average_precision(dets, one) == pytest.approx(0.5)
    assert average_precision(dets, one, ignore_outside=lambda k, b: b[0] < 300) == pytest.approx(1.0)


class _FixedDetector:
    """Returns the ground-truth boxes of the mock, optionally shifted sideways, plus optional junk."""
    def __init__(self, gt, paths, shift=0.0, junk=False):
        self.by_path = {}
        for cam, plist in paths.items():
            for f, p in enumerate(plist):
                boxes = [((x1 + shift, y1, x2 + shift, y2), 0.9) for ff, _, x1, y1, x2, y2 in gt[cam] if ff == f]
                if junk:
                    boxes.append(((560.0, 250.0, 610.0, 295.0), 0.95))
                self.by_path[p] = boxes

    def __call__(self, path):
        return self.by_path[path]


def _mock_eval(tmp_path, **kw):
    from ai.detect_eval import evaluate_detector
    from ai.wildtrack import load_wildtrack
    images, gt = load_wildtrack(make_mock(tmp_path))
    return evaluate_detector(images, gt, _FixedDetector(gt, images, **kw), conf_op=0.5)


def test_perfect_detector_scores_100_everywhere(tmp_path):
    r = _mock_eval(tmp_path)["total"]
    assert r["AP50"] == 100.0 and r["AP50_95"] == 100.0 and r["recall_at_conf"] == 100.0 and r["precision_at_conf"] == 100.0
    assert r["gt_boxes"] == 4 * N


def test_ap50_95_punishes_loose_boxes_that_ap50_accepts(tmp_path):
    r = _mock_eval(tmp_path, shift=15.0)["total"]       # IoU about 0.54: passes 0.5, fails 0.55 and above
    assert r["AP50"] == 100.0 and r["AP50_95"] == pytest.approx(10.0, abs=0.1)


def test_junk_boxes_lower_precision_and_the_area_variant_forgives_them(tmp_path):
    r = _mock_eval(tmp_path, junk=True)
    t = r["total"]
    assert t["recall_at_conf"] == 100.0 and t["precision_at_conf"] < 100.0 and t["AP50"] < 100.0
    # camera C2 sees two people, so it has an estimable area, and the far-away junk box lies outside it
    c2 = r["cameras"]["C2"]
    assert c2["AP50"] < 100.0 and c2["AP50_ignoring_outside_area"] == 100.0 and c2["precision_ignoring_outside_area"] == 100.0


def test_recall_is_split_by_person_height(tmp_path):
    t = _mock_eval(tmp_path)["total"]["recall_by_person_height"]
    assert t["large_gt120px"]["gt"] == 4 * N and t["small_lt60px"]["gt"] == 0 and t["small_lt60px"]["recall"] is None


def test_detect_eval_cli_runs_with_a_stand_in_detector(tmp_path, capsys, monkeypatch):
    from ai import detect_eval, recording
    root = make_mock(tmp_path / "d")

    class Fake:
        device = "cpu"
        def __init__(self, *a, **k):
            pass
        def __call__(self, path):
            return [((60.0, 60.0, 110.0, 220.0), 0.8)]
    monkeypatch.setattr(detect_eval, "YoloPersonDetector", Fake)
    assert recording.main(["detect-eval", "--root", str(root), "--weights", "a.pt", "b.pt", "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "a@1280" in out and "b@1280" in out and (tmp_path / "o" / "detect_a_1280.json").exists()
