"""Phase 4 unit tests: Pose estimation, fall detection, and violence detection.

All tests are runnable offline without model downloads or GPU dependencies
(using synthetic keypoint arrays and mocked detectors).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from ai.pose.detector import PoseDetector, _box_iou
from ai.pose.fall import FallDetector, TrackState
from ai.pose.pipeline import PosePipeline
from ai.pose.schema import (
    COCO_KEYPOINTS,
    FallAlert,
    FallState,
    FightAlert,
    Keypoints,
    PoseRecord,
)
from ai.pose.violence import ViolenceDetector
from ai.schema import TrackRecord


# ---------------------------------------------------------------- helpers
def make_upright_kpts(x=100.0, y=100.0, scale=1.0) -> Keypoints:
    """Generate a synthetic upright standing person keypoint array."""
    data = np.zeros((17, 3), dtype=np.float32)
    # Nose, eyes, ears near head (y=100)
    data[0] = [x, y, 0.9]       # nose
    data[1] = [x - 5, y - 5, 0.9] # left eye
    data[2] = [x + 5, y - 5, 0.9] # right eye
    data[3] = [x - 10, y - 5, 0.9] # left ear
    data[4] = [x + 10, y - 5, 0.9] # right ear

    # Shoulders at y=130
    data[5] = [x - 20 * scale, y + 30 * scale, 0.9] # left shoulder
    data[6] = [x + 20 * scale, y + 30 * scale, 0.9] # right shoulder

    # Elbows at y=160
    data[7] = [x - 25 * scale, y + 60 * scale, 0.9] # left elbow
    data[8] = [x + 25 * scale, y + 60 * scale, 0.9] # right elbow

    # Wrists at y=190
    data[9] = [x - 25 * scale, y + 90 * scale, 0.9] # left wrist
    data[10] = [x + 25 * scale, y + 90 * scale, 0.9] # right wrist

    # Hips at y=200
    data[11] = [x - 15 * scale, y + 100 * scale, 0.9] # left hip
    data[12] = [x + 15 * scale, y + 100 * scale, 0.9] # right hip

    # Knees at y=250
    data[13] = [x - 15 * scale, y + 150 * scale, 0.9] # left knee
    data[14] = [x + 15 * scale, y + 150 * scale, 0.9] # right knee

    # Ankles at y=300
    data[15] = [x - 15 * scale, y + 200 * scale, 0.9] # left ankle
    data[16] = [x + 15 * scale, y + 200 * scale, 0.9] # right ankle

    return Keypoints(data)


def make_horizontal_kpts(x=100.0, y=300.0, scale=1.0) -> Keypoints:
    """Generate a synthetic horizontal lying-down / fallen person keypoint array."""
    data = np.zeros((17, 3), dtype=np.float32)
    # Head at x=50, y=300
    data[0] = [x - 100 * scale, y, 0.9]     # nose
    data[5] = [x - 70 * scale, y - 10, 0.9] # left shoulder
    data[6] = [x - 70 * scale, y + 10, 0.9] # right shoulder
    data[9] = [x - 40 * scale, y - 15, 0.9] # left wrist
    data[10] = [x - 40 * scale, y + 15, 0.9] # right wrist
    data[11] = [x, y - 10, 0.9]             # left hip
    data[12] = [x, y + 10, 0.9]             # right hip
    data[15] = [x + 80 * scale, y - 5, 0.9] # left ankle
    data[16] = [x + 80 * scale, y + 5, 0.9] # right ankle
    return Keypoints(data)


# ---------------------------------------------------------------- Keypoints tests
def test_keypoints_shape_validation():
    with pytest.raises(ValueError):
        Keypoints(np.zeros((16, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        Keypoints(np.zeros((17, 2), dtype=np.float32))

    kpts = make_upright_kpts()
    assert kpts.data.shape == (17, 3)
    assert len(COCO_KEYPOINTS) == 17


def test_keypoints_centers_and_torso_angle():
    kpts_up = make_upright_kpts(x=100.0, y=100.0)
    sc = kpts_up.shoulders_center
    hc = kpts_up.hips_center
    tc = kpts_up.torso_center
    assert sc is not None and hc is not None and tc is not None
    assert sc[0] == pytest.approx(100.0, abs=1.0)
    assert hc[0] == pytest.approx(100.0, abs=1.0)
    # Upright person has torso angle close to 0 degrees
    assert kpts_up.torso_angle < 15.0

    kpts_horiz = make_horizontal_kpts(x=200.0, y=300.0)
    # Horizontal person has torso angle close to 90 degrees
    assert kpts_horiz.torso_angle > 70.0


def test_keypoints_bbox_and_aspect_ratio():
    kpts_up = make_upright_kpts(x=100.0, y=100.0)
    bbox_up = kpts_up.bbox
    assert bbox_up[2] > bbox_up[0]
    assert bbox_up[3] > bbox_up[1]
    # Standing person: height >> width, aspect ratio < 0.6
    assert kpts_up.aspect_ratio < 0.6

    kpts_horiz = make_horizontal_kpts(x=200.0, y=300.0)
    # Lying down: width > height, aspect ratio > 1.0
    assert kpts_horiz.aspect_ratio > 1.0


# ---------------------------------------------------------------- PoseRecord tests
def test_poserecord_to_dict():
    kpts = make_upright_kpts()
    rec = PoseRecord("cam1", 5, 42, (80.0, 95.0, 120.0, 305.0), kpts, 0.95, global_id=7)
    d = rec.to_dict()
    assert d["camera_id"] == "cam1"
    assert d["frame_idx"] == 5
    assert d["track_id"] == 42
    assert d["global_id"] == 7
    assert d["conf"] == 0.95
    assert "torso_angle" in d
    assert "aspect_ratio" in d


# ---------------------------------------------------------------- FallDetector tests
def test_fall_detector_upright_no_alert():
    det = FallDetector()
    for i in range(15):
        kpts = make_upright_kpts(x=100.0, y=100.0)
        rec = PoseRecord("c0", i, 1, kpts.bbox, kpts, 0.9)
        alert = det.update(rec, timestamp=i * 0.1)
        assert alert is None


def test_fall_detector_bending_no_alert():
    det = FallDetector()
    # Person bending forward slightly (angle ~ 40 deg)
    data = make_upright_kpts().data.copy()
    data[5:7, 0] += 50.0  # tilt shoulders forward
    kpts_bent = Keypoints(data)
    for i in range(15):
        rec = PoseRecord("c0", i, 1, kpts_bent.bbox, kpts_bent, 0.9)
        alert = det.update(rec, timestamp=i * 0.1)
        assert alert is None


def test_fall_detector_rapid_fall_triggers_alert():
    det = FallDetector(inactivity_seconds=0.5, descent_velocity_threshold=0.8)
    # Frames 0..3: upright at y=100
    for i in range(4):
        k = make_upright_kpts(x=100.0, y=100.0)
        rec = PoseRecord("c0", i, 1, k.bbox, k, 0.9)
        det.update(rec, timestamp=i * 0.1)

    # Frame 4: rapid drop to horizontal at y=300
    alert_emitted = None
    for i in range(4, 15):
        k = make_horizontal_kpts(x=100.0, y=300.0)
        rec = PoseRecord("c0", i, 1, k.bbox, k, 0.9)
        a = det.update(rec, timestamp=i * 0.1)
        if a is not None:
            alert_emitted = a

    assert alert_emitted is not None
    assert alert_emitted.state == FallState.FALLEN
    assert alert_emitted.track_id == 1
    assert alert_emitted.torso_angle > 60.0
    assert alert_emitted.confidence > 0.7


def test_fall_detector_recovery_event():
    det = FallDetector(inactivity_seconds=0.3, descent_velocity_threshold=0.8)
    # Upright
    for i in range(3):
        k = make_upright_kpts(x=100.0, y=100.0)
        det.update(PoseRecord("c0", i, 1, k.bbox, k, 0.9), timestamp=i * 0.1)
    # Fall down
    for i in range(3, 8):
        k = make_horizontal_kpts(x=100.0, y=300.0)
        det.update(PoseRecord("c0", i, 1, k.bbox, k, 0.9), timestamp=i * 0.1)

    # Stands back up
    recovery_alert = None
    for i in range(8, 12):
        k = make_upright_kpts(x=100.0, y=100.0)
        a = det.update(PoseRecord("c0", i, 1, k.bbox, k, 0.9), timestamp=i * 0.1)
        if a is not None and a.state == FallState.RECOVERED:
            recovery_alert = a

    assert recovery_alert is not None
    assert recovery_alert.state == FallState.RECOVERED
    assert "stood back up" in recovery_alert.details.lower()


# ---------------------------------------------------------------- ViolenceDetector tests
def test_violence_detector_distant_people_no_alert():
    vd = ViolenceDetector(proximity_threshold=1.2)
    p1 = PoseRecord("c0", 0, 1, (10, 10, 50, 200), make_upright_kpts(x=30, y=10), 0.9)
    p2 = PoseRecord("c0", 0, 2, (500, 10, 540, 200), make_upright_kpts(x=520, y=10), 0.9)
    alerts = vd.update([p1, p2], timestamp=0.0)
    assert len(alerts) == 0


def test_violence_detector_calm_proximity_no_alert():
    vd = ViolenceDetector(proximity_threshold=1.5, motion_energy_threshold=1.5)
    for i in range(10):
        # People close to each other, but standing still
        p1 = PoseRecord("c0", i, 1, (100, 100, 140, 300), make_upright_kpts(x=120, y=100), 0.9)
        p2 = PoseRecord("c0", i, 2, (150, 100, 190, 300), make_upright_kpts(x=170, y=100), 0.9)
        alerts = vd.update([p1, p2], timestamp=i * 0.1)
        assert len(alerts) == 0


def test_violence_detector_rapid_strikes_triggers_alert():
    vd = ViolenceDetector(proximity_threshold=1.5, motion_energy_threshold=1.0, consecutive_frames=3)
    alerts_emitted = []
    for i in range(8):
        ts = i * 0.1
        # Alternate wrist positions rapidly (violent swinging motion)
        k1 = make_upright_kpts(x=120, y=100)
        k1.data[9, 0] += 80.0 if (i % 2 == 0) else -80.0 # violent wrist oscillation
        k2 = make_upright_kpts(x=170, y=100)
        k2.data[10, 0] += 80.0 if (i % 2 == 0) else -80.0

        p1 = PoseRecord("c0", i, 1, (100, 100, 140, 300), k1, 0.9)
        p2 = PoseRecord("c0", i, 2, (150, 100, 190, 300), k2, 0.9)
        al = vd.update([p1, p2], timestamp=ts)
        alerts_emitted.extend(al)

    assert len(alerts_emitted) > 0
    alert = alerts_emitted[-1]
    assert alert.track_ids == (1, 2)
    assert alert.motion_intensity >= 1.0


# ---------------------------------------------------------------- Pipeline & Drawing tests
def test_box_iou():
    assert _box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _box_iou((0, 0, 10, 10), (10, 10, 20, 20)) == pytest.approx(0.0)
    assert _box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50.0 / 150.0)


def test_pose_pipeline_mocked_end_to_end():
    mock_detector = MagicMock()
    k_up = make_upright_kpts(x=100, y=100)
    pose_rec = PoseRecord("c0", 0, 1, k_up.bbox, k_up, 0.95)
    mock_detector.detect.return_value = [pose_rec]
    mock_detector.match_to_tracks.side_effect = lambda poses, tracks: poses
    mock_detector.draw.side_effect = lambda frame, poses: frame.copy()

    fall_det = FallDetector()
    viol_det = ViolenceDetector()
    pipe = PosePipeline(detector=mock_detector, fall_detector=fall_det, violence_detector=viol_det)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    track = TrackRecord("c0", 0.0, 0, 1, k_up.bbox, "person", 0.9)
    poses, falls, fights = pipe.process(frame, "c0", 0, 0.0, tracks=[track])

    assert len(poses) == 1
    assert len(falls) == 0
    assert len(fights) == 0

    vis = pipe.draw(frame, poses, falls, fights)
    assert vis.shape == frame.shape


# ---------------------------------------------------------------- CLI Runner tests
def test_cmd_demo_offline(tmp_path, monkeypatch):
    from ai.pose.runner import cmd_demo

    class MockDetector:
        def __init__(self, *args, **kwargs):
            pass
        def detect(self, frame, camera_id="demo", frame_idx=0):
            k = make_upright_kpts(x=100, y=100)
            return [PoseRecord(camera_id, frame_idx, 1, k.bbox, k, 0.9)]
        def draw(self, frame, poses):
            return frame.copy()

    class MockCap:
        def __init__(self, *args, **kwargs):
            self._count = 0
        def isOpened(self): return True
        def get(self, prop): return 25.0
        def read(self):
            if self._count < 3:
                self._count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None
        def release(self): pass

    monkeypatch.setattr("ai.pose.runner.PoseDetector", MockDetector)
    monkeypatch.setattr("cv2.VideoCapture", MockCap)

    out_file = tmp_path / "alerts.json"
    args = SimpleNamespace(
        source="dummy.mp4",
        weights="dummy.pt",
        conf=0.25,
        run_every=1,
        device="cpu",
        show=False,
        save_video=None,
        out=str(out_file),
        max_frames=3,
    )
    ret = cmd_demo(args)
    assert ret == 0
    assert out_file.exists()


def test_cmd_run_eval_offline(tmp_path, monkeypatch):
    from ai.pose.runner import cmd_run_eval
    from ai.runcache import Event

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meta = {
        "cameras": {
            "c0": {
                "video": "c0.avi",
                "stride": 1,
                "fps": 25.0,
                "last_idx": 3,
            }
        }
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))

    class MockDetector:
        def __init__(self, *args, **kwargs):
            pass
        def detect(self, frame, camera_id="c0", frame_idx=0):
            k = make_upright_kpts(x=100, y=100)
            return [PoseRecord(camera_id, frame_idx, 1, k.bbox, k, 0.9)]
        def match_to_tracks(self, poses, tracks):
            return poses
        def draw(self, frame, poses):
            return frame.copy()

    class MockReader:
        def __init__(self, *args, **kwargs):
            self.fps = 25.0
        def frames(self):
            for i in range(3):
                yield i, np.zeros((100, 100, 3), dtype=np.uint8)

    mock_run = SimpleNamespace(
        dir=run_dir,
        events=[
            Event("c0", 0.0, 0, [TrackRecord("c0", 0.0, 0, 1, (10, 10, 50, 100), "person", 0.9)], {})
        ]
    )

    monkeypatch.setattr("ai.pose.runner.PoseDetector", MockDetector)
    monkeypatch.setattr("ai.recording.load_run", lambda p: mock_run)
    monkeypatch.setattr("ai.sources.MediaReader", MockReader)

    args = SimpleNamespace(
        run=str(run_dir),
        weights="dummy.pt",
        conf=0.25,
        run_every=1,
        device="cpu",
        render=True,
    )
    ret = cmd_run_eval(args)
    assert ret == 0
    assert (run_dir / "pose_alerts.json").exists()
    assert (run_dir / "pose_render" / "c0.mp4").exists()

