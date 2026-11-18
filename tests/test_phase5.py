"""Phase 5 unit tests: Weapon detection, Vehicle tracking & ANPR.

All tests are runnable offline without model downloads or GPU dependencies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from ai.schema import TrackRecord
from ai.vehicles.anpr import LicensePlateReader, clean_plate_text
from ai.vehicles.index import PlateIndex, normalize_plate
from ai.vehicles.pipeline import VehicleANPRPipeline
from ai.vehicles.schema import (
    PlateRecord,
    VehicleAlert,
    VehicleRecord,
    VehicleType,
)
from ai.vehicles.tracker import VehicleTracker
from ai.weapons.detector import WeaponDetector, _point_in_box
from ai.weapons.schema import WeaponAlert, WeaponRecord, WeaponType


# ---------------------------------------------------------------- Weapon tests
def test_weapon_schema_and_types():
    rec = WeaponRecord((10.0, 10.0, 50.0, 50.0), WeaponType.KNIFE, 0.88, held_by_track_id=3)
    d = rec.to_dict()
    assert d["weapon_type"] == "knife"
    assert d["conf"] == 0.88
    assert d["held_by_track_id"] == 3


def test_point_in_box():
    box = (100.0, 100.0, 200.0, 300.0)
    assert _point_in_box((150.0, 200.0), box)
    assert not _point_in_box((50.0, 50.0), box)
    # Inside with 15% margin
    assert _point_in_box((90.0, 100.0), box, margin=0.15)


def test_weapon_person_association():
    detector = WeaponDetector.__new__(WeaponDetector)
    # Weapon 1 inside Person 1 box
    w1 = WeaponRecord((120.0, 150.0, 140.0, 170.0), WeaponType.KNIFE, 0.9)
    # Weapon 2 on the floor far away
    w2 = WeaponRecord((600.0, 600.0, 620.0, 620.0), WeaponType.GUN, 0.85)

    person1 = TrackRecord("c0", 0.0, 0, 1, (100.0, 100.0, 160.0, 250.0), "person", 0.95)
    person2 = TrackRecord("c0", 0.0, 0, 2, (300.0, 100.0, 360.0, 250.0), "person", 0.95)

    updated = detector.associate_with_persons([w1, w2], [person1, person2])
    assert updated[0].held_by_track_id == 1
    assert updated[1].held_by_track_id is None


def test_weapon_process_and_draw():
    detector = WeaponDetector.__new__(WeaponDetector)
    detector.detect = MagicMock(return_value=[
        WeaponRecord((120.0, 150.0, 140.0, 170.0), WeaponType.KNIFE, 0.92)
    ])
    person = TrackRecord("c0", 0.0, 0, 42, (100.0, 100.0, 160.0, 250.0), "person", 0.95, global_id=7)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    weapons, alerts = detector.process(frame, "c0", 0, 0.0, person_tracks=[person])

    assert len(weapons) == 1
    assert weapons[0].held_by_track_id == 42
    assert len(alerts) == 1
    assert alerts[0].is_held is True
    assert alerts[0].global_id == 7
    assert "knife" in alerts[0].details.lower()

    vis = detector.draw(frame, weapons, alerts)
    assert vis.shape == frame.shape


# ---------------------------------------------------------------- ANPR & PlateIndex tests
def test_normalize_plate():
    assert normalize_plate("mh-12 de 1433") == "MH12DE1433"
    assert normalize_plate("DL 01 A 1234") == "DL01A1234"
    assert normalize_plate("b-123-xyz") == "B123XYZ"


def test_clean_plate_text():
    clean, fmt = clean_plate_text("MH12DE1433")
    assert clean == "MH12DE1433"
    assert fmt == "IND"

    clean2, fmt2 = clean_plate_text("ABC1234")
    assert clean2 == "ABC1234"
    assert fmt2 == "GENERIC"


def test_plate_index_exact_and_fuzzy_matching(tmp_path):
    idx = PlateIndex(max_edit_distance=1)
    idx.enroll("MH12DE1433", reason="Stolen vehicle", owner="John Doe")
    idx.enroll("KA01AB9999", reason="Unpaid fines")

    # Exact match
    res = idx.match("MH12DE1433")
    assert res is not None
    assert res.plate == "MH12DE1433"
    assert res.reason == "Stolen vehicle"

    # Fuzzy match (1 character OCR confusion: '0' vs 'D')
    res_fuzzy = idx.match("MH120E1433")
    assert res_fuzzy is not None
    assert res_fuzzy.plate == "MH12DE1433"

    # Distant plate should not match
    assert idx.match("XX99ZZ0000") is None

    # Persistence
    save_path = tmp_path / "watchlist.json"
    idx.save(save_path)
    loaded = PlateIndex.load(save_path)
    assert loaded.size == 2
    assert loaded.match("MH12DE1433") is not None


def test_license_plate_localization_morphology():
    reader = LicensePlateReader(gpu=False)
    # Create synthetic vehicle bumper with a high-contrast white plate on dark bumper
    veh_img = np.full((120, 200, 3), 40, dtype=np.uint8)  # dark vehicle
    # Draw white rectangular plate (aspect ratio ~ 3.3)
    cv2.rectangle(veh_img, (60, 75), (140, 95), (240, 240, 240), -1)
    # Add dark characters on the plate
    cv2.putText(veh_img, "MH12", (65, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1)

    candidates = reader.locate_plates(veh_img)
    assert len(candidates) > 0
    # Best candidate bounding box should be near the plate
    best = candidates[0]
    aspect = (best[2] - best[0]) / max(1.0, (best[3] - best[1]))
    assert 2.0 <= aspect <= 5.5


# ---------------------------------------------------------------- VehicleTracker tests
def test_vehicle_tracker_association_and_stationary():
    tracker = VehicleTracker(parking_limit_seconds=1.0)
    # Mock detect to return a fixed car at (100, 100, 200, 180)
    tracker.detect = MagicMock(return_value=[{
        "bbox": (100.0, 100.0, 200.0, 180.0),
        "conf": 0.9,
        "vehicle_type": VehicleType.CAR,
    }])

    # Frame 0 at t=0.0
    recs0, alerts0 = tracker.update(np.zeros((300, 300, 3), np.uint8), "c0", 0, 0.0)
    assert len(recs0) == 1
    assert recs0[0].track_id == 1
    assert recs0[0].vehicle_type == VehicleType.CAR
    assert len(alerts0) == 0

    # Frame 1 at t=0.5: same vehicle stationary
    recs1, alerts1 = tracker.update(np.zeros((300, 300, 3), np.uint8), "c0", 1, 0.5)
    assert recs1[0].track_id == 1
    assert len(alerts1) == 0

    # Frame 2 at t=1.2: stationary for > 1.0s -> triggers parking violation alert!
    recs2, alerts2 = tracker.update(np.zeros((300, 300, 3), np.uint8), "c0", 2, 1.2)
    assert recs2[0].track_id == 1
    assert len(alerts2) == 1
    assert alerts2[0].alert_type == "parking_violation"
    assert alerts2[0].track_id == 1


# ---------------------------------------------------------------- Pipeline & Drawing tests
def test_vehicle_pipeline_mocked_end_to_end():
    tracker = VehicleTracker()
    tracker.detect = MagicMock(return_value=[{
        "bbox": (100.0, 100.0, 200.0, 180.0),
        "conf": 0.9,
        "vehicle_type": VehicleType.CAR,
    }])

    plate_reader = LicensePlateReader(gpu=False)
    plate_reader.process_vehicle = MagicMock(return_value=PlateRecord(
        text="MH12DE1433",
        confidence=0.94,
        bbox=(130.0, 150.0, 170.0, 165.0),
        country_format="IND",
    ))

    watchlist = PlateIndex(max_edit_distance=1)
    watchlist.enroll("MH12DE1433", reason="Wanted Stolen Vehicle")

    pipe = VehicleANPRPipeline(tracker=tracker, plate_reader=plate_reader, watchlist=watchlist)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vehicles, alerts = pipe.process(frame, "c0", 0, 0.0)

    assert len(vehicles) == 1
    assert vehicles[0].plate is not None
    assert vehicles[0].plate.text == "MH12DE1433"
    assert len(alerts) == 1
    assert alerts[0].alert_type == "watchlist_match"
    assert "Wanted Stolen Vehicle" in alerts[0].details

    vis = pipe.draw(frame, vehicles, alerts)
    assert vis.shape == frame.shape


# ---------------------------------------------------------------- CLI Subcommand tests
def test_cmd_watchlist_cli(tmp_path):
    from ai.vehicles.runner import (
        cmd_watchlist_add,
        cmd_watchlist_list,
        cmd_watchlist_remove,
    )

    wl_file = tmp_path / "test_wl.json"
    args_add = SimpleNamespace(plate="KA05MJ1234", reason="Traffic fine", owner="Alice", notes="Red sedan", file=str(wl_file))
    assert cmd_watchlist_add(args_add) == 0

    args_list = SimpleNamespace(file=str(wl_file))
    assert cmd_watchlist_list(args_list) == 0

    args_rem = SimpleNamespace(plate="KA05MJ1234", file=str(wl_file))
    assert cmd_watchlist_remove(args_rem) == 0
    assert cmd_watchlist_remove(args_rem) == 1 # already removed


def test_cmd_demo_offline(tmp_path, monkeypatch):
    from ai.vehicles.runner import cmd_demo

    class MockCap:
        def __init__(self, *args, **kwargs):
            self._count = 0
        def isOpened(self): return True
        def get(self, prop): return 25.0
        def read(self):
            if self._count < 2:
                self._count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None
        def release(self): pass

    class MockTracker:
        def __init__(self, *args, **kwargs): pass
        def update(self, frame, camera_id, frame_idx, timestamp):
            rec = VehicleRecord(camera_id, frame_idx, 1, (10, 10, 80, 80), VehicleType.CAR, 0.9)
            return [rec], []

    class MockReader:
        def __init__(self, *args, **kwargs): pass
        def process_vehicle(self, frame, bbox):
            return PlateRecord("MH12DE1433", 0.95, (20, 20, 50, 30), "IND")

    monkeypatch.setattr("cv2.VideoCapture", MockCap)
    monkeypatch.setattr("ai.vehicles.runner.VehicleTracker", MockTracker)
    monkeypatch.setattr("ai.vehicles.runner.LicensePlateReader", MockReader)

    out_file = tmp_path / "events.json"
    args = SimpleNamespace(
        source="dummy.mp4",
        weights="dummy.pt",
        conf=0.25,
        watchlist=None,
        device="cpu",
        show=False,
        save_video=None,
        out=str(out_file),
        max_frames=2,
    )
    ret = cmd_demo(args)
    assert ret == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert len(data["vehicle_events"]) > 0


def test_cmd_weapons_runner_offline(tmp_path, monkeypatch):
    from ai.weapons.runner import cmd_demo as weapon_demo
    from ai.weapons.runner import cmd_image as weapon_image

    class MockCap:
        def __init__(self, *args, **kwargs):
            self._count = 0
        def isOpened(self): return True
        def get(self, prop): return 25.0
        def read(self):
            if self._count < 2:
                self._count += 1
                return True, np.zeros((100, 100, 3), dtype=np.uint8)
            return False, None
        def release(self): pass

    class MockWeaponDetector:
        def __init__(self, *args, **kwargs): pass
        def process(self, frame, camera_id="demo", frame_idx=0, timestamp=0.0):
            w = WeaponRecord((10, 10, 30, 30), WeaponType.KNIFE, 0.91)
            a = WeaponAlert(camera_id, frame_idx, None, None, WeaponType.KNIFE, 0.91, w.bbox, timestamp, False, "Threat detected")
            return [w], [a]
        def draw(self, frame, weapons, alerts):
            return frame.copy()

    monkeypatch.setattr("cv2.VideoCapture", MockCap)
    monkeypatch.setattr("ai.weapons.runner.WeaponDetector", MockWeaponDetector)

    # Test demo
    out_alerts = tmp_path / "weapon_alerts.json"
    demo_args = SimpleNamespace(
        source="dummy.mp4",
        weights="dummy.pt",
        conf=0.25,
        device="cpu",
        show=False,
        save_video=None,
        out=str(out_alerts),
        max_frames=2,
    )
    assert weapon_demo(demo_args) == 0
    assert out_alerts.exists()

    # Test image
    dummy_img = tmp_path / "threat.jpg"
    cv2.imwrite(str(dummy_img), np.zeros((100, 100, 3), dtype=np.uint8))
    out_img = tmp_path / "detected.jpg"
    img_args = SimpleNamespace(
        photo=str(dummy_img),
        weights="dummy.pt",
        conf=0.25,
        device="cpu",
        out_photo=str(out_img),
    )
    assert weapon_image(img_args) == 0
    assert out_img.exists()


