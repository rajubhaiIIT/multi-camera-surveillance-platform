"""Phase 6 unit tests: Image Enhancement, Stabilization, Background Subtraction & Optical Flow.

All tests run offline with zero external network or GPU requirements.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from ai.enhancement.enhance import (
    auto_enhance,
    clahe_enhance,
    gamma_correct,
    unsharp_mask,
)
from ai.enhancement.flow import DenseOpticalFlow
from ai.enhancement.motion import BackgroundMotionDetector
from ai.enhancement.stabilizer import VideoStabilizer


# ---------------------------------------------------------------- Enhancement tests
def test_clahe_enhances_dark_image():
    # Synthetic dark image with low contrast
    dark_img = np.full((100, 100, 3), 30, dtype=np.uint8)
    # Add a slight feature
    dark_img[40:60, 40:60] = 50

    enhanced = clahe_enhance(dark_img, clip_limit=3.0)
    assert enhanced.shape == dark_img.shape
    # Mean luminance should increase in dark regions
    assert float(np.mean(enhanced)) > float(np.mean(dark_img))


def test_gamma_correction():
    img = np.full((50, 50, 3), 64, dtype=np.uint8)
    brightened = gamma_correct(img, gamma=1.5)
    assert brightened.shape == img.shape
    assert int(brightened[0, 0, 0]) > 64

    darkened = gamma_correct(img, gamma=0.7)
    assert int(darkened[0, 0, 0]) < 64


def test_unsharp_mask():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)
    sharpened = unsharp_mask(img, sigma=1.0, strength=1.5)
    assert sharpened.shape == img.shape


def test_auto_enhance_modes():
    img = np.full((80, 80, 3), 50, dtype=np.uint8)
    res_ll = auto_enhance(img, mode="low_light")
    res_sh = auto_enhance(img, mode="sharpen")
    res_all = auto_enhance(img, mode="all")
    assert res_ll.shape == img.shape
    assert res_sh.shape == img.shape
    assert res_all.shape == img.shape


# ---------------------------------------------------------------- Stabilization tests
def test_stabilizer_static_and_jittered_frames():
    stab = VideoStabilizer(smoothing_radius=5)
    frame1 = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (30, 30), (80, 80), (255, 255, 255), -1)

    # Frame 1: initial
    out1 = stab.process_frame(frame1)
    assert out1.shape == frame1.shape

    # Frame 2: simulated camera shake (shifted by 4 pixels)
    frame2 = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (34, 30), (84, 80), (255, 255, 255), -1)

    out2 = stab.process_frame(frame2)
    assert out2.shape == frame2.shape


# ---------------------------------------------------------------- Motion Detection tests
def test_background_motion_detector():
    detector = BackgroundMotionDetector(history=10, min_area=50.0)
    bg = np.full((120, 160, 3), 100, dtype=np.uint8)

    # Train background model on static frames
    for _ in range(5):
        mask, boxes, pct = detector.process(bg)

    # Static frame should have negligible motion
    assert pct < 2.0

    # Introduce a moving bright rectangle
    moving_frame = bg.copy()
    moving_frame[40:80, 50:100] = 255  # 40x50 = 2000 pixels

    mask, boxes, pct = detector.process(moving_frame)
    assert pct > 5.0
    assert len(boxes) > 0

    vis = detector.draw(moving_frame, boxes, pct)
    assert vis.shape == moving_frame.shape


# ---------------------------------------------------------------- Dense Optical Flow tests
def test_dense_optical_flow():
    flow_engine = DenseOpticalFlow()
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(f1, (40, 50), 15, (255, 255, 255), -1)

    # Frame 1: first step
    flow1, mean1, max1 = flow_engine.compute(f1)
    assert flow1.shape == (100, 100, 2)
    assert mean1 == 0.0

    # Frame 2: circle shifted right
    f2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(f2, (50, 50), 15, (255, 255, 255), -1)

    flow2, mean2, max2 = flow_engine.compute(f2)
    assert mean2 > 0.0
    assert max2 > 0.0

    # Visualizations
    hsv_vis = flow_engine.visualize_flow_hsv(flow2)
    assert hsv_vis.shape == (100, 100, 3)

    arrows_vis = flow_engine.draw_flow_arrows(f2, flow2, step=10)
    assert arrows_vis.shape == (100, 100, 3)


# ---------------------------------------------------------------- CLI Runner tests
def test_cmd_runner_offline(tmp_path, monkeypatch):
    from ai.enhancement.runner import cmd_enhance, cmd_flow, cmd_motion, cmd_stabilize

    class MockCap:
        def __init__(self, *args, **kwargs):
            self._count = 0
        def isOpened(self): return True
        def get(self, prop): return 25.0
        def read(self):
            if self._count < 3:
                self._count += 1
                return True, np.full((80, 80, 3), 100, dtype=np.uint8)
            return False, None
        def release(self): pass

    monkeypatch.setattr("cv2.VideoCapture", MockCap)

    # Test enhance CLI
    out_en = tmp_path / "out_en.mp4"
    args_en = SimpleNamespace(source="dummy.mp4", mode="low_light", compare=False,
                              show=False, save_video=str(out_en), max_frames=2)
    assert cmd_enhance(args_en) == 0

    # Test stabilize CLI
    out_st = tmp_path / "out_st.mp4"
    args_st = SimpleNamespace(source="dummy.mp4", smoothing=5, compare=False,
                              show=False, save_video=str(out_st), max_frames=2)
    assert cmd_stabilize(args_st) == 0

    # Test motion CLI
    out_mo = tmp_path / "out_mo.mp4"
    args_mo = SimpleNamespace(source="dummy.mp4", history=10, var_threshold=25.0,
                              show=False, save_video=str(out_mo), max_frames=2)
    assert cmd_motion(args_mo) == 0

    # Test flow CLI
    out_fl = tmp_path / "out_fl.mp4"
    args_fl = SimpleNamespace(source="dummy.mp4", mode="arrows", step=16,
                              show=False, save_video=str(out_fl), max_frames=2)
    assert cmd_flow(args_fl) == 0
