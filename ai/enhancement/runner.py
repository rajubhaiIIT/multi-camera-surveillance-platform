"""CLI Runner for Image Enhancement, Stabilization, Background Subtraction, and Optical Flow.

Commands:
  enhance    apply CLAHE, gamma correction, or unsharp mask
  stabilize  cancel camera shake and pole vibrations
  motion     MOG2 background subtraction and moving region tracking
  flow       dense optical flow vector fields and motion heatmaps
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from .enhance import auto_enhance, clahe_enhance, gamma_correct, unsharp_mask
from .flow import DenseOpticalFlow
from .motion import BackgroundMotionDetector
from .stabilizer import VideoStabilizer


def _open_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    w, h = size
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def _fit_display(img: np.ndarray, max_w: int = 1500, max_h: int = 820) -> np.ndarray:
    """Scale frame down if it exceeds monitor bounds so the entire comparison fits on screen."""
    h, w = img.shape[:2]
    if w <= max_w and h <= max_h:
        return img
    scale = min(max_w / float(w), max_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def cmd_enhance(a) -> int:
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        print(f"error: cannot open source {a.source}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    n = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            enhanced = auto_enhance(frame, mode=a.mode)

            if a.compare:
                left_view = frame.copy()
                right_view = enhanced.copy()

                # High-contrast label badges with dark backgrounds
                cv2.rectangle(left_view, (15, 15), (280, 75), (0, 0, 0), -1)
                cv2.rectangle(left_view, (15, 15), (280, 75), (0, 0, 255), 2)
                cv2.putText(left_view, "ORIGINAL", (30, 56), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                label_r = f"ENHANCED ({a.mode.upper()})"
                badge_w = 40 + len(label_r) * 18
                cv2.rectangle(right_view, (15, 15), (badge_w, 75), (0, 0, 0), -1)
                cv2.rectangle(right_view, (15, 15), (badge_w, 75), (0, 255, 0), 2)
                cv2.putText(right_view, label_r, (30, 56), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                # Solid 10-pixel bright white divider bar with black border
                sep = np.full((frame.shape[0], 10, 3), 255, dtype=np.uint8)
                sep[:, :2] = (0, 0, 0)
                sep[:, -2:] = (0, 0, 0)
                vis = np.hstack([left_view, sep, right_view])
            else:
                vis = enhanced

            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = _open_writer(a.save_video, fps, (w, h))
                writer.write(vis)

            if a.show:
                try:
                    disp = _fit_display(vis)
                    cv2.imshow("Enhancement Preview (q to quit)", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    a.show = False

            n += 1
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if a.show:
            try: cv2.destroyAllWindows()
            except cv2.error: pass

    elapsed = time.time() - t0
    print(f"Enhanced {n} frames in {elapsed:.1f}s ({n / max(1e-6, elapsed):.1f} FPS)")
    if a.save_video:
        print(f"Saved video -> {a.save_video}")
    return 0


def cmd_stabilize(a) -> int:
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        print(f"error: cannot open source {a.source}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stabilizer = VideoStabilizer(smoothing_radius=a.smoothing)
    writer = None
    n = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            stabilized = stabilizer.process_frame(frame)
            if a.compare:
                left_view = frame.copy()
                right_view = stabilized.copy()

                cv2.rectangle(left_view, (15, 15), (360, 75), (0, 0, 0), -1)
                cv2.rectangle(left_view, (15, 15), (360, 75), (0, 0, 255), 2)
                cv2.putText(left_view, "ORIGINAL (SHAKY)", (30, 56), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.rectangle(right_view, (15, 15), (280, 75), (0, 0, 0), -1)
                cv2.rectangle(right_view, (15, 15), (280, 75), (0, 255, 0), 2)
                cv2.putText(right_view, "STABILIZED", (30, 56), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                sep = np.full((frame.shape[0], 10, 3), 255, dtype=np.uint8)
                sep[:, :2] = (0, 0, 0)
                sep[:, -2:] = (0, 0, 0)
                vis = np.hstack([left_view, sep, right_view])
            else:
                vis = stabilized

            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = _open_writer(a.save_video, fps, (w, h))
                writer.write(vis)

            if a.show:
                try:
                    disp = _fit_display(vis)
                    cv2.imshow("Stabilization (q to quit)", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    a.show = False

            n += 1
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if a.show:
            try: cv2.destroyAllWindows()
            except cv2.error: pass

    elapsed = time.time() - t0
    print(f"Stabilized {n} frames in {elapsed:.1f}s ({n / max(1e-6, elapsed):.1f} FPS)")
    if a.save_video:
        print(f"Saved video -> {a.save_video}")
    return 0


def cmd_motion(a) -> int:
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        print(f"error: cannot open source {a.source}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = BackgroundMotionDetector(history=a.history, var_threshold=a.var_threshold)
    writer = None
    n = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mask, boxes, motion_pct = detector.process(frame)
            vis = detector.draw(frame, boxes, motion_pct)

            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = _open_writer(a.save_video, fps, (w, h))
                writer.write(vis)

            if a.show:
                try:
                    disp = _fit_display(vis)
                    cv2.imshow("Motion Detection (q to quit)", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    a.show = False

            n += 1
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if a.show:
            try: cv2.destroyAllWindows()
            except cv2.error: pass

    elapsed = time.time() - t0
    print(f"Processed {n} frames in {elapsed:.1f}s ({n / max(1e-6, elapsed):.1f} FPS)")
    if a.save_video:
        print(f"Saved video -> {a.save_video}")
    return 0


def cmd_flow(a) -> int:
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        print(f"error: cannot open source {a.source}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    flow_engine = DenseOpticalFlow()
    writer = None
    n = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            flow, mean_mag, max_mag = flow_engine.compute(frame)

            if a.mode == "hsv":
                vis = flow_engine.visualize_flow_hsv(flow)
            else:
                vis = flow_engine.draw_flow_arrows(frame, flow, step=a.step)

            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = _open_writer(a.save_video, fps, (w, h))
                writer.write(vis)

            if a.show:
                try:
                    disp = _fit_display(vis)
                    cv2.imshow("Dense Optical Flow (q to quit)", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    a.show = False

            n += 1
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if a.show:
            try: cv2.destroyAllWindows()
            except cv2.error: pass

    elapsed = time.time() - t0
    print(f"Computed optical flow for {n} frames in {elapsed:.1f}s ({n / max(1e-6, elapsed):.1f} FPS)")
    if a.save_video:
        print(f"Saved video -> {a.save_video}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 1. Enhance
    en = sub.add_parser("enhance", help="enhance dark or blurry video")
    en.add_argument("--source", required=True)
    en.add_argument("--mode", default="low_light", choices=["low_light", "sharpen", "all"])
    en.add_argument("--compare", action="store_true", help="side-by-side original and enhanced view")
    en.add_argument("--show", action="store_true")
    en.add_argument("--save-video")
    en.add_argument("--max-frames", type=int)
    en.set_defaults(fn=cmd_enhance)

    # 2. Stabilize
    st = sub.add_parser("stabilize", help="cancel camera vibration/shake")
    st.add_argument("--source", required=True)
    st.add_argument("--smoothing", type=int, default=15)
    st.add_argument("--compare", action="store_true", help="side-by-side original and stabilized view")
    st.add_argument("--show", action="store_true")
    st.add_argument("--save-video")
    st.add_argument("--max-frames", type=int)
    st.set_defaults(fn=cmd_stabilize)

    # 3. Motion
    mo = sub.add_parser("motion", help="MOG2 background subtraction")
    mo.add_argument("--source", required=True)
    mo.add_argument("--history", type=int, default=500)
    mo.add_argument("--var-threshold", type=float, default=25.0)
    mo.add_argument("--show", action="store_true")
    mo.add_argument("--save-video")
    mo.add_argument("--max-frames", type=int)
    mo.set_defaults(fn=cmd_motion)

    # 4. Flow
    fl = sub.add_parser("flow", help="dense optical flow")
    fl.add_argument("--source", required=True)
    fl.add_argument("--mode", default="arrows", choices=["arrows", "hsv"])
    fl.add_argument("--step", type=int, default=16)
    fl.add_argument("--show", action="store_true")
    fl.add_argument("--save-video")
    fl.add_argument("--max-frames", type=int)
    fl.set_defaults(fn=cmd_flow)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
