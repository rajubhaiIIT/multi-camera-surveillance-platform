"""Frame sources: a live RTSP stream (threaded, always the newest frame, auto-reconnect) or a video file."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

# Must be set before OpenCV opens a stream: use TCP for RTSP (UDP drops packets on many networks).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


@dataclass
class Frame:
    image: np.ndarray
    ts: float
    idx: int


class FileSource:
    """Sequential read of a video file. Never drops frames. ts = seconds from start."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __iter__(self):
        idx = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                return
            yield Frame(img, idx / self.fps, idx)
            idx += 1

    def close(self):
        self.cap.release()


class RtspSource:
    """Live stream. A background thread keeps only the latest frame, so a slow consumer
    skips frames instead of falling behind (real-time behaviour). Reconnects with backoff."""

    def __init__(self, url: str, max_backoff: float = 10.0):
        self.url, self.max_backoff = url, max_backoff
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._count = 0
        self._stop = threading.Event()
        self.connected = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                self.connected = False
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                continue
            self.connected, backoff = True, 1.0
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break  # stream dropped -> reconnect
                with self._cond:
                    self._frame = Frame(img, time.time(), self._count)
                    self._count += 1
                    self._cond.notify_all()
            cap.release()
            self.connected = False

    def __iter__(self):
        last = -1
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._stop.is_set() or (self._frame and self._frame.idx != last),
                                    timeout=1.0)
                f = self._frame
            if f is not None and f.idx != last and not self._stop.is_set():
                last = f.idx
                yield f

    def close(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=3)


def open_source(src: str):
    """rtsp:// / rtmp:// / http(s):// -> live source; anything else is treated as a file path."""
    if src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return RtspSource(src)
    return FileSource(src)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(folder) -> list[str]:
    from pathlib import Path
    return [str(f) for f in sorted(Path(folder).iterdir()) if f.suffix.lower() in IMAGE_EXTS]


class MediaReader:
    """Frames from a video file, an image folder, or an explicit list of image paths.
    Yields (frame_index, image) only for every `stride`-th frame; skipped frames are not decoded."""

    def __init__(self, media, stride: int = 1, folder_fps: float = 2.0):
        from pathlib import Path
        self.stride, self.total = max(1, int(stride)), 0
        if isinstance(media, (list, tuple)):
            self.images, self.path = [str(m) for m in media], None
        elif Path(str(media)).is_dir():
            self.images, self.path = list_images(media), None
        else:
            self.images, self.path = None, str(media)
        if self.images is not None:
            if not self.images:
                raise IOError(f"no images in {media}")
            self.fps = float(folder_fps)
        else:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise IOError(f"cannot open video: {self.path}")
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

    @property
    def source(self):
        """What to store in a run's meta.json so the media can be found again."""
        from pathlib import Path
        return self.images if self.images is not None else str(Path(self.path).resolve())

    def frames(self):
        if self.images is not None:
            for idx, p in enumerate(self.images):
                self.total = idx + 1
                if idx % self.stride == 0:
                    img = cv2.imread(p)
                    if img is None:
                        raise IOError(f"cannot read image: {p}")
                    yield idx, img
            return
        cap = cv2.VideoCapture(self.path)
        idx = 0
        try:
            while True:
                ok, img = (cap.grab(), None) if idx % self.stride else cap.read()
                if not ok:
                    break
                self.total = idx + 1
                if idx % self.stride == 0:
                    yield idx, img
                idx += 1
        finally:
            cap.release()
