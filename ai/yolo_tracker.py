"""Detection + single-camera tracking: YOLO (Ultralytics) with its built-in ByteTrack.

Each camera needs its OWN Tracker instance, because the tracker keeps per-camera state.
"""
from __future__ import annotations

from .schema import TrackRecord

# COCO class ids: person, car, motorcycle, bus, truck
DEFAULT_CLASSES = (0, 2, 3, 5, 7)


def _precision_kwargs(fp16: bool) -> dict:
    """Ultralytics >= 8.4 replaced the `half` argument with `quantize` (and warns on every frame)."""
    if not fp16:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
    except Exception:
        return {"half": True}


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Tracker:
    def __init__(self, camera_id: str, weights: str = "yolo11n.pt", classes=DEFAULT_CLASSES,
                 conf: float = 0.1, iou: float = 0.7, imgsz: int = 640, device: str = "auto",
                 half: bool | None = None, tracker_cfg: str = "bytetrack.yaml"):
        from ultralytics import YOLO  # imported here so schema/tests work without torch

        self.camera_id = camera_id
        self.device = resolve_device(device)
        self.half = (not self.device.startswith("cpu")) if half is None else half
        self.classes = list(classes) if classes is not None else None
        self.conf, self.iou, self.imgsz, self.tracker_cfg = conf, iou, imgsz, tracker_cfg
        self._precision = _precision_kwargs(self.half)
        self.model = YOLO(weights)
        self.names = self.model.names

    def update(self, image, ts: float, idx: int) -> list[TrackRecord]:
        """Process one BGR frame; return the active tracks in it."""
        res = self.model.track(
            image, persist=True, tracker=self.tracker_cfg, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, classes=self.classes, device=self.device,
            verbose=False, **self._precision,
        )[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        out = []
        for tid, bb, cf, c in zip(ids, xyxy, confs, clss):
            if bb[2] > bb[0] and bb[3] > bb[1]:
                out.append(TrackRecord(self.camera_id, ts, idx, int(tid), tuple(bb),
                                       self.names[int(c)], min(max(float(cf), 0.0), 1.0)))
        return out
