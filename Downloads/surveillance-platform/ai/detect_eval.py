"""Detection-only evaluation on WILDTRACK: run just the detector (no tracker, no Re-ID) and score the boxes.

Reports AP50 and AP50-95 (the standard detection metrics, independent of any confidence threshold), an operating-point
report (recall / precision at one confidence), recall split by how tall the person is, and a variant that does not
punish boxes outside the estimated annotated area (WILDTRACK only labels people inside a fixed ground area).
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .diagnose import _inside, _iou, annotated_region

IOU_THRS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
SIZE_BUCKETS = (("small_lt60px", 0, 60), ("medium_60_120px", 60, 120), ("large_gt120px", 120, 1e9))


def _match(dets, gts, thr, ignore_outside=None):
    """Greedy matching in descending confidence. dets: [(frame_key, conf, box)], gts: {frame_key: ndarray [n,4]}.
    Returns (statuses, used): statuses is a list of (conf, 1=true positive / 0=false positive / -1=ignored), highest
    confidence first; used[frame_key] flags the ground-truth boxes that got matched. `ignore_outside(frame_key, box)`
    returns False for a box outside the annotated area: if it matches nothing it is ignored instead of counted as a FP."""
    used = {k: np.zeros(len(g), bool) for k, g in gts.items()}
    out = []
    for key, conf, box in sorted(dets, key=lambda d: -d[1]):
        g, best, bj = gts.get(key), -1.0, -1
        if g is not None and len(g):
            ious = np.where(used[key], -1.0, _iou(np.array([box], float), g)[0])
            bj = int(ious.argmax())
            best = float(ious[bj])
        if best >= thr:
            used[key][bj] = True
            out.append((conf, 1))
        elif ignore_outside is not None and not ignore_outside(key, box):
            out.append((conf, -1))
        else:
            out.append((conf, 0))
    return out, used


def average_precision(dets, gts, thr: float = 0.5, ignore_outside=None):
    """All-point interpolated AP (PASCAL VOC style). None if there is no ground truth."""
    n_gt = sum(len(g) for g in gts.values())
    if n_gt == 0:
        return None
    flags = [s for _, s in _match(dets, gts, thr, ignore_outside)[0] if s != -1]
    if not flags:
        return 0.0
    tp = np.cumsum([f == 1 for f in flags])
    fp = np.cumsum([f == 0 for f in flags])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _scores(dets, gts, ignore_outside, conf_op):
    pct = lambda v: None if v is None else round(100.0 * v, 1)
    ap50 = average_precision(dets, gts, 0.5)
    ap5095 = float(np.mean([average_precision(dets, gts, t) for t in IOU_THRS])) if ap50 is not None else None
    ap50_area = average_precision(dets, gts, 0.5, ignore_outside)
    op = [d for d in dets if d[1] >= conf_op]
    status, used = _match(op, gts, 0.5, ignore_outside)
    tp = sum(s == 1 for _, s in status)
    fp = sum(s == 0 for _, s in status)
    ign = sum(s == -1 for _, s in status)
    n_gt = sum(len(g) for g in gts.values())
    heights = np.concatenate([g[:, 3] - g[:, 1] for g in gts.values() if len(g)] or [np.zeros(0)])
    hit = np.concatenate([used[k] for k, g in gts.items() if len(g)] or [np.zeros(0, bool)])
    by_size = {}
    for name, lo, hi in SIZE_BUCKETS:
        m = (heights >= lo) & (heights < hi)
        by_size[name] = {"gt": int(m.sum()), "recall": pct(hit[m].mean()) if m.any() else None}
    return {"gt_boxes": n_gt, "AP50": pct(ap50), "AP50_95": pct(ap5095), "AP50_ignoring_outside_area": pct(ap50_area),
            "recall_at_conf": pct(tp / n_gt) if n_gt else None,
            "precision_at_conf": pct(tp / (tp + fp + ign)) if tp + fp + ign else None,
            "precision_ignoring_outside_area": pct(tp / (tp + fp)) if tp + fp else None,
            "boxes_at_conf": len(op), "recall_by_person_height": by_size}


def evaluate_detector(images: dict, gt: dict, detect, conf_op: float = 0.25, margin: float = 0.2) -> dict:
    """images {cam: [paths]}, gt {cam: [[frame_idx, pid, x1, y1, x2, y2]]} from ai.wildtrack.load_wildtrack.
    `detect(path) -> [(box, conf), ...]` person boxes at a LOW confidence (about 0.001) so the full curve is available."""
    t0, n_img = time.time(), 0
    dets, gts, regions = [], {}, {}
    for cam, paths in images.items():
        per_frame = defaultdict(list)
        for f, _, x1, y1, x2, y2 in gt.get(cam, []):
            per_frame[f].append((x1, y1, x2, y2))
        regions[cam] = annotated_region(gt.get(cam, []), margin)
        for f, p in enumerate(paths):
            gts[(cam, f)] = np.array(per_frame.get(f, []), float).reshape(-1, 4)
            for box, conf in detect(p):
                dets.append(((cam, f), float(conf), tuple(float(v) for v in box)))
            n_img += 1
    secs = time.time() - t0
    inside = lambda key, box: _inside(regions[key[0]], box)
    cams = {}
    for cam in images:
        sub = lambda d: [x for x in d if x[0][0] == cam]
        cams[cam] = _scores(sub(dets), {k: v for k, v in gts.items() if k[0] == cam}, inside, conf_op)
    return {"total": _scores(dets, gts, inside, conf_op), "cameras": cams, "images": n_img,
            "seconds": round(secs, 1), "images_per_second": round(n_img / secs, 2) if secs else None,
            "conf_for_recall_precision": conf_op}


class YoloPersonDetector:
    """Person boxes from an Ultralytics model, with no tracker involved."""

    def __init__(self, weights: str, imgsz: int = 1280, device: str = "auto", conf: float = 0.001, person_class: int = 0):
        from ultralytics import YOLO

        from .yolo_tracker import _precision_kwargs, resolve_device
        self.device = resolve_device(device)
        self.model, self.imgsz, self.conf, self.cls = YOLO(weights), imgsz, conf, person_class
        self.precision = _precision_kwargs(not self.device.startswith("cpu"))

    def __call__(self, path: str):
        r = self.model.predict(path, imgsz=self.imgsz, conf=self.conf, classes=[self.cls], device=self.device,
                               verbose=False, **self.precision)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return list(zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()))


def format_comparison(results: dict) -> str:
    f = lambda v: "-" if v is None else f"{v:.1f}"
    head = f"{'model':26s}{'AP50':>7s}{'AP50-95':>9s}{'AP50 (area)':>13s}{'recall':>8s}{'prec':>7s}{'rec small':>11s}{'rec med':>9s}{'rec large':>11s}{'img/s':>7s}"
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        t, s = r["total"], r["total"]["recall_by_person_height"]
        lines.append(f"{name[:25]:26s}{f(t['AP50']):>7s}{f(t['AP50_95']):>9s}{f(t['AP50_ignoring_outside_area']):>13s}"
                     f"{f(t['recall_at_conf']):>8s}{f(t['precision_at_conf']):>7s}"
                     f"{f(s['small_lt60px']['recall']):>11s}{f(s['medium_60_120px']['recall']):>9s}"
                     f"{f(s['large_gt120px']['recall']):>11s}{f(r['images_per_second']):>7s}")
    return "\n".join(lines)
