"""Why is a multi-camera score low? Measure detection on its own, ignoring identities.

For each camera and frame, predicted boxes are matched to ground-truth boxes (IoU >= 0.5, one-to-one).
WILDTRACK only annotates people standing inside a fixed ground area, so a detector that correctly finds
people OUTSIDE that area gets "false positives" in the main score. To show how much of the damage that is,
false positives are split by whether the box's feet fall inside an APPROXIMATION of the annotated area
(the convex hull of all ground-truth feet positions in that camera, enlarged by `margin`).
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def annotated_region(gt_rows, margin: float = 0.2):
    """Approximate annotated ground area as seen in one camera, or None if it cannot be estimated."""
    pts = np.array([[(r[2] + r[4]) / 2, r[5]] for r in gt_rows], np.float32)
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts)[:, 0, :]
    if cv2.contourArea(hull) < 1.0:
        return None
    c = hull.mean(axis=0)
    return ((hull - c) * (1 + margin) + c).astype(np.float32)


def _inside(region, box) -> bool:
    if region is None:
        return True
    foot = (float((box[0] + box[2]) / 2), float(box[3]))
    return cv2.pointPolygonTest(region.reshape(-1, 1, 2), foot, False) >= 0


def detection_report(run, gt: dict, iou_thr: float = 0.5, margin: float = 0.2) -> dict:
    """Returns {"cameras": {cam: stats}, "total": stats}. Percentages are 0-100."""
    out = {}
    for cam in run.meta["cameras"]:
        rows = gt.get(cam, [])
        gt_by, pr_by, frames = defaultdict(list), defaultdict(list), set()
        for f, _, x1, y1, x2, y2 in rows:
            gt_by[f].append((x1, y1, x2, y2))
        for ev in run.events:
            if ev.cam == cam:
                frames.add(ev.idx)
                pr_by[ev.idx] += [r.bbox for r in ev.recs if r.cls == "person"]
        region = annotated_region(rows, margin)
        tp = fn = fp_in = fp_out = fp_unk = n_gt = n_pr = 0
        for f in frames:
            g = np.array(gt_by.get(f, []), float).reshape(-1, 4)
            p = np.array(pr_by.get(f, []), float).reshape(-1, 4)
            n_gt, n_pr = n_gt + len(g), n_pr + len(p)
            hit_p, hit_g = set(), 0
            if len(g) and len(p):
                iou = _iou(g, p)
                for i, j in zip(*linear_sum_assignment(-iou)):
                    if iou[i, j] >= iou_thr:
                        hit_p.add(j)
                        hit_g += 1
            tp, fn = tp + hit_g, fn + len(g) - hit_g
            for j in range(len(p)):
                if j not in hit_p:
                    if region is None:
                        fp_unk += 1                      # cannot tell: no estimable area for this camera
                    elif _inside(region, p[j]):
                        fp_in += 1
                    else:
                        fp_out += 1
        out[cam] = _stats(tp, fn, fp_in, fp_out, fp_unk, n_gt, n_pr, region is not None)
    keys = ("tp", "fn", "fp_inside", "fp_outside", "fp_unknown", "gt_boxes", "pred_boxes")
    tot = {k: sum(c[k] for c in out.values()) for k in keys}
    total = _stats(tot["tp"], tot["fn"], tot["fp_inside"], tot["fp_outside"], tot["fp_unknown"],
                   tot["gt_boxes"], tot["pred_boxes"], any(c["area_estimated"] for c in out.values()))
    with_area = [c for c in out.values() if c["area_estimated"]]          # in-area figures: only cameras with an area
    if with_area:
        tp_a, fi, fo = (sum(c[k] for c in with_area) for k in ("tp", "fp_inside", "fp_outside"))
        total["precision_inside_area"] = round(100.0 * tp_a / (tp_a + fi), 1) if tp_a + fi else None
        total["fp_outside_share"] = round(100.0 * fo / (fi + fo), 1) if fi + fo else None
    return {"cameras": out, "total": total}


def _stats(tp, fn, fp_in, fp_out, fp_unk, n_gt, n_pr, has_region) -> dict:
    pct = lambda a, b: round(100.0 * a / b, 1) if b else None
    return {"gt_boxes": n_gt, "pred_boxes": n_pr, "tp": tp, "fn": fn, "fp_inside": fp_in, "fp_outside": fp_out,
            "fp_unknown": fp_unk, "recall": pct(tp, tp + fn), "precision_all": pct(tp, tp + fp_in + fp_out + fp_unk),
            "precision_inside_area": pct(tp, tp + fp_in) if has_region else None,
            "fp_outside_share": pct(fp_out, fp_in + fp_out) if has_region else None,
            "area_estimated": has_region}


def format_report(rep: dict) -> str:
    h = f"{'camera':8s}{'GT':>7s}{'pred':>7s}{'recall%':>9s}{'prec% all':>11s}{'prec% in-area':>15s}{'FP outside%':>13s}"
    lines = [h, "-" * len(h)]
    f = lambda v: "-" if v is None else f"{v:.1f}"
    for name, s in [*rep["cameras"].items(), ("TOTAL", rep["total"])]:
        lines.append(f"{name:8s}{s['gt_boxes']:7d}{s['pred_boxes']:7d}{f(s['recall']):>9s}{f(s['precision_all']):>11s}"
                     f"{f(s['precision_inside_area']):>15s}{f(s['fp_outside_share']):>13s}")
    return "\n".join(lines)
