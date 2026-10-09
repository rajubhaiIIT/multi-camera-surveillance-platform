"""Cross-camera (MTMC) scoring. All cameras are laid end to end as ONE sequence, so a person only keeps
their identity score if the SAME global ID is used in every camera they appear in."""
from __future__ import annotations

from pathlib import Path

from .mot_eval import evaluate
from .mot_io import write_mot


def evaluate_mtmc(root: Path, cams: dict[str, dict], tracker_name: str = "mtmc", print_results: bool = True) -> dict:
    """cams[name] = {"gt":   [(frame, global_id, x1, y1, x2, y2), ...],
                     "pred": [(frame, global_id, x1, y1, x2, y2, conf), ...],
                     "n_frames": int}          # frames are 1-based within each camera"""
    root = Path(root)
    seq = root / "_mtmc_in" / "MTMC-01"
    (seq / "gt").mkdir(parents=True, exist_ok=True)
    gt_lines, pred_rows, offset = [], [], 0
    for name in cams:
        c = cams[name]
        seen_gt: set = set()
        for f, gid, x1, y1, x2, y2 in c["gt"]:
            key = (gid, f + offset)
            if key not in seen_gt:
                gt_lines.append(f"{f + offset},{gid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,1,1.0")
                seen_gt.add(key)
        seen_pred: dict = {}
        for f, gid, x1, y1, x2, y2, conf in c["pred"]:
            key = (gid, f + offset)
            if key not in seen_pred or conf > seen_pred[key][-1]:
                seen_pred[key] = (f + offset, gid, x1, y1, x2, y2, conf)
        pred_rows.extend(seen_pred.values())
        offset += int(c["n_frames"])
    (seq / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n")
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=MTMC-01\nimDir=img1\nframeRate=30\n"
        f"seqLength={offset}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")
    pred_file = root / "_mtmc_in" / "pred" / "MTMC-01.txt"
    write_mot(pred_file, pred_rows)
    return evaluate(root / "te", {"MTMC-01": seq}, {"MTMC-01": pred_file}, tracker_name, print_results=print_results)


def tracklet_link_scores(items: list[tuple]) -> dict:
    """How well did the matcher link tracklets? items: [(camera, true_person, predicted_global_id_or_None)].
    A pair of tracklets is a hit when both share the true person AND the predicted ID.
    Reported separately for pairs from different cameras (the Re-ID problem) and from the same camera."""
    import itertools
    out = {}
    for scope in ("cross_camera", "same_camera"):
        tp = fp = fn = 0
        for (c1, l1, g1), (c2, l2, g2) in itertools.combinations(items, 2):
            if (c1 == c2) != (scope == "same_camera"):
                continue
            same_label, same_gid = l1 == l2, g1 is not None and g1 == g2
            tp += same_label and same_gid
            fp += (not same_label) and same_gid
            fn += same_label and not same_gid
        p = tp / (tp + fp) * 100 if tp + fp else None
        r = tp / (tp + fn) * 100 if tp + fn else None
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else None
        out[scope] = {"precision": p, "recall": r, "f1": f1, "pairs": int(tp + fp + fn)}
    return out
