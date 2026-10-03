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
        for f, gid, x1, y1, x2, y2 in c["gt"]:
            gt_lines.append(f"{f + offset},{gid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,1,1.0")
        for f, gid, x1, y1, x2, y2, conf in c["pred"]:
            pred_rows.append((f + offset, gid, x1, y1, x2, y2, conf))
        offset += int(c["n_frames"])
    (seq / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n")
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=MTMC-01\nimDir=img1\nframeRate=30\n"
        f"seqLength={offset}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")
    pred_file = root / "_mtmc_in" / "pred" / "MTMC-01.txt"
    write_mot(pred_file, pred_rows)
    return evaluate(root / "te", {"MTMC-01": seq}, {"MTMC-01": pred_file}, tracker_name, print_results=print_results)
