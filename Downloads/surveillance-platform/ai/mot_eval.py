"""Evaluate tracker output in MOTChallenge format with TrackEval (HOTA, CLEAR/MOTA, IDF1)."""
from __future__ import annotations

import builtins
import json
import shutil
import warnings
from pathlib import Path

import numpy as np

# TrackEval was written for old NumPy; restore the aliases NumPy removed.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _n in ("float", "int", "bool", "object", "str"):
        if _n not in vars(np):
            setattr(np, _n, getattr(builtins, _n))


def prepare_tree(root: Path, gt_dirs: dict[str, Path], pred_files: dict[str, Path],
                 tracker_name: str, benchmark: str = "MOT17", split: str = "train") -> dict:
    """Lay out the folders TrackEval expects. gt_dirs: seq -> folder containing gt/gt.txt and seqinfo.ini."""
    bs = f"{benchmark}-{split}"
    gt_root = root / "gt" / "mot_challenge"
    trk_dir = root / "trackers" / "mot_challenge" / bs / tracker_name / "data"
    if root.exists():
        shutil.rmtree(root)
    trk_dir.mkdir(parents=True)
    (gt_root / "seqmaps").mkdir(parents=True)
    for seq, src in gt_dirs.items():
        dst = gt_root / bs / seq
        (dst / "gt").mkdir(parents=True)
        shutil.copy(src / "gt" / "gt.txt", dst / "gt" / "gt.txt")
        shutil.copy(src / "seqinfo.ini", dst / "seqinfo.ini")
        shutil.copy(pred_files[seq], trk_dir / f"{seq}.txt")
    (gt_root / "seqmaps" / f"{bs}.txt").write_text("name\n" + "\n".join(gt_dirs) + "\n")
    return {"gt": gt_root, "trackers": root / "trackers" / "mot_challenge"}


def evaluate(root: Path, gt_dirs: dict[str, Path], pred_files: dict[str, Path],
             tracker_name: str = "run", benchmark: str = "MOT17", split: str = "train",
             print_results: bool = True) -> dict:
    """Return {"combined": {...}, "per_sequence": {seq: {...}}} with metrics in percent."""
    import trackeval  # noqa: E402  (after the NumPy shim above)

    paths = prepare_tree(root, gt_dirs, pred_files, tracker_name, benchmark, split)
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_RESULTS=print_results,
                    OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
                    TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True)
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(GT_FOLDER=str(paths["gt"]), TRACKERS_FOLDER=str(paths["trackers"]),
                  TRACKERS_TO_EVAL=[tracker_name], BENCHMARK=benchmark, SPLIT_TO_EVAL=split,
                  PRINT_CONFIG=False, OUTPUT_FOLDER=str(root / "out"))
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = trackeval.Evaluator(eval_cfg).evaluate(
        [trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)

    raw = res["MotChallenge2DBox"][tracker_name]

    def pick(r: dict) -> dict:
        r = r["pedestrian"]
        return {
            "HOTA": float(np.mean(r["HOTA"]["HOTA"])) * 100,
            "DetA": float(np.mean(r["HOTA"]["DetA"])) * 100,
            "AssA": float(np.mean(r["HOTA"]["AssA"])) * 100,
            "MOTA": float(r["CLEAR"]["MOTA"]) * 100,
            "IDF1": float(r["Identity"]["IDF1"]) * 100,
            "IDSW": int(r["CLEAR"]["IDSW"]),
            "FP": int(r["CLEAR"]["CLR_FP"]),
            "FN": int(r["CLEAR"]["CLR_FN"]),
        }

    out = {"combined": pick(raw["COMBINED_SEQ"]),
           "per_sequence": {s: pick(v) for s, v in raw.items() if s != "COMBINED_SEQ"}}
    (root / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
