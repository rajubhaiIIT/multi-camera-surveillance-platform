#!/usr/bin/env python3
"""Score a run from run_mot17.py with TrackEval (HOTA, MOTA, IDF1, ID switches).

    python eval/eval_mot17.py --run-name yolo11n_1280
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.mot_eval import evaluate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--mot-root")
    a = ap.parse_args()

    run = ROOT / "eval" / "runs" / a.run_name
    preds = {p.stem: p for p in sorted((run / "tracks").glob("*.txt"))}
    if not preds:
        sys.exit(f"no result files in {run / 'tracks'}. Run eval/run_mot17.py first.")
    if a.mot_root:
        mot = Path(a.mot_root)
    else:
        base = ROOT / "data" / "raw" / "mot17"
        mot = next((p for p in (base / "MOT17", base) if (p / "train").is_dir()), None)
        if mot is None:
            sys.exit("MOT17 not found")
    gts = {s: mot / "train" / s for s in preds}

    res = evaluate(run / "trackeval", gts, preds, tracker_name=a.run_name)
    c = res["combined"]
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}
    print(f"\n=== Combined ({len(preds)} MOT17 train sequences, persons) ===")
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in c.items()})
    print("\nPaste into eval/results.md:")
    print(f"| 1 | Tracking | MOT17 train | HOTA / IDF1 / MOTA | {c['HOTA']:.1f} / {c['IDF1']:.1f} / {c['MOTA']:.1f} "
          f"(IDSW {c['IDSW']}) | - | {meta.get('date', '')} | {meta.get('weights', '')} imgsz={meta.get('imgsz', '')} "
          f"conf={meta.get('conf', '')} {meta.get('overall_fps_incl_image_read', '?')} FPS on {meta.get('gpu') or meta.get('device', '')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
