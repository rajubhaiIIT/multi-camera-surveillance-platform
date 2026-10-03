#!/usr/bin/env python3
"""Rank-1 / mAP of a trained Re-ID checkpoint on Market-1501 (and optionally on another Market-style folder).

    python -m ai.reid.eval_market --ckpt models/reid/resnet50_market/best.pt --data data/raw/market1501
    python -m ai.reid.eval_market --ckpt ... --data data/raw/cuhk03_market_format   # cross-dataset test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ReIDDataset, find_market_root, load_split, test_transform
from .metrics import evaluate_market, suggest_threshold
from .model import load_reid_model


@torch.inference_mode()
def extract_features(model, items, device: str, batch_size: int = 256, workers: int = 4) -> np.ndarray:
    ds = ReIDDataset(items, test_transform())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=device.startswith("cuda"))
    feats = []
    model.eval()
    for imgs, _, _ in dl:
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            f = model(imgs.to(device, non_blocking=True))
        feats.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(feats).numpy()


def evaluate_folder(model, data: str, device: str, workers: int = 4, with_threshold: bool = True) -> dict:
    root = find_market_root(data)
    q, g = load_split(root, "query"), load_split(root, "gallery")
    qf, gf = extract_features(model, q, device, workers=workers), extract_features(model, g, device, workers=workers)
    qp, qc = (np.array(v) for v in zip(*[(p, c) for _, p, c in q]))
    gp, gc = (np.array(v) for v in zip(*[(p, c) for _, p, c in g]))
    res = evaluate_market(qf, qp, qc, gf, gp, gc)
    res["queries"], res["gallery"] = len(q), len(g)
    if with_threshold:
        res["threshold_suggestion"] = suggest_threshold(qf, qp, qc, gf, gp, gc)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="write the metrics to this json file")
    a = ap.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    model, meta = load_reid_model(a.ckpt, device)
    res = evaluate_folder(model, a.data, device, a.workers)
    res.update(checkpoint=str(a.ckpt), data=str(a.data), backbone=meta["backbone"], feat_dim=meta["feat_dim"])
    print(json.dumps(res, indent=2))
    print(f"\nRank-1 {res['rank1']:.1f}% | Rank-5 {res['rank5']:.1f}% | mAP {res['mAP']:.1f}%")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
