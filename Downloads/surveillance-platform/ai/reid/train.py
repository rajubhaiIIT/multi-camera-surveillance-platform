#!/usr/bin/env python3
"""Train the Re-ID model on Market-1501 (ID loss + triplet loss, PK sampling, mixed precision).

    python -m ai.reid.train --data data/raw/market1501 --out models/reid/resnet50_market
Runs on one consumer GPU (about 4 GB with the defaults). Add --epochs 5 for a quick trial.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PKSampler, ReIDDataset, find_market_root, load_split, train_transform
from .eval_market import evaluate_folder
from .losses import LabelSmoothCE, batch_hard_triplet
from .model import ReIDNet, save_checkpoint


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_init(model: ReIDNet, path: str) -> tuple[int, list[str]]:
    """Start from an existing checkpoint (e.g. Market-1501) for fine-tuning. The classifier is skipped: it has
    one output per training person, and a new dataset has different people."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("feat_dim") != model.feat_dim:
        raise ValueError(f"checkpoint embedding size {ck.get('feat_dim')} != model {model.feat_dim}; "
                         f"pass the same --feat-dim and --backbone")
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("classifier.")}
    res = model.load_state_dict(sd, strict=False)
    if res.unexpected_keys:
        raise ValueError(f"checkpoint does not match this backbone (unexpected: {res.unexpected_keys[:3]}...)")
    return len(sd), list(res.missing_keys)


def lr_factor(epoch: int, epochs: int, warmup: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    prog = (epoch - warmup) / max(1, epochs - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing (or above) Market-1501")
    ap.add_argument("--out", default="models/reid/resnet50_market")
    ap.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--feat-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--p", type=int, default=16, help="identities per batch")
    ap.add_argument("--k", type=int, default=4, help="images per identity per batch")
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--init-ckpt", help="fine-tune: start from this checkpoint (same --backbone / --feat-dim)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline / tests)")
    ap.add_argument("--max-batches", type=int, help="debug: batches per epoch")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    set_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    use_cuda = device.startswith("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    root = find_market_root(a.data)
    ds = ReIDDataset(load_split(root, "train"), train_transform(), relabel=True)
    sampler = PKSampler(ds.labels, a.p, a.k, num_batches=a.max_batches, seed=a.seed)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=a.workers, pin_memory=use_cuda,
                    persistent_workers=a.workers > 0)
    print(f"train: {len(ds)} images, {ds.num_classes} people | device={device} | "
          f"{len(sampler)} batches/epoch of {a.p * a.k}")

    model = ReIDNet(a.backbone, ds.num_classes, a.feat_dim, pretrained=not a.no_pretrained).to(device)
    if a.init_ckpt:
        n, missing = load_init(model, a.init_ckpt)
        print(f"initialised {n} tensors from {a.init_ckpt} (new, untrained: {missing})")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: lr_factor(e, a.epochs, a.warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    ce = LabelSmoothCE()

    best, history = -1.0, []
    for epoch in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for imgs, labels, _ in dl:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_cuda):
                logits, feats = model(imgs)
            loss = ce(logits, labels) + batch_hard_triplet(feats, labels, a.margin)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item(), n + 1
        sched.step()
        rec = {"epoch": epoch + 1, "loss": round(tot / max(n, 1), 4), "lr": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        last = epoch + 1 == a.epochs
        if (epoch + 1) % a.eval_every == 0 or last:
            m = evaluate_folder(model, a.data, device, a.workers, with_threshold=last)
            rec.update(rank1=round(m["rank1"], 2), mAP=round(m["mAP"], 2))
            if m["mAP"] > best:
                best = m["mAP"]
                save_checkpoint(out / "best.pt", model, a.backbone, ds.num_classes, epoch + 1, m)
            if last:
                (out / "metrics.json").write_text(json.dumps(m, indent=2))
        history.append(rec)
        print(rec, flush=True)
        save_checkpoint(out / "last.pt", model, a.backbone, ds.num_classes, epoch + 1, None)
        (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best mAP {best:.2f}. checkpoints in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
