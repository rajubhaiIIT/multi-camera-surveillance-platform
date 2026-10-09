#!/usr/bin/env python3
"""Convert a Kaggle-style CUHK03 folder into the Market-1501 layout that ai.reid.eval_market reads.

Supports two item formats found in the wild:
  Format A: [pid, cam, rel_path]   (older Kaggle versions)
  Format B: [rel_path, cam, pid]   (newer Kaggle versions, e.g. from your download)

Usage:
    python scripts/convert_cuhk03.py --data data/raw/cuhk03 --out data/raw/cuhk03_market
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def find_cuhk03_root(path: str | Path) -> Path:
    p = Path(path)
    for candidate in [p, *p.rglob("splits_new_detected.json")]:
        root = candidate if candidate.is_dir() else candidate.parent
        if (root / "splits_new_detected.json").exists():
            return root
    raise FileNotFoundError(
        f"CUHK03 not found under {path}. "
        "Expected splits_new_detected.json and images_detected/ in the same folder.")


def parse_item(item) -> tuple[int, int, str] | None:
    """Return (pid, cam, rel_path) regardless of item format, or None to skip."""
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        a, b, c = item[0], item[1], item[2]
        # Format A: [pid, cam, path]
        if isinstance(a, int) and isinstance(b, int) and isinstance(c, str):
            return int(a), int(b), str(c)
        # Format B: [path, cam, pid]  <-- this is what your Kaggle version uses
        if isinstance(a, str) and isinstance(c, int):
            rel, cam, pid = str(a), int(b), int(c)
            return pid, cam, rel
        # Format C: all strings, pid/cam encoded in the filename
        if isinstance(a, str):
            rel = str(a)
            parts = Path(rel).stem.replace("\\", "/").split("/")[-1].split("_")
            try:
                pid = int(parts[0])
                cam = int(b) if isinstance(b, int) else 0
            except (IndexError, ValueError):
                return None
            return pid, cam, rel
    elif isinstance(item, dict):
        pid = int(item.get("pid", item.get("id", 0)))
        cam = int(item.get("camid", item.get("cam", 0)))
        rel = str(item.get("img_path", item.get("path", "")))
        return pid, cam, rel
    return None


def find_image(root: Path, rel: str) -> Path | None:
    """Try several path variants to locate the image file."""
    # normalise separators
    rel_clean = rel.replace("\\", "/").lstrip("./")
    stem = Path(rel_clean).name
    for candidate in [
        root / rel_clean,
        root / stem,
        root / "images_detected" / stem,
        root / "images_labeled" / stem,
        # handle ./data\cuhk03\images_detected\... prefix from your version
        root / Path(rel_clean).parts[-1] if Path(rel_clean).parts else root / stem,
    ]:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def convert(data: str, out: str, split_idx: int = 0) -> dict:
    root = find_cuhk03_root(data)
    raw = json.loads((root / "splits_new_detected.json").read_text())
    split = raw[split_idx] if isinstance(raw, list) else raw
    out_root = Path(out)
    counts = {"bounding_box_train": 0, "query": 0, "bounding_box_test": 0}
    missing, skipped = 0, 0

    for folder in counts:
        (out_root / folder).mkdir(parents=True, exist_ok=True)

    def write(folder: str, pid: int, cam: int, idx: int, src: Path):
        name = f"{pid:04d}_c{cam + 1}s1_{idx:06d}_00{src.suffix}"
        dst = out_root / folder / name
        if not dst.exists():
            shutil.copy2(src, dst)
        counts[folder] += 1

    folder_map = (
        ("bounding_box_train", "train"),
        ("query", "query"),
        ("bounding_box_test", "gallery"),
    )

    for folder, key in folder_map:
        items = split.get(key, [])
        for i, item in enumerate(items):
            parsed = parse_item(item)
            if parsed is None:
                skipped += 1
                continue
            pid, cam, rel = parsed
            src = find_image(root, rel)
            if src is None:
                missing += 1
                if missing <= 5:
                    print(f"  cannot find image: {rel}", file=sys.stderr)
                continue
            write(folder, pid, cam, i, src)

    train_pids = {Path(f).name.split("_")[0]
                  for f in (out_root / "bounding_box_train").glob("*.*")}
    test_pids = {Path(f).name.split("_")[0]
                 for f in (out_root / "query").glob("*.*")}
    info = {**counts, "missing_images": missing, "skipped_items": skipped,
            "train_people": len(train_pids), "test_people": len(test_pids)}
    (out_root / "split_info.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="folder containing splits_new_detected.json")
    ap.add_argument("--out", required=True,
                    help="where to write bounding_box_train/, query/, bounding_box_test/")
    ap.add_argument("--split-idx", type=int, default=0,
                    help="which split to use when the JSON is a list (default: 0)")
    a = ap.parse_args(argv)

    print(f"reading from {a.data}")
    try:
        root = find_cuhk03_root(a.data)
    except FileNotFoundError as e:
        print(f"error: {e}")
        return 1

    raw = json.loads((root / "splits_new_detected.json").read_text())
    n_splits = len(raw) if isinstance(raw, list) else 1
    split = raw[a.split_idx] if isinstance(raw, list) else raw
    print(f"splits_new_detected.json: {n_splits} split(s), keys: {list(split)[:6]}")
    for k in ("train", "query", "gallery"):
        items = split.get(k, [])
        print(f"  {k}: {len(items)} items, first: {items[0] if items else 'empty'!r}")
        if items:
            p = parse_item(items[0])
            print(f"  parsed as: pid={p[0] if p else '?'}, cam={p[1] if p else '?'}, path={p[2] if p else '?'}")

    info = convert(a.data, a.out, a.split_idx)
    print(json.dumps(info, indent=2))

    if info["missing_images"] > 0:
        print(f"\nwarning: {info['missing_images']} images could not be found.")
    if info["query"] == 0:
        print("\nerror: no query images written. Paste the output above and I will fix it.")
        return 1

    print(f"\nDone. Next:\n"
          f"  python -m ai.reid.eval_market "
          f"--ckpt models/reid/resnet50_market/best.pt --data {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
