#!/usr/bin/env python3
"""Convert a Kaggle-style CUHK03 folder into the Market-1501 layout that ai.reid.eval_market reads.

Uses the "new protocol" (767/700 split, same split style as Market-1501) with detected boxes.
The output has bounding_box_train/, query/ and bounding_box_test/ folders with file names in
the Market-1501 format: <pid:04d>_c<cam+1>s1_<idx:06d>_00.jpg

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
        f"CUHK03 not found under {path}. Expected splits_new_detected.json "
        "and images_detected/ in the same folder.")


def find_image(root: Path, rel: str) -> Path | None:
    """Try several common path styles for the image."""
    for candidate in [
        root / rel,
        root / "images_detected" / Path(rel).name,
        root / "images_detected" / rel,
        root / rel.lstrip("/"),
    ]:
        if candidate.exists():
            return candidate
    return None


def convert(data: str, out: str, split_idx: int = 0) -> dict:
    root = find_cuhk03_root(data)
    raw = json.loads((root / "splits_new_detected.json").read_text())

    # Handle both a list of splits and a single split dict
    split = raw[split_idx] if isinstance(raw, list) else raw
    counts = {"bounding_box_train": 0, "query": 0, "bounding_box_test": 0}
    missing = 0

    out_root = Path(out)
    for folder in counts:
        (out_root / folder).mkdir(parents=True, exist_ok=True)

    def write(folder: str, pid: int, cam: int, idx: int, src: Path):
        name = f"{pid:04d}_c{cam + 1}s1_{idx:06d}_00.jpg"
        dst = out_root / folder / name
        if not dst.exists():
            shutil.copy2(src, dst)
        counts[folder] += 1

    for folder, key in (("bounding_box_train", "train"),
                        ("query", "query"),
                        ("bounding_box_test", "gallery")):
        items = split.get(key, [])
        for i, item in enumerate(items):
            # item may be [pid, cam, rel_path] or just rel_path or a dict
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                pid, cam, rel = int(item[0]), int(item[1]), str(item[2])
            elif isinstance(item, str):
                # path like "images_detected/0001_c1_00003.jpg"
                p = Path(item)
                parts = p.stem.split("_")
                try:
                    pid, cam = int(parts[0]), int(parts[1].lstrip("c")) - 1
                except (IndexError, ValueError):
                    pid, cam = i, 0
                rel = item
            elif isinstance(item, dict):
                pid = int(item.get("pid", item.get("id", i)))
                cam = int(item.get("camid", item.get("cam", 0)))
                rel = str(item.get("img_path", item.get("path", "")))
            else:
                print(f"  skipping unknown item format: {item!r}", file=sys.stderr)
                continue

            src = find_image(root, rel)
            if src is None:
                missing += 1
                if missing <= 5:
                    print(f"  cannot find image: {rel}", file=sys.stderr)
                continue
            write(folder, pid, cam, i, src)

    info = {**counts, "missing_images": missing,
            "train_people": len({int(Path(f).name.split("_")[0])
                                 for f in (out_root / "bounding_box_train").glob("*.jpg")}),
            "test_people": len({int(Path(f).name.split("_")[0])
                                for f in (out_root / "query").glob("*.jpg")})}
    (out_root / "split_info.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
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

    # Print a short preview of the JSON format to help diagnose issues
    raw = json.loads((root / "splits_new_detected.json").read_text())
    n_splits = len(raw) if isinstance(raw, list) else 1
    split = raw[a.split_idx] if isinstance(raw, list) else raw
    print(f"splits_new_detected.json: {n_splits} split(s), keys: {list(split)[:6]}")
    for k in ("train", "query", "gallery"):
        items = split.get(k, [])
        print(f"  {k}: {len(items)} items, first: {items[0] if items else 'empty'!r}")

    info = convert(a.data, a.out, a.split_idx)
    print(json.dumps(info, indent=2))
    if info["missing_images"] > 0:
        print(f"\nwarning: {info['missing_images']} images could not be found. "
              "Check that images_detected/ is inside the same folder as the JSON files.")
    if info["query"] == 0:
        print("\nerror: no query images were written. "
              "The JSON format may not match any supported layout. "
              "Paste the 'first:' lines above and I will fix the converter.")
        return 1
    print(f"\nDone. Next:\n"
          f"  python -m ai.reid.eval_market "
          f"--ckpt models/reid/resnet50_market/best.pt --data {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
