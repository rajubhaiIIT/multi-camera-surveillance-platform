#!/usr/bin/env python3
"""Download / extract / verify the Phase 0 starter datasets.

Commands
    list                 show every dataset and how to get it
    fetch [names...]     download direct-link datasets, extract any archives you
                         placed in data/downloads/, then verify
    verify [names...]    report which datasets are in place (exit 1 if any missing)

Two kinds of dataset:
    url     downloaded automatically (resumable)
    manual  hosted on Google Drive / Kaggle / a form page. Download it in your
            browser, drop the archive into data/downloads/, name it after the
            dataset key (e.g. market1501.zip), then run `fetch` again to extract.

No dataset here needs an approval request. Links move over time: if an automatic
URL fails, the script says so and prints the manual instructions instead.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data" / "downloads"
RAW = ROOT / "data" / "raw"

# expect = folder/file name patterns (searched recursively) that prove the data is in place.
DATASETS: dict[str, dict] = {
    "mot17": {
        "title": "MOT17 (single-camera tracking)",
        "method": "url",
        "url": "https://motchallenge.net/data/MOT17.zip",
        "expect": ["train", "img1"],
        "size": "about 5 GB",
        "used_in": "Phase 1 (tracking metrics)",
        "page": "https://motchallenge.net/data/MOT17/",
    },
    "lfw": {
        "title": "LFW (face verification)",
        "method": "url",
        "url": "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
        "expect": ["lfw"],
        "size": "about 170 MB",
        "used_in": "Phase 3 (face recognition)",
        "page": "http://vis-www.cs.umass.edu/lfw/",
    },
    "market1501": {
        "title": "Market-1501 (person Re-ID)",
        "method": "manual",
        "expect": ["bounding_box_train", "bounding_box_test", "query"],
        "size": "about 150 MB",
        "used_in": "Phase 2 (Re-ID training and test)",
        "how": "Search 'Market-1501' on Kaggle or Hugging Face Datasets (community copies), "
               "or the official page linked from the paper. Expect a folder with "
               "bounding_box_train / bounding_box_test / query.",
    },
    "cuhk03": {
        "title": "CUHK03 (cross-dataset Re-ID test)",
        "method": "manual",
        "expect": ["*cuhk03*"],
        "size": "about 1 GB",
        "used_in": "Phase 2 (cross-dataset check)",
        "how": "Search 'CUHK03' on Kaggle or Hugging Face. The torchreid docs also describe "
               "the expected layout (cuhk03_release + the new-protocol files).",
    },
    "rwf2000": {
        "title": "RWF-2000 (violence recognition)",
        "method": "manual",
        "expect": ["Fight", "NonFight"],
        "size": "several GB",
        "used_in": "Phase 4 (violence classifier)",
        "how": "GitHub repo 'RWF2000-Video-Database-for-Violence-Detection' lists the download "
               "links; Kaggle mirrors also exist. Expect train/val folders each with "
               "Fight and NonFight.",
    },
    "urfall": {
        "title": "UR Fall Detection (fall detection)",
        "method": "manual",
        "expect": ["fall-*", "adl-*"],
        "size": "start with a few sequences",
        "used_in": "Phase 4 (fall detection)",
        "page": "http://fenix.ur.edu.pl/mkepski/ds/uf.html",
        "how": "Open the page and download a handful of fall-XX and adl-XX RGB sequences "
               "(camera 0 is enough). Unzip them into one folder.",
    },
    "wildtrack": {
        "title": "WILDTRACK (multi-camera tracking)",
        "method": "manual",
        "expect": ["annotations_positions", "Image_subsets"],
        "size": "about 2 GB",
        "used_in": "Phase 2 (cross-camera IDF1)",
        "page": "https://www.epfl.ch/labs/cvlab/data/data-wildtrack/",
        "how": "Download the dataset from the EPFL page. Expect annotations_positions, "
               "calibrations and Image_subsets folders.",
    },
    "weapons": {
        "title": "One Roboflow / Kaggle weapon dataset (guns, knives)",
        "method": "manual",
        "expect": ["data.yaml"],
        "size": "varies",
        "used_in": "Phase 5 (weapon detection)",
        "how": "On Roboflow Universe search 'weapon detection' or 'gun knife', pick a set with "
               "a permissive license and many images, export as 'YOLOv8' format (needs a free "
               "account). Check the license and note it in docs/.",
    },
}

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2")


# ---------------------------------------------------------------- helpers
def dest_of(name: str) -> Path:
    return RAW / name


def is_present(name: str) -> tuple[bool, list[str]]:
    """Return (all expected patterns found?, list of missing patterns)."""
    d = dest_of(name)
    if not d.exists():
        return False, list(DATASETS[name]["expect"])
    missing = [p for p in DATASETS[name]["expect"] if next(d.rglob(p), None) is None]
    return not missing, missing


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download(url: str, target: Path) -> None:
    """Resumable HTTP download."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        resumed = resp.status == 206
        if have and not resumed:
            have = 0  # server ignored Range; restart
        total = resp.headers.get("Content-Length")
        total = int(total) + have if total else None
        mode = "ab" if resumed else "wb"
        done = have
        with open(part, mode) as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {human(done)} / {human(total)} ({done * 100 // total}%)", end="", flush=True)
                else:
                    print(f"\r  {human(done)}", end="", flush=True)
    print()
    part.rename(target)


def safe_extract(archive: Path, out: Path) -> None:
    """Extract zip/tar, refusing paths that escape the output folder."""
    out.mkdir(parents=True, exist_ok=True)
    base = out.resolve()

    def check(p: str) -> None:
        if not (base / p).resolve().is_relative_to(base):
            raise RuntimeError(f"unsafe path in archive: {p}")

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                check(n)
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                check(m.name)
            t.extractall(out)


def find_archive(name: str) -> Path | None:
    if not DOWNLOADS.exists():
        return None
    for p in sorted(DOWNLOADS.iterdir()):
        if p.name.lower().startswith(name) and p.name.lower().endswith(ARCHIVE_SUFFIXES):
            return p
    return None


def print_manual(name: str) -> None:
    d = DATASETS[name]
    print(f"  MANUAL: {d['title']}")
    if d.get("page"):
        print(f"    page : {d['page']}")
    print(f"    how  : {d['how']}")
    print(f"    then : save the archive as data/downloads/{name}.zip (or .tar/.tgz) and run 'fetch' again,")
    print(f"           or unpack it yourself into data/raw/{name}/")


# ---------------------------------------------------------------- commands
def cmd_list(_: argparse.Namespace) -> int:
    for name, d in DATASETS.items():
        print(f"{name:11s} [{d['method']:6s}] {d['title']}  ({d['size']}; {d['used_in']})")
    return 0


def fetch_one(name: str) -> None:
    d = DATASETS[name]
    ok, _ = is_present(name)
    if ok:
        print(f"[{name}] already in place")
        return

    archive = find_archive(name)
    if archive is None and d["method"] == "url":
        target = DOWNLOADS / Path(d["url"]).name
        if target.exists():
            archive = target
        else:
            print(f"[{name}] downloading {d['url']} ({d['size']})")
            try:
                download(d["url"], target)
                archive = target
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"[{name}] automatic download failed: {e}")
                print("  The link may have moved. Download it from the page below instead.")
                print(f"  page : {d.get('page', d['url'])}")
                print(f"  then : save as data/downloads/{name}.zip and run 'fetch' again.")
                return

    if archive is None:
        print(f"[{name}] not found")
        print_manual(name)
        return

    print(f"[{name}] extracting {archive.name} -> data/raw/{name}/")
    try:
        safe_extract(archive, dest_of(name))
    except (zipfile.BadZipFile, tarfile.TarError, RuntimeError) as e:
        print(f"[{name}] extraction failed: {e} (file may be incomplete; delete it and retry)")


def selected(names: list[str]) -> list[str]:
    bad = [n for n in names if n not in DATASETS]
    if bad:
        sys.exit(f"unknown dataset(s): {', '.join(bad)}. Try: python scripts/download_data.py list")
    return names or list(DATASETS)


def cmd_verify(args: argparse.Namespace) -> int:
    all_ok = True
    for name in selected(args.names):
        ok, missing = is_present(name)
        all_ok &= ok
        print(f"[{'OK' if ok else 'MISSING':7s}] {name:11s} {DATASETS[name]['title']}"
              + ("" if ok else f"   (not found: {', '.join(missing)})"))
    return 0 if all_ok else 1


def cmd_fetch(args: argparse.Namespace) -> int:
    for name in selected(args.names):
        fetch_one(name)
        print()
    print("---- verification ----")
    return cmd_verify(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    for c, fn in (("fetch", cmd_fetch), ("verify", cmd_verify)):
        sp = sub.add_parser(c)
        sp.add_argument("names", nargs="*", help="dataset keys (default: all)")
        sp.set_defaults(fn=fn)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
