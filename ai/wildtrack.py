"""WILDTRACK: 7 synchronized, overlapping cameras, 400 annotated frames at 2 fps, ground-truth person IDs.

Expected folder (as in the official download):
    Wildtrack_dataset/Image_subsets/C1 ... C7/<frame>.png      (frame = 00000000, 00000005, ...)
    Wildtrack_dataset/annotations_positions/<frame>.json       one file per frame, a list of people:
        {"personID": 12, "positionID": 3456, "views": [{"viewNum": 0, "xmin": .., "ymin": .., "xmax": .., "ymax": ..}, ...]}
        (a view where the person is not visible has xmin = -1)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .schema import TrackRecord


def find_root(path: str | Path) -> Path:
    p = Path(path)
    if (p / "annotations_positions").is_dir() and (p / "Image_subsets").is_dir():
        return p
    hit = next(p.rglob("annotations_positions"), None) if p.exists() else None
    if hit is not None and (hit.parent / "Image_subsets").is_dir():
        return hit.parent
    raise FileNotFoundError(f"WILDTRACK not found under {p}: need annotations_positions/ and Image_subsets/ folders.")


def load_wildtrack(root, cams: list[str] | None = None, max_frames: int | None = None):
    """Returns (images, gt): images {camera: [image paths in time order]}, gt {camera: [[frame_idx, pid, x1, y1, x2, y2]]}.
    frame_idx is the position in the list, identical in every camera (the cameras are synchronized)."""
    root = find_root(root)
    cam_dirs = {d.name: d for d in (root / "Image_subsets").iterdir() if re.fullmatch(r"C\d+", d.name)}
    if not cam_dirs:
        raise FileNotFoundError(f"no C1..C7 folders in {root / 'Image_subsets'}")
    chosen = sorted(cams or cam_dirs, key=lambda c: int(c[1:]))
    missing = [c for c in chosen if c not in cam_dirs]
    if missing:
        raise ValueError(f"unknown camera(s) {missing}; available: {sorted(cam_dirs, key=lambda c: int(c[1:]))}")

    imgs = {c: {f.stem: f for f in cam_dirs[c].iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")} for c in chosen}
    ann = {f.stem: f for f in (root / "annotations_positions").glob("*.json")}
    stems = sorted(set(ann).intersection(*[set(v) for v in imgs.values()]))
    if not stems:
        raise ValueError("no frame has both an annotation file and an image in every chosen camera "
                         f"(annotation names like {sorted(ann)[:2]}, image names like {sorted(next(iter(imgs.values())))[:2]})")
    if max_frames:
        stems = stems[:max_frames]

    data = [json.loads(ann[s].read_text()) for s in stems]
    nums = sorted({int(v["viewNum"]) for frame in data[:20] for ped in frame for v in ped.get("views", []) if "viewNum" in v})
    base = nums[0] if nums else 0                          # viewNum may start at 0 or at 1
    images = {c: [str(imgs[c][s]) for s in stems] for c in chosen}
    gt = {c: [] for c in chosen}
    for idx, frame in enumerate(data):
        for ped in frame:
            pid = int(ped["personID"])
            for v in ped.get("views", []):
                cam = f"C{int(v['viewNum']) - base + 1}"
                if cam not in gt:
                    continue
                x1, y1, x2, y2 = (float(v[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                if min(x1, y1, x2, y2) < 0 or x2 <= x1 or y2 <= y1:      # not visible in this camera
                    continue
                gt[cam].append([idx, pid, x1, y1, x2, y2])
    return images, gt


class OracleTracker:
    """A perfect detector + tracker: returns the ground-truth boxes (track id = person id).
    Running the pipeline with it isolates Re-ID + cross-camera matching from detection quality."""

    def __init__(self, camera_id: str, gt_rows):
        self.camera_id, self.by_frame = camera_id, defaultdict(list)
        for f, pid, x1, y1, x2, y2 in gt_rows:
            self.by_frame[f].append((int(pid), (float(x1), float(y1), float(x2), float(y2))))

    def update(self, image, ts, idx):
        return [TrackRecord(self.camera_id, ts, idx, pid, bb, "person", 1.0) for pid, bb in self.by_frame.get(idx, [])]


def export_reid_dataset(images: dict, gt: dict, out_dir, test_frames: int = 50, train_stride: int = 2,
                        min_h: int = 48, min_w: int = 20) -> dict:
    """Cut ground-truth person boxes out of WILDTRACK into a Market-1501-style folder, so the Re-ID tools
    (ai.reid.eval_market, ai.reid.train) work on it unchanged.

    Person-disjoint split, so scores are honest:
      test people  = everyone who appears in frames < test_frames (the same frames the matching benchmark uses)
                     -> query: one crop per (person, camera); gallery: all their other crops in those frames
      train people = everyone else, from frames >= test_frames (every `train_stride`-th frame)
    images/gt must cover ALL frames: load with load_wildtrack(root) and no max_frames.
    """
    import cv2
    out = Path(out_dir)
    test_pids = {int(r[1]) for rows in gt.values() for r in rows if r[0] < test_frames}
    plan: dict = defaultdict(list)                      # (cam, frame) -> [(folder, pid, box)]
    per_key: dict = defaultdict(list)
    for cam, rows in gt.items():
        for f, pid, x1, y1, x2, y2 in rows:
            if y2 - y1 < min_h or x2 - x1 < min_w:
                continue
            if int(pid) in test_pids:
                if f < test_frames:
                    per_key[(cam, int(pid))].append((f, (x1, y1, x2, y2)))
            elif f >= test_frames and f % train_stride == 0:
                plan[(cam, f)].append(("bounding_box_train", int(pid), (x1, y1, x2, y2)))
    for (cam, pid), items in per_key.items():
        items.sort()
        qf = items[len(items) // 2][0]                 # the middle appearance becomes the query
        for f, box in items:
            plan[(cam, f)].append(("query" if f == qf else "bounding_box_test", pid, box))
    counts = {"bounding_box_train": 0, "query": 0, "bounding_box_test": 0}
    if not any(v[0] == "query" for items in plan.values() for v in items):
        raise ValueError("no usable test crops (are the boxes all smaller than the minimum size?)")
    if not any(v[0] == "bounding_box_train" for items in plan.values() for v in items):
        raise ValueError("no training people: everyone appears in the first test_frames frames. "
                         "Use the full 400 frames (do not pass --max-frames) or a smaller --test-frames.")
    for d in counts:
        (out / d).mkdir(parents=True, exist_ok=True)
    for (cam, f), items in sorted(plan.items()):
        img = cv2.imread(images[cam][f])
        if img is None:
            raise IOError(f"cannot read {images[cam][f]}")
        h, w = img.shape[:2]
        for k, (folder, pid, (x1, y1, x2, y2)) in enumerate(items):
            crop = img[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
            if crop.size == 0:
                continue
            cv2.imwrite(str(out / folder / f"{pid:04d}_c{int(cam[1:])}s1_{f:06d}_{k:02d}.jpg"), crop)
            counts[folder] += 1
    info = {**counts, "test_people": len(test_pids), "train_people": len({v[1] for items in plan.values()
                                                                         for v in items if v[0] == "bounding_box_train"}),
            "test_frames": test_frames, "train_stride": train_stride}
    (out / "split.json").write_text(json.dumps({**info, "test_pids": sorted(test_pids)}, indent=2))
    return info
