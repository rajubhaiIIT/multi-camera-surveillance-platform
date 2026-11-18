"""Label WHO each tracked person is (instead of drawing boxes), then score cross-camera matching.

Boxes come from the tracker; you only say which real person each track is. So the score measures
identity matching (the Re-ID problem), not detection quality.
"""
from __future__ import annotations

import csv
import dataclasses
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .mtmc_eval import evaluate_mtmc, tracklet_link_scores
from .runcache import Run


def tracklet_table(run: Run, person_class: str = "person") -> dict:
    """{(camera, track_id): {"frames", "first_ts", "last_ts"}} for every person track in the run."""
    t: dict = {}
    for ev in run.events:
        for r in ev.recs:
            if r.cls != person_class:
                continue
            e = t.setdefault((ev.cam, r.track_id), {"frames": 0, "first_ts": ev.ts, "last_ts": ev.ts})
            e["frames"] += 1
            e["last_ts"] = max(e["last_ts"], ev.ts)
            e["first_ts"] = min(e["first_ts"], ev.ts)
    return t


def write_label_template(run: Run, min_frames: int = 15, force: bool = False) -> Path:
    path = run.dir / "labels.csv"
    if path.exists() and not force:
        return path                                          # never overwrite someone's hard work
    rows = sorted(((c, tid, v) for (c, tid), v in tracklet_table(run).items() if v["frames"] >= min_frames),
                  key=lambda x: (x[0], x[2]["first_ts"]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "track_id", "frames", "first_ts", "last_ts", "person"])
        for c, tid, v in rows:
            w.writerow([c, tid, v["frames"], round(v["first_ts"], 2), round(v["last_ts"], 2), ""])
    return path


def read_labels(path: Path) -> dict:
    """{(camera, track_id): person}. Blank or 'ignore' rows are left out (strangers, false detections)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            p = (row.get("person") or "").strip()
            if p and p.lower() not in ("ignore", "other", "x"):
                out[(row["camera"], int(row["track_id"]))] = p
    return out


def make_contact_sheets(run: Run, min_frames: int = 15, per_sheet: int = 8, crop_h: int = 170) -> list[Path]:
    """One picture per ~8 tracks: a caption plus up to 4 crops each, so you can read off who is who."""
    table = tracklet_table(run)
    out_dir = run.dir / "sheets"
    out_dir.mkdir(exist_ok=True)
    by_cam = defaultdict(list)
    for (cam, tid), v in table.items():
        if v["frames"] >= min_frames:
            by_cam[cam].append((tid, v))
    paths = []
    for cam, tracks in by_cam.items():
        tracks.sort(key=lambda x: x[1]["first_ts"])
        for s in range(0, len(tracks), per_sheet):
            rows = []
            for tid, v in tracks[s:s + per_sheet]:
                cap = np.full((crop_h, 260, 3), 245, np.uint8)
                for i, txt in enumerate((f"{cam}  track {tid}", f"{v['frames']} frames",
                                         f"t = {v['first_ts']:.1f} - {v['last_ts']:.1f} s")):
                    cv2.putText(cap, txt, (10, 40 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                tiles = [cap]
                for k in (1, 4, 8, 12):
                    f = run.dir / "crops" / cam / f"{tid}_{k}.jpg"
                    img = cv2.imread(str(f)) if f.exists() else None
                    if img is not None:
                        tiles.append(cv2.resize(img, (max(20, int(img.shape[1] * crop_h / img.shape[0])), crop_h)))
                rows.append(np.hstack(tiles))
            width = max(r.shape[1] for r in rows)
            sheet = np.vstack([np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0)), constant_values=200) for r in rows])
            p = out_dir / f"sheet_{cam}_{s // per_sheet + 1:02d}.jpg"
            cv2.imwrite(str(p), sheet)
            paths.append(p)
    return paths


def score_run(run: Run, records: list, labels: dict) -> dict:
    """records: person records with global_id back-filled (see ai.replay.replay). Returns metrics."""
    people = {p: i + 1 for i, p in enumerate(sorted(set(labels.values())))}
    cams: dict = {}
    for name, info in run.meta["cameras"].items():
        cams[name] = {"gt": [], "pred": [], "n_frames": int(info["last_idx"]) + 1}
    gid_of: dict = {}
    for r in records:
        key = (r.camera_id, r.track_id)
        if key not in labels:
            continue
        gid_of.setdefault(key, r.global_id)
        row = (r.frame_idx + 1, people[labels[key]], *r.bbox)
        cams[r.camera_id]["gt"].append(row)
        if r.global_id is not None:
            cams[r.camera_id]["pred"].append((r.frame_idx + 1, r.global_id, *r.bbox, r.conf))
    if not any(c["gt"] for c in cams.values()):
        raise ValueError("no labeled tracks found. Fill the 'person' column of labels.csv first.")
    if not any(c["pred"] for c in cams.values()):
        raise ValueError("the matcher assigned no global IDs to any labeled track (are the tracks too short?)")
    m = evaluate_mtmc(run.dir / "trackeval_mtmc", cams, print_results=False)["combined"]
    items = [(c, labels[(c, t)], g) for (c, t), g in gid_of.items()]
    return {"mtmc": m, "links": tracklet_link_scores(items),
            "true_people": len(people), "predicted_identities": len({g for g in gid_of.values() if g is not None}),
            "labeled_tracklets": len(gid_of), "unmatched_tracklets": sum(g is None for g in gid_of.values())}


def _dedup_overlapping(cams: dict) -> dict:
    """When cameras overlap, the same (global_id, frame) can appear multiple times in the
    merged timeline (once per camera). Keep only the highest-confidence prediction per
    (global_id, offset_frame) and one gt box per (person_id, offset_frame)."""
    import copy
    out = copy.deepcopy(cams)
    # deduplicate predictions: keep best-confidence box per (gid, frame)
    for c in out.values():
        best: dict = {}
        for row in c["pred"]:
            f, gid, x1, y1, x2, y2, conf = row
            if (gid, f) not in best or conf > best[(gid, f)][-1]:
                best[(gid, f)] = row
        c["pred"] = list(best.values())
    # deduplicate gt: keep first box per (pid, frame) within each camera
    for c in out.values():
        seen: set = set()
        deduped = []
        for row in c["gt"]:
            f, pid = row[0], row[1]
            if (pid, f) not in seen:
                deduped.append(row)
                seen.add((pid, f))
        c["gt"] = deduped
    return out


def score_run_gt(run: Run, records: list, gt: dict) -> dict:
    """Score against ground-truth boxes with global person IDs (e.g. WILDTRACK).
    gt = {camera: [[frame_idx, person_id, x1, y1, x2, y2], ...]}. Unlike labeled runs, detection quality counts too."""
    cams = {n: {"gt": [(f + 1, pid, x1, y1, x2, y2) for f, pid, x1, y1, x2, y2 in gt.get(n, [])], "pred": [],
                "n_frames": int(i["last_idx"]) + 1} for n, i in run.meta["cameras"].items()}
    tracks, unmatched = {}, set()
    for r in records:
        if r.cls != "person":
            continue
        tracks[(r.camera_id, r.track_id)] = r.global_id
        if r.global_id is None:
            unmatched.add((r.camera_id, r.track_id))
        else:
            cams[r.camera_id]["pred"].append((r.frame_idx + 1, r.global_id, *r.bbox, r.conf))
    if not any(c["gt"] for c in cams.values()):
        raise ValueError("ground truth is empty")
    if not any(c["pred"] for c in cams.values()):
        raise ValueError("the matcher assigned no global IDs to any track")
    # Overlapping cameras: deduplicate so each (global_id, timeline_frame) appears at most once.
    cams = _dedup_overlapping(cams)
    m = evaluate_mtmc(run.dir / "trackeval_mtmc", cams, print_results=False)["combined"]
    none = {"precision": None, "recall": None, "f1": None, "pairs": 0}
    return {"mtmc": m, "links": {"cross_camera": none, "same_camera": none},
            "true_people": len({row[1] for c in cams.values() for row in c["gt"]}),
            "predicted_identities": len({g for g in tracks.values() if g is not None}),
            "labeled_tracklets": len(tracks), "unmatched_tracklets": len(unmatched)}
