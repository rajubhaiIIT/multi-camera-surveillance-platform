#!/usr/bin/env python3
"""Everything for the own-recording experiment.

  1. python -m ai.recording sync    cam1=a.mp4 cam2=b.mp4 cam3=c.mp4 --out runs/rec01/offsets.json
  2. python -m ai.recording extract --video cam1=a.mp4 --video cam2=b.mp4 --offsets runs/rec01/offsets.json --out runs/rec01
  3. (look at runs/rec01/sheets/*.jpg and fill the `person` column of runs/rec01/labels.csv)
  4. python -m ai.recording score   runs/rec01 --render
  5. python -m ai.recording sweep   runs/rec01

Public footage instead of your own (WILDTRACK, ground truth included, no labeling):
  python -m ai.recording wildtrack --root data/raw/wildtrack --out runs/wt --max-frames 50
  python -m ai.recording score runs/wt --stale-after 6 --render
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .crosscam import MatcherConfig, Topology, load_config
from .labeling import make_contact_sheets, read_labels, score_run, score_run_gt, write_label_template
from .replay import render_run, replay
from .runcache import load_run


def _cfg(a, run_dir: Path | None = None) -> tuple[MatcherConfig, Topology]:
    auto = Path(run_dir) / "topology.json" if run_dir else None
    cfgfile = a.config or (str(auto) if auto and auto.exists() else None)       # runs can ship their own layout
    cfg, topo = load_config(cfgfile) if cfgfile else (MatcherConfig(), Topology())
    if a.threshold is not None:
        cfg = dataclasses.replace(cfg, sim_threshold=a.threshold)
    return cfg, topo


def _score(run, records) -> dict:
    """Ground-truth boxes (gt.json, e.g. WILDTRACK) if the run has them, otherwise your labels.csv."""
    gt = run.dir / "gt.json"
    if gt.exists():
        return score_run_gt(run, records, json.loads(gt.read_text()))
    return score_run(run, records, read_labels(run.dir / "labels.csv"))


def _fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def _summary(res: dict) -> dict:
    m, cross = res["mtmc"], res["links"]["cross_camera"]
    return {"IDF1": m["IDF1"], "HOTA": m["HOTA"], "AssA": m["AssA"], "IDSW": m["IDSW"],
            "link_P": cross["precision"], "link_R": cross["recall"], "link_F1": cross["f1"],
            "identities": res["predicted_identities"], "people": res["true_people"],
            "unmatched": res["unmatched_tracklets"]}


def cmd_extract(a):
    from .multicam import FrameEmbedder
    from .multicam_extract import extract_run
    from .reid.embedder import ReIDEmbedder
    from .yolo_tracker import Tracker

    videos = dict(kv.split("=", 1) for kv in a.video)
    offsets = json.loads(Path(a.offsets).read_text()) if a.offsets else {}
    trackers = {c: Tracker(c, a.weights, classes=(0,), conf=a.conf, imgsz=a.imgsz, device=a.device) for c in videos}
    emb = ReIDEmbedder(a.ckpt, device=a.device)
    print(f"device={emb.device}, cameras={list(videos)}, stride={a.stride}")
    fe = FrameEmbedder(trackers, emb, embed_every=a.embed_every)
    out = extract_run(videos, offsets, fe, a.out, stride=a.stride, max_seconds=a.max_seconds,
                      meta_extra={"reid_ckpt": a.ckpt, "yolo": a.weights, "imgsz": a.imgsz})
    run = load_run(out)
    sheets = make_contact_sheets(run, a.min_frames)
    print(f"\nlabels template: {write_label_template(run, a.min_frames)}\n{len(sheets)} contact sheets in {out / 'sheets'}")
    print("Next: open the sheets, write a person name for each track in labels.csv (blank = ignore), then run 'score'.")
    return 0


def cmd_wildtrack(a):
    from .multicam import FrameEmbedder
    from .multicam_extract import extract_run
    from .reid.embedder import ReIDEmbedder
    from .wildtrack import OracleTracker, load_wildtrack
    from .yolo_tracker import Tracker

    images, gt = load_wildtrack(a.root, a.cams, a.max_frames)
    n = len(next(iter(images.values())))
    print(f"WILDTRACK: {len(images)} cameras x {n} frames, {sum(len(v) for v in gt.values())} ground-truth boxes")
    if a.oracle:
        trackers = {c: OracleTracker(c, gt[c]) for c in images}
        print("ORACLE mode: ground-truth boxes and per-camera ids replace YOLO (tests Re-ID + matching only)")
    else:
        trackers = {c: Tracker(c, a.weights, classes=(0,), conf=a.conf, imgsz=a.imgsz, device=a.device) for c in images}
    emb = ReIDEmbedder(a.ckpt, device=a.device)
    print(f"device={emb.device}")
    fe = FrameEmbedder(trackers, emb, embed_every=a.embed_every)
    out = extract_run(images, {}, fe, a.out, stride=1, folder_fps=2.0,
                      meta_extra={"dataset": "WILDTRACK", "reid_ckpt": a.ckpt, "imgsz": a.imgsz,
                                  "yolo": "oracle (ground-truth boxes)" if a.oracle else a.weights})
    (out / "gt.json").write_text(json.dumps(gt))
    (out / "topology.json").write_text(json.dumps({"all_overlapping": True, "matcher": {"max_gap_s": 60.0}}, indent=2))
    print(f"\nsaved to {out}. Next:\n  python -m ai.recording score {out} --stale-after 6 --render\n"
          f"  python -m ai.recording sweep {out} --stale-after 6")
    return 0


def cmd_detect_eval(a):
    from pathlib import Path as P

    from .detect_eval import YoloPersonDetector, evaluate_detector, format_comparison
    from .wildtrack import load_wildtrack
    images, gt = load_wildtrack(a.root, a.cams, a.max_frames)
    n = len(next(iter(images.values())))
    print(f"WILDTRACK: {len(images)} cameras x {n} frames, {sum(len(v) for v in gt.values())} ground-truth boxes")
    out = P(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for w in a.weights:
        print(f"\ndetecting with {w} at imgsz {a.imgsz} ...")
        det = YoloPersonDetector(w, a.imgsz, a.device)
        res = evaluate_detector(images, gt, det, a.conf)
        res.update(weights=w, imgsz=a.imgsz, device=det.device, frames_per_camera=n)
        name = P(w).stem
        (out / f"detect_{name}_{a.imgsz}.json").write_text(json.dumps(res, indent=2))
        results[f"{name}@{a.imgsz}"] = res
    print("\n" + format_comparison(results))
    print(f"\nrecall/precision are at confidence {a.conf}; 'AP50 (area)' ignores unmatched boxes outside the "
          f"ESTIMATED annotated area. Details per camera: {out}/detect_*.json")
    return 0


def cmd_export_reid(a):
    from .wildtrack import export_reid_dataset, load_wildtrack
    images, gt = load_wildtrack(a.root)                     # all frames: training people come from later frames
    info = export_reid_dataset(images, gt, a.out, a.test_frames, a.train_stride)
    print(json.dumps(info, indent=2))
    print(f"\nWrote a Market-style dataset to {a.out}. Measure the domain gap of your Market-trained model:\n"
          f"  python -m ai.reid.eval_market --ckpt models/reid/resnet50_market/best.pt --data {a.out}")
    return 0


def cmd_diagnose(a):
    from .diagnose import detection_report, format_report
    run = load_run(a.run)
    gt = run.dir / "gt.json"
    if not gt.exists():
        raise FileNotFoundError(f"{gt} not found: diagnose needs a ground-truth run (the 'wildtrack' command)")
    rep = detection_report(run, json.loads(gt.read_text()), margin=a.margin)
    (run.dir / "diagnose.json").write_text(json.dumps(rep, indent=2))
    print(format_report(rep))
    t = rep["total"]
    na = lambda v: "n/a" if v is None else f"{v}%"
    print(f"\nDetection alone: found {na(t['recall'])} of ground-truth people. Precision {na(t['precision_all'])} overall, "
          f"{na(t['precision_inside_area'])} counting only extra boxes inside the (estimated) annotated area; "
          f"{na(t['fp_outside_share'])} of the classifiable extra boxes are outside it.")
    return 0


def cmd_sheets(a):
    run = load_run(a.run)
    make_contact_sheets(run, a.min_frames)
    print("sheets:", write_label_template(run, a.min_frames, force=a.force).parent / "sheets")
    return 0


def cmd_face_run(a):
    """Run face recognition on every camera video in an existing extraction run.

    Loads the enrolled FaceIndex from --index, detects + identifies faces frame by frame,
    and writes:
      <run>/face_events.jsonl   — one JSON line per identification event
      <run>/face_summary.json  — per-person identity counts + confidence stats
    """
    import ai.face.detector
    from .face.index import FaceIndex
    from .face.pipeline import FacePipeline

    run = load_run(a.run)

    # Load index
    idx_path = Path(a.index)
    if not idx_path.exists():
        print(f"error: face index not found at {idx_path}. "
              f"Enroll people first:\n  python -m ai.face.enroll enroll --id <id> --name <name> --photos <folder>")
        return 1
    idx = FaceIndex.load(idx_path)
    idx.threshold = a.threshold
    if idx.size == 0:
        print("error: face index is empty. Enroll people first.")
        return 1
    print(f"loaded face index: {idx.size} enrolled people  (threshold={a.threshold})")

    det = ai.face.detector.FaceDetector(det_size=(a.det_size, a.det_size), min_det_score=a.min_det_score,
                                       min_face_height=a.min_face_height,
                                       model_root=a.model_root)

    pipe = FacePipeline(det, idx, run_every=a.run_every, min_hits=a.min_hits)

    events = []
    # Discover video files from run meta: can be meta["videos"] or meta["cameras"][cam]["video"]
    meta_path = run.dir / "meta.json"
    if not meta_path.exists():
        print(f"error: {meta_path} not found. Run 'extract' first.")
        return 1
    meta = json.loads(meta_path.read_text())
    videos = meta.get("videos", {})
    if not videos and "cameras" in meta:
        videos = {cam: cdata["video"] for cam, cdata in meta["cameras"].items() if "video" in cdata}
    if not videos:
        print("error: no video paths in meta.json")
        return 1

    import cv2
    from .sources import MediaReader

    cfg, topo = _cfg(a, run.dir)
    # Replay cached embeddings to get global-id-assigned records
    recs_by_frame: dict = {}
    all_recs = replay(run, cfg, topo, stale_after_s=a.stale_after)
    for r in all_recs:
        recs_by_frame.setdefault((r.camera_id, r.frame_idx), []).append(r)

    render_writers = {}
    render_dir = None
    if getattr(a, "render", False):
        render_dir = run.dir / "face_render"
        render_dir.mkdir(parents=True, exist_ok=True)

    for cam_id, vid_path in videos.items():
        cam_info = meta.get("cameras", {}).get(cam_id, {})
        stride = int(cam_info.get("stride", meta.get("stride", 1)))
        folder_fps = float(cam_info.get("fps", 25.0))
        last_idx = cam_info.get("last_idx", float("inf"))

        try:
            reader = MediaReader(vid_path, stride=stride, folder_fps=folder_fps)
        except Exception as e:
            print(f"  warning: cannot open media for camera {cam_id} ({vid_path}): {e}")
            continue

        pipe._frame_count.clear()
        pipe._hits.clear()
        print(f"  processing camera {cam_id} ({vid_path}) stride={stride} ...")
        
        writer = None
        for frame_idx, frame in reader.frames():
            if frame_idx > last_idx:
                break
            recs = recs_by_frame.get((cam_id, frame_idx), [])
            hits = pipe.process(frame, recs)
            
            # Map hits by track_id for visual overlay
            hit_map = {h.track_id: h for h in hits}
            
            for h in hits:
                gid = next((r.global_id for r in recs if r.track_id == h.track_id), None)
                ev = {"camera_id": h.camera_id, "frame_idx": h.frame_idx,
                      "track_id": h.track_id, "global_id": gid,
                      "person_id": h.person_id,
                      "name": h.name, "similarity": round(h.similarity, 4),
                      "det_score": round(h.det_score, 4),
                      "face_bbox": [round(v, 1) for v in h.face_bbox]}
                events.append(ev)

            if render_dir is not None:
                vis = frame.copy()
                for r in recs:
                    x1, y1, x2, y2 = (int(v) for v in r.bbox)
                    label = f"P{r.global_id}" if r.global_id is not None else f"#{r.track_id}"
                    color = (0, 255, 0)
                    if r.track_id in hit_map:
                        f_hit = hit_map[r.track_id]
                        label += f" | {f_hit.name} ({f_hit.similarity:.2f})"
                        color = (0, 215, 255) # golden for face matched
                        fx1, fy1, fx2, fy2 = [int(v) for v in f_hit.face_bbox]
                        cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(vis, f"{cam_id}  f={frame_idx}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                if writer is None:
                    h, w = vis.shape[:2]
                    out_path = render_dir / f"{cam_id}.mp4"
                    fps_out = max(reader.fps / stride, 1.0)
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
                writer.write(vis)

        if writer:
            writer.release()

    # Write events
    events_path = run.dir / "face_events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events))
    print(f"\nwrote {len(events)} identification events -> {events_path}")

    # Build summary
    from collections import Counter, defaultdict
    per_person: dict = defaultdict(lambda: {"count": 0, "sims": [], "cameras": set(), "tracks": set()})
    for ev in events:
        p = per_person[ev["person_id"]]
        p["count"] += 1
        p["sims"].append(ev["similarity"])
        p["cameras"].add(ev["camera_id"])
        p["tracks"].add((ev["camera_id"], ev["track_id"]))
    summary = {}
    for pid, p in per_person.items():
        name = idx._persons[pid].name if pid in idx._persons else pid
        summary[pid] = {
            "name": name,
            "hits": p["count"],
            "mean_sim": round(float(sum(p["sims"]) / len(p["sims"])), 4),
            "cameras_seen": sorted(p["cameras"]),
            "unique_tracks": len(p["tracks"]),
        }
    summary_path = run.dir / "face_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


    print("\nFace identification summary:")
    print(f"{'name':20s}  {'hits':>5}  {'mean_sim':>8}  cameras")
    for pid, s in sorted(summary.items(), key=lambda x: -x[1]["hits"]):
        print(f"{s['name']:20s}  {s['hits']:5}  {s['mean_sim']:8.3f}  {', '.join(s['cameras_seen'])}")
    print(f"\nwrote summary -> {summary_path}")
    return 0



def cmd_score(a):
    run = load_run(a.run)
    cfg, topo = _cfg(a, run.dir)
    recs = replay(run, cfg, topo, no_reid=a.no_reid, stale_after_s=a.stale_after)
    res = _score(run, recs)
    (run.dir / "score.json").write_text(json.dumps(res, indent=2))
    s = _summary(res)
    print(json.dumps(res, indent=2))
    print(f"\nIDF1 {s['IDF1']:.1f} | HOTA {s['HOTA']:.1f} | ID switches {s['IDSW']} | "
          f"identities {s['identities']} for {s['people']} real people | cross-camera link F1 {_fmt(s['link_F1'])}")
    if a.render:
        print("rendered:", [str(p) for p in render_run(run, recs)])
    return 0


def cmd_sweep(a):
    run = load_run(a.run)
    base_cfg, topo = _cfg(a, run.dir)
    rows = []

    def one(name, cfg, no_reid=False, topology=topo):
        try:
            s = _summary(_score(run, replay(run, cfg, topology, no_reid=no_reid, stale_after_s=a.stale_after)))
        except ValueError as e:
            s = {"error": str(e)}
        rows.append({"setting": name, "threshold": cfg.sim_threshold, **s})

    for t in a.thresholds:
        one(f"full matcher, threshold {t}", dataclasses.replace(base_cfg, sim_threshold=t))
    one("appearance only (no timing/layout rules)", dataclasses.replace(base_cfg, use_constraints=False))
    one("rules only (no Re-ID appearance)", base_cfg, no_reid=True)
    (run.dir / "sweep.json").write_text(json.dumps(rows, indent=2))

    print("| setting | IDF1 | HOTA | ID sw. | identities / people | link F1 | unmatched |\n|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['setting']} | {r['error']} |")
        else:
            print(f"| {r['setting']} | {r['IDF1']:.1f} | {r['HOTA']:.1f} | {r['IDSW']} | "
                  f"{r['identities']} / {r['people']} | {_fmt(r['link_F1'])} | {r['unmatched']} |")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="clap-based time offsets")
    s.add_argument("videos", nargs="+")
    s.add_argument("--out", default="offsets.json")
    s.set_defaults(fn=lambda a: __import__("ai.sync", fromlist=["main"]).main(a.videos + ["--out", a.out]))

    e = sub.add_parser("extract", help="run detection + tracking + Re-ID on the videos (GPU)")
    e.add_argument("--video", action="append", required=True, help="cam=path (repeat per camera)")
    e.add_argument("--offsets", help="json from the sync step")
    e.add_argument("--out", required=True)
    e.add_argument("--ckpt", default="models/reid/resnet50_market/best.pt")
    e.add_argument("--weights", default="yolo11n.pt")
    e.add_argument("--imgsz", type=int, default=960)
    e.add_argument("--conf", type=float, default=0.1)
    e.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    e.add_argument("--embed-every", type=int, default=3)
    e.add_argument("--max-seconds", type=float, help="debug: only the first N seconds of each video")
    e.add_argument("--min-frames", type=int, default=15, help="tracks shorter than this are not offered for labeling")
    e.add_argument("--device", default="auto")
    e.set_defaults(fn=cmd_extract)

    w = sub.add_parser("wildtrack", help="run on the public WILDTRACK dataset (ground truth included, no labeling)")
    w.add_argument("--root", default="data/raw/wildtrack")
    w.add_argument("--out", required=True)
    w.add_argument("--cams", nargs="+", help="e.g. C1 C4 C7 (default: all 7)")
    w.add_argument("--max-frames", type=int, help="quick trial: only the first N frames per camera")
    w.add_argument("--ckpt", default="models/reid/resnet50_market/best.pt")
    w.add_argument("--weights", default="yolo11n.pt")
    w.add_argument("--imgsz", type=int, default=1280)
    w.add_argument("--conf", type=float, default=0.1)
    w.add_argument("--embed-every", type=int, default=2)
    w.add_argument("--device", default="auto")
    w.add_argument("--oracle", action="store_true",
                   help="use ground-truth boxes instead of a detector: isolates Re-ID + matching from detection")
    w.set_defaults(fn=cmd_wildtrack)

    de = sub.add_parser("detect-eval", help="detector only (no tracker, no Re-ID): AP50, AP50-95, recall by person size")
    de.add_argument("--root", default="data/raw/wildtrack")
    de.add_argument("--weights", nargs="+", required=True, help="one or more models, compared side by side")
    de.add_argument("--imgsz", type=int, default=1280)
    de.add_argument("--conf", type=float, default=0.25, help="confidence used for the recall / precision columns")
    de.add_argument("--cams", nargs="+")
    de.add_argument("--max-frames", type=int, help="quick trial: only the first N frames per camera")
    de.add_argument("--out", default="runs/detect_eval")
    de.add_argument("--device", default="auto")
    de.set_defaults(fn=cmd_detect_eval)

    ex = sub.add_parser("export-reid", help="WILDTRACK ground-truth crops as a Market-style Re-ID dataset (person-disjoint split)")
    ex.add_argument("--root", default="data/raw/wildtrack")
    ex.add_argument("--out", default="data/wt_reid")
    ex.add_argument("--test-frames", type=int, default=50, help="people seen in the first N frames are TEST people")
    ex.add_argument("--train-stride", type=int, default=2, help="use every Nth frame for training crops")
    ex.set_defaults(fn=cmd_export_reid)

    dg = sub.add_parser("diagnose", help="detection-only report for a WILDTRACK run (recall, precision, FPs outside the annotated area)")
    dg.add_argument("run")
    dg.add_argument("--margin", type=float, default=0.2, help="enlarge the estimated annotated area by this fraction")
    dg.set_defaults(fn=cmd_diagnose)

    sh = sub.add_parser("sheets", help="(re)make the contact sheets and labels.csv template")
    sh.add_argument("run")
    sh.add_argument("--min-frames", type=int, default=15)
    sh.add_argument("--force", action="store_true", help="overwrite labels.csv (DELETES your labels)")
    sh.set_defaults(fn=cmd_sheets)

    fr = sub.add_parser("face-run",
                        help="run face recognition on a completed extraction run; identifies enrolled people in the videos")
    fr.add_argument("run", help="path to an existing run directory (must have meta.json from 'extract')")
    fr.add_argument("--index", default="models/face_index.json",
                    help="path to the enrolled FaceIndex JSON (built with: python -m ai.face.enroll enroll ...)")
    fr.add_argument("--threshold", type=float, default=0.4,
                    help="cosine similarity threshold for a positive identification")
    fr.add_argument("--model-root", default="models/insightface")
    fr.add_argument("--det-size", type=int, default=320, help="SCRFD input size")
    fr.add_argument("--min-det-score", type=float, default=0.4)
    fr.add_argument("--min-face-height", type=float, default=40.0)
    fr.add_argument("--run-every", type=int, default=5,
                    help="run face detection every N frames per track (reduces GPU load)")
    fr.add_argument("--min-hits", type=int, default=2,
                    help="require this many matching frames before reporting an identification")
    fr.add_argument("--stale-after", type=float, default=3.0)
    fr.add_argument("--config", help="json: matcher settings + camera layout")
    fr.add_argument("--render", action="store_true", help="write annotated videos with face IDs to <run>/face_render")
    fr.set_defaults(fn=cmd_face_run, threshold=0.4)


    for name, fn in (("score", cmd_score), ("sweep", cmd_sweep)):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("--threshold", type=float)
        p.add_argument("--config", help="json: matcher settings + camera layout (default: <run>/topology.json if present)")
        p.add_argument("--stale-after", type=float, default=3.0,
                       help="seconds without seeing a track before it counts as ended (use 6 for WILDTRACK's 2 fps)")
        if name == "score":
            p.add_argument("--no-reid", action="store_true")
            p.add_argument("--render", action="store_true", help="write annotated videos to <run>/render")
        else:
            p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
        p.set_defaults(fn=fn)


    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, FileNotFoundError, IOError) as e:        # clear message instead of a stack trace
        print(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
