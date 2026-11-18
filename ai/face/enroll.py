#!/usr/bin/env python3
"""Enrollment CLI: register people from photo folders, evaluate the index, and run LFW.

Commands
--------
enroll      add one person from a folder of photos (or a single photo)
remove      remove a person from the index
list        show who is enrolled
search      find a person by uploading a photo
demo        run face detection + matching on a video or RTSP stream (no GUI needed: save to video)
bias        measure per-group accuracy gap on a folder of labeled photos

Usage
-----
python -m ai.face.enroll enroll --id aryan --name "Aryan" --photos data/faces/aryan/
python -m ai.face.enroll list
python -m ai.face.enroll search --photo query.jpg
python -m ai.face.enroll demo --source rtsp://localhost:8554/cam1 --save-video out.mp4
python -m ai.face.enroll bias --data data/faces/bias_set --out eval/runs/face_bias.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = str(ROOT / "models" / "face_index.json")
DEFAULT_MODEL = str(ROOT / "models" / "insightface")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_index(path: str, threshold: float = 0.4):
    from .index import FaceIndex
    p = Path(path)
    if p.exists():
        idx = FaceIndex.load(p)
        idx.threshold = threshold
        return idx
    return FaceIndex(threshold=threshold)


def embed_photos(detector, paths: list[Path]) -> tuple[list, list[str]]:
    """Return (embeddings, warnings) for a list of image paths."""
    embs, warnings = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            warnings.append(f"cannot read {p.name}")
            continue
        faces = detector.detect(img)
        if not faces:
            warnings.append(f"no face found in {p.name}")
            continue
        if len(faces) > 1:
            warnings.append(f"{len(faces)} faces in {p.name}: using the highest-confidence one")
        best = max(faces, key=lambda f: f.det_score)
        embs.append(best.embedding)
    return embs, warnings


def cmd_enroll(a) -> int:
    from .detector import FaceDetector
    det = FaceDetector(model_root=a.model_root,
                       min_det_score=getattr(a, "min_det_score", 0.3),
                       min_face_height=getattr(a, "min_face_height", 30.0))
    photos = Path(a.photos)
    if photos.is_file():
        paths = [photos]
    elif photos.is_dir():
        paths = sorted(f for f in photos.iterdir() if f.suffix.lower() in IMG_EXTS)
    else:
        print(f"error: {a.photos} is not a file or folder")
        return 1
    if not paths:
        print(f"error: no images found in {a.photos}")
        return 1
    print(f"reading {len(paths)} photo(s) for {a.name!r} ...")
    embs, warnings = embed_photos(det, paths)
    for w in warnings:
        print(f"  warning: {w}")
    if not embs:
        print("error: no usable face found in any photo")
        return 1
    idx = load_index(a.index, a.threshold)
    idx.enroll(a.id, a.name, embs)
    idx.save(a.index)
    print(f"enrolled {a.name!r} (id={a.id}) with {len(embs)} embedding(s). "
          f"Index now has {idx.size} people.")
    return 0


def cmd_remove(a) -> int:
    idx = load_index(a.index)
    if idx.remove(a.id):
        idx.save(a.index)
        print(f"removed {a.id}. Index now has {idx.size} people.")
    else:
        print(f"error: {a.id} is not in the index")
    return 0


def cmd_list(a) -> int:
    idx = load_index(a.index)
    if not idx.size:
        print("index is empty")
        return 0
    print(f"{'id':20s}  {'name':20s}  embeddings")
    for pid, p in idx._persons.items():
        print(f"{pid:20s}  {p.name:20s}  {len(p.embeddings)}")
    return 0


def cmd_search(a) -> int:
    from .detector import FaceDetector
    det = FaceDetector(model_root=a.model_root,
                       min_det_score=getattr(a, "min_det_score", 0.3),
                       min_face_height=getattr(a, "min_face_height", 10.0))
    idx = load_index(a.index, a.threshold)
    if not idx.size:
        print("index is empty. Enroll some people first.")
        return 1
    img = cv2.imread(a.photo)
    if img is None:
        print(f"error: cannot read {a.photo}")
        return 1
    faces = det.detect(img)
    if not faces:
        print("no face detected in the photo")
        return 1
    best = max(faces, key=lambda f: f.det_score)
    results = idx.search(best.embedding, top_k=a.top_k)
    print(f"query face: det_score={best.det_score:.3f}")
    for i, r in enumerate(results, 1):
        flag = "MATCH" if r.matched else "below threshold"
        print(f"  {i}. {r.name:20s} (id={r.person_id}) sim={r.similarity:.3f} [{flag}]")
    return 0


def cmd_demo(a) -> int:
    import time
    from .detector import FaceDetector
    from .index import FaceIndex
    det = FaceDetector(model_root=a.model_root, min_det_score=0.4)
    idx = load_index(a.index, a.threshold)
    src = a.source
    cap = cv2.VideoCapture(src if not src.isdigit() else int(src))
    if not cap.isOpened():
        print(f"error: cannot open {src}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    n, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = det.detect(frame)
            vis = det.draw(frame, faces)
            for f in faces:
                r = idx.identify(f.embedding)
                label = f"{r.name} {r.similarity:.2f}" if r else "unknown"
                x1, y1 = int(f.bbox[0]), int(f.bbox[1])
                cv2.putText(vis, label, (x1, max(y1 - 18, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            if a.show:
                cv2.imshow("face demo (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if a.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.save_video,
                                            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)
            n += 1
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if a.show:
            cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"processed {n} frames in {elapsed:.1f}s ({n / elapsed:.1f} fps)")
    return 0


def cmd_bias(a) -> int:
    """Measure per-group Rank-1 accuracy.

    Folder layout: data/faces/bias_set/<group>/<person_id>/<photo>.jpg
    Each person needs at least 2 photos: one is the query, the rest go in the gallery.
    """
    from .detector import FaceDetector
    from .index import FaceIndex
    det = FaceDetector(model_root=a.model_root, min_det_score=0.3, min_face_height=30.0)
    root = Path(a.data)
    if not root.is_dir():
        print(f"error: {a.data} is not a folder")
        return 1
    groups = [d for d in root.iterdir() if d.is_dir()]
    if not groups:
        print(f"error: no group subfolders in {a.data}. "
              "Expected layout: <data>/<group>/<person_id>/<photos>")
        return 1
    results: dict = {}
    for group in sorted(groups):
        persons = [d for d in group.iterdir() if d.is_dir()]
        if len(persons) < 2:
            print(f"  {group.name}: only {len(persons)} person(s), need at least 2; skipping")
            continue
        idx = FaceIndex(threshold=a.threshold)
        queries, query_ids = [], []
        for person_dir in persons:
            photos = sorted(f for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
            if len(photos) < 2:
                continue
            # first photo is the query, rest go in the gallery
            query_photos, gallery_photos = [photos[0]], photos[1:]
            gallery_embs, _ = embed_photos(det, gallery_photos)
            if gallery_embs:
                idx.enroll(person_dir.name, person_dir.name, gallery_embs)
            query_embs, _ = embed_photos(det, query_photos)
            if query_embs:
                queries.append(query_embs[0])
                query_ids.append(person_dir.name)
        if not queries or not idx.size:
            print(f"  {group.name}: no usable photos; skipping")
            continue
        correct = sum(
            idx.search(e, top_k=1)[0].person_id == pid
            for e, pid in zip(queries, query_ids)
            if idx.search(e, top_k=1))
        acc = round(100.0 * correct / len(queries), 1)
        results[group.name] = {"rank1": acc, "queries": len(queries), "people": idx.size}
        print(f"  {group.name:15s}: Rank-1 {acc}%  ({correct}/{len(queries)})")
    if not results:
        print("no groups could be evaluated")
        return 1
    accs = [v["rank1"] for v in results.values()]
    gap = round(max(accs) - min(accs), 1)
    results["_summary"] = {"max_gap_pp": gap, "mean": round(sum(accs) / len(accs), 1)}
    print(f"\nMax accuracy gap across groups: {gap} percentage points")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--model-root", default=DEFAULT_MODEL)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll")
    e.add_argument("--id", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--photos", required=True,
                   help="a single photo or a folder of photos")
    e.add_argument("--min-det-score", type=float, default=0.3)
    e.add_argument("--min-face-height", type=float, default=30.0)
    e.set_defaults(fn=cmd_enroll)

    r = sub.add_parser("remove")
    r.add_argument("--id", required=True)
    r.set_defaults(fn=cmd_remove)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    s = sub.add_parser("search")
    s.add_argument("--photo", required=True)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--min-det-score", type=float, default=0.3)
    s.add_argument("--min-face-height", type=float, default=10.0)
    s.set_defaults(fn=cmd_search)

    d = sub.add_parser("demo")
    d.add_argument("--source", required=True, help="video file, RTSP URL or webcam index")
    d.add_argument("--show", action="store_true")
    d.add_argument("--save-video", help="save annotated video to this path")
    d.add_argument("--max-frames", type=int)
    d.set_defaults(fn=cmd_demo)

    b = sub.add_parser("bias")
    b.add_argument("--data", required=True,
                   help="root folder: <data>/<group>/<person_id>/<photos>")
    b.add_argument("--out", help="write results to this json file")
    b.set_defaults(fn=cmd_bias)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
