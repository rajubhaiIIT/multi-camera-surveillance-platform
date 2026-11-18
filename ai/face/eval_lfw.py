#!/usr/bin/env python3
"""Evaluate face verification on LFW (6,000 standard pairs, 10-fold cross-validation).

    python -m ai.face.eval_lfw --data data/raw/lfw --detector-only
    python -m ai.face.eval_lfw --data data/raw/lfw
    python -m ai.face.eval_lfw --data data/raw/lfw --tar-at-far 0.01 0.001
    python -m ai.face.eval_lfw --data data/raw/lfw --degrade-scale 0.5
    python -m ai.face.eval_lfw --data data/raw/lfw --degrade-jpeg 30

LFW folder layout expected:
    data/raw/lfw/
        lfw/               (image folders per person)
        pairs.txt          (official 6,000 pairs file)
        OR
        data/raw/lfw/      (images directly if the zip extracted without a subfolder)

Download pairs.txt from http://vis-www.cs.umass.edu/lfw/pairs.txt if it isn't in the folder.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np


PAIRS_URLS = ["https://ndownloader.figshare.com/files/5976006",      # mirror listed in scikit-learn's LFW loader
              "http://vis-www.cs.umass.edu/lfw/pairs.txt"]             # official host (failed to resolve for some users)


def find_lfw_root(path: str | Path) -> Path:
    """Folder that directly contains one sub-folder per person (Name/Name_0001.jpg)."""
    p = Path(path)
    hit = next(p.rglob("*_0001.jpg"), None) if p.exists() else None
    if hit is None:
        raise FileNotFoundError(
            f"no LFW images under {p}. Expected folders like Aaron_Eckhart/Aaron_Eckhart_0001.jpg. "
            "Run: python scripts/download_data.py fetch lfw")
    return hit.parent.parent


def find_pairs(root: Path) -> Path:
    """pairs.txt next to the images or up to two folders above; downloaded from a mirror if missing."""
    for c in (root / "pairs.txt", root.parent / "pairs.txt", root.parent.parent / "pairs.txt"):
        if c.exists():
            return c
    target = root.parent / "pairs.txt"
    errors = []
    for url in PAIRS_URLS:
        try:
            print(f"pairs.txt not found; downloading from {url} ...", flush=True)
            urllib.request.urlretrieve(url, target)
            if target.stat().st_size > 1000:
                return target
        except Exception as e:                       # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {e}")
    raise FileNotFoundError("cannot get pairs.txt (" + "; ".join(errors) + f"). Download it manually and save it as {target}")


def load_pairs(root: Path) -> tuple[list, list]:
    """Return (pairs, labels): pairs is [(path_a, path_b)], labels is 1 (same) / 0 (different)."""
    lines = find_pairs(root).read_text().strip().splitlines()

    def img(name: str, idx: int) -> Path:
        return root / name / f"{name}_{idx:04d}.jpg"

    pairs, labels = [], []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:                          # same person
            pairs.append((img(parts[0], int(parts[1])), img(parts[0], int(parts[2]))))
            labels.append(1)
        elif len(parts) == 4:                        # different people
            pairs.append((img(parts[0], int(parts[1])), img(parts[2], int(parts[3]))))
            labels.append(0)
    return pairs, labels


def degrade_image(img: np.ndarray, scale: float | None = None,
                  jpeg_quality: int | None = None) -> np.ndarray:
    """Optionally shrink and/or JPEG-compress an image to simulate CCTV conditions.

    scale=0.5 halves width and height (quarter resolution).
    jpeg_quality=30 applies lossy JPEG compression then decodes back.
    Both can be combined.
    """
    if scale is not None and scale < 1.0:
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        img = cv2.resize(small, (w, h))   # upscale back to original size so the detector sees the same input size
    if jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def tar_at_far(sims: np.ndarray, labels: list[int], valid: list[bool],
               far_targets: list[float] | None = None) -> dict:
    """Compute True Accept Rate (TAR) at fixed False Accept Rate (FAR) targets.

    TAR@FAR is the verification ROC metric: at a threshold where <FAR> of
    impostor pairs are wrongly accepted, what fraction of genuine pairs are
    correctly accepted?

    Parameters
    ----------
    sims   : cosine similarities for all pairs
    labels : 1 (same person) or 0 (different person)
    valid  : which pairs have embeddings for both images
    far_targets : e.g. [0.01, 0.001] for FAR=1% and FAR=0.1%

    Returns
    -------
    dict mapping "tar@far=X" -> TAR value (0..1), plus "auc" (area under ROC).
    """
    if far_targets is None:
        far_targets = [0.01, 0.001]

    sims = np.array(sims, np.float32)
    labels = np.array(labels, int)
    valid = np.array(valid, bool)

    pos_sims = sims[valid & (labels == 1)]   # genuine pairs
    neg_sims = sims[valid & (labels == 0)]   # impostor pairs

    if len(pos_sims) == 0 or len(neg_sims) == 0:
        return {"error": "not enough valid pairs to compute TAR@FAR"}

    # Sweep thresholds from -1 to 1 at fine steps
    thresholds = np.arange(-1.0, 1.01, 0.001)
    tars, fars = [], []
    for t in thresholds:
        tars.append(float((pos_sims >= t).mean()))
        fars.append(float((neg_sims >= t).mean()))
    tars_arr = np.array(tars)
    fars_arr = np.array(fars)

    # AUC via trapezoidal integration (FAR decreasing as threshold rises → reverse)
    auc = float(np.trapezoid(tars_arr[::-1], x=fars_arr[::-1]))

    result: dict = {"auc": round(auc, 5)}
    for far_target in far_targets:
        # Find the threshold where FAR is just at or below far_target
        idx = np.where(fars_arr <= far_target)[0]
        if len(idx) == 0:
            result[f"tar@far={far_target}"] = None   # FAR never drops this low
        else:
            result[f"tar@far={far_target}"] = round(float(tars_arr[idx[0]]), 5)

    result["genuine_pairs"] = int(len(pos_sims))
    result["impostor_pairs"] = int(len(neg_sims))
    return result


def pick_face(faces: list, image_shape: tuple, pick: str = "center"):
    """LFW photos are centred on the labelled person, so 'center' (the face nearest the middle of the image)
    avoids choosing a bystander. 'score' takes the most confident detection."""
    if pick == "score":
        return max(faces, key=lambda f: f.det_score)
    cx, cy = image_shape[1] / 2.0, image_shape[0] / 2.0
    return min(faces, key=lambda f: (((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2))


def pairs_to_embeddings(pairs: list, detector, batch: int = 64, pick: str = "score",
                        stats: dict | None = None,
                        degrade_scale: float | None = None,
                        degrade_jpeg: int | None = None) -> tuple[np.ndarray, list[bool]]:
    """Cosine similarity for every pair. Returns (similarities, valid_mask). If `stats` is a dict it is filled with
    counts: images, with_face, multi_face (more than one face found), pick_differs (centre pick != top-score pick).

    degrade_scale : if < 1.0, downscale then upscale each image before embedding (simulates low-res CCTV).
    degrade_jpeg  : if set, apply JPEG compression at this quality (0-100) before embedding.
    """
    unique = {p for pair in pairs for p in pair}
    emb_cache: dict = {}
    n_multi = n_differs = n_img = 0
    missing = set()
    for p in sorted(unique):
        img = cv2.imread(str(p)) if Path(p).exists() else None
        if img is None:
            missing.add(p)
            continue
        n_img += 1
        if degrade_scale is not None or degrade_jpeg is not None:
            img = degrade_image(img, scale=degrade_scale, jpeg_quality=degrade_jpeg)
        faces = detector.detect(img)
        if not faces:
            continue
        best = pick_face(faces, img.shape, pick)
        if len(faces) > 1:
            n_multi += 1
            n_differs += pick_face(faces, img.shape, "center") is not pick_face(faces, img.shape, "score")
        emb_cache[str(p)] = best.embedding
    if missing:
        print(f"  warning: {len(missing)} images missing or unreadable", file=sys.stderr)
    if stats is not None:
        stats.update(images=n_img, with_face=len(emb_cache), multi_face=n_multi, pick_differs=n_differs)
    sims, valid = [], []
    for a_, b_ in pairs:
        ea, eb = emb_cache.get(str(a_)), emb_cache.get(str(b_))
        ok = ea is not None and eb is not None
        sims.append(float(ea @ eb) if ok else 0.0)
        valid.append(ok)
    return np.array(sims, np.float32), valid



def worst_pairs(pairs: list, labels: list, sims: np.ndarray, valid: list, n: int = 10) -> dict:
    """The pairs the model got most wrong: same-person pairs with the lowest similarity, different-person pairs with the highest."""
    idx = [i for i, v in enumerate(valid) if v]
    pos = sorted((i for i in idx if labels[i] == 1), key=lambda i: sims[i])[:n]
    neg = sorted((i for i in idx if labels[i] == 0), key=lambda i: -sims[i])[:n]
    show = lambda i: {"similarity": round(float(sims[i]), 3), "a": str(pairs[i][0]), "b": str(pairs[i][1])}
    return {"same_person_lowest": [show(i) for i in pos], "different_person_highest": [show(i) for i in neg]}


def eval_lfw(sims: np.ndarray, labels: list[int], valid: list[bool],
             n_folds: int = 10) -> dict:
    """10-fold cross-validation: threshold on 9 folds, evaluate on the 1 held-out fold."""
    sims, labels = np.array(sims), np.array(labels)
    valid = np.array(valid, bool)
    n = len(sims)
    fold_size = n // n_folds
    accs, thresholds = [], []
    for fold in range(n_folds):
        val_idx = np.zeros(n, bool)
        val_idx[fold * fold_size:(fold + 1) * fold_size] = True
        tr_idx = ~val_idx & valid
        best_t, best_acc = 0.0, 0.0
        for t in np.arange(-1.0, 1.01, 0.01):
            pred = (sims[tr_idx] >= t).astype(int)
            acc = (pred == labels[tr_idx]).mean()
            if acc > best_acc:
                best_acc, best_t = acc, t
        thresholds.append(best_t)
        pred_val = (sims[val_idx & valid] >= best_t).astype(int)
        accs.append((pred_val == labels[val_idx & valid]).mean())
    return {"accuracy": round(float(np.mean(accs)) * 100, 3),
            "std": round(float(np.std(accs)) * 100, 3),
            "mean_threshold": round(float(np.mean(thresholds)), 3),
            "valid_pairs": int(valid.sum()),
            "total_pairs": int(n)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/lfw")
    ap.add_argument("--model-root", default="models/insightface")
    ap.add_argument("--det-size", type=int, default=320,
                    help="SCRFD input size. LFW images are 250x250 so 320 is plenty.")
    ap.add_argument("--min-det-score", type=float, default=0.3,
                    help="lower threshold finds more faces in LFW's controlled conditions")
    ap.add_argument("--min-face-height", type=float, default=30.0)
    ap.add_argument("--pick", choices=["center", "score"], default="center",
                    help="which face to use when an image has several: nearest the centre (LFW is centred on the person) or most confident")
    ap.add_argument("--worst", type=int, default=0, help="also list the N most wrongly scored pairs of each kind")
    ap.add_argument("--detector-only", action="store_true",
                    help="only count how many images have a detectable face; skip verification")
    ap.add_argument("--tar-at-far", type=float, nargs="*", metavar="FAR",
                    help="compute TAR at these FAR targets (e.g. --tar-at-far 0.01 0.001 0.0001). "
                         "Pass with no values to use defaults [0.01, 0.001].")
    ap.add_argument("--degrade-scale", type=float, default=None,
                    help="downscale images by this factor before embedding (e.g. 0.5 = half resolution). "
                         "Simulates CCTV low-resolution conditions.")
    ap.add_argument("--degrade-jpeg", type=int, default=None,
                    help="apply JPEG compression at this quality (0-100) before embedding. "
                         "Simulates CCTV compression artifacts.")
    ap.add_argument("--out", help="write metrics to this json file")
    a = ap.parse_args(argv)

    from .detector import FaceDetector
    print("loading model ...")
    det = FaceDetector(det_size=(a.det_size, a.det_size), min_det_score=a.min_det_score,
                       min_face_height=a.min_face_height, model_root=a.model_root)
    root = find_lfw_root(a.data)
    pairs, labels = load_pairs(root)
    print(f"loaded {len(pairs)} pairs from {root}")

    # Report any degradation that's active
    if a.degrade_scale is not None or a.degrade_jpeg is not None:
        parts = []
        if a.degrade_scale is not None:
            parts.append(f"scale={a.degrade_scale} ({a.degrade_scale*100:.0f}% resolution)")
        if a.degrade_jpeg is not None:
            parts.append(f"jpeg_quality={a.degrade_jpeg}")
        print(f"image degradation: {', '.join(parts)}")

    if a.detector_only:
        found, total = 0, 0
        for i, (pa, pb) in enumerate(pairs):
            for p in (pa, pb):
                if not Path(p).exists():
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                if a.degrade_scale is not None or a.degrade_jpeg is not None:
                    img = degrade_image(img, scale=a.degrade_scale, jpeg_quality=a.degrade_jpeg)
                total += 1
                if det.detect(img):
                    found += 1
            if (i + 1) % 500 == 0:
                print(f"  checked {i + 1}/{len(pairs)} pairs ...", flush=True)
        print(f"face detection rate: {found}/{total} = {100 * found / max(total, 1):.1f}%")
        return 0

    print("extracting embeddings ...")
    st: dict = {}
    sims, valid = pairs_to_embeddings(pairs, det, pick=a.pick, stats=st,
                                      degrade_scale=a.degrade_scale,
                                      degrade_jpeg=a.degrade_jpeg)
    res = eval_lfw(sims, labels, valid)
    res.update(pick=a.pick, det_size=a.det_size, **st)
    if a.degrade_scale is not None:
        res["degrade_scale"] = a.degrade_scale
    if a.degrade_jpeg is not None:
        res["degrade_jpeg"] = a.degrade_jpeg
    print(f"images: {st['images']}, with a face: {st['with_face']}, with more than one face: {st['multi_face']} "
          f"({100 * st['multi_face'] / max(st['images'], 1):.1f}%), where centre pick differs from top-score pick: {st['pick_differs']}")
    if a.worst:
        res["worst"] = worst_pairs(pairs, labels, sims, valid, a.worst)

    # TAR@FAR
    if a.tar_at_far is not None:
        far_targets = a.tar_at_far if a.tar_at_far else [0.01, 0.001]
        taf = tar_at_far(sims, labels, valid, far_targets)
        res["tar_at_far"] = taf
        print("\nTAR @ FAR:")
        for k, v in taf.items():
            if k.startswith("tar@"):
                val_str = f"{v*100:.3f}%" if v is not None else "n/a (FAR target unreachable)"
                print(f"  {k}: {val_str}")
        print(f"  AUC: {taf['auc']:.5f}")

    print(json.dumps(res, indent=2))
    print(f"\nLFW accuracy: {res['accuracy']} ± {res['std']}%  "
          f"(threshold {res['mean_threshold']}, {res['valid_pairs']}/{res['total_pairs']} valid pairs)")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

