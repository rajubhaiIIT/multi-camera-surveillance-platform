"""Market-1501 evaluation protocol (CMC / Rank-k / mAP) and a similarity-threshold helper."""
from __future__ import annotations

import numpy as np


def evaluate_market(qf, qp, qc, gf, gp, gc, max_rank: int = 50) -> dict:
    """qf/gf: L2-normalised features [N, D]. qp/gp: person ids. qc/gc: camera ids.

    Standard protocol: for each query, gallery images of the SAME person from the SAME camera are ignored,
    and junk images (id -1) are ignored. Distractors (id 0) stay in the gallery as wrong answers.
    """
    qf, gf = np.asarray(qf, np.float32), np.asarray(gf, np.float32)
    qp, qc, gp, gc = (np.asarray(a) for a in (qp, qc, gp, gc))
    sim = qf @ gf.T
    order_all = np.argsort(-sim, axis=1)
    cmcs, aps, skipped = [], [], 0
    for i in range(len(qf)):
        order = order_all[i]
        keep = ~(((gp[order] == qp[i]) & (gc[order] == qc[i])) | (gp[order] == -1))
        matches = (gp[order][keep] == qp[i]).astype(np.int32)
        n_rel = matches.sum()
        if n_rel == 0:                       # this person never appears in another camera
            skipped += 1
            continue
        cmc = np.minimum(matches.cumsum(), 1)[:max_rank]
        cmcs.append(np.pad(cmc, (0, max_rank - len(cmc)), mode="edge"))
        prec = matches.cumsum() / (np.arange(len(matches)) + 1.0)
        aps.append(float((prec * matches).sum() / n_rel))
    if not cmcs:
        raise ValueError("no valid queries (does every query have a match in another camera?)")
    cmc = np.mean(cmcs, axis=0)
    return {"rank1": float(cmc[0] * 100), "rank5": float(cmc[4] * 100), "rank10": float(cmc[9] * 100),
            "mAP": float(np.mean(aps) * 100), "valid_queries": len(cmcs), "skipped_queries": skipped}


def suggest_threshold(qf, qp, qc, gf, gp, gc, n_pairs: int = 200_000, seed: int = 0) -> dict:
    """Pick the cosine-similarity threshold with the best F1 for 'same person?' on cross-camera pairs.

    Start point for the cross-camera matcher only: random negatives are easier than real
    look-alikes, so re-tune on your own multi-camera recording.
    """
    rng = np.random.default_rng(seed)
    qf, gf = np.asarray(qf, np.float32), np.asarray(gf, np.float32)
    qi, gi = rng.integers(0, len(qf), n_pairs), rng.integers(0, len(gf), n_pairs)
    ok = (gp[gi] != -1) & (gp[gi] != 0) & ~((qp[qi] == gp[gi]) & (qc[qi] == gc[gi]))
    qi, gi = qi[ok], gi[ok]
    s = np.einsum("ij,ij->i", qf[qi], gf[gi])
    pos = qp[qi] == gp[gi]
    # random pairs are ~99.9% negatives; also add true positives so F1 is meaningful
    pq = np.flatnonzero(np.isin(gp, np.unique(qp)))
    pj = rng.choice(pq, min(len(pq), 50_000))
    qs = np.array([rng.choice(np.flatnonzero(qp == gp[j])) if (qp == gp[j]).any() else 0 for j in pj])
    good = ~((qp[qs] == gp[pj]) & (qc[qs] == gc[pj])) & (qp[qs] == gp[pj])
    s = np.concatenate([s, np.einsum("ij,ij->i", qf[qs[good]], gf[pj[good]])])
    pos = np.concatenate([pos, np.ones(good.sum(), bool)])
    best = {"f1": -1.0}
    for t in np.arange(0.0, 1.0, 0.01):
        pred = s >= t
        tp = (pred & pos).sum()
        p, r = tp / max(pred.sum(), 1), tp / max(pos.sum(), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best["f1"]:
            best = {"threshold": round(float(t), 2), "precision": float(p), "recall": float(r), "f1": float(f1)}
    best["same_person_mean_sim"] = float(s[pos].mean())
    best["different_person_mean_sim"] = float(s[~pos].mean())
    return best
