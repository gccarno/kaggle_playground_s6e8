#!/usr/bin/env python3
"""Can a tabular foundation model be made competitive on S6E8, and is subsetting the obstacle?

TabPFN v2's ~10k-row ceiling was the reason this repo never tried a TFM. That reason has
expired: TabPFN-3 states 1,000,000 x 200 and TabICLv2 states 500K with CPU/disk offload,
against our 691,369 x 12. So the question is no longer "does it fit" but "is it any good
here", and the natural follow-up -- fit on subsets and extend to the whole dataset -- is
not even a workaround for an ICL model, since the training set IS the context.

TabICLv2 is used rather than TabPFN-3 for one non-technical reason: TabICL is BSD-3
(soda-inria), while the TabPFN-3 weights carry a licence forbidding commercial and
production use. A no-prize Playground competition is very probably fine, but "very
probably fine" is not a licence review, and the BSD option costs nothing.

THREE ARMS, all on fold 0 of the frozen partition, all scored on the same 20,000 query
rows so they are comparable to each other. They are NOT directly comparable to the OOF
numbers in runs.csv -- one fold, a 20k query subset, AUC standard error about +/- 0.002 --
which is fine, because the gaps that matter here are ten times that.

  A. SINGLE CONTEXT, growing. The context-size scaling curve, on raw features.
  B. CONTEXT ENSEMBLE. K DISJOINT contexts partition the training fold, so every training
     row is used exactly once, and the K predictions are averaged in logit space. This is
     "train on subsets and extend to the whole dataset" done as well as it can be done.
  C. LOOKUP REPRESENTATION. The suspected real cause of A's weakness. This data is a
     value->target LOOKUP TABLE (rates 0.119-0.986, Pearson(value, rate) = -0.044,
     cardinality up to 1,459); a model that normalises an integer code and reads it as a
     magnitude cannot see that, and no amount of context repairs it. Here the encoder does
     the 553k-row work and the TFM only combines 12 smooth features.

MEASURED, 2026-08-11, RTX 3060 laptop, ~15 min total:

    A  ctx 4k 0.933739 | 8k 0.936880 | 16k 0.938499 | 32k 0.939397 | 69k 0.940215
    B  8 disjoint 16k blocks (128k rows total), logit-averaged     0.939787
    C  one 16k context on the target-encoded representation       0.958762  (upper bound)

    reference: weakest leg ever admitted 0.9405, weakest leg in the pool 0.9622,
               best leg 0.9687, 23-leg stack OOF 0.969434

CONCLUSION. Subsetting is not the obstacle and neither is context size. Arm A's curve is
flat by 32k -- the last doubling buys +0.0008 -- and arm B, which sees 128k rows, does not
beat a single 69k context (0.939787 vs 0.940215, inside the +/- 0.002 noise). Both plateau
near 0.940, our worst leg ever. Arm C moves the SAME model with the SAME 16k context by
+0.019 by changing only the representation, and is still below every leg in the pool while
being an optimistic upper bound. So the ceiling is representational, which is Phase 4's
finding arriving from a third direction, and a TFM leg is not worth GPU hours: it would
enter around 0.958 at best against a pool floor of 0.9622, and Phase 3 measured that a leg
weaker than the pool contributes nothing however decorrelated it is.

LEAKAGE. The categorical mapping is built over train u test values, which is unsupervised
and not a leak. The target encoding in arm C is fit on fold-0 TRAINING rows only and
applied to the query rows, so nothing about the validation fold reaches the model. Context
rows do carry their own labels through their TE values -- standard, smoothed at 50 -- which
inflates arm C. That is deliberate: arm C only needs to be an upper bound to settle this.

Usage:
    pip install tabicl          # BSD-3, ~250kB, pulls no new dependencies on this env
    python scripts/tfm_probe.py
"""
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED, N_FOLDS, TARGET, ID = 42, 5, "addicted_label", "id"
QUERY, CTX, K = 20_000, 16_000, 8
CONTEXTS = (4_000, 8_000, 16_000, 32_000, 64_000)
SMOOTH = 50.0
EPS = 1e-6


def load():
    tr = pd.read_csv(REPO / "data" / "train.csv")
    te = pd.read_csv(REPO / "data" / "test.csv")
    feats = [c for c in tr.columns if c not in (ID, TARGET)]
    X = tr[feats].copy()
    for c in (c for c in feats if tr[c].dtype == object):
        vocab = pd.Index(sorted(set(tr[c].dropna()) | set(te[c].dropna())))
        X[c] = pd.Categorical(tr[c], categories=vocab).codes.astype(float)
        X.loc[tr[c].isna(), c] = np.nan
    return tr, X.to_numpy(dtype=np.float32), feats, tr[TARGET].to_numpy()


def target_encode(raw, y, fit_idx, feats):
    """Smoothed value->rate map per column, fit ONLY on fit_idx. Returns full-length array."""
    prior = y[fit_idx].mean()
    out = np.empty((len(raw), len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        g = pd.DataFrame({"v": raw[c].iloc[fit_idx], "y": y[fit_idx]}).groupby("v")["y"]
        rate = (g.sum() + SMOOTH * prior) / (g.count() + SMOOTH)
        out[:, j] = raw[c].map(rate).fillna(prior).to_numpy()
    return out


def strat_partition(idx, y, k, rng):
    """k disjoint stratified blocks covering idx, each internally shuffled.

    The shuffle is load-bearing: blocks are built pos-then-neg, so an unshuffled prefix
    slice would be entirely one class and predict_proba would return a single column.
    """
    pos, neg = idx[y[idx] == 1].copy(), idx[y[idx] == 0].copy()
    rng.shuffle(pos); rng.shuffle(neg)
    out = []
    for p, n in zip(np.array_split(pos, k), np.array_split(neg, k)):
        b = np.concatenate([p, n]); rng.shuffle(b); out.append(b)
    return out


def main():
    raw, X, feats, y = load()
    tr_idx, va_idx = next(iter(StratifiedKFold(N_FOLDS, shuffle=True,
                                               random_state=SEED).split(X, y)))
    q = va_idx[:QUERY]
    rng = np.random.default_rng(SEED)

    from tabicl import TabICLClassifier

    def fit_predict(Xc, yc, Xq):
        clf = TabICLClassifier(device="cuda", random_state=SEED)
        t0 = time.time()
        clf.fit(Xc, yc)
        return clf.predict_proba(Xq)[:, 1], time.time() - t0

    print(f"fold 0: {len(tr_idx):,} train rows, query {len(q):,}\n")

    print("A. SINGLE CONTEXT, raw features")
    for n in CONTEXTS:
        blk = strat_partition(tr_idx, y, max(1, len(tr_idx) // n), rng)[0]
        p, s = fit_predict(X[blk], y[blk], X[q])
        print(f"   ctx {len(blk):>7,}   {s:>6.0f}s   AUC {roc_auc_score(y[q], p):.6f}")

    print(f"\nB. CONTEXT ENSEMBLE, raw features, {K} disjoint blocks capped at {CTX:,}")
    acc, t = [], 0.0
    for i, blk in enumerate(strat_partition(tr_idx, y, K, rng)):
        sub = blk[:CTX]
        p, s = fit_predict(X[sub], y[sub], X[q])
        acc.append(logit(np.clip(p, EPS, 1 - EPS))); t += s
        print(f"   block {i+1}/{K}  solo {roc_auc_score(y[q], p):.6f}   "
              f"ensemble-so-far {roc_auc_score(y[q], np.mean(acc, 0)):.6f}   ({t:.0f}s)")

    print("\nC. LOOKUP REPRESENTATION (TE fit on the training fold), one 16k context")
    TE = target_encode(raw, y, tr_idx, feats)
    blk = strat_partition(tr_idx, y, len(tr_idx) // CTX, rng)[0]
    p, s = fit_predict(TE[blk], y[blk], TE[q])
    print(f"   ctx {len(blk):>7,}   {s:>6.0f}s   AUC {roc_auc_score(y[q], p):.6f}   "
          "(UPPER BOUND -- context rows see their own labels through TE)")

    print("\nreference: pool floor 0.9622, best leg 0.9687, 23-leg stack OOF 0.969434")


if __name__ == "__main__":
    main()
