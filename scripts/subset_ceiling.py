#!/usr/bin/env python3
"""Ceiling diagnostic: does ANY equal-weight subset of our legs clear the gate?

This exists to answer one question honestly -- "is the +0.0002 gate reachable at all by
recombining what we already own, or has the ensemble axis genuinely closed?" -- after three
separate hand-picked combinations landed at 0.9x the gate.

IT IS A DIAGNOSTIC AND NOTHING MAY BE SHIPPED FROM IT. The reported maximum is a maximum
over thousands of subsets scored on the SAME OOF rows, which is precisely the selection
bias KAGGLE_PLAYBOOK.md section 5 warns about -- a hill-climb over many models found
+0.00014 in S6E7 and it was pure noise. A high number here means "the ceiling is at least
this", not "ship this".

Two stages, because 16k subsets x 691k rows of roc_auc_score is ~40 minutes:
  1. screen every subset on a fixed 200k-row stratified subsample using a rank-based AUC;
  2. re-score the top survivors on the FULL OOF, which is the only number quoted.
"""
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
LEGS = {"B6": "f64e2781", "B7": "6e3dd7c3", "B8": "9e68c7ce", "B10x": "7ebde432",
        "C1": "6a091632", "C2": "5daf6c12", "C3x": "0c32463c", "C4x": "fdeaa047",
        "D1cat": "7c1e9334", "E1mlp": "2fb89920", "E2emb": "8bd89dee",
        "E3raw": "a594ffe2", "F1real": "53c678f6", "G1ftt": "b2f35b4a",
        "G2tab": "f1424102"}
CHAMPION = ["B10x", "C4x", "D1cat", "E2emb", "F1real"]   # shipped ffb65555, LB 0.96892
GATE = 0.0002


def fast_auc(y, s):
    """Mann-Whitney AUC -- one argsort instead of sklearn's full curve machinery."""
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties, or ties would bias the statistic
    s_sorted = s[order]
    ties = np.flatnonzero(np.diff(s_sorted)) + 1
    for a, b in zip(np.r_[0, ties], np.r_[ties, len(s)]):
        if b - a > 1:
            ranks[order[a:b]] = (a + b + 1) / 2.0
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    y = pd.read_csv(REPO / "data" / "train.csv",
                    usecols=["addicted_label"]).addicted_label.to_numpy()
    names, cols, folds = [], [], None
    for n, r in LEGS.items():
        f = glob.glob(str(REPO / "experiments" / "preds" / r / "oof_proba_*.csv"))
        if not f:
            print(f"  (skip {n}: no artifact)"); continue
        d = pd.read_csv(f[0])
        if folds is None:
            folds = d.fold.to_numpy()
        else:
            assert (d.fold.to_numpy() == folds).all(), f"{n}: different CV partition"
        names.append(n); cols.append(d.proba.to_numpy())
    A = np.vstack(cols)
    idx = {n: i for i, n in enumerate(names)}
    champ_full = roc_auc_score(y, A[[idx[n] for n in CHAMPION]].mean(0))
    print(f"{len(names)} legs; champion {'+'.join(CHAMPION)} = {champ_full:.6f}; gate +{GATE}")

    rng = np.random.default_rng(42)
    sub = np.sort(rng.choice(len(y), 200_000, replace=False))
    ys, As = y[sub], A[:, sub]

    combos = [c for r in range(2, len(names) + 1)
              for c in itertools.combinations(range(len(names)), r)]
    print(f"stage 1: screening {len(combos):,} equal-weight subsets on 200k rows", flush=True)
    scored = [(fast_auc(ys, As[list(c)].mean(0)), c) for c in combos]
    scored.sort(reverse=True)

    print("stage 2: re-scoring the top 25 on the FULL OOF\n", flush=True)
    final = sorted(((roc_auc_score(y, A[list(c)].mean(0)), c) for _, c in scored[:25]),
                   reverse=True)
    print(f"{'full-OOF':>10}{'vs champ':>11}  legs")
    for s, c in final:
        flag = "  <-- clears (DIAGNOSTIC ONLY)" if s - champ_full >= GATE else ""
        print(f"{s:>10.6f}{s - champ_full:>+11.6f}  {'+'.join(names[i] for i in c)}{flag}")

    n_clear = sum(1 for s, _ in final if s - champ_full >= GATE)
    print(f"\n{n_clear} of the top 25 clear +{GATE} on full OOF.")
    print("Selection bias warning: this is a maximum over "
          f"{len(combos):,} subsets scored on the same rows. Section 5 of the playbook")
    print("says a hill-climb at this scale is noise. Nothing ships from this table.")


if __name__ == "__main__":
    main()
