#!/usr/bin/env python3
"""Does a fitted meta-learner beat our equal-weight average? (idea audit, public notebooks)

Everything we have ever shipped is an EQUAL-WEIGHT mean of legs -- make_blend.py,
subset_ceiling.py and nested_subset.py all search over *which* legs, never over *how much*
of each. The public frontier does not do that. Both notebooks at the top of the public LB
(szymonkapiski 0.97084, riponce 0.97068) fit a logistic regression on the OOF matrix in
LOGIT space, and both report negative coefficients on some members.

That is the mechanism worth testing here, and it is not "hill climbing with more steps":

  * an equal-weight mean can only ADD. A leg that is weak but wrong in a useful direction
    can only dilute it, which is exactly why our 0.965 screening rule kept finding that
    weak legs "damage the blend" -- that finding is a property of the COMBINER, not of the
    legs. A linear stacker can give such a leg a negative weight and use it as a
    correction.
  * logits, not probabilities. This target saturates (the top screen-time decile is ~100%
    positive), so near p=1 the probability scale has no resolution left and the mean is
    dominated by rows where nothing is in doubt. log(p/(1-p)) keeps resolving there.

PROTOCOL. The meta-model is refit inside each outer fold and only ever predicts rows it did
not see, so its OOF is honest in the same sense make_blend.py's weights are. The champion
needs no fitting, so it is scored directly. Known and accepted caveat, shared with every
public stack: the base legs' OOF values come from models fitted on this same partition, so
a meta-model training on folds 1-4 consumes predictions from base models that saw fold 0.
That inflates ALL stack numbers here by a small constant; it does not affect the comparison
between two combiners on identical inputs, which is what this script is for.

GATE: +0.0002 OOF over the equal-weight champion, the same pre-registered gate as every
other probe. Nothing ships from this file -- it decides whether a stacked leg is worth a
real run.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from subset_ceiling import CHAMPION, LEGS

REPO = Path(__file__).resolve().parent.parent
EPS = 1e-6
GATE = 0.0002


def load():
    y = pd.read_csv(REPO / "data" / "train.csv",
                    usecols=["addicted_label"]).addicted_label.to_numpy()
    names, cols, folds = [], [], None
    for n, r in LEGS.items():
        f = glob.glob(str(REPO / "experiments" / "preds" / r / "oof_proba_*.csv"))
        if not f:
            continue
        d = pd.read_csv(f[0])
        if folds is None:
            folds = d.fold.to_numpy()
        else:
            assert (d.fold.to_numpy() == folds).all(), f"{n}: different CV partition"
        names.append(n)
        cols.append(d.proba.to_numpy())
    return y, names, np.column_stack(cols), folds


def honest_stack(X, y, folds, C=1.0):
    """Refit the meta-model per outer fold; predict only held-out rows."""
    oof = np.zeros(len(y))
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=5000, tol=1e-5)
        m.fit(sc.transform(X[tr]), y[tr])
        oof[va] = m.predict_proba(sc.transform(X[va]))[:, 1]
    return oof


def main():
    y, names, P, folds = load()
    idx = {n: i for i, n in enumerate(names)}
    L = logit(np.clip(P, EPS, 1 - EPS))
    print(f"{len(names)} legs, {len(y):,} rows, folds {sorted(set(folds))}\n")

    champ = P[:, [idx[n] for n in CHAMPION]].mean(1)
    a_champ = roc_auc_score(y, champ)
    best_solo = max(roc_auc_score(y, P[:, i]) for i in range(len(names)))
    print(f"best single leg                        {best_solo:.6f}")
    print(f"equal-weight champion ({len(CHAMPION)} legs)      {a_champ:.6f}   <- what we ship\n")

    # Equal weight over ALL legs, for contrast: this is the thing our 0.965 rule rejected.
    print(f"equal-weight over all {len(names)} legs          "
          f"{roc_auc_score(y, P.mean(1)):.6f}")
    print(f"equal-weight over all {len(names)} legs, logit   "
          f"{roc_auc_score(y, L.mean(1)):.6f}\n")

    for C in (0.03, 0.1, 1.0):
        a_p = roc_auc_score(y, honest_stack(P, y, folds, C))
        a_l = roc_auc_score(y, honest_stack(L, y, folds, C))
        print(f"stack C={C:<5g} proba {a_p:.6f} ({a_p - a_champ:+.6f})   "
              f"logit {a_l:.6f} ({a_l - a_champ:+.6f})"
              f"{'   <- CLEARS' if a_l - a_champ >= GATE else ''}")

    # Which legs does a fitted combiner actually use, and with what sign? The interesting
    # cases are the ones an equal-weight search threw away -- NODE and the sub-0.965 legs.
    sc = StandardScaler().fit(L)
    w = LogisticRegression(C=1.0, max_iter=5000, tol=1e-5).fit(sc.transform(L), y).coef_[0]
    t = pd.DataFrame({"leg": names, "solo": [roc_auc_score(y, P[:, i]) for i in range(len(names))],
                      "weight": w, "in_champ": [n in CHAMPION for n in names]})
    print("\n" + t.reindex(t.weight.abs().sort_values(ascending=False).index)
          .to_string(index=False, float_format="%.4f"))
    print(f"\nnegative weights: {(w < 0).sum()} of {len(w)}  "
          "(an equal-weight mean cannot express any of these)")


if __name__ == "__main__":
    main()
