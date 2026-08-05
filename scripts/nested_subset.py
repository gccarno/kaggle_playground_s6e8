#!/usr/bin/env python3
"""Does SUBSET SELECTION generalise, or is the ceiling scan just selection bias?

scripts/subset_ceiling.py reported a best-of-16,383 at +0.000282 over the champion. That
number is meaningless on its own -- it is a maximum over thousands of candidates scored on
the same OOF rows, which is exactly what KAGGLE_PLAYBOOK.md section 5 says produced a fake
+0.00014 in S6E7.

But its top entries were not random: C4x, E2emb and F1real appeared in all of them, and
none of the three is in the champion. Structure like that deserves a real test rather than
a verdict in either direction.

The test: nest the SELECTION itself. For each outer fold k, choose the best subset using
only the other four folds, then score that subset on fold k, which the chooser never saw.
Pool the five held-out predictions and compare to the champion scored the same way. If
subset selection is noise, the held-out number collapses to zero or negative. If it holds,
the gain is real and survives the same protocol make_blend.py uses for weights.

Reports the per-fold chosen subsets too -- a procedure that picks a wildly different subset
every fold is unstable even if its mean looks good.
"""
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_ceiling import fast_auc, LEGS, CHAMPION

REPO = Path(__file__).resolve().parent.parent
SCREEN_ROWS = 150_000
TOP_K = 20


def main():
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
        names.append(n); cols.append(d.proba.to_numpy())
    A = np.vstack(cols)
    idx = {n: i for i, n in enumerate(names)}
    champ_rows = [idx[n] for n in CHAMPION]

    combos = [c for r in range(2, len(names) + 1)
              for c in itertools.combinations(range(len(names)), r)]
    print(f"{len(names)} legs, {len(combos):,} subsets, nesting the selection over 5 folds\n",
          flush=True)

    rng = np.random.default_rng(42)
    held_new = np.empty(len(y))
    held_champ = np.empty(len(y))
    chosen = []

    for k in sorted(set(folds)):
        tr = np.flatnonzero(folds != k)
        va = np.flatnonzero(folds == k)
        screen = np.sort(rng.choice(tr, min(SCREEN_ROWS, len(tr)), replace=False))

        # stage 1: cheap screen of every subset on training folds only
        ys, As = y[screen], A[:, screen]
        top = sorted(((fast_auc(ys, As[list(c)].mean(0)), c) for c in combos),
                     reverse=True)[:TOP_K]
        # stage 2: exact re-score of the survivors, still training folds only
        best = max(((roc_auc_score(y[tr], A[list(c)][:, tr].mean(0)), c) for _, c in top))
        held_new[va] = A[list(best[1])][:, va].mean(0)
        held_champ[va] = A[champ_rows][:, va].mean(0)
        chosen.append(best[1])
        print(f"  fold {k}: chose {'+'.join(names[i] for i in best[1])}", flush=True)

    a_new = roc_auc_score(y, held_new)
    a_champ = roc_auc_score(y, held_champ)
    print(f"\nheld-out (selection never saw the fold it is scored on):")
    print(f"  champion        {a_champ:.6f}")
    print(f"  selected subset {a_new:.6f}")
    print(f"  delta           {a_new - a_champ:+.6f}   gate +0.0002")

    per_fold = [(roc_auc_score(y[folds == k], held_new[folds == k])
                 - roc_auc_score(y[folds == k], held_champ[folds == k]))
                for k in sorted(set(folds))]
    print(f"  per-fold        {['%+.6f' % d for d in per_fold]}")
    print(f"  folds positive  {sum(d > 0 for d in per_fold)}/5")

    stable = len({frozenset(c) for c in chosen})
    print(f"\nstability: {stable} distinct subsets chosen across 5 folds "
          f"({'stable' if stable <= 2 else 'UNSTABLE -- procedure is fitting fold noise'})")
    core = set.intersection(*[set(c) for c in chosen])
    print(f"legs chosen by EVERY fold: {'+'.join(sorted(names[i] for i in core)) or '(none)'}")


if __name__ == "__main__":
    main()
