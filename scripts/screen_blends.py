"""Offline blend screening -- evaluate candidate blends WITHOUT archiving or submitting.

Reproduces make_blend.py's nested protocol exactly: for each fold k the weight is
chosen on the other 4 folds and applied to fold k, so the reported AUC is never the
number the weight was fitted on. Pooled over all folds it is directly comparable
across candidates.
"""
import glob, itertools, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

REPO = "C:/Users/gccar/Documents/kaggle_contests/playground_s6e8"
RATIOS2 = [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]
RATIOS3 = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]

y = pd.read_csv(f"{REPO}/data/train.csv", usecols=["addicted_label"]).addicted_label.values

LEGS = {"B6": "f64e2781", "B7": "6e3dd7c3", "B8": "9e68c7ce",
        "B10x": "7ebde432", "C1": "6a091632"}
for extra in sys.argv[1:]:
    name, rid = extra.split("=")
    LEGS[name] = rid

P, folds = {}, None
for n, r in LEGS.items():
    f = glob.glob(f"{REPO}/experiments/preds/{r}/oof_proba_*.csv")
    if not f:
        print(f"  (skip {n}: no artifact for {r})"); continue
    d = pd.read_csv(f[0])
    if folds is None:
        folds = d.fold.values
    else:
        assert (d.fold.values == folds).all(), f"{n}: DIFFERENT CV PARTITION -- not blendable"
    P[n] = d.proba.values


def nested(names):
    ratios = RATIOS2 if len(names) == 2 else RATIOS3
    held = np.empty(len(y))
    chosen = []
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        best = max(ratios, key=lambda r: roc_auc_score(
            y[tr], sum(w * P[n][tr] for w, n in zip(r, names)) / sum(r)))
        held[va] = sum(w * P[n][va] for w, n in zip(best, names)) / sum(best)
        chosen.append(best)
    return roc_auc_score(y, held), chosen


CHAMP = ("B6", "B10x")
champ_auc, _ = nested(CHAMP)
print(f"champion blend {CHAMP} nested-CV held-out = {champ_auc:.6f}\n")

cands = [c for c in itertools.combinations(P, 2) if set(c) != set(CHAMP)]
cands += list(itertools.combinations(P, 3))
rows = []
for c in cands:
    auc, chosen = nested(c)
    dis = np.mean([((P[a] > .5) != (P[b] > .5)).mean() * 100
                   for a, b in itertools.combinations(c, 2)])
    rows.append((auc - champ_auc, auc, "+".join(c), dis,
                 max(set(chosen), key=chosen.count)))
rows.sort(reverse=True)
print(f"{'blend':<16}{'nested AUC':>12}{'vs champ':>11}{'mean dis%':>11}  modal ratio")
for d, auc, name, dis, ratio in rows:
    flag = "  <-- CLEARS +0.0002" if d >= 0.0002 else ""
    print(f"{name:<16}{auc:>12.6f}{d:>+11.6f}{dis:>10.3f}%  {ratio}{flag}")
