#!/usr/bin/env python3
"""Optuna HPO for the XGBoost leg, on the local GPU. KAGGLE_PLAYBOOK.md sections 2 and 3.

Why XGBoost and why now:
  * the feature set is settled -- every FE family probe (B1, B2/B3 screens, D2, D2b) missed,
    and target encoding is the one feature lever that paid. Section C of the phase plan says
    tuning comes AFTER that, because HPO results are conditional on the feature set;
  * the offline blend screen showed XGBoost is load-bearing: every blend containing it beat
    every blend without it. It is also the least-tuned leg -- one probe (C3, depth 6->8,
    +0.000001) against LightGBM's three.

Leakage discipline (section 2), which HPO is a named usage site for: the target encoder is
re-fit INSIDE every fold of every trial. It is never fit once outside the loop, even though
that would make the search several times faster.

Honest caveat on the protocol: the search runs on a stratified subsample with its own
3-fold split, so the rows it selects on overlap the frozen 5-fold OOF used for the gate.
That makes the SEARCH's own score optimistic. It is not the number anything ships on --
the winning config is re-run through the full frozen 5-fold and judged there.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
SEED = 42
RAW_CAT = ["gender", "stress_level", "academic_work_impact"]
RAW_NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
           "work_study_hours", "sleep_hours", "notifications_per_day",
           "app_opens_per_day", "weekend_screen_time"]


def rate_table(codes_fit, y_fit, k, prior, m):
    s = np.bincount(codes_fit, weights=y_fit, minlength=k)
    n = np.bincount(codes_fit, minlength=k)
    return (s + prior * m) / (n + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--sample", type=int, default=300_000)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    train = pd.read_csv(REPO / "data" / "train.csv")
    test = pd.read_csv(REPO / "data" / "test.csv")

    # Unsupervised label mapping over train + test -- never touches y, so it is safe here.
    codes = {}
    for c in RAW_NUM:
        cc, _ = pd.factorize(pd.concat([train[c], test[c]]), use_na_sentinel=False)
        codes[c] = (cc[:len(train)], int(cc.max()) + 1)

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(train), size=min(args.sample, len(train)), replace=False))
    y = train["addicted_label"].to_numpy()[idx]
    X = train.iloc[idx][RAW_NUM + RAW_CAT].copy()
    for c in RAW_CAT:
        levels = pd.Index(sorted(set(train[c].dropna()) | set(test[c].dropna())))
        X[c] = pd.Categorical(X[c], categories=levels)
    sub_codes = {c: codes[c][0][idx] for c in RAW_NUM}
    print(f"tuning on {len(X):,} rows, {args.folds}-fold, {args.trials} trials, GPU")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y))

    def objective(trial):
        p = dict(
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        )
        aucs = []
        for itr, iva in splits:
            Xtr, Xva = X.iloc[itr].copy(), X.iloc[iva].copy()
            ytr = y[itr]
            prior = float(ytr.mean())
            # Re-fit the encoder inside THIS fold of THIS trial (section 2).
            for c in RAW_NUM:
                ctr, k = sub_codes[c], codes[c][1]
                enc = np.empty(len(itr))
                inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                for i_fit, i_app in inner.split(np.zeros(len(itr)), ytr):
                    tbl = rate_table(ctr[itr][i_fit], ytr[i_fit], k, prior, 20.0)
                    enc[i_app] = tbl[ctr[itr][i_app]]
                full = rate_table(ctr[itr], ytr, k, prior, 20.0)
                Xtr[f"{c}__te"], Xva[f"{c}__te"] = enc, full[ctr[iva]]
            model = xgb.XGBClassifier(
                n_estimators=4000, learning_rate=0.05, device="cuda", tree_method="hist",
                enable_categorical=True, eval_metric="auc", early_stopping_rounds=200,
                random_state=SEED, **p)
            model.fit(Xtr, ytr, eval_set=[(Xva, y[iva])], verbose=False)
            aucs.append(roc_auc_score(y[iva], model.predict_proba(Xva)[:, 1]))
        trial.set_user_attr("fold_aucs", [round(a, 6) for a in aucs])
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()

    out = REPO / ".kaggle_output" / "xgb_best_params.json"

    def cb(st, tr):
        print(f"  trial {tr.number:3d}  auc={tr.value:.6f}  best={st.best_value:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # Checkpoint every trial. A long GPU search that dies at trial 40 should not
        # throw away 40 trials, and this one already did once.
        out.write_text(json.dumps({"best_params": st.best_params,
                                   "best_value": st.best_value,
                                   "n_trials": len(st.trials)}, indent=1), encoding="utf-8")

    study.optimize(objective, n_trials=args.trials, callbacks=[cb])

    print(f"\nbest sub-sample AUC {study.best_value:.6f} (optimistic -- see module docstring)")
    print("best params:", json.dumps(study.best_params, indent=1))
    out = REPO / ".kaggle_output" / "xgb_best_params.json"
    out.write_text(json.dumps(study.best_params, indent=1), encoding="utf-8")
    print(f"wrote {out}")

    base = study.trials[0]
    print(f"\nspread over {len(study.trials)} trials: "
          f"{min(t.value for t in study.trials):.6f} .. {study.best_value:.6f}")


if __name__ == "__main__":
    main()
