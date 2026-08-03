#!/usr/bin/env python3
"""Blend archived per-learner predictions at FIXED ROUND WEIGHTS and log the result.

KAGGLE_PLAYBOOK.md §5, which this deliberately implements the boring way:

  * fixed round ratios (1:1, 2:1, 3:1), never a hill-climbed simplex. A hill-climb over
    many models found +0.00014 in S6E7 -- about 0.5 sigma, i.e. pure selection bias --
    and one collapsed to weight 1.0 on a single model, a degenerate "blend";
  * the ratio is chosen by NESTED CV (pick on 4 folds, score on the held-out 5th), so
    the number used to justify shipping is not the number the weight was fitted on;
  * two strong decorrelated learners beat a big stack with a meta-learner.

Usage:
    python scripts/make_blend.py --runs f64e2781 7ebde432 --submit
    python scripts/make_blend.py --runs a b c --weights 2 1 1     # skip the search
"""
import argparse, csv, itertools, json, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDS = REPO_ROOT / "experiments" / "preds"
COMPETITION = "playground-series-s6e8"
RATIOS = [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]


def load_run(run_id):
    d = PREDS / run_id
    oof = next(d.glob("oof_proba_*.csv"), None)
    tst = next(d.glob("test_proba_*.csv"), None)
    if oof is None or tst is None:
        raise SystemExit(f"{run_id}: missing oof/test proba artifacts in {d}")
    return pd.read_csv(oof), pd.read_csv(tst), oof.stem.replace("oof_proba_", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run_ids under experiments/preds/")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="fixed weights; omit to search round ratios (2 runs only)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    truth = pd.read_csv(REPO_ROOT / "data" / "train.csv", usecols=["id", "addicted_label"])
    y = truth["addicted_label"].to_numpy()

    oofs, tests, names, folds = [], [], [], None
    for r in args.runs:
        o, t, name = load_run(r)
        assert (o["id"].to_numpy() == truth["id"].to_numpy()).all(), f"{r}: OOF ids misaligned"
        if folds is None:
            folds = o["fold"].to_numpy()
        else:
            # Different partitions make the OOF matrices incomparable and any blend
            # built from them invalid -- refuse rather than silently average them.
            assert (o["fold"].to_numpy() == folds).all(), f"{r}: DIFFERENT CV PARTITION -- not blendable"
        oofs.append(o["proba"].to_numpy()); tests.append(t); names.append(f"{r}:{name}")
    P = np.vstack(oofs)

    for r, p in zip(args.runs, oofs):
        print(f"  {r:10} solo OOF {roc_auc_score(y, p):.6f}")

    if args.weights:
        w = np.array(args.weights, dtype=float)
    elif len(args.runs) == 2:
        print("\nnested CV over round ratios (choose on 4 folds, score on the 5th):")
        held, chosen = [], []
        for k in sorted(set(folds)):
            tr_m, va_m = folds != k, folds == k
            best = max(RATIOS, key=lambda r: roc_auc_score(
                y[tr_m], (r[0] * P[0][tr_m] + r[1] * P[1][tr_m]) / sum(r)))
            s = roc_auc_score(y[va_m], (best[0] * P[0][va_m] + best[1] * P[1][va_m]) / sum(best))
            solo = max(roc_auc_score(y[va_m], p[va_m]) for p in oofs)
            held.append(s - solo); chosen.append(best)
            print(f"  fold {k}: chose {best[0]}:{best[1]}  blend-minus-best-solo {s - solo:+.6f}")
        w = np.array(max(set(chosen), key=chosen.count), dtype=float)
        print(f"\nheld-out mean gain over best solo: {np.mean(held):+.6f}  (gate +0.0002)")
        print(f"modal ratio -> weights {w.tolist()}")
    else:
        w = np.ones(len(args.runs))
        print("\n3+ runs: defaulting to equal weights (no simplex search -- §5).")

    w = w / w.sum()
    blend_oof = (w[:, None] * P).sum(axis=0)
    auc = roc_auc_score(y, blend_oof)
    print(f"\nfull-OOF blend AUC = {auc:.6f}   weights {np.round(w, 4).tolist()}")
    for r, p in zip(args.runs, oofs):
        print(f"    vs {r}: {auc - roc_auc_score(y, p):+.6f}")

    run_id = uuid.uuid4().hex[:8]
    dest = PREDS / run_id
    dest.mkdir(parents=True, exist_ok=True)

    ids = tests[0]["id"].to_numpy()
    for t in tests:
        assert (t["id"].to_numpy() == ids).all(), "test ids misaligned across runs"
    blend_test = np.clip(sum(wi * t["proba"].to_numpy() for wi, t in zip(w, tests)), 0.0, 1.0)

    pd.DataFrame({"id": truth["id"], "fold": folds, "proba": blend_oof}).to_csv(
        dest / "oof_proba_blend.csv", index=False)
    pd.DataFrame({"id": ids, "proba": blend_test}).to_csv(dest / "test_proba_blend.csv", index=False)
    sub = pd.DataFrame({"id": ids, "addicted_label": blend_test})
    sub.to_csv(dest / "submission.csv", index=False)
    (dest / "manifest.json").write_text(json.dumps(
        {"sources": args.runs, "learners": names, "weights": w.tolist(),
         "oof_auc": auc}, indent=1), encoding="utf-8")

    ss = pd.read_csv(REPO_ROOT / "data" / "sample_submission.csv")
    assert len(sub) == len(ss) and (sub["id"].values == ss["id"].values).all()
    assert sub["addicted_label"].notna().all() and sub["addicted_label"].between(0, 1).all()
    print(f"submission validated -> {dest / 'submission.csv'}")

    score = ""
    if args.submit:
        msg = f"blend {'+'.join(args.runs)} w={np.round(w, 3).tolist()}"
        out = subprocess.run(["kaggle", "competitions", "submit", COMPETITION,
                              "-f", str(dest / "submission.csv"), "-m", msg],
                             capture_output=True, text=True, cwd=REPO_ROOT)
        print(out.stdout or out.stderr)
        if out.returncode != 0:
            raise SystemExit(f"submit failed: {out.stderr}")
        import time
        for _ in range(40):
            q = subprocess.run(["kaggle", "competitions", "submissions", COMPETITION, "--csv"],
                               capture_output=True, text=True, cwd=REPO_ROOT)
            rows = list(csv.DictReader(q.stdout.splitlines()))
            if rows:
                s = rows[0].get("publicScore") or rows[0].get("public_score")
                if s not in (None, "", "None", "pending"):
                    score = s; break
            time.sleep(30)
        print(f"Public LB = {score}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("cr", Path(__file__).with_name("collect_run.py"))
    cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)
    cr.append_run_row({
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=REPO_ROOT).stdout.strip(),
        "kernel_ref": "blend", "run_tag": "blend",
        "description": f"blend {'+'.join(args.runs)} weights {np.round(w, 3).tolist()}",
        "final_oof_auc": round(float(auc), 6), "public_lb_score": score,
        "preds_dir": dest.relative_to(REPO_ROOT).as_posix(), "notes": args.notes,
    })
    print(f"Appended blend run {run_id} to experiments/runs.csv")


if __name__ == "__main__":
    main()
