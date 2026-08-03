# Kaggle Playground Series S6E8 — Predicting Smartphone Addiction

This file is the **contract**. The decisions recorded here are made once and never revisited, because
every prediction artifact in `experiments/preds/` is keyed to them. Changing any of them
retroactively invalidates cross-run blending. See `KAGGLE_PLAYBOOK.md` §1.

## Competition facts

| | |
|---|---|
| Slug | `playground-series-s6e8` |
| Title | Predicting Smartphone Addiction |
| Task | Binary classification |
| Target | `addicted_label` ∈ {0, 1} |
| **Submission** | `id,addicted_label` where the value is a **probability in [0,1]**, not a hard label |
| **Metric** | **ROC AUC** |
| Train | 691,369 rows × 12 features |
| Test | 296,302 rows |
| Positive rate | 0.709424 |
| Deadline | 2026-08-31 23:59 UTC |
| Limits | 10 submissions/day, max team size 3 |
| Kernel | `gcarno/eda-s6e8` |
| Data (kernel) | `/kaggle/input/competitions/playground-series-s6e8/` |
| Data (local) | `data/` — gitignored; `kaggle competitions download -c playground-series-s6e8 -p data` |

## The frozen CV split

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # stratified on addicted_label
```

**Frozen for the entire competition.** Every `oof_proba_*.csv` under `experiments/preds/` is aligned
to this split and to the row order of `train.csv`, which is what makes OOF matrices from runs weeks
apart directly blendable. Do not change `n_splits`, `shuffle`, or `random_state` — not for a "quick
test", not for a new learner.

Seed-bagging *inside* a fold (averaging several model seeds per fold) is fine and was a real gain in
S6E7. Re-running the whole stack at a different *split* seed was not (§5) — it is a probe, not a
default, and it never changes the numbers above.

## Metric discipline

`sklearn.metrics.roc_auc_score` is the objective **everywhere**: CV scoring, any future Optuna
objective, and any neural early-stopping. Not accuracy, not logloss as a silent proxy.

Two consequences specific to AUC, which differ from S6E7's balanced accuracy:

- **AUC is rank-based, so there is no decision threshold and no per-class weight to tune.** Submit
  raw probabilities. Any monotone transform of the scores leaves the metric unchanged — so
  probability *calibration* cannot help the leaderboard, only ranking can.
- Class imbalance is mild (71/29) and `scale_pos_weight` / class weighting does not change the
  ranking much. Treat it as a probe with a pre-registered gate, never as a default.

## Leakage discipline

The load-bearing constraint (§2). Target encoders, quantile bin edges, scalers, imputers and
category vocabularies are:

- **fit on the training fold only**, applied to the val/test fold;
- **re-fit at every usage site** — feature selection, HPO, and the final stack each get their own fit,
  never one global fit before the CV loop;
- unseen categories at inference map to a reserved "unknown" level, never raise.

An *unsupervised* label mapping (ordinal-coding a category vocabulary over train ∪ test) is not a
leak, because it never touches `y`. A *supervised* one (target/count encoding) is. When the
distinction is non-obvious in code, say which one it is in a comment.

## Pre-registered improvement gate

**Provisional: +0.0002 OOF AUC.** Measured, not invented — but a floor, not the final gate.

§4 says the gate should be ≈1σ of the OOF↔LB residual, which needs ~10 paired runs. Until those
exist, the gate comes from a **seed twin**: run `22de9888` is the champion with only LightGBM's
`random_state` changed (42→43), the CV partition untouched.

| measurement | value |
|---|---|
| seed-twin \|ΔOOF\| — OOF movement when *nothing meaningful* changed | 0.000066 |
| seed-twin \|ΔLB\| | 0.000090 |
| 3 × paired-bootstrap SE of the ΔAUC | 0.000089 |
| seed-twin disagreement rate at 0.5 | 1.45% |
| **provisional gate** (≈3× the seed noise) | **+0.0002** |

Why this is a floor and not the answer: it captures *estimation* noise but not OOF→LB *transfer*
noise, so it can only ever be too permissive. Replace it with the measured residual σ as soon as
there are enough pairs, and re-judge any probe that cleared it by less than the new σ.

**A paired-bootstrap CI excluding zero is not a pass.** The seed twin's ΔOOF of 0.000066 is
"distinguishable" by that test — two seeds genuinely are two different models — and it is still pure
noise for decision-making, because another seed would move it again. Only the gate decides shipping.

### OOF↔LB calibration (n=5) — and its one known limit

| run | tag | OOF | LB | offset |
|---|---|---|---|---|
| `ba6c676a` | anchor | 0.963320 | 0.96500 | +0.001680 |
| `22de9888` | seed twin | 0.963254 | 0.96491 | +0.001656 |
| `66e3dede` | B1 interaction | 0.963047 | 0.96478 | +0.001733 |
| `6b671f87` | champion @ES400 | 0.963405 | 0.96521 | +0.001805 |
| `f64e2781` | **B6 target encoding** | **0.966384** | **0.96788** | **+0.001496** |

Spearman(OOF, LB) = **+1.000** across all five. Offset is positive because an OOF model sees 4/5 of
the data while the submission averages all 5 fold-models.

**The offset is NOT model-family invariant.** The four non-TE runs sit at +0.001719 ± 0.000066. B6
came in at +0.001496 — **3.37σ low**. Before submitting B6 I pre-registered a predicted LB of
0.96810 ± 0.0002 from this calibration; it landed at 0.96788, outside the band and in the direction
that target-encoding optimism predicts (the val-fold encoding is built on 4/5 of train, so OOF
benefits marginally more than test does).

Consequences, which bind on every later decision:

- **Within a model family, the gate stands.** Every probe is a strict twin, so this is the usual case.
- **Across families, do not compare OOF at gate resolution.** A TE model needs ≈0.0002 subtracted
  from its OOF before being compared with a non-TE model. Blend weights fitted on OOF across families
  inherit the same bias.
- The 0.0002 family offset is itself estimated from a single TE run. Treat it as an order of
  magnitude, not a constant, and re-estimate as more TE runs land.

## Experiment log

`experiments/runs.csv` is append-only and tracked in git — one row per run, with a long free-text
`notes` column (the single most valuable column in S6E7). `experiments/preds/<run_id>/` holds that
run's `submission.csv`, `oof_proba_*.csv`, `test_proba_*.csv`; it is gitignored but never deleted.

Runs are collected with:

```
python scripts/collect_run.py --submit --description "..." --notes "..."
```

which pushes the kernel, polls to completion, parses the `RUN_METRICS_JSON:` line from the kernel
log, archives the artifacts, optionally submits, and appends the row.
