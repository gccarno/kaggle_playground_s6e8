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

### OOF↔LB calibration (n=9)

| run | tag | OOF | LB | offset |
|---|---|---|---|---|
| `ba6c676a` | anchor | 0.963320 | 0.96500 | +0.001680 |
| `22de9888` | seed twin | 0.963254 | 0.96491 | +0.001656 |
| `66e3dede` | B1 interaction ratios | 0.963047 | 0.96478 | +0.001733 |
| `6b671f87` | champion @ES400 | 0.963405 | 0.96521 | +0.001805 |
| `f64e2781` | B6 target encoding | 0.966384 | 0.96788 | +0.001496 |
| `6e3dd7c3` | B7 TE cross | 0.966353 | 0.96806 | +0.001707 |
| `9e68c7ce` | B8 num_leaves 255 | 0.966135 | 0.96781 | +0.001675 |
| `6a091632` | C1 lr 0.02 | 0.966604 | 0.96802 | +0.001416 |
| **`b171c0cf`** | **blend (champion)** | **0.966902** | **0.96831** | +0.001408 |

**offset +0.001620 ± 0.000143, Spearman(OOF, LB) = +0.950, residual σ = 0.000107.**

Gate stays **+0.0002** ≈ 1.9σ of the residual. Offset is positive because an OOF model sees 4/5 of the
data while a submission averages all 5 fold-models.

**The first rank inversion, at n=9.** C1 beat B7 on OOF (0.966604 vs 0.966353, +0.000251) and *lost*
to it on LB (0.96802 vs 0.96806). That is why Spearman fell from 0.976 to 0.950 — not a worse
calibration, a more honest one, because the first eight runs never contained two models close enough
to test the ordering. Read it as the intended warning: **at ΔOOF ≈ 0.00025 the two metrics do not
reliably agree on which model is better**, so a probe clearing the gate by a hair is not a decision.
This argues the gate should move *up* when §4's ~10 pairs land, not down.

#### A retracted claim, kept on purpose

After B6 I recorded that "the offset is not model-family invariant — TE models carry ≈0.0002 more OOF
optimism," because B6's +0.001496 sat 3.37σ below the four non-TE runs then available. **B7 refuted
it**: also target-encoded, it landed at +0.001707, inside the non-TE range. Against all eight runs B6
is an unremarkable low draw.

The lesson is about the estimate, not the encoder. A σ computed from four nearly-identical models
spanning 0.00036 of OOF was too small (0.000064) and too confident; over a 3× wider span it is
0.000096. **Do not treat a σ estimated from a narrow OOF range as valid outside that range, and do
not promote a single outlier to a structural finding.** This is the same error the playbook warns
about for public-LB scores, committed against our own metric.

## Strength vs decorrelation (§6) — first measurements

The playbook's central late-competition test: for a blend to improve, a leg must be both *strong solo*
and *decorrelated* from the champion's other legs. In S6E7 those two were strongly anti-correlated
(Spearman −0.84), which is what closes the ensemble axis. First three points here, all against the
XGBoost leg, with the seed-noise disagreement floor at 1.450%:

| leg | solo OOF | disagreement vs XGB |
|---|---|---|
| B6 LightGBM (lr 0.05) | 0.966384 | **1.931%** |
| C1 LightGBM (lr 0.02) | 0.966604 | **1.843%** |

Already the expected direction: **the stronger leg is the more correlated one.** C1 beat B6 solo by
+0.000220 (clearing the gate) yet moved the blend only +0.000073 (missing it), because most of the
solo gain was cancelled by lost diversity.

**Operational consequence:** when the champion is a blend, a leg clearing the gate is *not* grounds to
ship. Re-derive the blend and judge it on nested-CV held-out AUC. Runs `6a091632` (leg cleared) and
`4886aef5` (blend did not) are the worked example.

### The existing ensemble space is exhausted

Screened offline from archived OOF artifacts — **all 10 pairs and 10 triples** over the five legs
(B6, B7, B8 LightGBM; B10 XGBoost; C1 LightGBM lr 0.02), each scored by the same nested protocol
`make_blend.py` uses (ratio chosen on 4 folds, applied to the held-out 5th, pooled). Champion
`B6+B10x` = 0.966895.

| blend | nested-CV held-out | vs champion | mean disagreement |
|---|---|---|---|
| B7+B10x+C1 | 0.967022 | +0.000127 | 1.829% |
| B7+B8+B10x | 0.967014 | +0.000119 | 2.129% |
| B6+B7+B10x | 0.966999 | +0.000104 | 1.894% |
| … 16 more … | | | |
| B6+C1 | 0.966641 | −0.000254 | 0.975% |

**Nothing clears +0.0002.** The best of twenty candidates reaches +0.000127, and the sixteen worst are
flat or negative. Two structural facts fall out:

- **Every blend that contains the XGBoost leg beats every blend that does not.** B10x supplies all the
  diversity; the LightGBM legs are near-substitutes for one another (B6+C1 disagree on 0.975%, below
  the 1.450% seed floor — two *different configs* that agree more than one config at two seeds).
- **Disagreement alone does not buy score.** B8+B10x is the most decorrelated pair on the board at
  2.222% and lands at −0.000002, because B8 is the weakest leg solo. Both halves of §6 are required.

So more LightGBM tuning cannot open the ensemble axis — only a different family can. That is what
run `D1` (CatBoost) tests, and it is the *reason* it is worth a kernel cycle.

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
