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

### OOF↔LB calibration (n=10)

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
| `b171c0cf` | blend, 2 legs | 0.966902 | 0.96831 | +0.001408 |
| `9e165a90` | blend, 3 families | 0.967197 | 0.96847 | +0.001273 |
| **`facfe8de`** | **blend, 4 families (champion)** | **0.967451** | **0.96865** | +0.001199 |

**Spearman(OOF, LB) = +0.973, residual σ = 0.000131.**

§4's ~10 paired runs now exist, so the gate stops being provisional. **The gate stays at +0.0002**,
which is 1.65σ of the measured residual rather than the 1σ §4 suggests as a floor — because the n=9
rank inversion showed OOF and LB disagreeing on ordering at ΔOOF ≈ 0.00025, i.e. *above* 1σ. A 1σ gate
would have shipped that inversion. Offset is positive because an OOF model sees 4/5 of the data while
a submission averages all 5 fold-models.

#### A hypothesis about blends, explicitly not a finding

The two blends carry the two smallest offsets on the board, and the ranges do not overlap:

| | n | mean offset | range |
|---|---|---|---|
| single models | 8 | +0.001646 | +0.001416 … +0.001805 |
| blends | 3 | +0.001293 | +0.001199 … +0.001408 |

Consistent with it, the 3-family blend's OOF gain of +0.000295 converted to +0.00016 on the LB, and
the 4-family blend's +0.000254 converted to +0.00018. A mechanism is available — part of what blending
cancels is *OOF-specific* noise, which has nothing to cancel on the test set — and if it is real,
**blend OOF gains should be discounted before being compared to the gate.**

**Still a hypothesis, still not being acted on.** The near-identical claim about target-encoded models
was made at n=1 earlier in this file and refuted by the next run.

**Falsification test, first of three: SURVIVED.** The pre-registered rule was that any blend landing
above +0.001416 (the lowest single-model offset) strikes this subsection. `facfe8de` landed at
+0.001199 — below the range, and the lowest offset yet. Two more blends to go before this graduates
from hypothesis to finding. Note also that the blend offsets are *monotonically decreasing* with the
number of legs (2 legs +0.001408, 3 +0.001273, 4 +0.001199), which the noise explanation does not
predict — but three points and an ordering is exactly the kind of pattern that fooled the last claim.

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

### D1 CatBoost — the first leg to break the tradeoff

Run `7c1e9334`, 2.2 hours on a kernel (impossible locally), `best_iter` 3320–4552 against the 8000 cap.

| leg | solo OOF | disagreement vs XGB |
|---|---|---|
| B6 LightGBM lr 0.05 | 0.966384 | 1.931% |
| C3 XGBoost depth 8 | 0.966633 | 1.702% |
| C1 LightGBM lr 0.02 | 0.966604 | 1.843% |
| **D1 CatBoost** | **0.966890** | **2.063%** |

**Strongest solo *and* most decorrelated.** Every previous leg traded one for the other; C1 bought
+0.000220 of strength at the cost of diversity and moved the blend +0.000073. The 2.063% threshold was
written into the run's notes *before* it ran, as the condition for the family change to have meant
anything.

D1's solo OOF of 0.966890 alone nearly equals the entire two-leg champion **blend** (0.966902).

All 25 CatBoost-containing blends cleared +0.0002, spanning +0.00027…+0.00038 — a spread that is
noise, so taking the argmax would be precisely the §5 selection bias. The shipped blend was picked by
a **rule stated before looking**: add the new family to the existing champion, one leg per family,
equal weights (which fit nothing at all). Result `9e165a90`: +0.000295 full-OOF, positive on all five
folds within +0.000276…+0.000331, **LB 0.96831 → 0.96847**.

### E1 MLP — decorrelation without strength, and it still pays

A fourth family (`2fb89920`), PyTorch, GPU, ~6 minutes. It is the **weakest leg on the board solo**
and by far the **most decorrelated**:

| leg | solo OOF | disagreement vs XGB |
|---|---|---|
| E1 MLP | **0.965935** (weakest) | **3.165%** (highest) |
| D1 CatBoost | 0.966890 (strongest) | 2.063% |
| B6 LightGBM | 0.966384 | 1.931% |

Both halves were pre-registered: disagreement above D1's 2.063%, solo *below* the trees, judged on the
blend and not on solo. A neural net shares no inductive bias with a tree — smooth global function
versus axis-aligned partition — and the 2.8–3.2% spread against every tree leg is that showing up in
the predictions. It is also the first model here that needs scaling and imputation, both fit on the
training fold only, under the same rule as the target encoder.

Shipped by the same stated rule: `facfe8de`, four families at equal weight, +0.000254 full-OOF,
positive on all five folds (+0.000197…+0.000307), **LB 0.96847 → 0.96865**. Again not the argmax —
dropping B6 scored marginally higher and was not taken.

**This is the clearest statement of §6 in the whole competition.** B8 was decorrelated but weak and
its blend was worth −0.000002. C1 was strong but correlated and its blend was worth +0.000073. The MLP
is *extremely* decorrelated and merely adequate, and it pays. Decorrelation is the scarce quantity
here; solo strength is not.

## The unsupervised categorical lever is closed (D2, D2b)

The EDA's central finding was that `notifications_per_day` is a **lookup table, not a function** —
per-value positive rates spanning 0.119…0.986 with Pearson(value, rate) = −0.044 and only 47.8% of
consecutive steps increasing. That explains its 0.492 solo AUC alongside its 1st place in split usage,
and it is why B1's ratio feature failed: a ratio is a monotone transform and cannot represent a lookup.

If values are labels rather than magnitudes, declaring them **categorical** is the unsupervised lever
on that structure. Both scopes were tried and both lost badly:

| run | scope | OOF | vs B6 | mean `best_iter` |
|---|---|---|---|---|
| B6 | numerics stay numeric | 0.966384 | — | 1860 |
| `D2` | all 9 numerics categorical | 0.962343 | **−0.004041** | 450 |
| `D2b` | only the 2 lookup columns | 0.964549 | **−0.001835** | 503 |

D2's failure had an obvious diagnosis, and acting on it was still wrong. Seven of the nine columns are
strongly **monotone** — `daily_screen_time_hours` scores 0.8896 solo across 1389 levels,
`weekend_screen_time` 0.8810 across 1437 — so categorifying them discards real ordinal signal *and*
hands LightGBM a 1400-level feature to overfit. D2b therefore restricted the treatment to the only two
columns the lookup-table property was ever measured on, and **pre-registered the prediction that
`best_iter` would recover to ~1800**. It did not: 503. The prediction failed and the score agreed.

**Mechanism:** LightGBM's categorical splitter sorts levels by their per-level gradient statistics on
the training fold and cuts — that *is* target encoding, fitted with no nesting and no smoothing. At
166 and 231 levels it overfits, which is exactly what the collapsed `best_iter` reports. B6 already
extracts this structure the safe way, with inner-OOF nesting and a smoothing prior, for +0.003. There
was never room for both.

The cheap LightGBM screen is the point: two runs, seven minutes, no submissions, and it retired a
CatBoost variant that would have cost a multi-hour kernel cycle to reach the same answer.

## A knob that improves the instrument without improving the score

Run `5daf6c12` (C2) raised the target encoder's inner folds 5 → 20. Its pre-registered mechanism
prediction was that `best_iter` should *rise*, because training rows would be fed a cleaner version of
the feature. Both predicted effects landed, and the score did not:

| | B6 (5 inner folds) | C2 (20 inner folds) |
|---|---|---|
| OOF AUC | 0.966384 | 0.966446 (**+0.000062 — misses the gate**) |
| mean `best_iter` | 1860 | 2286 |
| sd of `best_iter` | 462 | **148** |
| range of `best_iter` | 1453 | **382** |

The mismatch it targets is real: training rows are encoded from an inner fit on 80% of the fold, val
and test rows from a fit on 100% of it. Closing that gap to 95% demonstrably cleans the feature —
23% more trees before overfitting — but buys **+0.000062, at the seed-noise floor of 0.000066**. So
the mismatch was never costing much score at 5 folds.

**The useful result is the third row.** Early-stopping scatter is what wrecked probe B1's
interpretation, where one fold's `best_iter` collapsed 3000 → 1055 and produced ~0.001 of per-fold
movement against a 0.0002 gate. C2 cuts that scatter by a factor of three. It is worth adopting as a
default **as a measurement-noise reduction, not as a score claim** — the distinction matters, and the
gate correctly rejects it on score.

## Hyperparameter tuning (C4) — the biggest solo win since target encoding, worth nothing

Optuna, 30 trials, 250k stratified subsample, 3-fold, ROC AUC objective, **on the local GPU** — the
target encoder re-fit inside every fold of every trial, because §2 names HPO as a usage site. XGBoost
was the target: the blend screen showed it is load-bearing, and it was the least-tuned leg.

The search itself looked weak — span 0.963503…0.964603, best only +0.000184 over trial 0, and it
returned `max_depth: 6`, our existing default, independently reproducing C3's finding that depth is
not the lever. On the frozen 5-fold it was anything but weak:

| | solo OOF | vs B10x | blend Δ vs champion |
|---|---|---|---|
| B10x XGBoost (untuned) | 0.966632 | — | — |
| **C4 XGBoost (Optuna)** | **0.967176** | **+0.000544** | **+0.000068 — miss** |

**+0.000544 solo, 2.7× the gate, the largest single-leg gain since target encoding — and the blend
moves +0.000068.** `best_iter` roughly doubled (901 → 2008), which is the subsample/colsample
regularisation buying depth of fitting. Swapping it in, keeping both XGBoost legs, and a five-leg
variant were all tried: +0.000068, +0.000085, +0.000183. **None clears. Nothing shipped.**

This is C1's lesson again and much sharper, because the solo gain is 2.5× larger and the blend gain is
smaller. What the blend already had, it already had.

*Protocol caveat, stated because it cuts against the result being even better:* the search subsample
overlaps the frozen OOF rows, so the search score is optimistic. Only the frozen 5-fold run counts,
which is the number above.

## E2 — a decorrelation record that buys nothing

The regularised answer to D2. A tree *split* on a 166/231-level lookup column overfits; an
**embedding** is the same per-value lookup fitted under weight decay. Mechanism prediction: higher
disagreement than E1. Confirmed — and it set the record:

| pair | disagreement |
|---|---|
| E2 vs XGBoost | **3.352%** (highest measured) |
| E2 vs E1 MLP | 2.990% |
| E1 vs CatBoost | 2.827% |

The two MLPs disagree with *each other* more than E1 disagrees with CatBoost. The representation
really did change. **Blend Δ: +0.000137 — miss.**

## E3 — decorrelation that is pure weakness

Every leg up to this point used the **identical** B6 target-encoded feature set, so feature
representation had been held constant across the whole portfolio. E3 is the same MLP on **raw
features, no target encoding**:

| | solo OOF | disagreement vs trees | blend Δ |
|---|---|---|---|
| E1 MLP (TE features) | 0.965935 | 2.827–3.165% | +0.000254 → shipped |
| **E3 MLP (raw features)** | **0.940496** | **10.165%** | **−0.000647** |

Target encoding is worth ~0.025 to a neural net, against ~0.003 to a tree — the net has no mechanism
for learning a non-monotone value→rate lookup from a raw column, which is the same structural fact
that drove B6 and D2.

E3 triples the decorrelation record and **damages the blend at every weight tried** (1, ½, ¼ — all
negative). This is §6's second warning measured on our own data: *"high disagreement is usually
weakness, not diversity — the disagreement was the model being wrong in new places."* Disagreement is
necessary and nowhere near sufficient.

## The wall (§6)

| | |
|---|---|
| **Spearman(strength, decorrelation), 11 legs** | **−0.645, p = 0.032** |
| S6E7 at the closed wall | −0.84 over 38 learners |
| all 11 legs, equal weight | 0.967507 — **+0.000056** vs the 4-leg champion |

Earlier in this session the same test gave −0.583 at p=0.099. It is now significant and heading toward
S6E7's number. The "everything" blend confirms §5 from the other side: eleven legs beat four by
+0.000056, which is nothing.

**The §8 signal is the sequence, not any single probe:**

| probe | blend Δ | |
|---|---|---|
| D1 CatBoost → 3 families | +0.000295 | shipped, LB 0.96847 |
| E1 MLP → 4 families | +0.000254 | shipped, LB 0.96865 |
| E2 embedding MLP | +0.000137 | miss |
| C4 + E2 together | +0.000183 | miss |
| C4 Optuna XGBoost | +0.000068 | miss |

Monotonically decreasing, and the last three miss despite one of them being the strongest solo leg
ever built here and another holding the decorrelation record. Both jaws of §6's wall, in consecutive
probes.

### How far that claim actually reaches — a correction

The −0.645 was computed over 11 legs that are **three tree libraries plus one MLP shape, every one of
them on the same target-encoded feature set**. Architecture varied; representation did not; and the
neural side was a single design. S6E7's −0.84 came from 38 learners spanning genuinely different
architectures. **Describing that measurement as "the wall" overstated it** — the defensible claim is
that *the tree axis plus one MLP shape* is exhausted, which is a much narrower statement.

Being explicit about what has **not** been tried, since the gap is the point:

| architecture | status |
|---|---|
| RealMLP (pytabkit) | implemented, running — PLR numeric embeddings, own scaling pipeline and schedule |
| FT-Transformer (rtdl) | implemented, queued |
| TabTransformer | implemented, queued |
| NODE | not implemented — no maintained package, would be from scratch |
| TabICL | **not applicable**: in-context learning needs the training set as context and targets thousands of rows, not 691k. A subsampled context makes it E3's failure mode by construction — weak, and decorrelated for the wrong reason |

E3 already complicates the picture in both directions: varying the feature set *does* produce
decorrelation far beyond anything architecture variation reached (10.2% vs 3.4%), and that
decorrelation was worthless. So the wall may well be real — but it had not been tested where it
mattered when the claim was made.

### F1 RealMLP — the test the claim needed, and it holds

Run `53c678f6`. A real fifth family, not an MLP reshape: PLR periodic-linear numeric embeddings, its
own scaling pipeline and LR schedule. It lands **in the useful quadrant** — the thing §6 says should
not exist once the axis is closed:

| | solo OOF | disagreement vs trees |
|---|---|---|
| F1 RealMLP | **0.966135** | 2.685 – 3.037% |
| E1 our MLP | 0.965935 | 2.827 – 3.165% |
| B6 LightGBM | 0.966384 | — |

Stronger than our own MLP and decorrelated from every tree. And the blend:

| blend | Δ vs champion | |
|---|---|---|
| champion + F1 (5 legs) | +0.000046 | miss |
| swap E1 → F1 | −0.000005 | miss |
| champion + F1 + E2 (6 legs) | +0.000154 | miss |

**None clears.** The diagnostic number is `F1 vs E1 = 1.884%` — the two neural legs disagree with each
other *less* than either disagrees with the trees, despite being independently designed
architectures. The neural family contributes one direction of disagreement, and E1 already spent it.

This is the strongest available evidence that the wall is real rather than an artefact of narrow
sampling: a properly different, properly strong architecture was added exactly where the theory says
a gain should appear, and it produced +0.000046.

The closest miss, `B6+C4x+D1cat+E1mlp+E2emb` at +0.000183, is **not** being shipped. It is 0.9× the
gate, the gate was pre-registered, and moving it now to admit the run that just missed it is precisely
what a pre-registered gate exists to prevent.

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
