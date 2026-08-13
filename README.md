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

### OOF↔LB calibration (n=13)

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
| `facfe8de` | blend, 4 families | 0.967451 | 0.96865 | +0.001199 |
| **`ffb65555`** | **nested-selected 5-leg — FINAL A** | **0.967733** | **0.96892** | +0.001187 |
| `2b54a858` | 13-leg no-selection — FINAL B | 0.967567 | 0.96873 | +0.001163 |

**Spearman(OOF, LB) = +0.984, residual σ = 0.000135.**

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
| blends | 5 | +0.001246 | +0.001163 … +0.001408 |

Consistent with it, the 3-family blend's OOF gain of +0.000295 converted to +0.00016 on the LB, and
the 4-family blend's +0.000254 converted to +0.00018. A mechanism is available — part of what blending
cancels is *OOF-specific* noise, which has nothing to cancel on the test set — and if it is real,
**blend OOF gains should be discounted before being compared to the gate.**

**Still a hypothesis, still not being acted on.** The near-identical claim about target-encoded models
was made at n=1 earlier in this file and refuted by the next run.

**All three falsification tests passed — this graduates from hypothesis to finding.** The
pre-registered rule was that any blend landing above +0.001416 (the lowest single-model offset)
strikes the subsection. Five blends have now landed at +0.001408, +0.001273, +0.001199, +0.001187,
+0.001163 — every one below, and **monotonically decreasing in the number of legs** (2, 3, 4, 5, 13).
The single-model and blend ranges still do not overlap at n=13.

**Operational consequence, now licensed:** a blend's OOF gain does not transfer to the LB at par.
Observed conversions: +0.000295 OOF → +0.00016 LB; +0.000254 → +0.00018; +0.000282 → +0.00027.
Discount blend OOF gains before comparing them to the gate.

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

### G1 FT-Transformer — three architectures, one direction

Run `b2f35b4a`. Attention over feature tokens: a third inductive bias, neither partitioning nor dense
mixture. Solo 0.965498, competitive. The pre-registered test was whether it would disagree with the
*existing neural legs* at ~2% (direction spent) or 3%+ (axis open).

| G1 vs | disagreement |
|---|---|
| F1 RealMLP | **2.208%** |
| E1 MLP | **2.429%** |
| B10x XGBoost | 3.157% |
| B6 LightGBM | 2.879% |
| D1 CatBoost | 2.870% |

**Every neural pair is closer to every other neural leg than any of them is to a tree.** Three
independently designed architectures — dense mixture, PLR embeddings, feature-token attention —
converge on the same alternative to tree partitioning. E1 spent that direction; F1 and G1 bought
strength, not diversity. Adding G1 to the champion: **−0.000019**.

### The architecture sweep, complete picture

| leg | architecture | solo OOF | vs other neural legs | blend Δ |
|---|---|---|---|---|
| E1 | MLP | 0.965935 | — | +0.000254 → shipped |
| E2 | MLP + embeddings | 0.965702 | 2.990% | +0.000137 |
| F1 | RealMLP (PLR) | 0.966135 | 1.884% | +0.000046 |
| G1 | FT-Transformer | 0.965498 | 2.208 – 2.429% | −0.000019 |
| E3 | MLP, raw features | 0.940496 | 9.571% | −0.000647 |

Wall test over 8 legs: **Spearman −0.738, p = 0.037** (was −0.583 ns at 9 tree-ish legs, then −0.645).
S6E7's closed wall was −0.84.

**Not attempted, and why:** NODE has no maintained package and would be a from-scratch build, only
justified if something above had surprised. TabICL is *not applicable* — in-context learning needs the
training set as context and targets thousands of rows, not 691k; a subsampled context reproduces E3's
failure mode by construction, weak and decorrelated for the wrong reason.

### The 0.9× pattern

Three independent combinations have landed at **+0.000183, +0.000154, +0.000197** — all ≈0.9× the
gate, none over. Piling on legs asymptotes just below the line. Calling +0.000197 "basically +0.0002"
is exactly what a pre-registered gate exists to prevent, and the blend-offset hypothesis predicts a
+0.0002 OOF gain converts to roughly +0.00013 on the LB anyway.

§7 gives a *separate* and legitimate reason to submit such a blend before the deadline: finals may
only be selected from entries actually submitted, and an all-legs equal-weight blend is a sound
variance-reduced **Final B** candidate — which is how S6E7's Final B was chosen, not on public LB.
That is a deadline decision, not a champion decision.

The closest miss, `B6+C4x+D1cat+E1mlp+E2emb` at +0.000183, is **not** being shipped. It is 0.9× the
gate, the gate was pre-registered, and moving it now to admit the run that just missed it is precisely
what a pre-registered gate exists to prevent.

## Nested subset selection — the wall was mis-measured twice

After three hand-picked combinations landed at ~0.9× the gate, `scripts/subset_ceiling.py` scored
**all 16,369 equal-weight subsets** of the 14 legs. Best: +0.000282. **That number alone is worthless**
— a maximum over thousands of candidates on the same OOF rows is precisely the §5 selection bias that
produced a fake +0.00014 in S6E7, and the script says so in its own docstring.

What made it worth testing rather than dismissing: the top 13 subsets *all* contained C4x, E2emb and
F1real, and none of the three was in the champion. Bias-driven maxima scatter their membership.

So `scripts/nested_subset.py` nested **the selection itself** — each outer fold chose a subset using
only the other four, then was scored on the fold the chooser never saw:

| | |
|---|---|
| champion, held out | 0.967451 |
| selected subset, held out | **0.967733** |
| **delta** | **+0.000282** — clears |
| per-fold | +0.000245, +0.000374, +0.000195, +0.000247, +0.000342 |
| folds positive | **5/5** |
| distinct subsets chosen | **1** |

**All five folds chose the identical subset from disjoint selection data**, so the nested and full-OOF
numbers coincide — the procedure is deterministic under resampling, which selection bias cannot be.

Shipped as `ffb65555` — `B10x + C4x + D1cat + E2emb + F1real`, equal weights. **LB 0.96865 → 0.96892.**

### Correction: C4 was not worth nothing

This file previously recorded C4 (Optuna XGBoost) as "the biggest solo win since target encoding,
worth nothing." That was wrong, and wrong in a specific, repeatable way: **I tested one pairing** —
swapping C4 into the existing champion — measured +0.000068, and generalised from a single
configuration to the whole lever. C4 appears in every subset the nested procedure selected.

Every change the selected subset makes to the champion swaps a weak leg for a strong one: C4x added,
B6 (weakest tree) dropped, E1 replaced by E2emb and F1real (the two stronger neural legs). That is a
mechanism, and it is the one the disagreement table already implied — the neural direction has a
single axis, so its *strongest* representatives should hold it, not its first.

**This is the second time in this competition a "closed" verdict came from testing one configuration
instead of the space.** The first was declaring the ensemble axis closed over three tree libraries and
one MLP shape, before RealMLP, FT-Transformer or any feature-set variation had been run. Both times
the lever was real and the test was too narrow. The pattern to distrust is not the gate — the gate has
been right every time — it is concluding *"X does not help"* from *"X did not help in the one place I
put it."*

## Running neural legs on Kaggle GPU

Four architectures were run as dedicated GPU kernels (`s6e8-realmlp`, `s6e8-ftt`, `s6e8-tabtf`,
`s6e8-node`), one notebook each with its own pinned `KAGGLE_CFG`, so they run concurrently rather
than serially.

### The P100 trap — read this before adding a GPU kernel

The first push errored on all three within minutes:

```
CUDA error: no kernel image is available for execution on the device
torch 2.10.0+cu128   Tesla P100-PCIE-16GB
```

`enable_gpu: true` in `kernel-metadata.json` gets you a **Tesla P100** (compute capability 6.0), and
the Kaggle image ships **torch 2.10.0+cu128**, whose builds no longer contain Pascal kernels. Kaggle's
default GPU cannot run PyTorch on its own image.

**Fix:** `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`. Valid values are
`NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`. This flag exists **only on the push command** — there
is no `kernel-metadata.json` equivalent — so `enable_gpu: true` alone silently allocates an unusable
machine. Verified this was *not* our `pip install` upgrading torch: the logs say `(nothing
reinstalled)`.

Second trap: `kaggle kernels output` hit a `charmap` encoding error writing the log on a Windows
console and left a **0-byte file**, which looks exactly like "the kernel died before starting". Run it
with `PYTHONUTF8=1` to get the real traceback.

### Local ↔ Kaggle reproducibility

| model | local (RTX 3060, 6 GB) | Kaggle T4 (16 GB) | Δ | runtime |
|---|---|---|---|---|
| RealMLP | 0.966135 | 0.966116 | −0.000019 | 3870 s → 1664 s |
| FT-Transformer | 0.965498 | 0.965436 | −0.000062 | 2176 s → 1662 s |
| TabTransformer | 0.954146 | 0.953339 | −0.000807 | 18351 s → 2432 s |

The first two deltas are inside the 0.000066 seed-noise floor, so the two hardware paths agree.

### TabTransformer: capacity was never the problem

Locally it OOM'd on 6 GB and had to run at batch 1024 / d_token 32, recorded at the time as *"not a
like-for-like comparison with FT-Transformer, do not read the number"*. The T4 ran it at the full
batch 4096 / d_token 64 — **4× the batch and 2× the token width moved it −0.000807**, i.e. slightly
*worse* and negligible against a 0.012 deficit. **The caveat resolves in favour of the original
number.** FT-Transformer at identical full capacity scores 0.965436, so the gap is architectural:
TabTransformer attends over *categorical* tokens only, and this dataset has 3 low-cardinality
categoricals against 18 continuous columns, so nearly all the signal bypasses the attention by
construction. Capacity cannot route signal down a path it does not traverse.

### NODE — a failed prediction worth more than the confirmation

`976c2703`, solo **0.962186**, 1.4 h on the T4. NODE is the one architecture whose inductive bias is
deliberately *tree-shaped*: ensembles of oblivious decision trees made differentiable by entmax
feature selection and soft thresholds. The pre-registered prediction was that it would therefore sit
**below 2.7% from the trees** — closer to them than any neural leg.

| leg | solo | mean disagreement vs trees |
|---|---|---|
| F1 RealMLP | 0.966135 | 2.841% |
| E1 MLP | 0.965935 | 2.986% |
| G1 FT-Transformer | 0.965498 | 2.988% |
| E2 embed-MLP | 0.965702 | 3.217% |
| **K4 NODE** | **0.962186** | **3.427%** |

**The prediction failed.** NODE is not closer to the trees — it is *further* from them than every
actual neural leg, and roughly equidistant from both camps (3.427% vs trees, 3.245% vs neural).
Making decision trees differentiable changes what they learn more than it preserves how they
partition. A tree-shaped inductive bias did not produce tree-like predictions.

### The 0.965 threshold, now predictive

Three legs below ~0.965 solo have each damaged the blend, with the damage ordered by the deficit:

| leg | solo | best blend contribution |
|---|---|---|
| K4 NODE | 0.962186 | −0.000353 |
| G2 TabTransformer | 0.953339 | −0.000517 |
| E3 raw-feature MLP | 0.940496 | −0.000647 |

Every one is monotone toward zero as its weight shrinks — **optimal weight zero**. The threshold
correctly forecast NODE's sign *before* the blend was computed, so it is now a screening rule and not
a post-hoc description: a leg below ~0.965 solo does not enter the pool, whatever its disagreement.

**TabICL remains the one architecture not run**, and it is inapplicable rather than skipped:
in-context learning needs the training set *as context* and targets thousands of rows, not 691k. A
subsampled context reproduces E3's failure mode by construction — weak, and decorrelated for the
wrong reason.

## Final submission selection (§7)

Both finals are submitted, so both are selectable at the deadline (2026-08-31).

| | run | construction | OOF | LB |
|---|---|---|---|---|
| **Final A** | `7f69fcf6` | 23 legs, gated pool, `C` selected on the grid | **0.969434** | **0.97060** |
| **Final B** | `c21b8df8` | 25 legs, **fixed rule**, `C` pinned at 0.1 | 0.969425 | 0.97058 |

**Final A** is the honest champion: best OOF *and* best LB, so the two selection criteria agree and
there is nothing to trade off. (It beats the 22-leg `0a3c852e` at 0.97056 by 0.00004 against a
resolution of 0.00014 — those two are tied, and the tie is broken on OOF.)

**Final B** is the variance-reduced twin, and it is *worse* on both axes — −0.000009 OOF and
−0.00002 LB. That is the point, and it is the same shape as S6E7's Final B, which scored 0.00010
worse and was correct. Every step fitted on data is removed:

- **No admission gate.** Final A's 23 legs are the survivors of a per-probe +0.0002 gate — a
  procedure run on data, with variance on an unseen split. Final B applies one threshold, solo
  OOF ≥ 0.965, to every leg we own, and takes whatever passes.
- **No grid search.** `C` is pinned at 0.1 rather than chosen from `C_GRID`.
- **Membership is mechanical.** `stack_logit.fixed_rule_legs()` derives the 25 run_ids from
  `runs.csv` plus the `experiments/preds/` inventory; nothing is hand-typed, and re-running
  reproduces the list exactly.

**What the rule changes, in both directions.** It *admits* three legs the gates rejected — `N1`
`fe_normalization` (0.968099), `N2` `fe_interaction` (0.968162), `N3` `num_as_cat` (0.966291), the
Phase 3 cheap round, all of which cleared solo and missed the stack gate. A fixed rule cannot honour
a gate, because the gate is exactly the fitted step Final B exists to remove. It *drops* one:
`K4 NODE` at 0.962186, below the threshold — and NODE took the third-largest |weight| in the fitted
16-leg stack, so that is the expensive half. In the event the two roughly cancelled: the three
admitted probes take small weights (+0.3181, −0.2446, +0.1977), matching Phase 3's measurement that
they contribute nothing. 12 of 25 weights are negative, against 10 of 23 in Final A.

**Phase 5 is why this hedge is better motivated than the previous one.** The CV→LB offset decays as
legs are added, which means the fitted meta-model's OOF grows more optimistic with member count, and
`honest_oof` cannot remove it because every leg's OOF came from this same partition. Final A is the
larger fitted selection; Final B pins what it can and admits by rule.

**Superseded:** the original finals were `ffb65555` (5 legs, nested-selected, 0.967733 / 0.96892)
and `2b54a858` (13 legs, no selection step, 0.967567 / 0.96873), both equal-weight means from before
the fitted stack existed and both ~0.0017 LB below what is above. That pair's Final B cost −0.00019;
this one costs −0.00002.

> **A deviation found while rewriting this section, recorded rather than fixed.** The old Final B
> deliberately excluded `K1 RealMLP` (`cd2dbce4`) and `K2 FT-Transformer` (`46859c68`), the Kaggle-T4
> reruns, on the grounds that they are duplicate *recipes* of `F1real` and `G1ftt` and averaging them
> is seed-bagging at the blend level. **Both pairs are in both current pools** — they entered when
> the Phase 2 stack pool was assembled, and nobody noticed the earlier principle had lapsed. The
> fitted weights show what the stacker does with them: `F1real` +0.1469 against `K1` −0.1096, and
> `K2` −0.1819 against `G1ftt` +0.0229 — near-cancelling opposite signs, i.e. the meta-model fitting
> the *difference* between two runs of one recipe, which is seed noise. It has not measurably cost
> anything (Final A is the LB champion), and removing them now would itself be a selection step made
> after seeing scores, which is exactly what Final B exists to avoid. Flagged, not repaired.

## Public-frontier audit (2026-08-07)

Read the top public notebooks to find what we had not tried. Public LB then topped out at 0.97086
against our 0.96892. Three of their findings are directly actionable and one of them is measured
below; the rest of the audit is recorded because knowing what the frontier tried and rejected is
worth as much as the wins.

**Their CV→LB offset is +0.0012, and so is ours** (0.967733 → 0.96892 = +0.00119). Two independent
pipelines agreeing on the offset to the fifth decimal means our §4 calibration is sound and their
OOF numbers are directly comparable to ours. It also means the public 0.97086 corresponds to roughly
0.9697 OOF — about **+0.0019 above our champion**, so the gap is real signal, not LB noise.

### 1. We have only ever built equal-weight averages — and that is why NODE looked useless

`make_blend.py`, `subset_ceiling.py` and `nested_subset.py` all search over *which* legs, never over
*how much*. Both top public notebooks instead fit a logistic regression on the OOF matrix **in logit
space**, and both report negative coefficients. `scripts/logit_stack.py` tests that on our own 16
legs, meta-model refit inside each outer fold:

| combiner | OOF | vs champion |
|---|---|---|
| best single leg (`C4x`) | 0.967176 | −0.000557 |
| equal-weight over all 16 legs | 0.967430 | −0.000303 |
| **equal-weight champion, 5 legs** | **0.967733** | — |
| logistic stack on **probabilities** | 0.967760 | +0.000027 |
| **logistic stack on logits** | **0.967976** | **+0.000243** ✅ |

Flat in `C` (0.03/0.1/1.0 differ by 3e-6), so it is not a knife-edge fit. Two mechanisms, and the
second is the one that matters:

- **Logits, not probabilities.** Probabilities buy +0.000027, logits +0.000243 — nearly ten times as
  much. This target saturates (the top screen-time decile is ~100% positive), so near p=1 the
  probability scale has no resolution left while log(p/(1−p)) still does.
- **A fitted combiner can subtract.** 4 of 16 weights are negative, and **`K4node` takes the third
  largest weight in the whole stack, at −0.796.** That reverses this repo's conclusion about NODE.
  The §6 finding that legs below ~0.965 solo "damage the blend" is a property of the *combiner*, not
  of the legs: an average can only dilute a weak-but-differently-wrong model, so it got thrown out of
  all 65,519 subsets. Given a sign, it is load-bearing. Same for `B6`, `B8`, `C1`.

This is the third instance of the pattern already named in this file — concluding *"X does not help"*
from *"X did not help in the one place I put it."* The 0.965 screening rule should now read *"below
0.965 a leg cannot help an equal-weight mean"*, which is a much narrower claim.

Caveat, shared with every public stack: the base legs' OOF come from models fitted on this same
partition, so a meta-model training on folds 1–4 consumes predictions from base models that saw
fold 0. That inflates all stack rows above by a small constant. It does not affect the comparison
between two combiners on identical inputs, which is what the script exists for.

### 2. The budget constraint was never tested on the target-encoded feature set

`daily_screen_time_hours ≥ social + gaming + work` holds in 100.00000% of all 859,029 train+test
rows (min gap exactly 0.000, 546 rows on the boundary) — a generator invariant, and the remainder is
a real latent variable at solo AUC 0.765. `fe_composition` exists in the generator (`build_model_nb.py:64`)
and computes exactly this. **The `engineered` column is empty on all five champion legs.** Only B1
(the ratio family) was ever run; it failed, we moved to target encoding at B6, and the composition
family was never revisited on top of it.

tamerlanomralinov ablates it at 10-fold: **+0.00096 per fold**, matching its 3-fold measurement of
+0.00092. That is ~5× our gate. His stated mechanism is that a 4-term linear combination is something
axis-aligned splits cannot build *at any data volume* — which is why it survives more folds, while
his log-ratio features gave +0.00036 at 3-fold and +0.00001 at 10-fold. **Validate FE at the fold
count you will submit with.**

### 3. Imputation as extra columns, never as a replacement

tomasa2 measures the same XGB imputer with opposite signs: imputed values *replacing* NaNs is
negative, imputed values *alongside* the retained NaN columns is +0.0012. A NaN-native GBM learns a
default split direction per node, which is strictly more expressive than one imputed point estimate;
but the imputed column restores ratio-feature coverage on the 39% of rows where some input is
missing. We currently do neither.

### Measured dead ends on the public frontier — do not re-run these

| idea | reported effect |
|---|---|
| concatenating the original `jayjoshi37` dataset | −0.0001 (also true here: 7,500 rows ≈ 1.1%) |
| NA-indicator features / `n_missing` | ≈ 0 — missingness is MCAR (target rate 0.708/0.710/0.711 by regime) |
| pairwise / multi-resolution / adaptive-smoothing TE | −0.0004 to +0.00002; plain single-column TE at smoothing 10 is the optimum |
| `age` as a categorical | −0.0006 |
| naive mean or rank-average of a large library | −0.0012 |
| tree depth 9–13 | to −0.0011 |
| seed-averaging a member to strengthen it | +0.00015 solo, **+0.000000 to the blend** (corr 0.9994) |
| record linkage / duplicate rows | none exist — 1 duplicate pair in 565,846 rows |

The last row but one independently confirms the standing no-seed-bagging constraint, with a number.

### The architecture we do not have

`lookup` (tamerlanomralinov's Lookup-Transformer) is the single load-bearing member of the 74-model
public stack: dropping it costs −0.000106, **6.6× the next member**, and 73 of 74 members cost less
than 0.00002 to drop. Its premise is that the generator memorised value→label associations, so each
column's *exact value* is a lookup key rather than a magnitude — `notifications_per_day` has
univariate AUC 0.492 yet its per-value residuals correlate 0.72 across two independent halves of the
data. It gives every exact observed value its own 128-d embedding, adds a periodic-linear (learned
Fourier) numeric embedding for the smooth trend, gives NaN its own learned per-column vector, and
runs attention over feature tokens.

That premise is the same one behind our B6 target encoding, which is our largest single win — TE
collapses a lattice cell to one scalar fitted in closed form; the embedding learns it end to end.
Its value is decorrelation, not strength: 0.96853 solo (below our `C4x`) but max correlation 0.9869
against a pack that correlates 0.986–0.999 internally. This is the counter-example to our own
neural-axis finding: MLP → embedding-MLP → RealMLP → FT-Transformer → NODE all bought strength and no
diversity, because they read the same *representation*. Reading the same structure through a
different *mechanism* is what pays.

### A note on the public OOF library

`szymonkapiski/s6e8-oof-library-47-models` (now 74 members) publishes aligned OOF+test predictions on
`StratifiedKFold(5, shuffle=True, random_state=42)` — **byte-identical to our frozen split**, so it
would stack against our legs with no realignment. Using it is permitted and is what the top of the
public LB is built from, but it means shipping other people's models, which is a strategy call rather
than a technical one and is not taken here unilaterally. His warning applies to any such reuse and is
worth repeating: most public OOF is on a *different* split (7-fold, 10-fold, or seed-averaged across
three partitions), all of it looks healthy, and **no distributional statistic detects the mismatch** —
a deliberately leaked model inflates AUC by 0.0054 while moving a KS statistic by 0.0002. Read the
code for `n_splits` and `random_state`, and check whether `random_state` is a loop variable.

## Phase 2 — implementing the audit findings

### B1 — the fitted stacker ships, and the optimism worry was wrong

`scripts/stack_logit.py`, run `cdbffba5`, 16 legs, logit space, meta-model refit inside each
outer fold, `C=0.1` chosen from two adjacent points on a plateau flat to 1e-6.

| | OOF | LB | offset |
|---|---|---|---|
| equal-weight champion `ffb65555` | 0.967733 | 0.96892 | +0.00119 |
| **logit stack `cdbffba5`** | **0.967976** | **0.96928** | **+0.00130** |
| delta | +0.000243 | **+0.00036** | |

**The pre-registered prediction was LB 0.96917 and it came in at 0.96928** — 0.00011 *above*.
That was the whole point of submitting: the stack's OOF carries a meta-fit optimism the
equal-weight champion does not (the base legs' OOF come from models fitted on this same
partition, so a meta-model training on folds 1–4 consumes predictions from base models that saw
fold 0). If that optimism were material the LB gain would have come in *below* the OOF gain.
It came in **larger** — +0.00036 on the LB against +0.000243 on OOF. The falsification test was
pre-registered at "below ~0.96900 demotes stacking"; it passed with room.

Stacking is now the shipping combiner. **New best LB: 0.96928.**

Fitted weights, and why this matters more than the number:

| leg | learner | solo | weight |
|---|---|---|---|
| `fdeaa047` | XGBoost (Optuna) | 0.9672 | +1.692 |
| `8bd89dee` | embedding-MLP | 0.9657 | +1.180 |
| **`976c2703`** | **NODE** | **0.9622** | **−0.792** |
| `7c1e9334` | CatBoost | 0.9669 | +0.665 |
| … | | | |
| `f64e2781` | LightGBM (B6) | 0.9664 | −0.288 |

**NODE takes the third-largest weight in the stack, negative.** Four of sixteen weights are
negative. The 65,519-subset equal-weight scan put NODE in exactly zero of its top 25 — correctly,
*for an average*, which can only add and so can only dilute a weak-but-differently-wrong model.
§6's "legs below ~0.965 solo damage the blend" is a property of the **combiner**, not of the legs,
and should now be read as *"below 0.965 a leg cannot help an equal-weight mean."*

### B2 — the budget constraint, the largest feature win since target encoding

`daily_screen_time_hours ≥ social + gaming + work` holds in **100.00000%** of all 859,029
train+test rows, minimum gap exactly 0.000. `fe_composition` has existed in the generator since
Phase 1, but was only ever run in the **pre-target-encoding** B1 era, where the ratio family
failed. `engineered` was empty on all five champion legs. Rerun as strict twins on the
target-encoded feature set:

| engine | baseline | + composition | delta |
|---|---|---|---|
| XGBoost (C4 `fdeaa047`) | 0.967176 | **0.968180** `f047412f` | **+0.001004** |
| LightGBM (B6 `f64e2781`) | 0.966384 | **0.967547** `7d1de67e` | **+0.001163** |
| CatBoost (D1 `7c1e9334`) | 0.966890 | *replication arm running* | |

Five to six times the gate, and within 0.00005 of the public 10-fold ablation of the same
feature (+0.00096). **Fold-count independence is the mechanism's own prediction** — a 4-term
linear combination is not something axis-aligned splits can build at *any* data volume — and it
is what separates this from the same author's log-ratio features, which collapse from +0.00036
at 3 folds to +0.00001 at 10 as the trees learn those relationships from splits by themselves.

Fourth instance of the pattern this file already names: concluding *"X does not help"* from
*"X did not help in the one place I put it."*

### L1 — the Lookup-Transformer breaks the §6 wall

Pre-registered on **decorrelation, not strength**. It passed both halves, and the second is the
interesting one:

| | measured | gate |
|---|---|---|
| solo OOF | **0.968626** — our strongest leg ever | ≥ 0.9655 |
| max logit-space \|corr\| vs the other 18 legs | **0.9796** | ≤ 0.990 |

For scale, **the other legs' own median max-correlation is 0.9941.** Its nearest neighbours are
the FT-Transformers at 0.979, not the trees.

This is the first leg here that is simultaneously the **strongest** and the **most
decorrelated**, which is the direct counter-example to §6's wall
(`Spearman(strength, decorrelation) = −0.645`). The wall was a property of legs that all read the
**same representation** — MLP, embedding-MLP, RealMLP, FT-Transformer, NODE — not a law about
tabular models. Reading the same lattice structure through a *different mechanism* buys both at
once. It takes the largest weight in the 19-leg stack, at +2.337.

> **Corrected by Phase 4.** The credit here belongs to the *representation*, not to the
> Lookup-Transformer. ModernNCA — a retrieval model sharing no inductive bias with a transformer —
> reads the same per-value representation and is worth **+0.000530** to a pool with `L1lookupt`
> removed, against `L1lookupt`'s own **+0.000520**. Either one alone buys ≈ the same thing, and
> putting both in buys only +0.00064 of a possible +0.00105.

### Stage D — the stack offset is now the most reliable instrument in the repo

| | OOF | LB | predicted LB | error |
|---|---|---|---|---|
| equal-weight champion `ffb65555` | 0.967733 | 0.96892 | — | — |
| 16-leg stack `cdbffba5` | 0.967976 | 0.96928 | 0.96917 | +0.00011 |
| 19-leg stack `6da97a58` | 0.969228 | 0.97051 | 0.97053 | −0.00002 |
| **22-leg stack `0a3c852e`** | **0.969314** | **0.97056** | 0.97060 | −0.00004 |

**Three pre-registered predictions, three hits, max error 0.00011.** The stack offset is
`+0.00128 ± 0.000029` over three points — an order of magnitude tighter than the ±0.00053 the
public bootstrap gives for a single AUC score, because both sides move together. Ten of the
twenty-two weights are negative. Public LB top was 0.97086 on 2026-08-07.

> **Corrected by the fourth point.** The offset is **not a constant** — it decays monotonically
> with pool size, and reading three points on a slope as `0.00128 ± 0.000029` was wrong. The
> fourth submission fell 3.8σ below that mean. See **§ Phase 5**; the header claim above
> ("the most reliable instrument in the repo") no longer holds in the form it was written.

The last submission is worth reading correctly: **+0.00005 over the previous one, which is below
the 0.00014 the leaderboard can resolve.** It was submitted so the OOF champion exists as a
selectable entry at the deadline, not to learn anything. Selection stays on OOF.

### Where the wall came back: two probes cleared solo and died in the stack

| probe | solo | vs its twin | stack gain | verdict |
|---|---|---|---|---|
| **M1 TabM** | 0.967160 | +0.00106 vs our best neural leg | **+0.000027** (with B2ccat) | miss |
| **B3 `fe_impute`** | 0.968425 | **+0.000245** vs B2, positive on 5/5 folds | **+0.000059** | miss |

Neither was submitted: the total move since the last submission is +0.000086, and the public
bootstrap puts the leaderboard's resolution at 0.00014, so a slot would buy a number we could
not read.

**TabM failed in an informative way.** The pre-registered falsification was "strong but a
variance-reduced twin of a leg we already own." Instead: its nearest neighbours are all **trees**
— 0.9867 with the B2 XGBoost, 0.9857 with C1, 0.9847 with C4x — and its solo score is C4x's to
four decimals. Fed the same target-encoded lattice the trees eat, a packed MLP ensemble
**re-derived the tree solution**. Publicly TabM is the strongest family after the library
author's own models; here it is a tree. The representation decides what a model becomes — the
same lesson the Lookup-Transformer taught from the other side.

**`fe_impute` was worth +0.0012 publicly and +0.000245 here**, and the gap is explicable rather
than mysterious: that measurement was against a baseline *without* the composition family, and
coverage for those derived features is the whole mechanism. B2 already bought most of it.

## Stage C — the public library as a measuring instrument (never shipped)

`scripts/public_gap.py`, read-only, on the 74-member public OOF library (verified on our exact
split). Nothing here was ever an input to a submission.

| honest per-fold logit stack | members | OOF |
|---|---|---|
| **ours only** | 21 | **0.969255** |
| public only | 74 | 0.969661 |
| union | 95 | 0.969733 |

**Our 21 legs land within 0.0004 of a 74-member public library.** Our legs contribute +0.000072
to a public-grade stack; the niche we are missing is worth **+0.000478**.

And that niche is essentially *one recipe*. Ranking public members by distance from our pack:

| member | solo | max corr vs ours | nearest leg of ours |
|---|---|---|---|
| **`naji05` / `naji03`** | **0.9688** | **0.9566** | `B2ccat` |
| `pub_tabm` | 0.9675 | 0.9687 | `B2xgb` |
| `pub_ravi` | 0.9665 | 0.9744 | `K2ftt` |
| `tabm_imp` | 0.9681 | 0.9811 | `M1tabm` |

For scale, our own legs' median max-correlation is **0.9945**. Two members sit at 0.9566 while
everything else in a 74-model library sits above 0.968 — and those two are `naji`'s, whose author
states plainly that the feature engineering and training code are private. **The remaining gap is
concentrated in a recipe nobody published.** That is a much more useful answer than "build more
models": copying the public frontier further has almost nothing left to give.

> This section originally closed by calling the redundancy in our own pack (median 0.9945) "the
> cheaper problem". **That was wrong, and Phase 3 below measures why** — max-correlation turns out
> not to predict stack contribution at all. The sentence is corrected rather than deleted because
> the reasoning that produced it is the reasoning worth not repeating.

### Durability problems found while implementing (both fixed)

- **`build_model_nb.py` was untracked**, living only in a session-scoped temp directory, while
  every notebook in the repo is generated from it. Now `scripts/build_model_nb.py`.
- **KCFG is parsed as JSON but baked into Python source.** The first config containing a boolean
  rendered `"fe_composition": true` and would have died with a `NameError` on the kernel's first
  cell. Now round-tripped through `json.loads` → `repr`, which also fails on a malformed CFG
  locally instead of on Kaggle.
- `collect_run.py` gained `--accelerator`; the P100 workaround was tribal knowledge until now.
- C4's Optuna parameters lived only in gitignored `.kaggle_output/`. They are now carried in the
  B2 row's `notes`, which is tracked.

## Phase 3 — max-correlation does not predict stack contribution

Phase 2 closed on an open worry: our 22 legs have a **median max-correlation of 0.9947**, every leg
has a near-twin, and the one leg that broke the pattern (`L1lookupt`, 0.9796) bought most of the
gain. The obvious reading was "redundancy is the constraint — reduce it cheaply and the stack
improves." Phase 3 opened by testing that reading three ways, and it does not survive.
`scripts/leg_diversity.py` reproduces all of it; nothing below was ever shipped.

| attack | result | what it rules out |
|---|---|---|
| LightGBM meta-learner over the 22 leg logits | **−0.000042** | a nonlinear combiner |
| the same, plus 9 raw features and a missing-count | **−0.000065** | feature-gated / region-switching stacking |
| tree-mean ↔ neural-mean correlation, by missingness regime | flat at **0.9785–0.9811** | heterogeneity for a gate to switch on |
| admitting the 9 Phase 1 **orphan legs**, all below 0.9947, free | **+0.000047** total | decorrelation as a goal in itself |

### The orphans, and a rule that expired without anyone noticing

Nine Phase 1 legs sit on disk with complete `oof`+`test` artifacts on the frozen partition. They were
excluded by `subset_ceiling.py`'s rule that legs below 0.965 solo are "strictly harmful" — a rule
[measured under an equal-weight mean](#1-we-have-only-ever-built-equal-weight-averages--and-that-is-why-node-looked-useless),
where a weak leg can only dilute. The fitted stack can subtract, so the rule did not carry over, and
`K4node` (0.9622 solo, third-largest |weight| in the stack) is the standing proof. Admitting them
costs nothing, so they were re-measured:

| leg | solo | max corr vs pool | contribution to the 22-leg stack |
|---|---|---|---|
| **`K3tabtf`** | 0.9533 | **0.9251** | **−0.000001** |
| `G2tabtf` | 0.9541 | 0.9311 | −0.000001 |
| `E3raw` | 0.9405 | 0.9505 | +0.000008 |
| `D2numcat` | 0.9623 | 0.9717 | **+0.000030** — the largest |
| `D2blookup` | 0.9645 | 0.9874 | +0.000015 |
| `v1anchor`, `A1seed`, `B1inter`, `A2es400` | ~0.963 | ~0.978 | −0.000001 … +0.000001 |
| **all nine together** | | | **+0.000047** |

`K3tabtf` is the finding. At max correlation **0.9251** against a pool whose median is 0.9947 it is
by a wide margin the most decorrelated thing we own, and it contributes **−0.000001** — because
0.015 AUC below the pack is noise, however uncorrelated that noise is.

**So max-correlation does not predict stack contribution, and the §6 framing needs one more turn of
the screw.** `L1lookupt` did not pay because it was decorrelated; it paid because it was
simultaneously our **strongest** leg (0.968626) *and* decorrelated. Decorrelation is necessary and
nowhere near sufficient — it is a multiplier on strength, not a substitute for it. The practical
consequence, and the reason this is written down: **"it will be different" is not a hypothesis.** A
probe justified only by expected decorrelation is now measured to be worth −0.000001.

Corollary for the seed question, priced rather than argued: `A1seed` — the one seed-only twin we own
— contributes **−0.000001**, matching the public library's measurement of a seed-averaged member at
+0.000000. The standing instruction not to spend on seed variants is correct on our own data.

### The cheap CPU round — three untried levers, three misses

Config-only strict twins, no new code, no GPU, no submissions. Each targeted *strength on a different
representation*, which is the only mechanism this pack has ever responded to. Gates pre-registered in
`notes` before the results were read: solo ≥ parent + 0.0002, **and** ≥ +0.0002 to the 22-leg stack.

| probe | run | solo | vs parent | max corr | stack Δ |
|---|---|---|---|---|---|
| **N1** `fe_normalization` | `8f40573b` | 0.968099 | −0.000081 | **0.9977** | −0.000000 |
| **N2** `fe_interaction` | `ceb7f8c0` | 0.968162 | −0.000018 | 0.9978 | +0.000001 |
| **N3** `num_as_cat` | `ad5e827a` | 0.966291 | −0.001256 | 0.9783 | +0.000029 |
| all three admitted | | | | | **+0.000032** |

- **N1 — the 24-hour budget is not a second composition.** `free_time = 24 − sleep − screen` is
  structurally identical to the screen-time constraint that cleared on all three engines at 5–6× the
  gate, and it had *more* rows to act on (80.9% coverage vs 61.0%). It still produced a model at
  **0.9977** correlation with its parent — the highest correlation anywhere in the pack, above even
  the `F1real`/`K1real` rerun pair. The composition family did not work because it was a budget; it
  worked because `other_hours` is a **latent variable at solo AUC 0.765** that the trees could not
  otherwise reach. `free_time` has no such quantity behind it.
- **N2 — `fe_interaction` did not lose because the base couldn't use it.** It lost by 0.0003 in the
  pre-target-encoding era and by 0.00002 now, on a base carrying TE, the composition family and
  Optuna params. That settles the ambiguity the probe was built for, in favour of the lookup reading:
  a ratio is a monotone transform and cannot represent a lookup table.
- **N3 is the informative one, and it prices the §6 wall directly.** It is the tree-side version of
  `L1lookupt`'s representation — value as *label*, not magnitude. Against the orphan `D2numcat`,
  which uses the same representation on a much weaker base:

  | | solo | max corr | stack Δ |
  |---|---|---|---|
  | `D2numcat` (old base) | 0.962343 | 0.9717 | +0.000030 |
  | `N3numcat` (modern base) | 0.966291 | 0.9783 | +0.000029 |

  **+0.0040 of solo strength bought −0.000001 of stack contribution.** The leg got stronger and, by
  exactly enough to cancel it, less decorrelated. That is the §6 wall as a *quantitative trade*
  rather than a correlation across legs: moving along the wall is free and worthless. `L1lookupt` did
  not move along it — it was simultaneously our strongest leg **and** our most decorrelated, which is
  why it was worth +0.0013 on the leaderboard and why nothing since has repeated it.

**Round closed with zero submissions.** `0a3c852e` (22 legs, OOF 0.969314, **LB 0.97056**) stands.
Seven consecutive probes have now landed inside ±0.00007 of the stack: TabM +0.000027, `fe_impute`
+0.000059, nonlinear meta −0.000042, orphan admission +0.000047, N1 −0.000000, N2 +0.000001,
N3 +0.000029. Per §8, that is the stop signal, and it is a much sharper one than a single miss.

## Phase 4 — the representation is the unit of ensemble value, not the model

`R1` (`9107a864`) is **ModernNCA** — a retrieval/metric learner that predicts a row by soft k-NN
over the training set in a learned metric space, `p(y=1|q) = Σⱼ softmax(−‖f(q) − f(xⱼ)‖₂) · yⱼ`.
It was run for one reason: this data's value→target map is a **lookup table**, and retrieval *is*
lookup. It reads `L1lookupt`'s per-value representation, and the `mnca` branch is spliced into the
`lookupt` branch so the vocabulary, rank-gauss and PLR code are shared verbatim — verified as the
original 119 lines reindented by exactly four spaces, so the two cannot drift apart.

It broke my own pre-registered prediction and still missed the gate:

| | measured | pre-registered |
|---|---|---|
| solo OOF | **0.968722** — our strongest leg ever | ≥ 0.9655 ✓ |
| max \|corr\| vs the pool | **0.9771** (vs `B2blgb`); pool median 0.9947 | — |
| corr vs `L1lookupt` | **0.9689** | ≥ 0.995, "expect a re-derivation" ✗ |
| \|weight\| rank in the fitted stack | **1 of 23** | — |
| **stack contribution** | **+0.000120** | ≥ +0.0002 ✗ |

So it is *simultaneously* the strongest leg we own and more decorrelated than `L1lookupt` was —
outcome (c), the quadrant §6 says is empty, and the **second** counter-example to the wall. And it
is still only worth +0.00012. The gate call was right; my reason for expecting it was wrong.

### Why: two unrelated architectures bought the same thing

Removing `L1lookupt` from the pool and adding each leg back alone:

| member set | OOF | Δ |
|---|---|---|
| 21 legs (neither lookup leg) | 0.968793 | — |
| + `L1lookupt` (= the shipped 22) | 0.969314 | +0.000520 |
| + `R1mnca` **instead** | 0.969323 | **+0.000530** |
| + **both** | 0.969434 | +0.000641 |

Either lookup leg **alone** is worth ≈ +0.00052. Together they are worth +0.00064, not +0.00105 —
**64% of their value is the same value.** The marginal is near-symmetric (mnca given L1 +0.000120,
L1 given mnca +0.000111), so this is not "mnca is redundant"; it is **the second lookup leg is
redundant, whichever one it is.**

A 4-layer transformer with attention over feature tokens and a soft-kNN metric learner share no
inductive bias whatsoever. Reading the same per-value representation they deliver the same +0.00052,
and most of it is the *same* +0.00052. **What `L1lookupt` bought was never "a transformer" — it was
access to the lookup representation, and that access is now saturated.** This is the generalisation
of the TabM lesson (§"the representation decides what a model becomes"): the representation is the
unit of ensemble value, and the architecture on top of it is close to a free variable. A third
architecture on the same representation is predicted to buy ≈ +0.0001 and must not be run on the
strength of this result.

### Correlation structure is not value structure

The Phase 3 finding was that a *decorrelated* leg can be worth nothing. This is the same point from
the other side: two legs correlated at only **0.9689** still share 64% of their contribution. Their
correlation *profiles* are not even similar —

| | vs `B2blgb` | vs `B2xgb` | vs `F1real` | vs `K4node` |
|---|---|---|---|---|
| `R1mnca` | 0.9771 | 0.9749 | **0.8628** | **0.9062** |
| `L1lookupt` | 0.9633 | 0.9688 | 0.9305 | 0.9536 |

Retrieval reads the lookup lattice roughly the way a **tree** does; the transformer reads it the way
a **net** does. Different neighbours, nearly the same value. Correlation is a poor proxy for
contribution in *both* directions, and only the add-one measurement settles it —
`scripts/leg_probe.py <run_id>` now does that in one command.

**Zero submissions on this probe's own terms.** The 23-leg stack (OOF **0.969434**) was submitted
afterwards as `7f69fcf6` on the `0a3c852e` precedent — a finals-selection call, not a gate
clearance. It landed at **LB 0.97060**, and what that number revealed is § Phase 5.

**Not run, and the one open question:** `mnca` on rank-gauss *scalars* instead of the per-value
embedding, which would attribute the above to the representation directly. Prediction pre-registered
before `R1` ran and unchanged: solo ≈ 0.966, max corr > 0.995, contribution ≈ 0.

## Phase 5 — the CV→LB offset is not a constant, and the gate has been deflating

`7f69fcf6` (23 legs) was submitted to give the OOF champion a selectable entry at the deadline. It
predicted LB 0.97071 and landed at **0.97060** — the first pre-registered stack prediction to miss,
by 0.00011, and the fourth point is what makes the series readable:

| stack | OOF | LB | offset = LB − OOF |
|---|---|---|---|
| 16 legs `cdbffba5` | 0.967976 | 0.96928 | 0.001304 |
| 19 legs `6da97a58` | 0.969228 | 0.97051 | 0.001282 |
| 22 legs `0a3c852e` | 0.969314 | 0.97056 | 0.001246 |
| **23 legs `7f69fcf6`** | **0.969434** | **0.97060** | **0.001166** |

The offset **decays monotonically with pool size**. The first three were read as a constant
`0.00128 ± 0.000029`; they were three points on a slope, and the fourth sits 3.8σ below their mean.

A falling offset means the OOF is becoming progressively more **optimistic** as legs are added —
exactly what a fitted meta-model does. More members, more parameters, more fit to the noise in the
OOF matrix. `honest_oof`'s per-outer-fold refit does not remove it, because every leg's OOF was
itself produced on this same partition (the residual optimism flagged at `stack_logit.py:22`, now
measured rather than assumed).

### The operational form: transfer ratio

| step | ΔOOF | ΔLB | transferred |
|---|---|---|---|
| 16 → 19 legs | +0.001252 | +0.001230 | **98%** |
| 19 → 22 legs | +0.000086 | +0.000050 | 58% |
| 22 → 23 legs | +0.000120 | +0.000040 | **33%** |

The one large genuine gain transferred essentially in full. The marginal gains transfer
progressively less.

**This changes what the gate means.** The +0.0002 gate is stated in OOF units and has been read
throughout as though OOF units were LB units. At the current pool size a leg that exactly clears it
is worth about **+0.00007 on the leaderboard — half the leaderboard's own resolution.** The gate was
not set too low; OOF and LB simply stopped being the same currency somewhere around 20 legs, and
nobody noticed because until now every prediction had landed. Judge any future leg on
*OOF contribution × the current transfer ratio*, and on that arithmetic **no single additional leg
of the kind we know how to build can move this leaderboard.**

It also retrospectively vindicates the `R1mnca` gate-2 miss: the gate said the leg was not worth
shipping, the leaderboard agreed to within a number too small to read, and the submission was
correctly framed as selection rather than as evidence.

**Standing:** `7f69fcf6` (23 legs, OOF 0.969434, **LB 0.97060**) is now both the OOF and the LB
champion, so Final A carries no conflict between the two criteria — but it beats `0a3c852e` by
0.00004 against a resolution of 0.00014. The two are tied; prefer the 23-leg on OOF.

### Tabular foundation models — subsetting is not the obstacle, the representation is

The reason this repo never tried a TFM was TabPFN v2's ~10k-row ceiling against our 691,369.
**That reason has expired:** TabPFN-3 states 1,000,000 × 200, TabICLv2 states 500K with CPU/disk
offload. Our table fits inside both. And for an in-context learner, "fit on a subset and extend to
the whole dataset" is not a workaround at all — the training set *is* the context, so ensembling
over subsets is free.

Measured on fold 0 of the frozen partition, 20k query rows, ~15 min on the local 3060 —
**zero Kaggle GPU hours, zero submissions.** `scripts/tfm_probe.py`. AUC standard error at
n=20k is ±0.002, so these are not comparable to `runs.csv` OOF at the fourth decimal; every gap
below is ten times that.

| arm | what | AUC |
|---|---|---|
| **A** single context, raw | 4k → 8k → 16k → 32k → 69k | 0.9337 → 0.9369 → 0.9385 → 0.9394 → **0.9402** |
| **B** 8 *disjoint* 16k contexts, 128k rows, logit-averaged | the proposal, done properly | **0.9398** |
| **C** one 16k context, **target-encoded** representation | changes only the representation | **0.9588** (upper bound) |

Arm A is flat by 32k — the last doubling buys +0.0008. Arm B sees nearly twice the rows of a
single 69k context and **does not beat it** (0.9398 vs 0.9402, inside noise); the ensemble had
converged by block 4. Both plateau at ~0.940, our worst leg ever, 0.022 below the pool floor.

Arm C moves the *same model* with the *same 16k context* by **+0.019 by changing only the
representation**, against +0.007 from a 17× larger context — and it is an *upper bound*, since its
context rows see their own labels through the TE (query rows are clean, so the comparison errs in
the honest direction).

**Mechanism:** this data is a value→target lookup table (rates 0.119–0.986, Pearson(value, rate)
= −0.044, cardinality to 1,459). A TFM normalises an integer code and reads it as a *magnitude*, so
it cannot see a non-monotonic lookup table however much context it is handed.

**Do not build a TFM leg.** The ceiling is representational — Phase 4's finding arriving from a
third independent direction, and the first time it has held for a model family pretrained by
someone else on data that is not ours. Even optimistic arm C enters at ~0.958 against a pool floor
of 0.9622, Phase 3 measured that a sub-pool-strength leg contributes nothing however decorrelated,
and Phase 5's 33% transfer ratio takes whatever is left. **What would change this answer:** a TFM
whose preprocessing treats high-cardinality integer columns as *categorical* rather than numeric.
Arm A is the two-line rerun that would test it.

*(TabICLv2 over TabPFN-3 for a non-technical reason: TabICL is BSD-3, while the TabPFN-3 weights
carry a licence forbidding commercial and production use. A no-prize Playground competition is very
probably fine under it — but "very probably fine" is not a licence review, and BSD cost nothing.)*

## Phase 6 — the remaining gap is a rule, not a model

Phase 5 concluded that no leg we know how to build can move this leaderboard. The obvious next
question is whether that is a fact about *us* or a fact about the *problem*. It is neither: it is a
fact about a constraint we chose. Measured 2026-08-12, 1,635 teams.

**We are rank 248 at 0.97060. #1 is 0.97124. The entire distance is 0.00064 AUC.**

### The leaderboard's shape is the finding

Scores in the top 400 do not form a continuum. They pile into identical spikes:

| score | teams |
|---|---|
| 0.97092 | **36** |
| 0.97086 | 23 |
| 0.97098 | 19 |
| 0.97084 | 17 |
| 0.97095 | 15 |

Agreement to five decimal places is not convergence — it is the same file. The #1-voted notebook,
`najiama/s6e8-addiction-lb-0-97097`, is **four cells that read `18_blend_submission.csv` from a
private dataset and write it to `submission.csv`.** Its author states plainly that the feature
engineering and training code stay private, and that the CSV is itself a weighted blend of Szymon's
OOF library, Omid's XGB/RealMLP/TabM, and Ravi's L2Stack. **That one CSV is rank 60.**

**180 teams sit in 0.97080–0.97100, and that band is a download.** Only 26 teams are above 0.97100
and 8 above 0.97110 — and the top of the leaderboard carries 58–75 submissions apiece, which is
public-split hill-climbing, the S6E7 failure mode.

### Is the gap signal or split noise? Both, in different places

Paired bootstrap over our own 691k OOF rows, our 23-leg stack against `naji05` — rank-correlated
0.984, i.e. genuinely different recipes rather than near-twins:

| | rows | SD of paired ΔAUC |
|---|---|---|
| public @20% | 59,260 | **0.000096** |
| private @80% | 237,041 | 0.000039 |
| **single absolute score, public** | 59,260 | **0.000631** |

Read those two scales together. The 0.0003 gap to the cluster is **3σ — real, and it would replicate.**
But a single score's absolute noise, 0.00063, **exceeds the whole spread from rank 248 to rank 1**.
Paired *differences* are readable; public *rank* at this density is mostly which rows landed in the
split. Both statements are true at once, and confusing them is how a leaderboard eats a month.

### What that leaves

Three questions that had been collapsing into one:

1. **Is our modelling exhausted?** Yes, and measured: Phase 5's transfer ratio puts a leg that
   exactly clears the +0.0002 OOF gate at +0.00007 LB, below what the split resolves.
2. **Is there headroom on the leaderboard?** Yes — and it is a *prediction-sharing* gap, not a
   modelling gap.
3. **Is that headroom reachable by us?** No. `public_gap.py` had already localised the entire
   missing niche at +0.000478 OOF, concentrated in `naji05`/`naji03` — the recipe whose code is
   private. We cannot reimplement what nobody published.

Using the public artifacts is **legal** under Kaggle rules and is what most of the teams above us
are doing. **We declined it, knowingly, at a measured cost of roughly 180 places.** The rule that
every model we ship is one we trained was put on the table with that price attached and reaffirmed
on 2026-08-12. It is the binding constraint on our final score, and it is a constraint, not a
finding — recorded here so that no future reader mistakes our rank for our ceiling.

> **Do not read the public leaderboard for guidance again.** Its top is a shared CSV, the ordering
> inside 0.0002 is split noise, and acting on either is the S6E7 failure mode.

### Final B, and what removing the selection apparatus cost

Final B (below) was rebuilt under a rule fixed in advance, which incidentally answered a question
Phase 3 had only half-answered. Replacing **every** fitted selection step — the admission gates that
curated Final A's 23 legs, and the `C` grid search — with one threshold and a pinned constant moved
the leaderboard by **−0.00002**, a seventh of its resolution.

Phase 3 showed that the nine gate-*rejected* orphans added +0.000047 when admitted. Phase 6 shows
the complementary thing: **the gates themselves are worth about +0.00002 LB in total.** The
apparatus was honest, carefully run, and not load-bearing.

The offset also got a fifth point, at 0.001155 for 25 legs against 0.001166 for 23. Phase 5 read
four points as a monotonic decay and pre-registered −0.00110; the truth came in 0.000055 above that.
**The decay is flattening, not continuing** — 22→23 fell 0.000080, 23→25 fell 0.000011 — so the
offset looks like it is asymptoting near 0.00115 rather than sliding to zero. Caveat, and it is not
small: Final B pins `C` and differs in pool *composition*, so this point is not on the same curve as
the other four. It is evidence against the offset falling without limit, not a fitted fifth point.
Phase 5's **transfer-ratio** argument is untouched by it, because that argument is about marginal
ΔOOF → ΔLB rather than the offset's level.

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
