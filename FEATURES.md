# Feature work in Playground S6E8 — importance, engineering, selection

Companion to `README.md` (the contract) and `KAGGLE_PLAYBOOK.md` (the method). This file collects
everything the competition established about **the features themselves**, separated into the three
activities that are usually conflated:

- **§1 Importance** — what the data said before any model was fitted, and what the models said about
  which columns they used.
- **§2 Engineering** — every derived-feature family that was built, organised by the model that
  consumed it, with its measured effect.
- **§3 Selection** — what was kept, what was dropped, and (mostly) what was *not* dropped and why.

Every number below is either quoted from `experiments/runs.csv` with its `run_id`, or recomputed
from `data/train.csv` for this document and marked *(recomputed)*.

**The single most important structural finding, stated once up front:** this dataset's columns split
into two kinds, and almost every feature result in the competition follows from which kind a column
is.

| kind | columns | signature | the lever that works |
|---|---|---|---|
| **monotone magnitudes** | `daily_screen_time_hours`, `weekend_screen_time`, `social_media_hours` | solo AUC 0.86–0.89, Pearson(value, per-value rate) ≈ +0.91 | ordinary numeric splits; **linear combinations** they cannot build |
| **lookup keys** | `notifications_per_day`, `app_opens_per_day` | solo AUC 0.49–0.54, Pearson(value, rate) ≈ −0.05, but per-value rates span 0.119–0.986 | **supervised per-value encodings** and **per-value embeddings** |

Reading a lookup column as a magnitude, or a magnitude column as a label, was the cause of every
feature-family failure recorded here.

---

## 1. Feature importance

### 1.1 The raw schema and univariate signal *(recomputed)*

12 features, 691,369 train rows, positive rate 0.7094.

| feature | kind | levels | % null | solo AUC | AUC of its missing-indicator |
|---|---|---:|---:|---:|---:|
| `daily_screen_time_hours` | num | 1389 | 13.86 | **0.8896** | 0.50063 |
| `weekend_screen_time` | num | 1437 | 16.21 | **0.8810** | 0.50048 |
| `social_media_hours` | num | 721 | 19.38 | **0.8578** | 0.50000 |
| `work_study_hours` | num | 600 | 7.45 | 0.6549 | 0.50022 |
| `gaming_hours` | num | 401 | 18.34 | 0.6220 | 0.50048 |
| `app_opens_per_day` | num | 166 | 11.67 | 0.5409 | 0.50095 |
| `sleep_hours` | num | 451 | 6.43 | 0.5270 | 0.50062 |
| `gender` | cat | 3 | 4.20 | 0.5120 | 0.49994 |
| `stress_level` | cat | 3 | 7.98 | 0.5031 | 0.49992 |
| `age` | num | 18 | 4.18 | 0.5023 | 0.50040 |
| `academic_work_impact` | cat | 2 | 6.40 | 0.5019 | 0.50000 |
| `notifications_per_day` | num | 231 | 9.78 | **0.4921** | 0.50019 |

Two things were settled here and never revisited:

**Missingness is uninformative.** The largest `|AUC(missing-indicator) − 0.5|` across all 12 columns
is **0.00095**. Missing-indicator features and an `n_missing` count are noise; the differing
train/test null rates are sampling noise, not drift. This is why NaN was left in place for the GBDTs
rather than imputed — and it is independently confirmed by the public frontier, which measured the
same features at ≈ 0 (`README.md` § *Measured dead ends*).

**Three columns are effectively flat.** `gender`, `stress_level`, `academic_work_impact` and `age`
sit within 0.012 of chance univariately. They were nonetheless never dropped — see §3.3.

### 1.2 The lookup-table finding — the discovery the whole competition ran on *(recomputed)*

Restricting to values observed ≥ 30 times and measuring the per-value positive rate:

| column | values | per-value rate range | Pearson(value, rate) | % of consecutive steps increasing |
|---|---:|---|---:|---:|
| `notifications_per_day` | 231 | **0.119 … 0.986** | **−0.050** | **47.8%** |
| `app_opens_per_day` | 166 | 0.114 … 0.985 | +0.126 | 49.7% |
| `daily_screen_time_hours` | 1108 | 0.029 … 1.000 | **+0.917** | 40.8% |
| `weekend_screen_time` | 1174 | 0.026 … 1.000 | +0.902 | 41.0% |
| `age` | 18 | 0.641 … 0.778 | +0.058 | 58.8% |

`notifications_per_day` carries an enormous amount of signal — individual values run from 12%
positive to 99% positive — and **AUC cannot see any of it**, because AUC only measures monotone
association and consecutive values step up barely half the time. Its solo AUC of 0.4921 is the
lowest of any column in the dataset, and it is the most informative column in the dataset. Contrast
`daily_screen_time_hours`, whose per-value rates rise almost perfectly with the value
(Pearson +0.917): there, the number *is* a magnitude.

The generator memorised value→label associations. The corroborating public measurement:
`notifications_per_day`'s per-value residuals correlate **0.72 across two independent halves of the
data** — the lattice is stable, not noise.

**Every consequence in this document descends from that table:**

- a ratio, a sum, a quantile bin — any monotone transform — cannot represent a lookup, which is why
  the `fe_interaction` family failed twice (§2.1, §2.7);
- a *supervised* per-value encoding can, which is why target encoding was the largest single win
  (§2.2);
- so can a learned *per-value embedding*, which is why the Lookup-Transformer was the only leg that
  ever broke the strength/decorrelation wall (§2.8);
- and declaring the columns categorical — the *unsupervised* imitation of the same idea — loses,
  because a GBDT's categorical splitter is an unsmoothed, unnested target encoder (§2.6).

### 1.3 What the model used vs. what the metric saw — the rank gap

The EDA's second finding was a **disagreement between univariate AUC and split usage**. Ranked by
`|solo AUC − 0.5|`, `notifications_per_day` is 9th of 12 and `app_opens_per_day` 6th. Ranked by how
many splits the baseline LightGBM spends on them, they are **1st and 2nd**.

Reproduced for this document *(recomputed: the anchor config — raw features, `lr` 0.05,
`num_leaves` 31, early stop 400 — on fold 0 of the frozen split; fold AUC 0.962597, `best_iter`
3395)*:

| feature | splits | % of splits | % of gain | solo AUC |
|---|---:|---:|---:|---:|
| **`notifications_per_day`** | **13,965** | **13.71%** | 7.56% | **0.4921** ← rank 12 of 12 |
| **`app_opens_per_day`** | **12,993** | **12.76%** | 6.58% | 0.5409 |
| `daily_screen_time_hours` | 12,091 | 11.87% | **45.27%** | 0.8896 |
| `weekend_screen_time` | 11,770 | 11.56% | 13.54% | 0.8810 |
| `work_study_hours` | 10,925 | 10.73% | 3.11% | 0.6549 |
| `social_media_hours` | 10,817 | 10.62% | 18.37% | 0.8578 |
| `gaming_hours` | 10,057 | 9.87% | 2.70% | 0.6220 |
| `sleep_hours` | 9,958 | 9.78% | 1.52% | 0.5270 |
| `age` | 5,495 | 5.40% | 0.91% | 0.5023 |
| `gender` | 1,458 | 1.43% | 0.18% | 0.5120 |
| `stress_level` | 1,392 | 1.37% | 0.16% | 0.5031 |
| `academic_work_impact` | 929 | 0.91% | 0.11% | 0.5019 |

**The two importance measures point in opposite directions, and both are right.** By *gain*,
`daily_screen_time_hours` dominates at 45% — it is the cheap, monotone signal a tree captures in a
handful of splits near the root. By *split count*, `notifications_per_day` is first: it earns only
7.6% of the gain while consuming 13.7% of the tree's budget, because the model is grinding out a
231-cell lookup table one axis-aligned cut at a time and each cut buys very little.

**That ratio — high usage, low yield — is the signature of a feature the model can only approximate,
and it is what motivated the entire B1→B6 sequence.** Handing the tree the lookup directly (B6) is
worth +0.003. Handing it a ratio instead (B1) is worth −0.0004.

The bottom three rows are the flat categoricals, at 0.11–0.18% of gain between them (§3.3).

### 1.4 Importance was measured by ablation, never by an attribution method

**No permutation importance, no SHAP, no gain-based feature ranking was ever used to make a
decision** in this repo. Every feature family's value is a **strict-twin ablation**: two runs
differing in exactly one `CFG` field, scored on the frozen 5-fold, compared against a
pre-registered +0.0002 gate. `scripts/run_local.py --diff-vs` enforces the one-field rule before a
run starts.

This is deliberate and it is the reason the numbers in §2 are trustworthy. Gain-based importance
would have ranked `daily_screen_time_hours` first and told us nothing about whether adding
`other_hours` helps; the ablation answers that question directly and in the units of the metric we
are scored on.

The one place attribution-style reasoning *was* used is the rank gap in §1.3 — and it was used to
**generate a hypothesis**, not to accept one.

### 1.5 Derived-feature importance *(recomputed)*

Solo AUC and coverage of the engineered columns, which is what §2's results are best read against.
Coverage matters as much as strength: a ratio is NaN whenever *either* operand is missing.

| engineered column | family | coverage | solo AUC | note |
|---|---|---:|---:|---|
| `screen_total` | composition | 74.9% | **0.9013** | strongest single feature in the project |
| **`other_hours`** | composition | **61.0%** | **0.7649** | the latent variable — see §2.3 |
| `share_social` | composition | 72.6% | 0.6603 | |
| `free_time` | normalization | 80.9% | 0.865 (inverted: raw 0.1351) | high coverage, no latent variable behind it |
| `opens_per_hour` | interaction | 76.6% | 0.698 (inverted: raw 0.3024) | monotone transform of a lookup column |

`free_time` is the instructive row: it has the *best coverage* of any derived column and a
respectable solo AUC, and it was worth **−0.000081**. Solo strength of a derived feature does not
predict its value (§2.7).

---

## 2. Feature engineering, by model

Every family lives in one function — `engineer()` in `scripts/build_model_nb.py:160` — behind a
boolean `CFG` flag, so a family is enabled identically for every learner and cannot drift between
them. That single-source property is what makes the reuse column in §2.10 meaningful rather than
aspirational.

Notation: **shipped** = in the final 23-leg stack's members; **rejected** = missed its gate;
**diagnostic** = run to settle a question, never intended to ship.

### 2.1 B1 `fe_interaction` — the ratios (rejected, twice)

```
opens_per_hour = app_opens_per_day / daily_screen_time_hours
notif_per_hour = notifications_per_day / daily_screen_time_hours
notif_per_open = notifications_per_day / app_opens_per_day
```

**Model:** LightGBM (`66e3dede`), later re-run on XGBoost (§2.7).
**Hypothesis:** the two columns with the biggest rank gap (§1.3) are being approximated split-by-split
because a GBDT cannot express `x/y` with axis-aligned cuts. Hand them over explicitly.
**Result: 0.963047 vs the 0.963405 champion — −0.000358.**

The hypothesis about *why* the trees spend splits there was right. The proposed remedy was wrong:
a ratio is a monotone transform of a lookup column, and §1.2 says no monotone transform can
represent a lookup table. This failure is what motivated the supervised version (B6) six runs later.

### 2.2 B6 per-value target encoding — the largest single win (**shipped, reused by every tree and MLP leg**)

Each raw *value* is mapped to the smoothed rate at which that value's rows are positive.

**Models:** LightGBM `f64e2781`, and thereafter **every** tree and MLP leg in the repo.
**Result: 0.966384 vs 0.963405 — +0.002979, roughly 15× the gate and 45× seed noise.**

Implementation, all of it load-bearing (`build_model_nb.py:341–407`, applied at `:456–462`):

- **Nested.** Training rows are encoded out-of-fold *within* the training fold (inner K-fold);
  val and test rows are encoded from a fit on the whole outer training fold. Without the nesting,
  each training row would see its own label through its value's mean.
- **Smoothed** with prior weight `m = 20`; an unseen value falls back to the training fold's prior
  exactly (`n = 0` ⇒ the smoothing term is all that remains), never to a global statistic and never
  raising.
- **Re-fit from scratch inside every fold.** This is the one transform in the notebook that sees
  `y`, and `README.md` §2's rule applies to it at every usage site, including HPO and the stack.
- **NaN gets its own code**, so "missing" carries its own rate rather than being dropped.
- The integer value→code map (`TE_CODES`) is factorised over train ∪ test *outside* the loop. That
  is an unsupervised label mapping and explicitly not a leak.

**The leak audit, run because a +0.003 jump on a supervised encoder deserves one:** train/val
encoding-AUC gaps of −0.0012…+0.0025 (negative on the singleton columns — the opposite sign to
leakage); fold std *fell* 0.000555→0.000413 and fold spread 0.001656→0.001130 (a leak inflates
both); `best_iter` fell from 3000–3998 to 1161–2614, i.e. the model converges faster when handed
information it previously had to build split-by-split, which was the pre-registered mechanism.

### 2.3 B2 `fe_composition` — the budget constraint (**shipped, reused by all six later families**)

```
other_hours   = daily_screen_time_hours − (social_media_hours + gaming_hours + work_study_hours)
share_{social,gaming,work,other} = each part / daily_screen_time_hours
weekend_ratio = weekend_screen_time / daily_screen_time_hours
screen_total  = daily_screen_time_hours + weekend_screen_time
```

**The invariant:** `daily_screen_time_hours ≥ social + gaming + work` holds in **100.00000%** of all
859,029 train + test rows, minimum gap exactly 0.000, 546 rows sitting on the boundary. That is a
generator invariant, not a correlation. The remainder is a genuine latent variable — unaccounted
screen time — at **solo AUC 0.765**.

**Replicated on three engines as strict twins:**

| engine | baseline | + composition | delta |
|---|---|---|---|
| XGBoost (C4 `fdeaa047`) | 0.967176 | **0.968180** `f047412f` | **+0.001004** |
| LightGBM (B6 `f64e2781`) | 0.966384 | **0.967547** `7d1de67e` | **+0.001163** |
| CatBoost (D1 `7c1e9334`) | 0.966890 | **0.968030** `e73f6257` | **+0.001140** |

Five to six times the gate, on all three, and within 0.00005 of an independent public 10-fold
ablation of the same feature (+0.00096).

**Mechanism, and why it survives more folds:** a 4-term linear combination is not something
axis-aligned splits can build *at any data volume*. Fold-count independence is the mechanism's own
prediction, and it is what separates this family from log-ratio features, which decay from +0.00036
at 3 folds to +0.00001 at 10 as the trees learn those relationships by themselves. **Validate FE at
the fold count you will submit with.**

**The process failure worth recording:** `fe_composition` had existed in the generator since Phase 1
and was only ever run in the *pre-target-encoding* era, alongside B1's failure. The `engineered`
column is empty on all five champion legs. It sat unused for weeks. This is the fourth instance of a
pattern `README.md` names by hand: concluding *"X does not help"* from *"X did not help in the one
place I put it."*

### 2.4 B3 `fe_impute` — imputation as extra columns (rejected on the second gate)

One `XGBRegressor` per numeric column, fit on the observed rows of train ∪ test, predicting that
feature from the other features. The imputed values are added as `imp_*` columns, and the whole
engineered family is **recomputed on them** as `impfe_*`. **The originals keep their NaNs.**

**Model:** XGBoost on the B2 composition base (`54db2990`), 44 features.
**Result: 0.968425 vs 0.968180 — +0.000245 solo (clears, positive on 5/5 folds); +0.000059 to the
stack (misses).** Not submitted.

- **Augment, never replace.** The same imputer measures with *opposite signs* publicly: replacing
  NaNs is negative, adding imputed values alongside them is +0.0012. A GBM's native NaN handling
  learns a default split *direction* per node, which is strictly more expressive than one imputed
  point estimate dragging the row to the middle of its distribution.
- **What it actually buys is coverage,** not better values: the composition family is available on
  only 61.0% of rows because a derived column is NaN whenever any operand is. It is missing exactly
  where it would say the most.
- **Leakage status:** the imputer predicts a *feature* from other features and never sees `y` —
  transductive unsupervised preprocessing, licensed outside the CV loop by the same argument as the
  category vocabulary. It is, however, the **only** family here that is not row-wise, so the
  notebook's row-wise purity assertion is *scoped to exclude it* rather than deleted, and the
  imputed block gets its own assertions instead (observed values untouched, originals still NaN).
- **Why it under-delivered vs. the public +0.0012:** that measurement was against a baseline
  *without* the composition family. B2 had already bought most of it.
- `engineer()` is reused verbatim to build the `impfe_*` block, specifically so the two copies of
  the formulas cannot drift apart.

### 2.5 B7 target-encoded value crosses (rejected)

Target-encode value *pairs* (`notifications_per_day × app_opens_per_day`) rather than single values —
the only encoding that expresses "this *combination* of values."

**Model:** LightGBM (`6e3dd7c3`). **Result: 0.966353 vs 0.966384 — −0.000031.**

Cardinality was the pre-registered risk and it is what the result reflects: up to 38k cells over
691k rows (~18 rows/cell) against ~2,700 rows/value for a single column, so per-row self-influence
rises from ~0.03% to ~3%. The nesting and smoothing stop being formalities. A disqualifier was
pre-registered (train/val encoding-AUC gap > +0.02 ⇒ leak, regardless of OOF) and was not tripped.

This run has a second life in the repo: it produced **the first OOF↔LB rank inversion** and, in doing
so, refuted a live worry that target encoding was inflating OOF (`README.md:145–163`).

### 2.6 D2 / D2b `num_as_cat` — the unsupervised lever on the lookup structure (rejected, both scopes)

If values are labels rather than magnitudes, declaring the numeric columns **categorical** is the
unsupervised way to act on that. Both scopes were tried on LightGBM and both lost badly:

| run | scope | OOF | vs B6 | mean `best_iter` |
|---|---|---|---|---|
| B6 `f64e2781` | numerics stay numeric | 0.966384 | — | 1860 |
| **D2** `b4326a68` | all 9 numerics categorical | 0.962343 | **−0.004041** | 450 |
| **D2b** `2b876e0b` | only the 2 lookup columns | 0.964549 | **−0.001835** | 503 |

D2's diagnosis was obvious — seven of the nine columns are strongly monotone, so categorifying them
discards real ordinal signal *and* hands LightGBM a 1400-level column to overfit. **Acting on that
diagnosis was still wrong.** D2b restricted the treatment to the only two columns the lookup property
was ever measured on, and pre-registered the prediction that `best_iter` would recover to ~1800. It
came back 503. The prediction failed and the score agreed.

**Mechanism:** LightGBM's categorical splitter sorts levels by their per-level gradient statistics on
the training fold and cuts. **That is target encoding, fitted with no nesting and no smoothing.** At
166 and 231 levels it overfits, which is precisely what the collapsed `best_iter` reports. B6 already
extracts this structure the safe way. There was never room for both.

The value of this pair is procedural: two LightGBM screens, seven minutes, zero submissions, and it
retired a CatBoost variant that would have cost a multi-hour kernel to reach the same answer.

### 2.7 The N-round — three untried levers on the modern base, three misses

Config-only strict twins on the Phase-2 base (TE + composition + Optuna params). Gates
pre-registered in `notes` before the results were read: solo ≥ parent + 0.0002 **and** ≥ +0.0002 to
the 22-leg stack.

| probe | family | model | run | solo | vs parent | max corr | stack Δ |
|---|---|---|---|---|---|---|---|
| **N1** | `fe_normalization` | XGBoost | `8f40573b` | 0.968099 | −0.000081 | **0.9977** | −0.000000 |
| **N2** | `fe_interaction` | XGBoost | `ceb7f8c0` | 0.968162 | −0.000018 | 0.9978 | +0.000001 |
| **N3** | `num_as_cat` | LightGBM | `ad5e827a` | 0.966291 | −0.001256 | 0.9783 | +0.000029 |

**N1 — the 24-hour budget is not a second composition.** `free_time = 24 − sleep − screen` is
structurally identical to the screen-time constraint that cleared at 5–6× the gate on all three
engines, and it had *more* rows to act on (80.9% vs 61.0% coverage). It produced a model at
**0.9977 correlation with its parent** — the highest correlation anywhere in the pack, above even
the two reruns of the same recipe. **The composition family did not work because it was a budget; it
worked because `other_hours` is a latent variable at solo AUC 0.765 that the trees could not
otherwise reach.** `free_time` has no such quantity behind it. This is the sharpest single lesson in
the feature record: *the arithmetic form of a feature is not the reason it works.*

**N2 settles B1's ambiguity.** The ratio family lost by 0.0003 in the pre-TE era, and by 0.00002 now
on a base carrying TE, the composition family and tuned parameters. It never lost because the base
could not use it. It lost because a ratio cannot represent a lookup table.

**N3 prices the wall.** Against the older `D2numcat` orphan on the same representation:

| | solo | max corr | stack Δ |
|---|---|---|---|
| `D2numcat` (old base) | 0.962343 | 0.9717 | +0.000030 |
| `N3numcat` (modern base) | 0.966291 | 0.9783 | +0.000029 |

**+0.0040 of solo strength bought −0.000001 of stack contribution.** The leg got stronger and, by
almost exactly enough to cancel it, less decorrelated.

### 2.8 L1 — the lookup representation (**shipped; the largest feature-side win after TE**)

Not a feature *column* but a feature *representation*, and the distinction is the point.

```
token_j = Embedding_j[exact_value_id] + PLR_j(rank_gauss value) · (1 − missing)
```

**Model:** Lookup-Transformer (`dada9e2d`), 4 layers, d=128, attention over feature tokens.
**Result: solo 0.968626 — the strongest leg in the repo at the time — at max |corr| 0.9796 against
a pack whose own median max-correlation is 0.9941. Worth ≈ +0.0013 on the leaderboard.**

Components, three of which are reused verbatim elsewhere (§2.10):

- **Per-value vocabulary** over train ∪ test: every exact observed value in every lattice column gets
  its own embedding row, with **id 0 reserved for NaN in every column**, so "missing" is a learned
  per-column vector rather than an imputed number. Unsupervised, so it is hoisted out of the CV loop
  by the same argument as the category vocabulary.
- **Rank-gauss (`QuantileTransformer`, normal output)** for the smooth branch. This one *learns* from
  the data, so it is fit on the **training fold only** — the same rule as the target encoder.
- **PLR (periodic-linear) numeric embedding** with learned Fourier frequencies, added to the lookup
  token and switched off where the value is missing.
- **The engineered composition columns get PLR tokens only** — they are continuous, not lattice-valued.
- **`te_cols` must be empty, enforced by an assertion.** The embedding *is* the lookup; a target
  encoding on top would read the same structure twice and recorrelate this leg with the tree pack,
  which would defeat its entire purpose.

**The correction Phase 4 forced.** The credit belongs to the **representation**, not to the
transformer. ModernNCA (`9107a864`) — a soft-kNN retrieval model sharing no inductive bias with a
transformer — reads the *same* per-value representation and is worth **+0.000530** to a pool with
`L1lookupt` removed, against `L1lookupt`'s own **+0.000520**. Both together buy +0.000641, not
+0.00105: **64% of their value is the same value.** A 4-layer transformer and a metric learner
deliver the same +0.0005 because they read the same features.

**The unit of ensemble value is the representation. The architecture on top of it is close to a free
variable.** This is the same lesson TabM taught from the other side: fed the tree-side target-encoded
representation, a packed MLP ensemble *re-derived the tree solution* (nearest neighbours 0.9847–0.9867
with the trees; solo score equal to the tuned XGBoost's to four decimals).

### 2.9 Per-model preprocessing — the same 12 columns, seven different renderings

Everything above is representation *choice*. This is the mandatory per-family plumbing, and it is
where the leakage rule bites most often.

| model | NaN handling | categoricals | numerics | consumes |
|---|---|---|---|---|
| **LightGBM** | native (learned split direction) | native `category`, vocabulary over train ∪ test | raw | TE, composition |
| **XGBoost** | native | `enable_categorical` | raw | TE, composition, impute |
| **CatBoost** | native for numerics | `cat_features`; NaN → explicit `__NA__` level (unsupervised relabel) | raw | TE, composition |
| **MLP** (E1/E2/E3) | **train-fold median** + an explicit **missingness mask column** | one-hot with column 0 = missing | train-fold standardise | TE; E2 adds embeddings |
| **RealMLP / TabM** | train-fold median + `__isna` indicator columns | strings, NaN → `__NA__` | pytabkit's own pipeline (incl. PLR) | TE (`53c678f6`, `cd2dbce4`); TE + composition (TabM `f1d249bd`) |
| **FT-Transformer / TabTransformer** | train-fold median, **no missingness mask** | `cat.codes + 1`, 0 = missing | train-fold standardise | TE only — never run on the composition family |
| **NODE** | train-fold median | strings, NaN → `__NA__` | raw | TE |
| **LookupT / ModernNCA** | **id 0 per column = a learned "missing" vector** | same per-value vocabulary as the numerics | rank-gauss + PLR | composition (PLR tokens only); **TE forbidden** |

Three notes on this table:

1. **The medians, scalers and quantile transformers are all fit on the training fold and applied to
   val/test.** The trees needed none of this; the MLP branch is the first place in the notebook where
   scaling discipline is load-bearing, and it is commented as such at `build_model_nb.py:538`.
2. **TabTransformer attends over categorical tokens only** — 3 categoricals against 18 continuous
   columns, so nearly all the signal bypasses the attention entirely. Its 0.9533/0.9541 solo scores
   are a *representation* failure, not a capacity one, which `K3_tabtf_gpu` confirmed by re-running it
   at full capacity on a T4 for 0.953339.
3. **The FT-Transformer branch drops the missingness mask** that the MLP branch keeps. Given §1.1
   this costs approximately nothing — but it is an inconsistency between two branches, not a decision,
   and it is recorded here as such.

### 2.10 Reuse map — what was built once and consumed many times

This is the column the user asked to have highlighted, and it is where most of the repo's engineering
leverage actually sits.

| what | built for | reused by | how the reuse is enforced |
|---|---|---|---|
| **`engineer()`** — all three row-wise FE families | B1, LightGBM | every learner; enabled by identical `CFG` booleans | one function, `build_model_nb.py:160`; a `CFG` flag, never a per-model copy |
| **`fe_composition` columns** | B2, XGBoost | LightGBM `7d1de67e`, CatBoost `e73f6257`, TabM `f1d249bd`, LookupT `dada9e2d`, ModernNCA `9107a864`, and the N1/N2/N3 bases | same |
| **`engineer()` inside `fe_impute`** | B3 | the `impfe_*` block | the imputed families call `engineer()` rather than restating the formulas — explicitly so the two copies cannot drift |
| **`TE_CODES`** (value→int, train ∪ test) | B6 target encoding | **E2's `nn.Embedding` blocks** and B7's crosses | one dict; E2 asserts its columns are present in `te_cols` |
| **Per-value lookup vocabulary + rank-gauss + PLR** | L1 Lookup-Transformer | **ModernNCA (R1)** | the `mnca` branch is *spliced into* the `lookupt` branch — verified as the original 119 lines reindented by exactly four spaces, so they cannot diverge |
| **Median-impute + explicit missingness indicator** | E1 MLP | RealMLP, TabM, NODE | same idea, separate code per library's interface — the one pattern that is re-implemented rather than shared |
| **`AS_CATEGORY` vocabulary** (train ∪ test levels) | the 3 raw categoricals | `num_as_cat` in D2/D2b/N3 | one code path builds both |
| **The target encoder itself** | B6 | every tree leg, every MLP leg, the HPO objective, and the stack | re-fit at each usage site by rule; never hoisted |

The two entries in bold are the ones that paid. `TE_CODES` shared between the encoder and E2's
embeddings is what made E2 a one-line config change instead of a new model. The lookup/rank-gauss/PLR
splice is what made Phase 4's central experiment — *is the value in the representation or the
architecture?* — a clean test rather than a confounded one, because the two legs are provably reading
identical inputs.

---

## 3. Feature selection

### 3.1 There was almost none, and that was the right call

**No column was ever dropped from any shipped model.** Selection in this competition happened at the
level of **feature families**, admitted or rejected by a pre-registered +0.0002 OOF gate, with the
run and its mechanism logged before the result was read.

Final tally over the whole competition:

| family | model(s) tested | verdict | delta |
|---|---|---|---|
| **B6 per-value target encoding** | LGB → all | **shipped** | **+0.002979** |
| **B2 `fe_composition`** | XGB, LGB, CatBoost | **shipped, 3/3 engines** | **+0.001004 / +0.001163 / +0.001140** |
| **L1 lookup representation** | LookupT, ModernNCA | **shipped** | +0.000520 to the stack |
| B3 `fe_impute` | XGB | cleared solo, **missed the stack gate** | +0.000245 / +0.000059 |
| N1 `fe_normalization` | XGB | rejected | −0.000081 |
| B1 / N2 `fe_interaction` | LGB, XGB | rejected twice | −0.000358 / −0.000018 |
| B7 TE value crosses | LGB | rejected | −0.000031 |
| D2 / D2b / N3 `num_as_cat` | LGB ×3 | rejected three times | −0.004041 / −0.001835 / −0.001256 |
| B5 `drop_flat_cats` | — | **never run** | — |

Two families shipped. Six were rejected. **The rejections are the more useful half of this table**,
and each one has a mechanism attached — which is why N2 could settle B1's open question three weeks
later instead of re-opening it.

### 3.2 The scope searches — the only within-family selection performed

Two probes searched over *which columns* a treatment applies to, and both are recorded because both
lost:

- **`num_as_cat`: all 9 numerics (D2) → the 2 lookup columns (D2b).** A textbook narrowing, made on
  a correct diagnosis, that still lost by 0.0018. §2.6.
- **`te_cols`: all 9 numerics.** Never narrowed. A subset search over which columns to target-encode
  was **not run** — see §3.4.

### 3.3 The three flat categoricals were kept, untested

`gender` (0.5120), `stress_level` (0.5031), `academic_work_impact` (0.5019) are within 0.012 of chance
univariately, and the baseline LightGBM spends **3.71% of its splits and 0.45% of its gain** on all
three combined (§1.3). `drop_flat_cats` exists as a `CFG` flag (`build_model_nb.py:70`) and **was
never run**.

This is a genuine gap in the record, stated plainly. The defensible reasons for it: dropping features
is a search over subsets with no mechanism behind it, and this repo's standing rule after Phase 3 is
that *"it will be different" is not a hypothesis* — a probe justified only by expected change was
measured to be worth −0.000001. Three columns out of 12 also cost nothing to carry at 691k×12. But it
was never *measured*, and "we did not test it" is the honest description rather than "it does not
matter."

### 3.4 What was deliberately not attempted, and why

| not done | reason |
|---|---|
| permutation / SHAP-driven column selection | attribution does not answer "does adding this help", and the ablation does — §1.4 |
| a subset search over `te_cols` | 2⁹ subsets scored on the same OOF rows is exactly the selection bias `KAGGLE_PLAYBOOK.md` §5 warns about; the ceiling diagnostic in `subset_ceiling.py` carries the same warning in its docstring and is barred from shipping |
| missing-indicator features / `n_missing` | measured at AUC 0.500 ± 0.001 across all 12 columns (§1.1), and independently confirmed at ≈ 0 publicly |
| `age` as a categorical | measured publicly at −0.0006; consistent with §1.2 (`age` has 18 levels, rates spanning only 0.641–0.778) |
| adding the original `jayjoshi37` dataset | 7,500 rows ≈ 1.1% of train; measured at −0.0001 |
| pairwise / multi-resolution / adaptive-smoothing TE | measured publicly at −0.0004 to +0.00002; our own B7 cross agrees at −0.000031 |
| dropping near-duplicate columns | none exist — 1 duplicate row-pair in 565,846 |

### 3.5 One combination that was never run, recorded rather than rationalised

The composition family cleared at 5–6× the gate on **all three tree engines**. Checking the
`engineered` column of `experiments/runs.csv` against every neural run:

| leg | run | got composition? |
|---|---|---|
| E1 MLP, E2 embedding-MLP, E3 raw-MLP | `2fb89920`, `8bd89dee`, `a594ffe2` | **no** |
| F1 / K1 RealMLP | `53c678f6`, `cd2dbce4` | **no** |
| G1 / K2 FT-Transformer | `b2f35b4a`, `46859c68` | **no** |
| G2 / K3 TabTransformer | `f1424102`, `9527c2d0` | **no** |
| K4 NODE | `976c2703` | **no** |
| M1 TabM | `f1d249bd` | yes |
| L1 LookupT, R1 ModernNCA | `dada9e2d`, `9107a864` | yes (as PLR tokens) |

Every neural leg built **before** Phase 2 ran without the strongest engineered family in the repo,
simply because it did not exist on the modern base yet, and none of them was re-run afterwards. This
is the same shape as the failure §2.3 describes — a family measured in one place and never carried to
the others — caught here by inspection rather than by a probe.

It is recorded, not acted on. The competition's modelling was closed by decision on 2026-08-12
(`README.md` § Phase 6), and Phase 5's transfer ratio puts a leg that exactly clears the +0.0002 OOF
gate at +0.00007 LB — below what the leaderboard's 0.00014 resolution can read. The honest statement
is that this was an oversight whose expected value, measured, is smaller than the instrument.

### 3.6 Feature *validation* — the assertions that stand in for a selection step

Because so little was dropped, correctness had to be enforced rather than inferred. Every one of
these runs on every execution of the notebook and fails loudly:

- **denominator positivity** for all four ratio denominators, in train *and* test, asserted rather
  than trusted from the EDA snapshot — otherwise a zero silently produces `inf`;
- **no `inf` in any engineered column**, both splits;
- **`other_hours ≥ 0` everywhere** — the composition assumption itself, re-checked every run;
- **row-wise purity**: recomputing the FE on a shuffled 2,000-row subset must give identical values.
  This is what proves no cross-row aggregation crept in, which *would* be a leak. The imputed block is
  the single scoped exemption, and it gets its own assertions instead (observed values untouched,
  originals still carrying their NaNs);
- **the target encoder's train/val encoding-AUC gap**, reported every run as a standing leak audit.

---

## 4. The answer: the most important features discovered

Ranked by what they were measured to be worth, not by any attribution score.

1. **`notifications_per_day` and `app_opens_per_day` read as lookup keys.** Solo AUC 0.492 and 0.541 —
   the two weakest columns in the dataset by the competition's own metric — carrying per-value
   positive rates from 0.119 to 0.986. Extracting that structure supervised was worth **+0.003**, the
   largest win in the competition. It is the whole reason target encoding, embeddings, and eventually
   the Lookup-Transformer and ModernNCA exist in this repo.

2. **`other_hours` — unaccounted screen time.** Not a column in the data: the residual of a generator
   invariant that holds in 100.00000% of 859,029 rows. A latent variable at **solo AUC 0.765** that
   axis-aligned splits cannot construct at any data volume. **+0.001 on all three tree engines**, and
   the only engineered column in the project that replicated across every model it was given to.

3. **`daily_screen_time_hours`, `weekend_screen_time`, `social_media_hours`** — the honest monotone
   signal, solo AUC 0.858–0.890, and `screen_total` at **0.9013** is the strongest single feature
   anywhere in the project. They are top of every importance ranking and they are **not** where any
   competitive edge was found: every model gets them right immediately, and nothing built on top of
   them ever cleared a gate.

4. **The per-value embedding of every lattice column** (id 0 = NaN, plus a PLR smooth branch).
   Worth +0.00052 to the stack and the only thing that ever broke the strength/decorrelation wall —
   and, per Phase 4, worth that same +0.00052 through *two* unrelated architectures, which is how we
   know the credit belongs to the representation.

5. **Missingness — importance zero, and worth knowing.** Every indicator sits at 0.500 ± 0.001.
   Resolving that in the first hour of EDA is what licensed leaving NaN in place for the GBDTs and
   what kept a whole family of indicator features from ever being built.

**The one-line summary.** The features that mattered were not the ones the metric ranked highest.
They were a column AUC calls worthless because its signal is non-monotone, and a column that does not
exist in the data at all until a generator invariant is noticed. Both were found by looking at the
*structure* of the columns rather than at an importance ranking, and both were confirmed by strict-twin
ablation at the fold count we submitted with.
