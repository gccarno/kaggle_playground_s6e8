# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Kaggle **Playground Series S6E8** ("Predicting Smartphone Addiction"). Binary classification of
`addicted_label` from 12 features about phone/screen usage, scored on **ROC AUC**. 691,369 train rows,
296,302 test rows, positive rate 0.709.

**`README.md` is the contract** — the frozen CV split, the metric, and the leakage rule live there and
are not up for renegotiation mid-competition. Read it before touching a model.
**`KAGGLE_PLAYBOOK.md`** is the method document distilled from S6E7 (which finished rank 120, +298
places in the shakeup). Its thesis: build the measuring instrument first, and stop when your own CV
says the signal is exhausted.

## Repo layout

- `eda-s6e8.ipynb` — the active notebook and the Kaggle kernel (`gcarno/eda-s6e8`). Two halves: EDA,
  then a raw-feature 5-fold LightGBM baseline. It emits `submission.csv`, `oof_proba_<learner>.csv`,
  `test_proba_<learner>.csv` and a final `RUN_METRICS_JSON:{...}` line. **Runs both locally and as a
  Kaggle kernel** via an `ON_KAGGLE` path switch — keep that property in every edit.
- `kernel-metadata.json` — `kaggle kernels push` config. `code_file` points at `eda-s6e8.ipynb`.
  CPU only (`enable_gpu: false`) — LightGBM on 691k×12 does not need a GPU and GPU kernels queue
  longer. Flip it only when a neural leg actually arrives.
- `scripts/collect_run.py` — push → poll → parse `RUN_METRICS_JSON` → archive artifacts → optionally
  submit → append a row to `experiments/runs.csv`.
- `experiments/runs.csv` — **tracked in git**, append-only, one row per run. The single most valuable
  asset in the repo; S6E7's end-of-competition ceiling analysis was pure re-analysis of this file.
- `experiments/preds/<run_id>/` — gitignored per-run probability artifacts. Never delete these.
- `data/` — gitignored local copy of the competition CSVs.

## Running / iterating

Data is local (`data/`) *and* mounted in the kernel at
`/kaggle/input/competitions/playground-series-s6e8/`. Iterate locally first — it is seconds instead
of a push-and-queue cycle — then push the same notebook.

```bash
# rebuild the notebook from its generator (if one is in use) and execute locally
python -m jupyter nbconvert --to notebook --execute eda-s6e8.ipynb --output <scratch>/out.ipynb

# push + poll + archive + submit + log, in one command
python scripts/collect_run.py --submit \
  --description "short: what changed this run" \
  --notes "hypothesis / mechanism / pre-registered gate / result"

kaggle kernels push -p .          # push only
kaggle kernels status gcarno/eda-s6e8
```

Kernels run with `enable_internet: true`, but prefer packages already in the `kaggle/python` image.
If a `!pip install` cell is ever added, keep it in sync with the notebook's actual imports — S6E7 lost
time to `pytabkit` not installing offline.

## Pipeline architecture

The notebook is linear and each stage depends on column-set decisions made earlier:

1. **Setup** — frozen constants (`SEED=42`, `N_FOLDS=5`, `TARGET`, `ID`) and the `ON_KAGGLE` path
   switch that picks `DATA_DIR` / `OUT_DIR`.
2. **Load + schema** — `FEATURES`, then the explicit `RAW_NUM` / `RAW_CAT` split. Every later stage
   is written against those two lists; declare new derived columns into them rather than inferring
   dtypes ad hoc.
3. **EDA** — target rate vs the `sample_submission` constant; missingness (rate *and* the AUC of each
   missing-indicator); train↔test drift (KS for numerics, Cramér's V for categoricals); single-feature
   AUC per column; correlation/near-duplicate scan.
4. **Encoding** — categoricals become pandas `category` with a vocabulary from `train ∪ test`, which
   LightGBM consumes natively. NaNs are left in place for LightGBM to route.
5. **Outer 5-fold CV** — the frozen `StratifiedKFold`. Per-fold model, OOF vector, and test
   predictions averaged across the 5 fold-models.
6. **Artifacts** — `oof_proba_<learner>.csv` (with its `fold` column), `test_proba_<learner>.csv`,
   `submission.csv`, each validated against `sample_submission.csv` by assertion before being trusted.
7. **`RUN_METRICS_JSON`** — one JSON line, the parsing contract with `collect_run.py`. Any
   `<learner>_oof_auc` key automatically gets its own column in `runs.csv`.

## Conventions to preserve when editing

- **Leakage discipline is the load-bearing constraint.** Target encoders, bin edges, scalers,
  imputers and category vocabularies are fit on the **train fold only**, re-fit at every usage site
  (feature selection, HPO, final stack), never once globally before the CV loop. Unseen categories at
  inference map to a reserved "unknown" level rather than raising. An unsupervised label mapping over
  train ∪ test is *not* a leak; a supervised (target/count) encoding is — say which one it is in a
  comment when it isn't obvious.
- **`roc_auc_score` is the objective everywhere** — CV scoring, any Optuna objective, any neural
  early-stopping. Never accuracy, never logloss as a silent proxy.
- **Submit probabilities, never hard labels.** AUC is rank-based: there is no threshold to tune, no
  per-class weight to search, and probability *calibration* cannot move the leaderboard — only
  ranking can. (This is the biggest single difference from the S6E7 codebase, which spent real effort
  on class weights and a decision rule.)
- **Never change the CV split.** `StratifiedKFold(5, shuffle=True, random_state=42)`. Every archived
  OOF matrix is aligned to it; changing it invalidates all cross-run blending retroactively.
- **Always write the per-learner OOF + test probability artifacts**, for every run, even throwaway
  ones. This is the highest-ROI habit in the playbook.
- **Fold-score spread is evaluation-fold difficulty, not test-prediction variance.** Do not respond to
  a wide spread by re-running at a new split seed; S6E7 spent 2.3 GPU-hours proving that buys nothing.
- **Every probe needs a written hypothesis and a pre-registered gate before it runs**, and its result
  is logged with the *mechanism*, not just the number. A well-written negative result is worth more
  than a +0.0001 blend. The gate value itself stays unset in `README.md` until the OOF↔LB residual σ
  has been measured over ~10 paired runs — do not invent one earlier.
