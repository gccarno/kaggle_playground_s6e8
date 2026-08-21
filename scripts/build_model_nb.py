#!/usr/bin/env python3
"""Build the config-driven modelling notebook.

  python build_model_nb.py                           -> s6e8-model.ipynb, local defaults
  python build_model_nb.py <out.ipynb> '<cfg json>'  -> a kernel pinned to that CFG

Kaggle offers no way to pass an env var into a kernel, so a kernel is configured by baking
KAGGLE_CFG into its own copy of the notebook. One copy per learner lets the GPU kernels run
concurrently instead of serially.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\gccar\Documents\kaggle_contests\playground_s6e8")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "s6e8-model.ipynb"
if not OUT.is_absolute():
    OUT = REPO / OUT
KCFG = sys.argv[2] if len(sys.argv) > 2 else '{"run_tag": "D1_catboost", "learner": "cat", "te_cols": "num"}'

C = []
def md(s): C.append(("markdown", s.strip("\n")))
def code(s): C.append(("code", s.strip("\n")))

md(r"""
# S6E8 — modelling pipeline

The experiment notebook. `eda-s6e8.ipynb` is the frozen EDA record; this is where probes run.

Everything that varies between runs lives in one `CFG` dict, overridable from the `S6E8_CFG`
environment variable. On Kaggle the variable is absent and the defaults below are used, so the same
file is both the local probe runner and the pushed kernel.

**Strict-twin discipline** (`KAGGLE_PLAYBOOK.md` §3): a probe changes exactly one `CFG` field from
the champion. `scripts/run_local.py --diff-vs` checks this before the run starts.
""")

code(r"""
import json, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)
T_START = time.time()

# ---- frozen contract (README.md). Never overridable by CFG. ----------------
SEED, N_FOLDS = 42, 5
TARGET, ID = "addicted_label", "id"
COMP = "playground-series-s6e8"

# ---- everything a probe is allowed to change ------------------------------
DEFAULTS = {
    "run_tag":       "champion",
    "learner":       "lgb",        # lgb | xgb | cat | mlp | realmlp | tabm | ftt | tabtf | node | lookupt | mnca
    "model_seed":    42,           # model randomness ONLY -- never the CV split seed
    "fe_interaction": False,       # B1: opens_per_hour, notif_per_hour, notif_per_open
    "fe_composition": False,       # B2: other_hours, share_*, weekend_ratio, screen_total
    "fe_normalization": False,     # B3: free_time, screen_per_sleep, screen_per_age, social_per_age
    "fe_impute":     False,        # P2-B3: model-impute the 9 numerics and add the results as
                                   # EXTRA columns (imp_*) plus the engineered families
                                   # recomputed on them (impfe_*). The originals keep their
                                   # NaNs. Augment, never replace -- see the cell below.
    "drop_flat_cats": False,       # B5: drop gender/stress_level/academic_work_impact
    "num_as_cat":    False,        # D2: declare the 9 numeric columns CATEGORICAL, acting on the
                                   # EDA finding that their value->target map is a lookup table
                                   # (rates 0.119..0.986, Pearson(value, rate) = -0.044) rather
                                   # than a function -- i.e. the values are labels, not magnitudes
    "te_cols":       [],           # B6: per-value TARGET encoding of these columns ("num" = all numeric)
    "te_crosses":    [],           # B7: target-encode VALUE PAIRS, e.g. [["a","b"]]
    "te_smooth":     20.0,         # prior weight in the smoothed rate
    "te_inner_folds": 5,           # inner OOF folds used to encode the training rows
    "mlp_embed":     [],           # E2: learn an nn.Embedding per listed column instead of
                                   # feeding its value. The regularised answer to D2: a split
                                   # on a 231-level column overfits, an embedding under weight
                                   # decay is the same lookup fitted with a penalty.
    "ftt_miss_mask": False,        # V-C1: give the ftt/tabtf branch the missingness-mask
                                   # block the MLP branch has always had. FEATURES.md 2.9
                                   # note 3 records the difference between the two branches
                                   # as "an inconsistency ... not a decision".
    "lookup_scalar_only": False,   # V-C3: lookupt/mnca WITHOUT the per-value embedding --
                                   # rank-gauss + PLR scalars only. Phase 4's one open
                                   # question, and the controlled test of the claim that
                                   # the REPRESENTATION (not the architecture) is the unit
                                   # of ensemble value.
    "lookup_allow_te": False,      # V-C4: relax the te_cols ban on the lookup learners.
                                   # FEATURES.md 2.8 backs that ban with an assert and a
                                   # mechanism argument, never a measurement.
    "lgb_params":    {},           # merged over the baseline params (all learners read it)
    "n_estimators":  8000,
    "early_stop":    400,
}
KAGGLE_DIR = Path("/kaggle/input/competitions") / COMP
ON_KAGGLE  = KAGGLE_DIR.exists()

# Pinned config for KERNEL runs. Kaggle offers no way to pass an env var into a
# kernel, so a kernel run is configured by editing this dict before
# `kaggle kernels push`. Ignored locally, where run_local.py sets S6E8_CFG instead.
# Kernels exist for runs too long for the ~10-minute local budget (Kaggle allows 9h).
KAGGLE_CFG = __KCFG__

CFG = {**DEFAULTS,
       **(KAGGLE_CFG if ON_KAGGLE else {}),
       **json.loads(os.environ.get("S6E8_CFG", "{}"))}
assert set(CFG) <= set(DEFAULTS), f"unknown CFG keys: {set(CFG) - set(DEFAULTS)}"
DATA_DIR   = KAGGLE_DIR if ON_KAGGLE else Path("data")
OUT_DIR    = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("S6E8_OUT", "experiments/preds/local"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"on_kaggle={ON_KAGGLE}  out={OUT_DIR}")
print("CFG:", json.dumps(CFG, indent=1))
""")

code(r"""
# Kaggle-only dependency install. The package is looked up FROM CFG["learner"], so this cell
# cannot drift out of sync with what the fold loop actually imports -- S6E7 lost time to
# exactly that failure (pytabkit missing at run time). Local runs already have these.
EXTRA_PKG = {"realmlp": "pytabkit", "tabm": "pytabkit", "ftt": "rtdl_revisiting_models",
             "tabtf": "tab-transformer-pytorch", "node": "pytorch_tabular"}
if ON_KAGGLE and CFG["learner"] in EXTRA_PKG:
    import subprocess, sys as _sys
    pkg = EXTRA_PKG[CFG["learner"]]
    print(f"installing {pkg} ...", flush=True)
    r = subprocess.run([_sys.executable, "-m", "pip", "install", "-q", pkg],
                       capture_output=True, text=True)
    print((r.stdout or "")[-1200:], (r.stderr or "")[-1200:])
    # Fail LOUDLY here rather than 40 minutes into fold 0 with an ImportError.
    assert r.returncode == 0, f"pip install {pkg} FAILED"

import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
""")

md(r"""
## Load and feature engineering

**Leakage note.** Every engineered feature below is *row-wise arithmetic on that row's own values* —
no aggregation across rows, no reference to `y`. It is therefore trivially leak-free and correctly
computed once, before the CV loop. This is the deliberate exception to `README.md`'s
fit-inside-the-fold rule: that rule exists for transforms that *learn* something from the data
(target encoders, bin edges, scalers), and none of these do. Do not "fix" this by moving it into the
fold loop — it would change nothing and cost 5× the compute.
""")

code(r"""
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
sub   = pd.read_csv(DATA_DIR / "sample_submission.csv")

RAW_CAT = ["gender", "stress_level", "academic_work_impact"]
RAW_NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
           "work_study_hours", "sleep_hours", "notifications_per_day",
           "app_opens_per_day", "weekend_screen_time"]

# Denominators must be strictly positive or the ratios below silently produce inf.
# Assert it rather than trusting the EDA snapshot.
for col, lo in [("daily_screen_time_hours", 0), ("app_opens_per_day", 0),
                ("sleep_hours", 0), ("age", 0)]:
    for name, df in (("train", train), ("test", test)):
        bad = (df[col].dropna() <= lo).sum()
        assert bad == 0, f"{name}.{col} has {bad} values <= {lo}; ratios would be inf"
print("denominator safety: all strictly positive in train and test")


def engineer(df):
    # Row-wise, unsupervised, leak-free by construction. Returns (df, added_cols).
    out, added = df.copy(), []

    if CFG["fe_interaction"]:
        # B1. The rank-gap finding: these two rank 9th/6th on marginal AUC but 1st/2nd on
        # split usage. A GBDT cannot express x/y with axis-aligned splits, so it burns
        # capacity approximating these; hand them over explicitly.
        out["opens_per_hour"] = df.app_opens_per_day / df.daily_screen_time_hours
        out["notif_per_hour"] = df.notifications_per_day / df.daily_screen_time_hours
        out["notif_per_open"] = df.notifications_per_day / df.app_opens_per_day
        added += ["opens_per_hour", "notif_per_hour", "notif_per_open"]

    if CFG["fe_composition"]:
        # B2. daily_screen_time is a confirmed composition: total - parts is NEVER negative
        # across 374,297 complete rows, so "other_hours" is a real latent variable.
        parts = df.social_media_hours + df.gaming_hours + df.work_study_hours
        out["other_hours"]   = df.daily_screen_time_hours - parts
        out["share_social"]  = df.social_media_hours / df.daily_screen_time_hours
        out["share_gaming"]  = df.gaming_hours / df.daily_screen_time_hours
        out["share_work"]    = df.work_study_hours / df.daily_screen_time_hours
        out["share_other"]   = out["other_hours"] / df.daily_screen_time_hours
        out["weekend_ratio"] = df.weekend_screen_time / df.daily_screen_time_hours
        out["screen_total"]  = df.daily_screen_time_hours + df.weekend_screen_time
        added += ["other_hours", "share_social", "share_gaming", "share_work",
                  "share_other", "weekend_ratio", "screen_total"]

    if CFG["fe_normalization"]:
        # B3. Absolute screen time means something different at 18 than at 35, and
        # against 5 vs 9 hours of sleep.
        out["free_time"]        = 24 - df.sleep_hours - df.daily_screen_time_hours
        out["screen_per_sleep"] = df.daily_screen_time_hours / df.sleep_hours
        out["screen_per_age"]   = df.daily_screen_time_hours / df.age
        out["social_per_age"]   = df.social_media_hours / df.age
        added += ["free_time", "screen_per_sleep", "screen_per_age", "social_per_age"]

    return out, added


train_fe, added = engineer(train)
test_fe,  _     = engineer(test)


def impute_numeric(tr, te):
    # One XGBRegressor per numeric column, fit on the OBSERVED rows of train + test.
    #
    # LEAKAGE STATUS -- read before moving this. The target is never involved: each model
    # predicts a FEATURE from other features. That makes it transductive UNSUPERVISED
    # preprocessing, which README.md section 2 licenses outside the CV loop, exactly like
    # the category vocabulary and the lookup-id mapping. A target encoder in this position
    # would be a leak; this is not one.
    #
    # It is, however, the one feature family here that is NOT row-wise: a row's imputed
    # value depends on other rows. That is why the row-wise purity assertion below is
    # scoped to exclude these columns rather than being deleted.
    # (Comments, not a docstring -- a triple-quoted string here would terminate the
    # generator's own code(r\"\"\"...\"\"\") block.)
    import xgboost as xgb
    n = len(tr)
    full = pd.concat([tr[RAW_NUM + RAW_CAT], te[RAW_NUM + RAW_CAT]], ignore_index=True)
    for c in RAW_CAT:
        full[c] = full[c].astype("category")
    out = full[RAW_NUM].copy()
    for col in RAW_NUM:
        obs = full[col].notna().to_numpy()
        feats = [c for c in RAW_NUM + RAW_CAT if c != col]
        m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                             tree_method="hist", enable_categorical=True, n_jobs=-1,
                             random_state=SEED)
        m.fit(full.loc[obs, feats], full.loc[obs, col])
        if (~obs).sum():
            out.loc[~obs, col] = m.predict(full.loc[~obs, feats])
    return out.iloc[:n].reset_index(drop=True), out.iloc[n:].reset_index(drop=True)


IMPUTED_COLS = []
if CFG["fe_impute"]:
    # AUGMENT, NEVER REPLACE. The same imputer measures with opposite signs publicly:
    # replacing the NaNs is negative, adding the imputed values alongside them is +0.0012.
    # A GBM's native NaN handling learns a default split DIRECTION per node, which is
    # strictly more expressive than one imputed point estimate dragging the row toward the
    # middle of the distribution. What the imputed copy buys is coverage: a ratio is NaN
    # whenever EITHER operand is missing, which is ~39% of rows, so the engineered families
    # are unavailable exactly where they would say the most.
    t0_imp = time.time()
    tr_imp, te_imp = impute_numeric(train, test)
    # The families recomputed on complete values. engineer() is reused rather than having
    # its formulas restated, so the two copies cannot drift apart.
    tr_imp_fe, imp_added = engineer(tr_imp)
    te_imp_fe, _         = engineer(te_imp)
    for dst, raw_src, fe_src in ((train_fe, tr_imp, tr_imp_fe), (test_fe, te_imp, te_imp_fe)):
        for c in RAW_NUM:
            dst[f"imp_{c}"] = raw_src[c].to_numpy()
        for c in imp_added:
            dst[f"impfe_{c}"] = fe_src[c].to_numpy()
    IMPUTED_COLS = [f"imp_{c}" for c in RAW_NUM] + [f"impfe_{c}" for c in imp_added]
    added = added + IMPUTED_COLS
    print(f"fe_impute: +{len(IMPUTED_COLS)} columns in {time.time() - t0_imp:.0f}s "
          f"(originals keep their NaNs)")

cat_cols = [] if CFG["drop_flat_cats"] else RAW_CAT
FEATURES = RAW_NUM + added + cat_cols
print(f"{len(FEATURES)} features = {len(RAW_NUM)} raw num + {len(added)} engineered + {len(cat_cols)} cat")
print(f"engineered: {added or '(none)'}")
""")

code(r"""
# ---- FE correctness, asserted not assumed ---------------------------------
for name, df in (("train", train_fe), ("test", test_fe)):
    for c in added:
        v = df[c].to_numpy(dtype="float64")
        finite = v[~np.isnan(v)]
        assert np.isfinite(finite).all(), f"{name}.{c} contains inf"
if CFG["fe_composition"]:
    for name, df in (("train", train_fe), ("test", test_fe)):
        neg = (df["other_hours"].dropna() < -1e-9).sum()
        assert neg == 0, f"{name}.other_hours negative on {neg} rows -- composition assumption broken"
    print("composition check: other_hours >= 0 everywhere in train and test")

# Row-wise purity: recomputing FE on a shuffled subset must give identical values.
# This is what proves no cross-row aggregation crept in (which WOULD be a leak).
#
# The imputed columns are DELIBERATELY excluded, and this is the only exemption in the
# notebook. Imputation is genuinely cross-row -- a row's imputed value is predicted from a
# model fitted on other rows -- so it cannot satisfy this invariant by construction. It is
# still leak-free with respect to the TARGET, which is the property that matters and which
# the certification cell checks separately. Scoping the assertion is the honest move;
# deleting it would silently stop checking the eight families that ARE row-wise.
row_wise = [c for c in added if c not in IMPUTED_COLS]
if row_wise:
    probe = train.sample(2000, random_state=0)
    re_fe, _ = engineer(probe)
    ref = train_fe.loc[probe.index, row_wise]
    assert np.allclose(re_fe[row_wise].to_numpy(dtype="float64"),
                       ref.to_numpy(dtype="float64"), equal_nan=True), \
        "FE is not row-wise -- values changed when computed on a subset"
    print(f"row-wise purity: {len(row_wise)} FE columns identical on a 2000-row shuffled "
          "subset -> no cross-row leakage")
if IMPUTED_COLS:
    # What we CAN assert about the imputed block: it never sees the target, it is complete
    # where the source column was observed, and it left the originals alone.
    for c in RAW_NUM:
        obs = train[c].notna().to_numpy()
        assert np.allclose(train_fe.loc[obs, f"imp_{c}"].to_numpy(dtype="float64"),
                           train[c][obs].to_numpy(dtype="float64")), \
            f"imp_{c} altered an OBSERVED value -- imputation must only fill gaps"
        assert train_fe[c].isna().sum() == train[c].isna().sum(), \
            f"{c} lost its NaNs -- fe_impute must AUGMENT, never replace"
    print(f"imputed block: {len(IMPUTED_COLS)} columns, observed values untouched, "
          "originals still carry their NaNs")

print(f"\nengineered-column coverage (non-null %):")
for c in added:
    print(f"  {c:<20} {train_fe[c].notna().mean()*100:5.1f}%")
""")

md(r"""
## Per-value target encoding (supervised — leakage discipline applies here)

Everything above is unsupervised row arithmetic. **This is not.** It maps each raw *value* to the
rate at which that value's rows are positive, so it touches `y` and must be fit strictly inside the
training fold — `README.md`'s load-bearing rule.

Why it should work here, and why B1 could not: the value→target relationship is **non-monotone**.
Individual `notifications_per_day` values carry positive rates spanning **0.119 to 0.986**, yet
Pearson(value, rate) is −0.044 and only 47.8% of consecutive steps increase. AUC only sees monotone
association, which is why the raw column scores 0.492; a tree *can* carve out individual values,
which is why it spent the most splits there. No unsupervised transform — ratio, sum, or quantile
bin — can reconstruct a lookup table. Playbook §6: *ask whether the lever is supervised before
building the unsupervised imitation of it.*

**Nesting.** A single fit on the outer training fold would let each training row see its own label
through its value's mean, so the model would learn to trust the encoding far more than it should and
the OOF would be inflated. Instead:

- training rows are encoded **out-of-fold within the training fold** (inner K-fold);
- validation and test rows are encoded with the mapping fit on the **whole** outer training fold;
- unseen values fall back to the training fold's prior, never to a global statistic and never raising.
""")

code(r"""
TE_COLS = RAW_NUM if CFG["te_cols"] == "num" else list(CFG["te_cols"])
TE_COLS = [c for c in TE_COLS if c in FEATURES]

# Integer codes per distinct value, computed ONCE over train+test so the code space
# is shared. This is an unsupervised label mapping (it never touches y), so doing it
# outside the fold loop is not a leak -- same argument as the categorical vocabulary.
# NaN gets its own code rather than being dropped, so "missing" carries its own rate.
TE_CODES = {}
for c in TE_COLS:
    codes, _ = pd.factorize(pd.concat([train_fe[c], test_fe[c]]), use_na_sentinel=False)
    TE_CODES[c] = (codes[:len(train_fe)], codes[len(train_fe):], int(codes.max()) + 1)

def _rate_table(codes_fit, y_fit, k, prior, m):
    # Smoothed per-value positive rate via bincount -- O(n), no groupby/map.
    s = np.bincount(codes_fit, weights=y_fit, minlength=k)
    n = np.bincount(codes_fit, minlength=k)
    return (s + prior * m) / (n + m)          # unseen value -> n=0 -> exactly the prior

def target_encode(c, itr, iva, seed, n_inner, m):
    # Returns (enc_train, enc_val, enc_test) for column c.
    # y is read ONLY at rows in itr -- the outer training fold.
    ctr_all, cte_all, k = TE_CODES[c]
    y_tr = y[itr]
    prior = float(y_tr.mean())

    enc_tr = np.empty(len(itr))
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for i_fit, i_app in inner.split(np.zeros(len(itr)), y_tr):
        tbl = _rate_table(ctr_all[itr][i_fit], y_tr[i_fit], k, prior, m)
        enc_tr[i_app] = tbl[ctr_all[itr][i_app]]

    full = _rate_table(ctr_all[itr], y_tr, k, prior, m)
    return enc_tr, full[ctr_all[iva]], full[cte_all]

# B7: value CROSSES. The supervised version of what B1 tried and failed at -- if the
# single-column relationship is a lookup table, the pairwise one may be too, and a
# cross is the only encoding that can express "this combination of values" rather than
# "this value of x, adjusted for y".
#
# Cardinality is the whole risk here. A cross of two columns with 231 and 166 levels
# has up to 38k cells over 691k rows (~18/cell), versus ~2700 rows/value for a single
# column. Self-influence per row rises from ~0.03% to ~3%, so the nesting and the
# smoothing stop being a formality -- and the train/val gap audit, which had no power
# on single columns, becomes a real test.
for a, b in CFG["te_crosses"]:
    ca_tr, ca_te, ka = TE_CODES[a] if a in TE_CODES else (None, None, None)
    if ca_tr is None:
        codes, _ = pd.factorize(pd.concat([train_fe[a], test_fe[a]]), use_na_sentinel=False)
        ca_tr, ca_te, ka = codes[:len(train_fe)], codes[len(train_fe):], int(codes.max()) + 1
    cb_tr, cb_te, kb = TE_CODES[b] if b in TE_CODES else (None, None, None)
    if cb_tr is None:
        codes, _ = pd.factorize(pd.concat([train_fe[b], test_fe[b]]), use_na_sentinel=False)
        cb_tr, cb_te, kb = codes[:len(train_fe)], codes[len(train_fe):], int(codes.max()) + 1
    name = f"{a}_X_{b}"
    joint = np.concatenate([ca_tr * kb + cb_tr, ca_te * kb + cb_te])
    dense, _ = pd.factorize(joint)                      # compact the sparse product space
    TE_CODES[name] = (dense[:len(train_fe)], dense[len(train_fe):], int(dense.max()) + 1)
    TE_COLS.append(name)
    occ = np.bincount(dense[:len(train_fe)])
    print(f"cross {name}: {TE_CODES[name][2]:,} observed cells, "
          f"{len(train_fe)/TE_CODES[name][2]:.1f} rows/cell median-occupancy {int(np.median(occ))}")

print(f"target-encoding {len(TE_COLS)} column(s): {TE_COLS or '(none)'}")
if TE_COLS:
    print(f"  smoothing m={CFG['te_smooth']}, inner folds={CFG['te_inner_folds']}", flush=True)
""")

md(r"""
## Cross-validation

The frozen split, unchanged: `StratifiedKFold(5, shuffle=True, random_state=42)`. `CFG["model_seed"]`
moves the *model's* randomness only — the partition never moves, which is what keeps every archived
OOF matrix blendable with every other.
""")

code(r"""
X, X_test = train_fe[FEATURES].copy(), test_fe[FEATURES].copy()

# D2: the numerics can also be declared categorical. The vocabulary is still built over
# train + test and never touches y, so this stays an UNSUPERVISED label mapping -- the
# same argument that licenses hoisting it out of the CV loop for the real categoricals.
# True = all nine; a list = just those columns. D2 showed the distinction is the whole
# point: the screen-time columns are strongly MONOTONE (solo AUC 0.86-0.89 over 700-1400
# levels), so categorifying them throws away real ordinal signal and hands LightGBM a
# 1400-level feature to overfit -- -0.004 OOF. The lookup-table structure was only ever
# a property of the low-AUC, high-split-usage columns.
NUM_AS_CAT  = (RAW_NUM if CFG["num_as_cat"] is True
               else [c for c in (CFG["num_as_cat"] or []) if c in FEATURES])
AS_CATEGORY = cat_cols + NUM_AS_CAT
for c in AS_CATEGORY:
    levels = pd.Index(sorted(set(train[c].dropna()) | set(test[c].dropna())))
    X[c]      = pd.Categorical(X[c], categories=levels)
    X_test[c] = pd.Categorical(X_test[c], categories=levels)
if NUM_AS_CAT:
    print("num_as_cat: " + ", ".join(f"{c}={X[c].cat.categories.size} levels" for c in NUM_AS_CAT))
y = train[TARGET].values

BASE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                   n_jobs=-1, verbose=-1)
params = {**BASE_PARAMS, **CFG["lgb_params"],
          "n_estimators": CFG["n_estimators"], "random_state": CFG["model_seed"]}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
assert skf.n_splits == 5 and skf.shuffle is True and skf.random_state == 42, "CV split drifted!"

oof        = np.zeros(len(X))
fold_id    = np.full(len(X), -1, dtype=int)
test_sum   = np.zeros(len(X_test), dtype=np.float64)
fold_aucs, best_iters = [], []

for k, (itr, iva) in enumerate(skf.split(X, y)):
    fold_id[iva] = k
    Xtr, Xva, Xte = X.iloc[itr].copy(), X.iloc[iva].copy(), X_test.copy()

    # Supervised encoding -- re-fit from scratch inside THIS fold every time.
    # Never hoist this out of the loop: it is the one transform here that sees y.
    for c in TE_COLS:
        e_tr, e_va, e_te = target_encode(
            c, itr, iva, seed=SEED + k,
            n_inner=CFG["te_inner_folds"], m=CFG["te_smooth"])
        Xtr[f"{c}__te"], Xva[f"{c}__te"], Xte[f"{c}__te"] = e_tr, e_va, e_te

    if CFG["learner"] == "lgb":
        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, y[itr], eval_set=[(Xva, y[iva])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(CFG["early_stop"], verbose=False)])
        best_it = model.best_iteration_
    elif CFG["learner"] == "xgb":
        import xgboost as xgb
        # lgb_params is the single knob dict for every learner, so XGB-specific keys
        # arrive through it and must OVERRIDE these defaults rather than collide with
        # them -- passing max_depth both ways is a duplicate-kwarg TypeError.
        xgb_kw = dict(
            n_estimators=CFG["n_estimators"], learning_rate=params["learning_rate"],
            max_depth=6, tree_method="hist", enable_categorical=True,
            eval_metric="auc", early_stopping_rounds=CFG["early_stop"],
            random_state=CFG["model_seed"], n_jobs=-1)
        xgb_kw.update({k: v for k, v in CFG["lgb_params"].items()
                       if k not in ("num_leaves", "min_child_samples")})
        model = xgb.XGBClassifier(**xgb_kw)
        model.fit(Xtr, y[itr], eval_set=[(Xva, y[iva])], verbose=False)
        best_it = model.best_iteration
    elif CFG["learner"] == "cat":
        from catboost import CatBoostClassifier
        # CatBoost rejects NaN inside a categorical feature, so missing gets its own
        # explicit level. That is an UNSUPERVISED relabelling (it never touches y) --
        # same argument as the pandas category vocabulary, not a leak.
        def _cat_ready(df):
            d = df.copy()
            for c in AS_CATEGORY:
                d[c] = d[c].astype(object).where(d[c].notna(), "__NA__").astype(str)
            return d
        Ctr, Cva, Cte = _cat_ready(Xtr), _cat_ready(Xva), _cat_ready(Xte)
        model = CatBoostClassifier(
            iterations=CFG["n_estimators"], learning_rate=params["learning_rate"],
            depth=6, eval_metric="AUC", random_seed=CFG["model_seed"],
            early_stopping_rounds=CFG["early_stop"], thread_count=-1,
            allow_writing_files=False, verbose=False,
            **{k: v for k, v in CFG["lgb_params"].items()
               if k not in ("learning_rate", "num_leaves", "min_child_samples")})
        model.fit(Ctr, y[itr], eval_set=(Cva, y[iva]), cat_features=AS_CATEGORY, verbose=False)
        best_it = model.get_best_iteration()   # read-only property, unlike lgb/xgb
        Xva, Xte = Cva, Cte      # predict from the same representation we trained on
    elif CFG["learner"] == "mlp":
        # A FOURTH FAMILY. Every within-family tuning probe has missed the gate (C1's
        # blend, C2, C3, D2, D2b); the only win since target encoding came from adding a
        # family (D1 CatBoost). A neural net shares no inductive bias with a tree at all,
        # so it is the largest remaining decorrelation bet.
        import torch, torch.nn as nn
        hp = {"hidden": [256, 128, 64], "dropout": 0.15, "lr": 1e-3, "wd": 1e-4,
              "batch": 4096, "epochs": 60, **CFG["lgb_params"]}
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(CFG["model_seed"])

        # E2 embeddings. Codes come from TE_CODES, which was factorised over train + test and
        # never touches y -- unsupervised, so it is safe outside the fold loop. The embedded
        # column is ALSO left in the numeric block, exactly as B1 kept its raw columns.
        emb_cols = [c for c in CFG["mlp_embed"] if c in TE_CODES]
        missing_emb = [c for c in CFG["mlp_embed"] if c not in TE_CODES]
        assert not missing_emb, f"mlp_embed needs these in te_cols to get codes: {missing_emb}"
        emb_spec = [(c, TE_CODES[c][2], int(min(24, round(TE_CODES[c][2] ** 0.5) + 1)))
                    for c in emb_cols]
        if emb_spec:
            print("    embeddings: " + ", ".join(f"{c}({k}->{d})" for c, k, d in emb_spec))

        def codes_for(which):
            if not emb_cols:
                return None
            cols = []
            for c in emb_cols:
                ctr, cte, _ = TE_CODES[c]
                cols.append({"tr": ctr[itr], "va": ctr[iva], "te": cte}[which])
            return torch.from_numpy(np.stack(cols, 1).astype("int64"))

        num_cols = [c for c in Xtr.columns if c not in AS_CATEGORY]

        # Imputation and standardisation are fit on the TRAINING FOLD ONLY and applied to
        # val/test -- the same rule as the target encoder. A tree needed neither, so this
        # is the first place in the notebook where scaling discipline is load-bearing.
        med = Xtr[num_cols].median()
        mu  = Xtr[num_cols].fillna(med).mean()
        sd  = Xtr[num_cols].fillna(med).std().replace(0, 1.0)

        def to_tensor(df):
            num = ((df[num_cols].fillna(med) - mu) / sd).to_numpy(dtype="float32")
            miss = df[num_cols].isna().to_numpy(dtype="float32")       # missingness is signal
            parts = [num, miss]
            for c in AS_CATEGORY:
                codes = df[c].cat.codes.to_numpy()                     # -1 for NaN
                oh = np.zeros((len(df), len(df[c].cat.categories) + 1), dtype="float32")
                oh[np.arange(len(df)), codes + 1] = 1.0                # column 0 == missing
                parts.append(oh)
            return torch.from_numpy(np.hstack(parts))

        Ttr, Tva, Tte = to_tensor(Xtr), to_tensor(Xva), to_tensor(Xte)
        Ctr, Cva, Cte = codes_for("tr"), codes_for("va"), codes_for("te")
        ytr_t = torch.from_numpy(y[itr].astype("float32"))

        class Net(nn.Module):
            def __init__(self, d_num):
                super().__init__()
                self.emb = nn.ModuleList([nn.Embedding(k, d) for _, k, d in emb_spec])
                d_in = d_num + sum(d for _, _, d in emb_spec)
                blocks = []
                for h in hp["hidden"]:
                    blocks += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.GELU(),
                               nn.Dropout(hp["dropout"])]
                    d_in = h
                blocks += [nn.Linear(d_in, 1)]
                self.body = nn.Sequential(*blocks)

            def forward(self, xn, xc=None):
                if self.emb:
                    xn = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], 1)
                return self.body(xn)

        net = Net(Ttr.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
        lossf = nn.BCEWithLogitsLoss()

        def infer(T, C=None):
            net.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(T), 16384):
                    cb = None if C is None else C[i:i+16384].to(dev)
                    out.append(torch.sigmoid(net(T[i:i+16384].to(dev), cb).squeeze(1)).cpu().numpy())
            return np.concatenate(out)

        # Early stopping on ROC AUC -- the competition metric, never a loss proxy (README).
        best_auc, best_state, best_ep, since = -1.0, None, 0, 0
        n = len(Ttr)
        for ep in range(hp["epochs"]):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, hp["batch"]):
                idx = perm[i:i+hp["batch"]]
                opt.zero_grad()
                cb = None if Ctr is None else Ctr[idx].to(dev)
                loss = lossf(net(Ttr[idx].to(dev), cb).squeeze(1), ytr_t[idx].to(dev))
                loss.backward(); opt.step()
            sched.step()
            a = roc_auc_score(y[iva], infer(Tva, Cva))
            if a > best_auc:
                best_auc, best_ep, since = a, ep, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since += 1
                if since >= 8:
                    break
        net.load_state_dict(best_state)

        class _Wrap:
            # Minimal predict_proba shim so the shared fold code below is untouched.
            def __init__(self, va, te): self.va, self.te = va, te
            def predict_proba(self, X_):
                p = self.va if len(X_) == len(Xva) else self.te
                return np.column_stack([1 - p, p])
        model = _Wrap(infer(Tva, Cva), infer(Tte, Cte))
        best_it = best_ep
        n_emb = sum(d for _, _, d in emb_spec)
        print(f"    mlp: {Ttr.shape[1]}+{n_emb} in, best epoch {best_ep} "
              f"(val AUC {best_auc:.6f}), dev={dev}", flush=True)
    elif CFG["learner"] in ("lookupt", "mnca"):
        # LOOKUP-TRANSFORMER. Architecture from tamerlanomralinov's public S6E8 notebook,
        # retrained here on OUR frozen 5-fold split (his uses 10, so his published OOF is
        # fitted on a different partition and would leak if stacked -- the architecture
        # transfers, the predictions do not).
        #
        # WHY THIS AND NOT ANOTHER NET. Every neural leg we own -- mlp, mlp_embed, realmlp,
        # ftt, node -- reads the same REPRESENTATION and so bought strength without
        # diversity (they sit 1.9-2.4% from each other). The generator memorised
        # value->label associations: notifications_per_day has univariate AUC 0.492 yet its
        # per-value residuals correlate 0.72 across two independent halves of the data.
        # B6 target encoding already exploits that, but collapses each lattice cell to ONE
        # closed-form scalar. This gives every exact observed value its own d-dim embedding,
        # learned end to end, and runs attention over feature tokens so pairwise
        # interactions are formed explicitly rather than by summation.
        #
        #     token_j = Embedding_j[exact_value_id] + PLR_j(rank_gauss value) * (1 - missing)
        #
        # THE PREDICTION IS DECORRELATION, NOT STRENGTH. Pre-registered: solo OOF >= 0.9655
        # and max |corr| against our 16 existing legs <= 0.990. Strong-but-correlated
        # refutes the mechanism regardless of the solo number.
        import math
        import torch, torch.nn as nn
        from sklearn.preprocessing import QuantileTransformer

        hp = {"d": 128, "k": 24, "layers": 4, "heads": 8, "drop": 0.1, "batch": 2048,
              "epochs": 32, "lr": 2e-3, "aug": 0.10, "wd_emb": 3e-4, "wd": 1e-5,
              "patience": 5, **CFG["lgb_params"]}
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(CFG["model_seed"] + k)

        # The 12 raw columns are lattice-valued and get lookup tables; the engineered
        # budget columns are continuous and get PLR tokens only. te_cols must be empty for
        # this learner -- the embedding IS the lookup, so a target encoding on top would be
        # the same structure read twice and would recorrelate this leg with the tree pack.
        # V-C4. The ban is real reasoning -- the embedding IS the lookup, so a target
        # encoding on top reads the same structure twice and should recorrelate this leg
        # with the tree pack, destroying the only thing it was built for. But it has only
        # ever been an assert, never a measurement, and this leg is worth +0.00052.
        # lookup_allow_te lifts it for exactly one probe; the default keeps the guard.
        assert not TE_COLS or CFG["lookup_allow_te"], (
            f"{CFG['learner']}: set te_cols=[] -- the embedding already IS the lookup "
            "(set lookup_allow_te to measure this rather than assume it)")
        LOOK_NUM = [c for c in RAW_NUM if c in FEATURES]
        LOOK_CAT = list(AS_CATEGORY)
        PLR_ONLY = [c for c in FEATURES if c not in LOOK_NUM + LOOK_CAT]

        # Built once and reused across folds: this is an UNSUPERVISED value->int mapping
        # over train + test (it never touches y), which README.md section 2 licenses
        # hoisting out of the CV loop -- the same argument as the pandas category
        # vocabulary. The fold loop body is module scope, so fold 0 builds it and folds
        # 1-4 reuse it.
        try:
            LOOK_VOCAB
        except NameError:
            LOOK_VOCAB, off, tot = {}, 0, 0
            for c in LOOK_NUM + LOOK_CAT:
                both_c = pd.concat([X[c], X_test[c]], ignore_index=True)
                levels = pd.Index(sorted(both_c.dropna().unique()))
                # id 0 is reserved for NaN in EVERY column, so "missing" gets its own
                # learned vector per column instead of an imputed value.
                LOOK_VOCAB[c] = (pd.Series(np.arange(1, len(levels) + 1), index=levels), off)
                off += len(levels) + 1
            tot = off
            LOOK_TOTV = tot
            print(f"    lookup vocab: {LOOK_TOTV:,} embedding rows over "
                  f"{len(LOOK_NUM) + len(LOOK_CAT)} columns", flush=True)

        def look_ids(df):
            cols = []
            for c in LOOK_NUM + LOOK_CAT:
                m, off = LOOK_VOCAB[c]
                v = df[c]
                if isinstance(v.dtype, pd.CategoricalDtype):
                    v = v.astype(object)
                cols.append(v.map(m).fillna(0).to_numpy(dtype="int64") + off)
            return torch.from_numpy(np.stack(cols, 1))

        # Rank-gauss for the smooth branch. THIS one learns from the data, so it is fit on
        # the TRAINING FOLD ONLY and applied to val/test -- same rule as the target encoder.
        def rank_gauss(cols):
            qts, out = {}, {}
            for c in cols:
                v = Xtr[c].to_numpy(dtype="float64")
                o = ~np.isnan(v)
                if o.sum() > 10:
                    q = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                            subsample=400_000, random_state=SEED)
                    q.fit(v[o].reshape(-1, 1))
                    qts[c] = q
            def apply(df):
                V = np.zeros((len(df), len(cols)), dtype="float32")
                M = np.zeros_like(V)
                for j, c in enumerate(cols):
                    v = df[c].to_numpy(dtype="float64")
                    o = ~np.isnan(v)
                    if c in qts and o.sum():
                        V[o, j] = qts[c].transform(v[o].reshape(-1, 1)).ravel()
                    M[~o, j] = 1.0
                return torch.from_numpy(V), torch.from_numpy(M)
            return apply

        # Categoricals carry no meaningful magnitude, so only the numeric lookup columns
        # get a smooth companion; a cat token is its embedding alone.
        num_apply = rank_gauss(LOOK_NUM)
        der_apply = rank_gauss(PLR_ONLY) if PLR_ONLY else None
        NLOOK, NNUM, NDER = len(LOOK_NUM) + len(LOOK_CAT), len(LOOK_NUM), len(PLR_ONLY)

        def pack(df):
            cn, cm = num_apply(df)
            if der_apply is None:
                dn = dm = torch.zeros((len(df), 0), dtype=torch.float32)
            else:
                dn, dm = der_apply(df)
            return look_ids(df), cn, cm, dn, dm

        Ptr, Pva, Pte = pack(Xtr), pack(Xva), pack(Xte)
        ytr_t = torch.from_numpy(y[itr].astype("float32"))

        class PLR(nn.Module):
            # Periodic-Linear numeric embedding with learned Fourier frequencies. Represents
            # a smooth trend far more efficiently than feeding a raw scalar into an MLP.
            def __init__(self, nfeat, kk, d, sigma=0.5):
                super().__init__()
                self.f = nn.Parameter(torch.randn(nfeat, kk) * sigma)
                self.w = nn.Parameter(torch.randn(nfeat, 2 * kk, d) / math.sqrt(2 * kk))
                self.b = nn.Parameter(torch.zeros(nfeat, d))
            def forward(self, x):
                z = 2 * math.pi * x.unsqueeze(-1) * self.f.unsqueeze(0)
                z = torch.cat([torch.sin(z), torch.cos(z)], -1)
                return torch.einsum("bfk,fkd->bfd", z, self.w) + self.b

        if CFG["learner"] == "lookupt":
            class LookupTransformer(nn.Module):
                def __init__(self):
                    super().__init__()
                    d = hp["d"]
                    self.emb = nn.Embedding(LOOK_TOTV, d)
                    nn.init.normal_(self.emb.weight, std=0.02)
                    self.plr_n = PLR(NNUM, hp["k"], d) if NNUM else None
                    self.plr_d = PLR(NDER, hp["k"], d) if NDER else None
                    self.cls = nn.Parameter(torch.zeros(1, 1, d))
                    self.pos = nn.Parameter(torch.randn(1, 1 + NLOOK + NDER, d) * 0.02)
                    self.edrop = nn.Dropout(hp["drop"])
                    enc = nn.TransformerEncoderLayer(d, hp["heads"], d * 2, hp["drop"],
                                                     activation="gelu", batch_first=True,
                                                     norm_first=True)
                    self.tr = nn.TransformerEncoder(enc, hp["layers"])
                    self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                              nn.Dropout(hp["drop"]), nn.Linear(d, 1))
                def forward(self, idx, cn, cm, dn, dm):
                    B = idx.shape[0]
                    # V-C3 lookup_scalar_only: below, the per-value token is ZEROED rather
                    # than the module skipped, so every shape, the position embedding and
                    # the parameter count stay identical and the ONLY thing that changes is
                    # whether the exact-value lookup is readable. That is what makes this an
                    # attribution test of the representation rather than a smaller model.
                    # Built by concatenation rather than in-place assignment into emb(idx):
                    # an in-place write into a tensor that requires grad is an autograd hazard
                    # for no gain here.
                    tok = self.emb(idx)
                    if CFG["lookup_scalar_only"]:
                        tok = torch.zeros_like(tok)
                    if self.plr_n is not None:
                        # Missing -> the smooth term is switched OFF and only Embedding[0]
                        # remains, i.e. a learned per-column "missing" vector.
                        smooth = self.plr_n(cn) * (1 - cm).unsqueeze(-1)
                        tok = torch.cat([tok[:, :NNUM] + smooth, tok[:, NNUM:]], 1)
                    parts = [self.cls.expand(B, -1, -1), tok]
                    if self.plr_d is not None:
                        parts.append(self.plr_d(dn) * (1 - dm).unsqueeze(-1))
                    t = torch.cat(parts, 1) + self.pos
                    return self.head(self.tr(self.edrop(t))[:, 0]).squeeze(-1)

            net = LookupTransformer().to(dev)
            embp = [p for n_, p in net.named_parameters() if n_.startswith("emb")]
            rest = [p for n_, p in net.named_parameters() if not n_.startswith("emb")]
            # Heavier decay on the lookup tables: they are the part with one free vector per
            # observed value and so the part that overfits.
            opt = torch.optim.AdamW([{"params": rest, "weight_decay": hp["wd"]},
                                     {"params": embp, "weight_decay": hp["wd_emb"]}], lr=hp["lr"])
            nstep = math.ceil(len(itr) / hp["batch"]) * hp["epochs"] + 10
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, hp["lr"], total_steps=nstep,
                                                       pct_start=0.15)
            lossf = nn.BCEWithLogitsLoss()

            G = [t.to(dev) for t in Ptr]
            ytr_d = ytr_t.to(dev)
            OFFT = torch.from_numpy(np.array(
                [LOOK_VOCAB[c][1] for c in LOOK_NUM + LOOK_CAT], dtype="int64")).to(dev)
            P_ = list(net.parameters())
            ema = [p.detach().clone() for p in P_]

            def infer(pk, chunk=16384):
                net.eval(); out = []
                with torch.no_grad():
                    for i in range(0, len(pk[0]), chunk):
                        out.append(torch.sigmoid(net(*[t[i:i+chunk].to(dev) for t in pk])
                                                 ).float().cpu().numpy())
                return np.concatenate(out)

            best_auc, best_w, best_ep, since = -1.0, None, 0, 0
            ntr = len(itr)
            # Skip validation over the first few epochs (the EMA is still cold and it costs a
            # full inference pass), but never skip ALL of them -- a short run must still leave
            # best_w set or the load-best below unpacks None.
            eval_from = min(4, hp["epochs"] - 1)
            for ep in range(hp["epochs"]):
                net.train()
                perm = torch.randperm(ntr, device=dev)
                for i in range(0, ntr, hp["batch"]):
                    sl = perm[i:i+hp["batch"]]
                    idx, cm = G[0][sl].clone(), G[2][sl].clone()
                    if hp["aug"] > 0:
                        # Hide extra values at random so the net sees every missingness
                        # pattern, not only the ones train happens to contain. Dropped cells
                        # map to their column's id 0, which is the learned "missing" vector.
                        drop = torch.rand(idx.shape, device=dev) < hp["aug"]
                        idx = torch.where(drop, OFFT.expand_as(idx), idx)
                        cm[:, :NNUM] = torch.maximum(cm[:, :NNUM], drop[:, :NNUM].float())
                    opt.zero_grad(set_to_none=True)
                    loss = lossf(net(idx, G[1][sl], cm, G[3][sl], G[4][sl]), ytr_d[sl])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(P_, 1.0)
                    opt.step(); sched.step()
                    with torch.no_grad():
                        torch._foreach_mul_(ema, 0.999)
                        torch._foreach_add_(ema, [p.detach() for p in P_], alpha=0.001)
                if ep >= eval_from:
                    # Validate the EMA copy, on ROC AUC -- the metric, never a loss proxy.
                    bak = [p.detach().clone() for p in P_]
                    with torch.no_grad():
                        for p, e in zip(P_, ema): p.copy_(e)
                    a = roc_auc_score(y[iva], infer(Pva))
                    if a > best_auc:
                        best_auc, best_ep, since = a, ep, 0
                        best_w = [e.detach().clone() for e in ema]
                    else:
                        since += 1
                    with torch.no_grad():
                        for p, b_ in zip(P_, bak): p.copy_(b_)
                    print(f"    lookupt f{k} ep{ep:3d} val {a:.6f} best {best_auc:.6f}", flush=True)
                    if since >= hp["patience"]:
                        break
            with torch.no_grad():
                for p, e in zip(P_, best_w): p.copy_(e)

            class _WrapL:
                def __init__(self, va, te): self.va, self.te = va, te
                def predict_proba(self, X_):
                    p = self.va if len(X_) == len(Xva) else self.te
                    return np.column_stack([1 - p, p])
            model = _WrapL(infer(Pva), infer(Pte))
            best_it = best_ep
            del G
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"    lookupt: {1 + NLOOK + NDER} tokens, best epoch {best_ep} "
                  f"(val AUC {best_auc:.6f}), dev={dev}", flush=True)

        else:
            # ---- ModernNCA: RETRIEVAL instead of function approximation -------------
            # ModernNCA (Ye et al., ICLR 2025, "Revisiting Nearest Neighbor for Tabular
            # Data") predicts a row by soft k-NN over the training set in a LEARNED
            # metric space:
            #
            #     p(y=1 | q) = sum_j softmax_j( -|| f(q) - f(x_j) ||_2 ) * y_j
            #
            # WHY IT IS RUN ON *THIS* REPRESENTATION AND NOT ON SCALED NUMERICS, which is
            # the entire experiment. Everything above this line is shared verbatim with
            # L1lookupt -- same vocabulary, same rank-gauss, same PLR companion, built by
            # the same code so the two CANNOT drift apart. The only thing that differs is
            # what sits on top of the representation.
            #
            # This repo's central EDA finding is that the value->target map is a LOOKUP
            # TABLE (rates 0.119-0.986, Pearson(value, rate) = -0.044, only 47.8% of
            # consecutive steps increasing), and retrieval IS lookup -- but only if the
            # metric encodes value IDENTITY. Fed rank-gauss scalars, "near in value" is
            # near in the metric, which is precisely the wrong geometry for a
            # non-monotone lattice, and is the same reason B1/N2's ratios failed: a
            # monotone transform cannot represent a lookup. Fed the per-value embedding,
            # retrieval operates in a space where the lookup structure IS the geometry.
            #
            # M1 TabM is the cautionary case and the reason this is worth exactly one
            # run: fed the TREE representation, a packed MLP ensemble re-derived the tree
            # solution (nearest neighbours 0.9847-0.9867 with the trees, solo score equal
            # to C4x's to four decimals). The representation decides what a model becomes.
            class NCAEncoder(nn.Module):
                # Deliberately SHALLOW. ModernNCA's own finding is that the retrieval does
                # the work and a deep encoder only overfits the metric -- which is also
                # why this is not merely "L1lookupt with a kNN head bolted on".
                def __init__(self):
                    super().__init__()
                    d = hp["d"]
                    self.emb = nn.Embedding(LOOK_TOTV, d)
                    nn.init.normal_(self.emb.weight, std=0.02)
                    self.plr_n = PLR(NNUM, hp["k"], d) if NNUM else None
                    self.plr_d = PLR(NDER, hp["k"], d) if NDER else None
                    self.edrop = nn.Dropout(hp["drop"])
                    self.mlp = nn.Sequential(
                        nn.Linear((NLOOK + NDER) * d, hp["hidden"]),
                        nn.BatchNorm1d(hp["hidden"]), nn.ReLU(), nn.Dropout(hp["drop"]),
                        nn.Linear(hp["hidden"], hp["dout"]))
                def forward(self, idx, cn, cm, dn, dm):
                    tok = self.emb(idx)
                    if CFG["lookup_scalar_only"]:
                        tok = torch.zeros_like(tok)
                    if self.plr_n is not None:
                        smooth = self.plr_n(cn) * (1 - cm).unsqueeze(-1)
                        tok = torch.cat([tok[:, :NNUM] + smooth, tok[:, NNUM:]], 1)
                    parts = [tok]
                    if self.plr_d is not None:
                        parts.append(self.plr_d(dn) * (1 - dm).unsqueeze(-1))
                    return self.mlp(self.edrop(torch.cat(parts, 1)).flatten(1))

            # A FINITE sentinel, not -inf: the streaming log-sum-exp below evaluates
            # exp(m - m) on an all-masked block, and -inf would turn that into NaN.
            NEG = -1e30

            def _lse(m, s, blk):
                # Running (max, sum) update for a log-sum-exp accumulated over blocks.
                nm = torch.maximum(m, blk.max(1).values)
                e = torch.where(blk > NEG / 2, (blk - nm.unsqueeze(1)).exp(),
                                torch.zeros_like(blk))
                return nm, s * torch.exp(m - nm) + e.sum(1)

            def nca_logp(zq, zc, yc, mask=None):
                # (log p1, log p0) for queries zq against the candidate bank zc.
                #
                # Distances are EUCLIDEAN, not squared. That is ModernNCA's reported
                # choice and it is load-bearing: squaring makes the softmax far peakier
                # and collapses the neighbourhood towards a hard 1-NN.
                sc = -torch.cdist(zq, zc)
                if mask is not None:
                    sc = sc.masked_fill(mask, NEG)
                yb = yc.unsqueeze(0) > 0.5
                allz = torch.logsumexp(sc, 1)
                pos = torch.logsumexp(torch.where(yb, sc, torch.full_like(sc, NEG)), 1)
                neg = torch.logsumexp(torch.where(yb, torch.full_like(sc, NEG), sc), 1)
                return pos - allz, neg - allz

            hp = {"d": 64, "k": 24, "hidden": 512, "dout": 128, "drop": 0.1,
                  "batch": 1024, "cand": 16384, "eval_cand": 65536, "epochs": 20,
                  "lr": 2e-3, "aug": 0.10, "wd_emb": 3e-4, "wd": 1e-5, "patience": 4,
                  **CFG["lgb_params"]}

            net = NCAEncoder().to(dev)
            embp = [p for n_, p in net.named_parameters() if n_.startswith("emb")]
            rest = [p for n_, p in net.named_parameters() if not n_.startswith("emb")]
            # Same split as the transformer above: heavier decay on the per-value tables,
            # which are the part with one free vector per observed value.
            opt = torch.optim.AdamW([{"params": rest, "weight_decay": hp["wd"]},
                                     {"params": embp, "weight_decay": hp["wd_emb"]}],
                                    lr=hp["lr"])
            nstep = math.ceil(len(itr) / hp["batch"]) * hp["epochs"] + 10
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, hp["lr"],
                                                        total_steps=nstep, pct_start=0.15)
            G = [t.to(dev) for t in Ptr]
            ytr_d = torch.from_numpy(y[itr].astype("float32")).to(dev)
            OFFT = torch.from_numpy(np.array(
                [LOOK_VOCAB[c][1] for c in LOOK_NUM + LOOK_CAT], dtype="int64")).to(dev)
            P_ = list(net.parameters())
            ema = [p.detach().clone() for p in P_]
            ntr = len(itr)

            def enc(sl, aug):
                idx, cm = G[0][sl], G[2][sl]
                if aug > 0:
                    idx, cm = idx.clone(), cm.clone()
                    drop = torch.rand(idx.shape, device=dev) < aug
                    idx = torch.where(drop, OFFT.expand_as(idx), idx)
                    cm[:, :NNUM] = torch.maximum(cm[:, :NNUM], drop[:, :NNUM].float())
                return net(idx, G[1][sl], cm, G[3][sl], G[4][sl])

            @torch.no_grad()
            def encode(pk, chunk=16384):
                net.eval()
                return torch.cat([net(*[t[i:i+chunk].to(dev) for t in pk])
                                  for i in range(0, len(pk[0]), chunk)])

            @torch.no_grad()
            def nca_predict(pk, ZC, YC, qchunk=2048, cchunk=65536):
                # Soft-kNN probability, streamed over candidate blocks. Materialising the
                # full (n_query x n_train) score matrix would be ~300 GB at inference; a
                # running (max, sum) pair is mathematically identical and needs only
                # (qchunk x cchunk).
                net.eval()
                out = []
                for i in range(0, len(pk[0]), qchunk):
                    zq = net(*[t[i:i+qchunk].to(dev) for t in pk])
                    n = len(zq)
                    mA = torch.full((n,), NEG, device=dev); sA = torch.zeros(n, device=dev)
                    mP = torch.full((n,), NEG, device=dev); sP = torch.zeros(n, device=dev)
                    for j in range(0, len(ZC), cchunk):
                        sc = -torch.cdist(zq, ZC[j:j+cchunk])
                        yb = YC[j:j+cchunk].unsqueeze(0) > 0.5
                        mA, sA = _lse(mA, sA, sc)
                        mP, sP = _lse(mP, sP,
                                      torch.where(yb, sc, torch.full_like(sc, NEG)))
                    out.append(torch.exp((mP + sP.log()) - (mA + sA.log()))
                               .float().cpu().numpy())
                return np.concatenate(out)

            # A FIXED evaluation bank, drawn once. A bank resampled per epoch would make
            # the monitored AUC move for a reason that is not the model.
            eval_idx = torch.randperm(
                ntr, generator=torch.Generator().manual_seed(SEED))[:min(hp["eval_cand"], ntr)]
            Pev, yev = [t[eval_idx] for t in Ptr], ytr_d[eval_idx.to(dev)]

            best_auc, best_w, best_ep, since = -1.0, None, 0, 0
            eval_from = min(3, hp["epochs"] - 1)
            for ep in range(hp["epochs"]):
                net.train()
                perm = torch.randperm(ntr, device=dev)
                for i in range(0, ntr, hp["batch"]):
                    sl = perm[i:i+hp["batch"]]
                    # Candidates are RESAMPLED every step. This is ModernNCA's scalability
                    # trick: the retrieval target is the whole training fold, and a fresh
                    # random subset per step is an unbiased view of it that keeps the
                    # (B x C) distance matrix resident. 16,384 of ~553k is ~3%.
                    cd = torch.randint(0, ntr, (hp["cand"],), device=dev)
                    zq, zc = enc(sl, hp["aug"]), enc(cd, hp["aug"])
                    # Self-retrieval would let a row read its OWN label -- the NCA form of
                    # a target leak. Masked explicitly rather than assumed rare.
                    lp1, lp0 = nca_logp(zq, zc, ytr_d[cd],
                                        sl.unsqueeze(1) == cd.unsqueeze(0))
                    yq = ytr_d[sl]
                    loss = -(yq * lp1 + (1 - yq) * lp0).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(P_, 1.0)
                    opt.step(); sched.step()
                    with torch.no_grad():
                        torch._foreach_mul_(ema, 0.999)
                        torch._foreach_add_(ema, [p.detach() for p in P_], alpha=0.001)
                if ep >= eval_from:
                    bak = [p.detach().clone() for p in P_]
                    with torch.no_grad():
                        for p, e in zip(P_, ema): p.copy_(e)
                    # Monitored on ROC AUC -- the metric, never the NCA loss as a proxy.
                    a = roc_auc_score(y[iva], nca_predict(Pva, encode(Pev), yev))
                    if a > best_auc:
                        best_auc, best_ep, since = a, ep, 0
                        best_w = [e.detach().clone() for e in ema]
                    else:
                        since += 1
                    with torch.no_grad():
                        for p, b_ in zip(P_, bak): p.copy_(b_)
                    print(f"    mnca f{k} ep{ep:3d} val {a:.6f} best {best_auc:.6f}",
                          flush=True)
                    if since >= hp["patience"]:
                        break
            with torch.no_grad():
                for p, e in zip(P_, best_w): p.copy_(e)

            class _WrapN:
                def __init__(self, va, te): self.va, self.te = va, te
                def predict_proba(self, X_):
                    p = self.va if len(X_) == len(Xva) else self.te
                    return np.column_stack([1 - p, p])

            # The FINAL artifacts retrieve over the ENTIRE training fold, which is
            # ModernNCA as published. The 65k-candidate bank above was used only to pick
            # the epoch and never produces a number this repo quotes.
            ZC = encode(Ptr)
            model = _WrapN(nca_predict(Pva, ZC, ytr_d), nca_predict(Pte, ZC, ytr_d))
            best_it = best_ep
            del G, ZC
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"    mnca: {NLOOK + NDER} tokens -> d{hp['dout']} metric, "
                  f"{hp['cand']:,} candidates per step, {ntr:,} at inference, "
                  f"best epoch {best_ep} (val AUC {best_auc:.6f}), dev={dev}", flush=True)
    elif CFG["learner"] in ("realmlp", "tabm"):
        # RealMLP (pytabkit) -- the strongest published tabular-NN baseline, and NOT a
        # reshaped version of our own MLP: it brings periodic-linear (PLR) numeric
        # embeddings, a bespoke scaling/tfms pipeline and its own LR schedule. Its own
        # validation split is passed in explicitly so it never sees the val fold's labels
        # through an internal random split.
        #
        # TabM shares this branch because it shares the pytabkit interface and the same
        # no-NaN constraint, so duplicating the preprocessing would only let the two copies
        # drift. It is NOT the same model: TabM is a PACKED ENSEMBLE of MLPs trained
        # jointly (k parallel heads over shared weights), which is a variance-reduction
        # mechanism rather than another architecture shape. The public library's evidence
        # for it is the reason it is here at all -- 9 members, best 0.96867, the strongest
        # family after the library author's own models and well above our best neural leg
        # at 0.9661.
        if CFG["learner"] == "tabm":
            from pytabkit import TabM_D_Classifier as _PTK
        else:
            from pytabkit import RealMLP_TD_Classifier as _PTK
        hp = {"n_epochs": 60, "batch_size": 1024, **CFG["lgb_params"]}
        Rtr, Rva, Rte = Xtr.copy(), Xva.copy(), Xte.copy()
        for c in AS_CATEGORY:                       # pytabkit wants plain strings, no NaN
            for d in (Rtr, Rva, Rte):
                d[c] = d[c].astype(object).where(d[c].notna(), "__NA__").astype(str)
        # RealMLP rejects NaN in continuous columns. Impute with the TRAINING FOLD median
        # and keep an explicit missing indicator -- the EDA measured a non-trivial AUC on
        # several missing-indicators, so dropping that signal would be a silent loss.
        r_num = [c for c in Xtr.columns if c not in AS_CATEGORY]
        r_med = Rtr[r_num].median()
        for d in (Rtr, Rva, Rte):
            for c in r_num:
                if Xtr[c].isna().any() or d[c].isna().any():
                    d[f"{c}__isna"] = d[c].isna().astype("float32")
            d[r_num] = d[r_num].fillna(r_med)
        model = _PTK(
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            random_state=CFG["model_seed"], verbosity=0,
            val_metric_name="1-auc_ovr",   # binary auc_ovr IS roc_auc -- the metric, not a proxy
            **hp)
        model.fit(Rtr, y[itr], X_val=Rva, y_val=y[iva], cat_col_names=AS_CATEGORY)
        Xva, Xte = Rva, Rte
        best_it = int(hp["n_epochs"])

    elif CFG["learner"] == "node":
        # NODE -- Neural Oblivious Decision Ensembles (pytorch_tabular). The one architecture
        # whose inductive bias is DELIBERATELY tree-shaped: ensembles of oblivious decision
        # trees made differentiable via entmax feature selection and soft thresholds. That is
        # what makes it interesting here and not just another net -- every neural leg so far
        # (MLP, embeddings, RealMLP, FT-Transformer) landed 1.9-2.4% from each other and
        # 2.7-3.2% from the trees, i.e. they all found the SAME alternative to partitioning.
        # NODE is the one that should sit between the two camps rather than in the neural one.
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        from pytorch_tabular.models import NodeConfig
        hp = {"num_layers": 2, "num_trees": 128, "depth": 6, "batch": 4096,
              "epochs": 20, "patience": 4, "lr": 1e-3, **CFG["lgb_params"]}

        ncat = [c for c in AS_CATEGORY]
        ncon = [c for c in Xtr.columns if c not in AS_CATEGORY]

        def frame(X_, y_=None):
            d = X_.copy()
            for c in ncat:                      # no NaN in categoricals; own level instead
                d[c] = d[c].astype(object).where(d[c].notna(), "__NA__").astype(str)
            d[ncon] = d[ncon].fillna(node_med)  # train-fold median only -- section 2
            if y_ is not None:
                d[TARGET] = y_
            return d

        node_med = Xtr[ncon].median()           # fit on the TRAINING FOLD, applied to val/test
        df_tr, df_va, df_te = frame(Xtr, y[itr]), frame(Xva, y[iva]), frame(Xte)

        tm = TabularModel(
            data_config=DataConfig(target=[TARGET], continuous_cols=ncon,
                                   categorical_cols=ncat),
            model_config=NodeConfig(task="classification", num_layers=hp["num_layers"],
                                    num_trees=hp["num_trees"], depth=hp["depth"],
                                    learning_rate=hp["lr"], metrics=["auroc"],
                                    metrics_prob_input=[True], seed=CFG["model_seed"]),
            optimizer_config=OptimizerConfig(),
            trainer_config=TrainerConfig(
                batch_size=hp["batch"], max_epochs=hp["epochs"],
                early_stopping="valid_auroc",        # the metric, never a loss proxy
                early_stopping_mode="max", early_stopping_patience=hp["patience"],
                load_best=True, checkpoints="valid_auroc", checkpoints_mode="max",
                accelerator="gpu" if __import__("torch").cuda.is_available() else "cpu",
                devices=1, progress_bar="none", trainer_kwargs=dict(enable_model_summary=False)),
            verbose=False, suppress_lightning_logger=True)
        tm.fit(train=df_tr, validation=df_va)

        def node_proba(df_):
            out = tm.predict(df_.drop(columns=[TARGET], errors="ignore"))
            col = [c for c in out.columns if c.endswith("_probability")]
            assert col, f"no probability column in {list(out.columns)}"
            # pytorch_tabular names these "<target>_0_probability" / "<target>_1_probability",
            # so match the SUFFIX. Matching a "1" prefix finds nothing and silently falls
            # through to a positional guess -- right by luck here, wrong the moment the
            # column order changes. Assert instead of guessing.
            one = [c for c in col if c.endswith("_1_probability")]
            assert len(one) == 1, f"cannot identify the positive-class column in {col}"
            return out[one[0]].to_numpy()

        class _WrapN:
            def __init__(self, va, te): self.va, self.te = va, te
            def predict_proba(self, X_):
                p = self.va if len(X_) == len(Xva) else self.te
                return np.column_stack([1 - p, p])
        model = _WrapN(node_proba(df_va), node_proba(df_te))
        best_it = int(hp["epochs"])
        print(f"    node: {hp['num_layers']}x{hp['num_trees']} trees depth {hp['depth']}",
              flush=True)

    elif CFG["learner"] in ("ftt", "tabtf"):
        # FT-Transformer / TabTransformer. Attention over feature tokens is a genuinely
        # different inductive bias from both an MLP and a tree: it models feature-to-feature
        # interaction explicitly rather than through partitioning or a dense mixture.
        import torch, torch.nn as nn
        hp = {"d_token": 64, "n_blocks": 3, "heads": 8, "dropout": 0.1, "lr": 1e-3,
              "wd": 1e-5, "batch": 4096, "epochs": 30, "patience": 5, **CFG["lgb_params"]}
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(CFG["model_seed"])
        num_cols = [c for c in Xtr.columns if c not in AS_CATEGORY]
        card = [len(Xtr[c].cat.categories) + 1 for c in AS_CATEGORY]

        med = Xtr[num_cols].median()
        mu  = Xtr[num_cols].fillna(med).mean()
        sd  = Xtr[num_cols].fillna(med).std().replace(0, 1.0)

        # V-C1. The MLP branch has always appended an isna() block here and this branch
        # never did -- FEATURES.md 2.9 note 3 calls that "an inconsistency between two
        # branches, not a decision". Same idiom as the mlp branch's to_tensor(), not a
        # second one, so the two cannot drift.
        N_CONT = len(num_cols) * (2 if CFG["ftt_miss_mask"] else 1)

        def tens(df):
            z = ((df[num_cols].fillna(med) - mu) / sd).to_numpy(dtype="float32")
            if CFG["ftt_miss_mask"]:
                z = np.hstack([z, df[num_cols].isna().to_numpy(dtype="float32")])
            n = torch.from_numpy(z)
            c = torch.from_numpy(np.stack(
                [df[col].cat.codes.to_numpy() + 1 for col in AS_CATEGORY], 1).astype("int64"))
            return n, c

        (Ntr, Ktr), (Nva, Kva), (Nte, Kte) = tens(Xtr), tens(Xva), tens(Xte)
        ytr_t = torch.from_numpy(y[itr].astype("float32"))

        if CFG["learner"] == "ftt":
            from rtdl_revisiting_models import FTTransformer
            net = FTTransformer(n_cont_features=N_CONT, cat_cardinalities=card,
                                d_out=1, n_blocks=hp["n_blocks"], d_block=hp["d_token"],
                                attention_n_heads=hp["heads"],
                                attention_dropout=hp["dropout"], ffn_d_hidden=None,
                                ffn_d_hidden_multiplier=4 / 3,
                                ffn_dropout=hp["dropout"], residual_dropout=0.0).to(dev)
            fwd = lambda n_, c_: net(n_, c_).squeeze(-1)
        else:
            from tab_transformer_pytorch import TabTransformer
            net = TabTransformer(categories=tuple(card), num_continuous=N_CONT,
                                 dim=hp["d_token"], depth=hp["n_blocks"], heads=hp["heads"],
                                 dim_out=1, attn_dropout=hp["dropout"],
                                 ff_dropout=hp["dropout"],
                                 mlp_hidden_mults=(4, 2),
                                 mlp_act=nn.ReLU()).to(dev)   # default None lands in a
                                 # Sequential and raises 'NoneType' is not callable
            fwd = lambda n_, c_: net(c_, n_).squeeze(-1)

        opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        lossf = nn.BCEWithLogitsLoss()

        def infer(N, K):
            net.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(N), 8192):
                    out.append(torch.sigmoid(
                        fwd(N[i:i+8192].to(dev), K[i:i+8192].to(dev))).cpu().numpy())
            return np.concatenate(out)

        best_auc, best_state, best_it, since = -1.0, None, 0, 0
        for ep in range(hp["epochs"]):
            net.train()
            perm = torch.randperm(len(Ntr))
            for i in range(0, len(Ntr), hp["batch"]):
                idx = perm[i:i+hp["batch"]]
                opt.zero_grad()
                loss = lossf(fwd(Ntr[idx].to(dev), Ktr[idx].to(dev)), ytr_t[idx].to(dev))
                loss.backward(); opt.step()
            a = roc_auc_score(y[iva], infer(Nva, Kva))          # AUC, never a loss proxy
            print(f"      ep {ep:2d} val AUC {a:.6f}", flush=True)
            if a > best_auc:
                best_auc, best_it, since = a, ep, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since += 1
                if since >= hp["patience"]:
                    break
        net.load_state_dict(best_state)

        class _WrapT:
            def __init__(self, va, te): self.va, self.te = va, te
            def predict_proba(self, X_):
                p = self.va if len(X_) == len(Xva) else self.te
                return np.column_stack([1 - p, p])
        model = _WrapT(infer(Nva, Kva), infer(Nte, Kte))
    else:
        raise ValueError(f"unknown learner {CFG['learner']}")
    oof[iva] = model.predict_proba(Xva)[:, 1]
    # Accumulate the SUM in float64 and divide once, rather than adding p/N_FOLDS each
    # fold. XGBoost returns float32 and does emit exactly 1.0; five rounded p/5 terms
    # then sum to 1.0000000149, which the submission's between(0,1) check rejects.
    test_sum += model.predict_proba(Xte)[:, 1].astype(np.float64)
    fold_aucs.append(roc_auc_score(y[iva], oof[iva]))
    best_iters.append(int(best_it))
    print(f"  fold {k}: AUC={fold_aucs[-1]:.6f}  best_iter={best_iters[-1]}", flush=True)

test_proba = np.clip(test_sum / N_FOLDS, 0.0, 1.0)   # clip is rank-preserving, so AUC is unaffected
oof_auc = roc_auc_score(y, oof)
assert (fold_id >= 0).all()
assert max(best_iters) < CFG["n_estimators"], \
    f"a fold hit the n_estimators cap ({max(best_iters)}) -- under-trained, raise it"

print(f"\nOOF AUC : {oof_auc:.6f}")
print(f"folds   : {np.mean(fold_aucs):.6f} +/- {np.std(fold_aucs):.6f}  (spread {max(fold_aucs)-min(fold_aucs):.6f})")
print("fold spread is EVALUATION-FOLD DIFFICULTY, not test-prediction variance.")
""")

code(r"""
if hasattr(model, "feature_importances_"):      # trees only; the MLP wrapper has none
    imp = (pd.Series(model.feature_importances_, index=list(Xtr.columns))
           / model.feature_importances_.sum() * 100).sort_values(ascending=False)
    print("feature importance (last fold, % of splits):")
    print(imp.round(2).to_string())
else:
    print(f"{CFG['learner']}: no split-based feature importance")
""")

code(r"""
# ---- leakage audit for the supervised encoder -----------------------------
# Reported honestly, because a train/val AUC gap is a WEAK test on this data and it
# would be worse than useless to treat a small gap as proof of safety.
#
# A row's own label moves its value's mean by 1/(n_rows_for_that_value + m). Columns
# like notifications_per_day have ~2700 rows per value, so self-influence is ~0.03%
# and even a naive (un-nested) fit shows almost no gap -- measured directly: a
# deliberately leaky fit produced a gap of just +0.0008. So a small gap here is
# evidence of low cardinality, NOT evidence the nesting works.
#
# What actually provides safety, in order of importance:
#   1. min rows-per-value (printed below) -- the structural bound on self-influence;
#   2. the inner-OOF nesting, which matters precisely for the singleton values that
#      columns like daily_screen_time_hours DO have (min n = 1);
#   3. smoothing, which shrinks a singleton toward the prior.
if TE_COLS:
    print(f"{'column':<42}{'med n/val':>10}{'AUC tr':>9}{'AUC va':>9}{'gap':>9}")
    for c in TE_COLS:
        occ = np.bincount(TE_CODES[c][0][itr])
        med_n = int(np.median(occ[occ > 0]))
        a_tr = roc_auc_score(y[itr], Xtr[f"{c}__te"])
        a_va = roc_auc_score(y[iva], Xva[f"{c}__te"])
        gap = a_tr - a_va
        flag = ""
        if med_n <= 30:
            flag = "  <-- sparse cells: gap test HAS power here"
        if gap > 0.02:
            flag = "  <-- LEAK SUSPECT"
        print(f"{c:<42}{med_n:>10}{a_tr:>9.4f}{a_va:>9.4f}{gap:>+9.4f}{flag}")
    print("\nInterpret the gap only as a smoke alarm for a GROSS error (e.g. encoding fit")
    print("on the full frame including val). It cannot certify the nesting; the nesting is")
    print("certified by construction -- the inner split never shows a row its own label.")
""")

md(r"""
## Artifacts
""")

code(r"""
LEARNER = CFG["learner"]
pd.DataFrame({ID: train[ID], "fold": fold_id, "proba": oof}).to_csv(
    OUT_DIR / f"oof_proba_{LEARNER}.csv", index=False)
pd.DataFrame({ID: test[ID], "proba": test_proba}).to_csv(
    OUT_DIR / f"test_proba_{LEARNER}.csv", index=False)

submission = pd.DataFrame({ID: test[ID], TARGET: test_proba})
submission.to_csv(OUT_DIR / "submission.csv", index=False)

assert list(submission.columns) == [ID, TARGET]
assert len(submission) == len(sub)
assert (submission[ID].values == sub[ID].values).all(), "ids must match sample_submission in ORDER"
assert submission[TARGET].notna().all() and submission[TARGET].between(0, 1).all()
print(f"submission validated: {len(submission):,} rows, "
      f"mean {submission[TARGET].mean():.5f} (train base rate 0.70942)")
""")

code(r"""
RUN_METRICS = {
    "run_tag":            CFG["run_tag"],
    "final_oof_auc":      round(float(oof_auc), 6),
    f"{LEARNER}_oof_auc": round(float(oof_auc), 6),
    "fold_aucs":          [round(float(a), 6) for a in fold_aucs],
    "fold_auc_mean":      round(float(np.mean(fold_aucs)), 6),
    "fold_auc_std":       round(float(np.std(fold_aucs)), 6),
    "best_iters":         best_iters,
    # Count the columns actually FED TO THE MODEL, not len(FEATURES): the __te columns
    # are created inside the fold loop, so len(FEATURES) under-reports by len(TE_COLS)
    # and the log would claim a 12-feature model that really saw 21.
    "n_features":         int(Xtr.shape[1]),
    "engineered":         added,
    "te_cols":            TE_COLS,
    "model_seed":         CFG["model_seed"],
    "n_folds":            N_FOLDS,
    "cv_seed":            SEED,
    "cfg":                CFG,
    "notebook_runtime_sec": round(time.time() - T_START, 1),
}
print("RUN_METRICS_JSON:" + json.dumps(RUN_METRICS))
""")

nb = {
    "cells": [
        {"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"execution_count": None, "outputs": []} if k == "code" else {})}
        for k, s in C
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0",
                          "mimetype": "text/x-python", "file_extension": ".py"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
# KCFG arrives as JSON on the command line but is baked into PYTHON source, and the two
# are not the same language: json's true/false/null are not Python literals, so injecting
# the raw text verbatim produces `KAGGLE_CFG = {..., "fe_composition": true}` and a
# NameError on the kernel's first cell. Round-trip through json.loads -> repr so booleans
# and nulls come out as Python, and so a malformed CFG fails HERE rather than on Kaggle.
KCFG_PY = repr(json.loads(KCFG))
OUT.write_text(json.dumps(nb, indent=1).replace("__KCFG__", json.dumps(KCFG_PY)[1:-1]),
                encoding="utf-8")
print(f"wrote {OUT} ({sum(1 for k,_ in C if k=='code')} code, {sum(1 for k,_ in C if k=='markdown')} md)")
