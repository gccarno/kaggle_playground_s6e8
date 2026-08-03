# Kaggle Playbook — transferable lessons from Playground S6E7

**Audience:** a Claude Code instance starting a new Kaggle competition (tabular, but most of this
generalizes). This is a *method* document, distilled from one full competition run. Every claim
below is backed by a number that was actually measured in that run; the numbers are cited so you can
recognize the same pattern rather than take the advice on faith.

**The result this method produced:** Playground Series S6E7 (3-class balanced accuracy, 690k train
rows). Public LB 0.95091 → **private LB 0.95050, rank 120, up 298 places in the shakeup**, 38
entries. The private #1 was 0.95085 — *below our own public score* — while the public LB leaders sat
at 0.95238/0.95264. Everyone who out-scored us on public without out-scoring us on honest OOF fell
below us on private.

That single fact is the thesis of this document: **the winning move was refusing to chase the public
leaderboard past the point where our own cross-validation said the signal was exhausted.**

---

## 1. Build the experiment harness before the first model

Do this on day one. It costs a few hours and it compounds for the entire competition.

**A one-command run collector.** In S6E7 this was `scripts/collect_run.py`: push the kernel → poll to
completion → parse a `RUN_METRICS_JSON` blob printed by the notebook → archive artifacts → append one
row to `experiments/runs.csv`. Never run an experiment whose result lands only in a scrollback buffer.

**An append-only experiment log in git.** One row per run: run id, git commit, config knobs, per-learner
OOF scores, final OOF, LB score, and a *long* free-text `notes` field. Commit it after every run. By
the end there were 41 rows and they were the single most valuable asset in the repo — the Phase 6
analysis that quantified the ceiling was pure re-analysis of this log, with no new training.

**Archive OOF and test probabilities for every learner, every run.** This is the highest-ROI decision
in the whole project. `experiments/preds/<run_id>/{oof_proba,test_proba}_<learner>.csv`, gitignored,
keyed by run id. Consequences:

- Any past model can be retro-blended with any future model without retraining.
- You can compute *disagreement* between any two models you have ever built.
- Late in the competition you can sweep the back catalogue for models judged only on OOF.

**Freeze the CV split globally.** Every model in the project used `StratifiedKFold(5, shuffle=True,
random_state=42)`. This is what makes OOF matrices from different runs, weeks apart, directly
blendable. Pick the split once, write it in the README, never change it.

**Automate submission slots.** Kaggle gives ~5 submissions/day and they do not roll over. A scheduled
task (`scripts/submit_queue.py`, hourly, Windows Task Scheduler) burned a priority queue before the
00:00 UTC reset. This turned an unused daily resource into 15 free OOF↔LB data points, which is what
made the reliability analysis in §4 possible.

---

## 2. Leakage discipline is the load-bearing constraint

Everything else is tuning; this is correctness. In S6E7 the rule was: target encoders, quantile bin
edges, scalers, imputers, and category vocabularies are **fit on a training fold only and applied to
the val/test fold** — never fit once on the full training set before the CV loop.

The same discipline has to be repeated identically in *every* usage site: feature selection, HPO, and
the final stack. It is tempting to fit the encoder once "just for feature selection." Don't — a
leak there silently inflates OOF and destroys the OOF↔LB relationship you will depend on in §4.

Corollaries that saved time:
- Unseen categories at inference must map to a reserved "unknown" level, never raise.
- Optimize the **actual competition metric** everywhere — CV scoring, the Optuna objective, and neural
  early stopping. In S6E7 that was `balanced_accuracy_score`, not accuracy or logloss. A heavily
  imbalanced 3-class problem where the majority class is 86% will happily converge to a model with
  33% balanced accuracy if you optimize the wrong thing.

---

## 3. Run experiments as falsifiable probes with a pre-registered gate

The format that worked:

> **Hypothesis.** X will help because Y. **Test.** Strict-twin ablation changing only X.
> **Gate.** Ship only if OOF improves by ≥ +0.0003. **Result.** …  **Mechanism.** …

Two rules make this actually bite:

**Pre-register the gate, before seeing the result.** In S6E7 the gate was +0.0003 OOF, chosen because
it was ≈1σ of the measured OOF↔LB residual (§4). About nine probes after the champion was found
cleared exactly zero of them, and because the gate was pre-registered none got rationalized into a
submission.

**Strict-twin ablations.** Change exactly one thing. The cleanest experiment in the project (P3) held
the features, backbone, embeddings, LR schedule, epoch count and seed-bagging identical to the
champion's neural leg and changed *only the output head and loss* (3-way softmax → ordinal CORAL).
Result: OOF 0.93560 vs the twin's 0.95051, **−0.0149**. Because it was a strict twin, the delta was
attributable to the ordinal *assumption*, not to capacity or tuning. That is a real finding; a
non-twin comparison would have been an anecdote.

**Log the mechanism, not just the number.** The CORAL failure decomposed into per-class recall:
`at-risk` 0.936 → 0.878, while `fit` and `unhealthy` both slightly *improved*, with true `at-risk`
rows bleeding symmetrically to both extremes. Mechanism: a 1-D latent severity score can only
predict the middle rank between two thresholds, and `at-risk` was not a midpoint on a continuum — it
was the diffuse majority *catch-all* class.

That mechanism, once written down, explained three other failures retroactively (a retrieval/metric
learner, a 1-D CNN, and the public library's ordinal XGBoost decompositions) and pre-emptively killed
a whole family of future ideas. **A well-written negative result is worth more than a +0.0001 blend.**

---

## 4. Measure whether your own OOF predicts the LB — then trust the answer

Do not accept folklore like "OOF inverts the LB here." Measure it. With ~26 runs having both an OOF
and an LB score, S6E7 got:

- **Spearman(OOF, LB) = 0.87**, residual **σ = 0.00029**, no run beyond ~2σ.
- Nothing that had been dismissed on OOF turned out to be a hidden LB winner.

This single measurement changes how you spend the rest of the competition:

1. **It sets the gate.** Any delta below ~1σ (0.0003) is noise. Chasing it is a coin flip.
2. **It licenses you to stop submitting to decide things.** With a reliable OOF you can kill probes
   offline and spend submissions on genuine finalists.
3. **It tells you when the public LB is lying.** Teams above you on public but not on OOF are either
   overfitting the public split or exploiting something that will not transfer.

Point 3 is what earned the +298 places. The public LB toppers at 0.95238/0.95264 were ~0.0015 above
an honest ceiling of 0.95091 — five times the measured residual σ. That is not a better model; that is
a different distribution. Our own public→private delta was −0.00041 (≈1.4σ), i.e. ordinary split
noise. **A public score that is many σ above what your CV can explain is a red flag, not a target.**

---

## 5. Ensembling: what actually moves the needle

**Two strong, decorrelated learners at fixed weights beat a big stack.** The champion was a
`bagged FT-PLR : XGBoost-with-raw-target-encoding` blend at a **fixed 2:1** ratio. Earlier versions
with 5 base learners and a logistic-regression meta-learner scored *worse*. The meta-learner dilutes:
it fits weights on OOF and spreads mass across redundant learners. Learners that turn out redundant
with the trees are an acceptable outcome, not a bug — but then drop them rather than stacking them.

**Fixed, round weights over hill-climbed ones.** A hill-climbed simplex search over 51 models found
+0.00014 OOF — about 0.5σ, i.e. pure selection bias over hundreds of configs. One S6E7 hill-climb
collapsed to weight 1.0 on a single model (a degenerate "blend"). Prefer a small grid of round ratios
(2:1, 3:1, 1:1) validated with nested CV: fit weights on 4 folds, score on the held-out 5th.

**Seed-bagging inside a fold: yes. Re-running at a new split seed: no.** Averaging 3 neural nets per
fold gave a real gain. Averaging the whole champion across two outer-CV seeds (rs=42 + rs=777) gave
**+0.00009 OOF / −0.00010 LB** — nothing. Mechanism: stack test predictions are *already* averaged
over 5 fold-models, so a new split seed re-randomizes variance that was largely cancelled. The two
seeds disagreed on 0.14% of test rows and flipped 210 of 295753 predictions. The 0.0036 spread across
per-fold scores is *evaluation-fold difficulty*, not test-prediction variance — do not mistake one for
the other and spend 2.3 GPU-hours on it like we did.

**Check the decision rule once, then leave it alone.** For balanced accuracy it is natural to hunt for
per-class probability multipliers. A 2-DOF search gave in-sample +0.00005 but held-out **−0.00010**,
negative in 3 of 5 folds, with unstable fitted weights. Plain argmax was already optimal. Prior
correction (`proba / priors`) was catastrophic (0.90127). Test these, don't assume them.

---

## 6. Recognizing the wall — the most important skill

Every competition has a point where the signal is exhausted and further work buys zero. Recognizing it
early is worth more than any model. Here is how to detect it.

**The strength ↔ decorrelation tradeoff.** For a blend to improve, you need a model that is both
*strong solo* and *decorrelated* from your champion's legs. Plot solo score against disagreement rate
across every model you own. In S6E7:

- **Spearman(strength, decorrelation) = −0.84** (p<1e-4) over 38 of our own learners;
- **−0.90** (p=0.0002) over 11 external models from other authors.

No probe among 49 was simultaneously ≥0.950 solo *and* >1.5% decorrelated. When that correlation is
strongly negative and no point sits in the useful quadrant, the ensemble axis is closed. Compute this
plot — it turns "I feel like we're stuck" into a number.

**High disagreement is usually weakness, not diversity.** Repeatedly, a novel architecture disagreed
with the champion on 5% of rows and looked promising — and got *literally zero* blend weight. The
disagreement was the model being wrong in new places. The test: does disagreement come *with*
competitive solo strength? If not, it is noise wearing a costume.

**Novel architectures mostly fail, and they fail for a discoverable reason.** S6E7 tried a tabular
foundation model (TabICL), a retrieval/metric learner (ModernNCA), neural oblivious decision trees
(NODE), a BatchEnsemble MLP (TabM), a 1-D CNN, and an ordinal CORAL head. All were either
*strong-but-correlated* or *decorrelated-but-weak* — the two jaws of the same wall. Three of them
broke on the *same* underlying property of the data (a diffuse majority catch-all class that
alternative inductive biases cannot carve out). Find that property and you can predict which
architectures will fail without training them.

**Distinguish supervised from unsupervised levers.** The biggest single feature win was raw per-value
target encoding, which is *supervised* (value → label rate). We then spent a probe testing whether
256-bin quantile quantization — the unsupervised analogue — could recover it. It could not, and could
not have: no unsupervised binning recreates a value→label mapping. Ask "is this lever supervised?"
before building the unsupervised imitation of it.

---

## 7. Final submission selection

Kaggle lets you pick 2 finals, and **only from entries you actually submitted** — so submit anything
you might want to select, even if you don't expect it to top the public LB.

The S6E7 choice, which worked:

- **Final A** = the honest champion: best OOF, best public LB, simplest recipe.
- **Final B** = the *variance-reduced* version of the same recipe (averaged over two independent fold
  partitions). Chosen not because public LB liked it — it scored 0.00010 *worse* — but because it has
  strictly lower prediction variance, which is the right property for an unseen split.

What we deliberately did **not** select: any of the banked public-notebook submissions scoring above
our ceiling on the public LB. Those were the entries that evaporated in the shakeup.

Rule of thumb: **one submission you believe for CV reasons, one that is the same idea with less
variance. Never spend a final slot on a public score you cannot explain.**

---

## 8. Checklist for the next competition

**Week 1 — infrastructure**
- [ ] Run collector: push → poll → parse metrics → archive artifacts → append log row.
- [ ] `experiments/runs.csv` in git, with a long free-text notes column.
- [ ] Per-learner OOF + test probability artifacts saved for *every* run, keyed by run id.
- [ ] Global frozen CV split, documented, never changed.
- [ ] Automated submission-slot queue so no daily slot is wasted.
- [ ] A single baseline end-to-end, submitted, logged. Establish the OOF↔LB pair early.

**Ongoing — per probe**
- [ ] Written hypothesis + mechanism + pre-registered gate.
- [ ] Strict-twin ablation: exactly one thing changes.
- [ ] Leakage audit of every fitted transform in the new path.
- [ ] Result logged with per-class breakdown and the *mechanism*, not just the score.
- [ ] Blend-tested against the champion's legs before being declared useless.

**Midpoint — reliability**
- [ ] Compute Spearman(OOF, LB) and the residual σ over all runs. Set the gate to ~1σ.
- [ ] Compute Spearman(solo strength, disagreement-vs-champion) over all models. If strongly
      negative with an empty useful quadrant, the ensemble axis is closed.
- [ ] Sweep the back catalogue: any model judged only on OOF that has never been blend-tested.

**Endgame**
- [ ] Stop when probes stop clearing the gate. Extra probes past the wall bought 0.00000 here.
- [ ] Final A = honest champion. Final B = lower-variance twin of it.
- [ ] Do not select a public score your CV cannot explain, however tempting the rank.

---

## 9. The two-sentence version

Build the logging and artifact infrastructure first, freeze one CV split, keep every OOF matrix you
ever produce, and measure how well your own OOF predicts the leaderboard so you know what a real
improvement looks like. Then use that number to tell the difference between a better model and a
luckier one — and when the strength↔decorrelation plot says the signal is exhausted, stop, submit the
model you can defend, and let the shakeup move you up.
