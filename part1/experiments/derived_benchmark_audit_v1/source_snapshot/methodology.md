# 02 · Methodology: evaluation protocols, leak test, objective and selection rule

How every number in these docs was produced, and why the builder set it up that way. The code is in `midterm/part1/p1lib/evaluate.py`, `p1lib/context.py`, `p1lib/search.py` and `scripts/leak_test.py`.

## 1. The core problem the protocols address

There are only 235 labeled rows, and the test set (2010-16) has a different label prior from anything labeled (see [01_data_findings.md](01_data_findings.md) §3-4). Any single validation scheme fails in one of two ways:

- **The official val (78 rows) is too small.** Its SE is 0.038, so a gap of about 0.07 is needed before two models can be told apart. It is also drawn from the same pre-2010 period as the training rows, so it contains almost no shift.
- **Random CV over the 235 rows is precise** (SE 0.017) but has no shift at all. It rewards models that copy the training prior, and it cannot see a prior correction's benefit. In the research report's checks, SLD *lowered* pooled-CV F1 (0.902 → 0.889) while raising the shifted scores.

So the builder used three protocols, optimized a blend of two, and kept the third as a tie-breaker.

## 2. Protocols A, B and C

All three run over the 235 labeled rows (157 labeled-train + 78 val). The 1,014 unlabeled rows always stay available as the pseudo-label pool, just as at submission time. `make_folds` in `p1lib/evaluate.py` builds them.

| Protocol | What it is | n eval | What it measures |
|---|---|---|---|
| **A** | `RepeatedStratifiedKFold(5 × R)`, stratified on the 8-class combination id | 235 per repeat | Fit quality with no shift. Precise, but blind to prior shift. |
| **B** | Chronological holdout: train on `start_year < cut`, evaluate on `≥ cut`, for cut ∈ {2004, 2005, 2006}. B is the **mean over the three cuts**. | 109 / 98 / 86 | Behaviour under a train→later shift in the same direction as the real one (Recruitment falls, Safety/Efficacy rise). Exaggerated: R drops to about 0.40 vs about 0.57 estimated for test. |
| **C** | The official split: train on the 157 labeled-train rows, score the 78 val rows | 78 | What the assignment says to report. Used as a tie-breaker, never optimized. |

**Settings by stage:**

| Stage | R | CV seed |
|---|---|---|
| Search | 4 | 0 |
| Leaderboard, experiments, selection re-evaluation, candidate manifests | 10 | 1 |
| Fresh-seed check | 10 | 2 |

**Metrics.** Each protocol reports sample-F1 with its SE, per-label F1, and macro-F1 with bootstrap SEs (200 resamples over rows, `p1lib/data.py::bootstrap_se`).

- **A.** Per-row F1 is averaged over the R repeats first, then SE = sd/√235.
- **B.** The SE reported is the *mean of the per-cut SEs*, not a pooled SE. The three cuts share most of their rows and are highly correlated (comment in `score()`).

### Why the chronological cuts were averaged

The research report used a single cut at 2005 (137 train / 98 eval). The builder's log says that cut is noisy (SE about 0.03 for 98 rows) and that "decisions flipped between cuts." So B became the mean over 2004, 2005 and 2006, with B2005 still reported separately for comparison with the research numbers (`notes/00_log.md`, "Protocol tweak").

Averaging reduces the variance from where the single cut falls. It does not add independent rows: the cuts overlap heavily. The paired-SE code therefore treats B's effective n as one cut (`paired_se`: `nB = len(dB) / B_ncuts`).

## 3. What may see what: leakage rules and the transductive toggle

Every fold builds a fresh `Ctx` (`p1lib/context.py`) holding four row sets:

- `train`: labeled rows whose labels may be read
- `eval`: rows to predict, with labels hidden
- `pool`: the 1,014 unlabeled rows, used for pseudo-labels
- `unsup`: rows whose *features* label-free steps may use

**Label-free steps:** scaler, PCA, SVD of the text blocks, K-means/GMM/HDBSCAN fits, the kNN graph.

- With `transductive=True` (the default, recorded in every manifest), `unsup` is all 1,561 rows (train + val + test features, no labels). The fit is then identical across folds and memoized.
- With `transductive=False`, the eval rows and the test rows are excluded. SLD then estimates the target prior on the unlabeled pool instead of on the eval batch.

**Label-dependent steps run inside `run_model(cfg, ctx, rows)` and read only `ctx.train` labels:**

| Step | How leakage is avoided |
|---|---|
| Classifiers | fit on fold-train rows only |
| Temperature | fit on inner 5-fold OOF predictions |
| Cluster→label maps | exact leave-one-out for training rows when used as features |
| Graph clamps and C&S residuals | C&S residuals use inner-fold OOF base probabilities, because in-sample residuals are near zero |
| Pseudo-labels | regenerated inside every fold |
| kNN votes | leave-one-out |
| Stacker meta-features | `oof_model`: nested inner 5-fold |

**One documented compromise.** For pipelines with pseudo-labels, temperature is fit on inner OOF of the classifier stage with the pseudo rows held fixed. The pseudo-labels are computed once from all fold-train labels. The builder notes this gives "slight optimism in T only" (`notes/00_log.md`, Decisions).

**The transductive toggle matters.** From `results/exp_transd.csv` (R=10, seed 1):

| Config | Transductive J (A / B) | Inductive J (A / B) |
|---|---|---|
| LR + SLD25 + exp-F1 | 0.881 (0.899 / 0.863) | 0.844 (0.884 / 0.804) |
| C&S as predictor | 0.870 (0.905 / 0.835) | 0.825 (0.875 / 0.775) |
| chosen pipeline | 0.892 (0.909 / 0.876) | 0.880 (0.908 / 0.852) |

- **For SLD, most of the inductive loss comes from estimating the prior on the wrong set.** Inductive SLD estimates on the 2003-2010 pool. Its estimated R prior on the B folds is 0.67-0.70 vs 0.49-0.52 when estimated on the eval batch; the true value is about 0.40-0.42 (`B_sld_prior` column).
- **Test features are never labeled**, so the builder treats the transductive mode as legitimate and records it everywhere. The research report flagged it as a question worth one line in the write-up (see [60_open_questions.md](60_open_questions.md)).

## 4. The leak test

`scripts/leak_test.py` writes `results/leak_test.txt`. It covers every reference config in `p1lib/configs.py::REFERENCE` (25 configs), plus the main-search argmax (`best0.json`), on three folds: an A fold, B2005 and C (folds built with seed 3). In each run it corrupts the labels of **every labeled row outside the fold's training set** with random combinations, re-runs the fold with caches cleared, and compares the calibrated eval probabilities. If any component reads a label it shouldn't, those probabilities change.

**Result:** `max|dP| = 0.00e+00` on all 78 checks (26 configs × 3 folds); `FAILURES: 0`. REPORT.md says "25 configs × 3"; the file has 26 config names because the argmax was added on top of the 25 references.

The test compares probabilities *before* the post-hoc head. SLD and decoding read only those probabilities, the training prior and (transductively) the eval rows' predictions, never labels.

**Positive control:** a planted leak (kNN votes computed from all labeled rows) gives `max|dP| = 0.159` and is detected. So the test can catch a leak, and found none.

The configs tested cover every component type: cluster, graph and C&S labelers, SLD pseudo-labels, both self-training variants, synthetic features, the HGB and LR stackers with nested OOF, and the ensembles.

## 5. The search objective J

```
J = 0.5 · A + 0.5 · B          J_se = 0.5 · sqrt(A_se² + B_se²)
```

- **C is logged on every trial and never optimized** (`p1lib/search.py` docstring).
- **Why a 50/50 blend:**
  - A alone has no shift, so it would pick models that copy the training prior.
  - B alone is noisier and exaggerates the shift.
  - The research report's protocol (§2.2) already required add-ons to be non-negative on both.
- **A and B are correlated but distinct.** Over the random-search trials their Spearman correlation is 0.73 ([`fig_sensitivity_continuous.png`](../midterm/part1/figures/fig_sensitivity_continuous.png), right panel). The low-A/low-B tail is mostly SLD collapses and cluster predictors.

## 6. The one-standard-error rule

Implemented in `p1lib/search.py::select_one_se` and applied by `scripts/select_final.py`:

1. **Re-evaluate on a fresh seed.** Take the top 40 distinct configs by search J (from the main, unsup and random logs), plus the simplest configs within 1 SE of the search best, plus the top rule-compliant ones. Re-evaluate all of them with **R=10 on CV seed 1**; the search used R=4 on seed 0. This de-biases the winner's curse (see [20_search_and_selection.md](20_search_and_selection.md)).
2. **Find the band.** Take the best re-evaluated J. Every config with `J ≥ J_best − J_se(best)` is "within noise."
3. **Pick the simplest config in the band.** Sort by `complexity`, then by J.
   - Complexity counts active components: +1 per synthetic feature, block or labeler; extra for multi-member or multi-round self-training; +0.25 OvR/NB, +0.5 kNN, +1 direct labeler, +2 HGB; +0.5 balanced class weights; +0.25 temperature; +0.5 SLD; +1 per extra ensemble member.
4. **Guard.** The choice's B must be within one B-SE of the plain powerset-LR baseline's B.
5. **Restrict the band for the assignment rule.** Only configs where clustering or a graph helps fill the labels count (`search.unsup_labeling`): a cluster, spreading or C&S labeler, or self-training with the K-means consistency filter or a C&S member.

**A paired variant was also computed**: within one *paired* SE of the argmax, using per-row differences. Both rules pick the same config, `search#787` (`results/selection.json`): 37 configs inside the plain 1-SE band (threshold 0.8829), 27 inside the paired band. The unconstrained 1-SE rule, without the compliance restriction, also picks `search#787`.

**Why 1-SE.** Hastie, Tibshirani & Friedman (ESL §7.10), plus the Cawley & Talbot (2010) argument that variance in the selection criterion itself causes overfitting of a size comparable to the gaps between algorithms. With an SE of about 0.016 on J and a top-40 spread of about 0.02, almost everything near the top is a tie, and the simplest tie generalizes best.

## 7. Nested check of the whole search procedure

Selecting the best of about 3,000 trials on 235 rows invites optimism, so the builder ran the *entire* procedure inside outer splits (`scripts/nested.py`):

- a compliant-only TPE search, about 400 trials at R=3, on the outer-train rows only
- re-evaluation of the top ~30 distinct configs at R=6 on a fresh seed
- the 1-SE rule with the B guard
- refit on outer-train, then score the held-out outer rows

There were five stratified outer folds, plus one chronological outer split: search only on `start_year < 2005` with inner cuts 2001/2002, then score on the 98 rows ≥2005. Results and implications are in [20_search_and_selection.md](20_search_and_selection.md) §6.

## 8. Final refit and candidate generation

`p1lib/submit.py`:

1. Freeze the config.
2. Refit on all 235 labeled rows with the 1,014 unlabeled rows as the pool, transductive.
3. Average **5 seeds**. Seeds change only the inner folds used for temperature and the C&S base OOF.
4. Apply the head on the 312 test rows.
5. Write `ID,Predicted` in the sample file's order. The order is checked against `sample_submission.csv`.
6. Write a manifest with the config, scores ± SE and predicted test label rates.

The research report suggested 10 bootstrap bags; the builder only seed-averaged (noted under "Not done").
