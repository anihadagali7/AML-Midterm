# dylan-model-v2: framework, search and results report

This is the full write-up from the agent that built the composable Part 1 framework, ran the hyperparameter searches and produced the candidate submissions. Paths are relative to this folder. In the original workspace the framework lived at `aml-midterm/midterm/part1/`.

**Summary.** The search-selected pipeline beats plain logistic regression clearly. It does not reliably beat the simpler LR + SLD + exp-F1 "anchor": the gap is about one standard error. A nested check shows the search procedure can fail badly when its validation protocol contains no time shift. So there are two main candidates to choose between, not one.

Nothing was submitted to Kaggle or pushed anywhere. The agent did not open the team's earlier-work folders.

## What was built and how to run it

Everything is in `midterm/part1/`. Pinned versions are in `midterm/part1/requirements.txt` (tested with Python 3.12).

**`p1lib/` is the framework.** Every component reads only the training labels of the current fold.

- **Preprocessors.** Standardize, PCA, and optional L2. Toggleable extra blocks: criteria, diseases, drugs and pathways text; Morgan fingerprints; drug counts; start_year. All are fit without labels, on the rows a fold is allowed to see.
- **Synthetic features**, each usable as input to downstream models:
  - GMM responsibilities
  - K-means distances or soft assignments
  - kNN label votes
  - graph spreading and Correct & Smooth (C&S) scores
  - cluster-to-label probabilities
  - out-of-fold probabilities of any sub-pipeline (this is also how stacking works)

  Label-dependent features are computed leave-one-out or inner-fold for training rows.
- **Labelers** that fill the 1,014 missing labels: cluster-then-label (K-means, GMM, seeded K-means, HDBSCAN), label spreading, C&S, SLD-EM, and guarded self-training.
- **Classifiers** over the 8 observed label combinations: multinomial LR, one-vs-rest LR, shrinkage LDA (with sample weights), kNN, HistGradientBoosting, Gaussian NB, or a labeler used directly as the predictor.
- **Shared head.** Temperature scaling on inner-fold predictions, one prior correction, then decoding to a non-empty label set (most probable combination, or expected F1). Combining SLD with `class_weight='balanced'` raises an error, since that would apply two prior corrections.
- **Composition.** Equal-weight or weighted ensembles, and stacks.

**Pipelines are config dicts.** Examples are in `p1lib/configs.py` and in the top cell of `notebooks/00_framework_source.ipynb`. Typical use: `evaluate(cfg, load(), R=10)`.

**Evaluation protocols.**

- **A:** 5-fold CV × R repeats over the 235 labeled rows.
- **B:** chronological holdout, averaged over cut years 2004, 2005 and 2006. The single 2005 cut was too noisy on its own; it is still reported separately as B2005.
- **C:** the official split, 157 labeled train rows → 78 val rows.

Each protocol reports sample-F1, per-label F1 and macro F1, with standard errors. The search objective is **J = 0.5·A + 0.5·B**.

**Leak test.** For every config, the labels of all rows outside the fold's training set were corrupted, and the predictions were checked for changes. They were bit-identical for 25 configs × 3 fold types, and a deliberately planted leak was caught. See `results/leak_test.txt`.

**Test-feature toggle.** `transductive` controls whether test features are used in the label-free steps. It is on by default and recorded in every manifest.

**Scripts** (in `scripts/`): `run_search.py`, `select_final.py`, `nested.py`, `leaderboard.py`, `experiments.py`, `ablate.py`, `make_candidates.py`, `leak_test.py`. The full command list is in `notes/01_how_to_run.md`.

**Notebooks** (in `notebooks/`, all executed):

| Notebook | Contents |
|---|---|
| `00_framework_source.ipynb` | All source code, as `%%writefile` cells, so the `.ipynb` files contain everything |
| `01_data_and_protocols.ipynb` | Data and evaluation protocols |
| `02_methods_and_leaderboard.ipynb` | Method comparison |
| `03_search_selection_overfitting.ipynb` | Search, selection and overfitting checks |
| `04_ablations_and_insights.ipynb` | Ablations |
| `05_final_pipeline_and_submissions.ipynb` | Final pipeline and submissions |

The 19 figures are in `figures/` as PNG and PDF. The working log is `notes/00_log.md`.

## Leaderboard

All rows use R=10, on a CV seed different from the search's.

| Config | A (±SE) | B (±SE) | C (±SE) | J |
|---|---|---|---|---|
| Majority (always Recruitment) | .623 ± .031 | .388 ± .048 | .624 ± .054 | .505 |
| LR, most-probable decoding | .902 ± .017 | .812 ± .036 | .838 ± .038 | .857 |
| LR + SLD(a=25) + exp-F1 ("anchor") | .899 ± .016 | .863 ± .028 | .853 ± .035 | .881 |
| LDA (balanced) + temperature, exp-F1 | .901 | .846 | .876 | .874 |
| C&S as predictor | .905 ± .016 | .835 ± .035 | .846 ± .037 | .870 |
| LR + LDA + C&S ensemble + SLD100 + exp-F1 | .912 ± .015 | .846 ± .030 | .865 ± .032 | .879 |
| HGB stacker | .900 | .684 | .859 | .792 |
| K-means cluster-to-label as predictor | .79 | .67 | .75 | .73 |
| **Selected (1-SE)** | **.909 ± .015** | **.876 ± .029** | **.882 ± .030** | **.892** |
| Best rule-compliant config | .908 | .890 | .872 | .899 |
| Best overall (does not satisfy the rule) | .901 | .910 | .885 | .905 |

## What the search chose and why

**Searches run** (logs in `results/search/`):

- a main TPE search, 1,200 trials
- a random search, 1,363 trials, used for the sensitivity plots
- a TPE search restricted to rule-compliant configs, 800 trials

**Selection.** The top distinct configs were re-evaluated on a fresh seed. The 1-SE rule then picked the simplest config within one standard error of the best. Selection was restricted to configs where clustering or a graph takes part in filling the labels: the plain top scorer fills them with self-training only, which arguably breaks the "unsupervised method must fill the labels" rule.

**Chosen pipeline.**

1. Fill the pool's missing labels with C&S pseudo-labels. They are hard labels, kept only where C&S is at least 50% confident, and given weight 0.05.
2. Train a balanced LDA on the labeled rows plus those pseudo-labeled rows.
3. Temperature scaling.
4. Expected-F1 decoding.

**Robustness of the choice.** The simplest-within-paired-SE rule and the unconstrained 1-SE rule pick the same config.

**Why it works.** The pseudo-labeled pool rows mainly improve LDA's shared covariance estimate; using them only for the covariance recovers the whole gain.

- Hard, confident pseudo-labels help LDA: J goes from .874 to .898.
- Soft labels hurt it (J .81–.84), because they leak between-class variation into the covariance.
- LR gains almost nothing from its own self-training labels. That explains why the research report saw self-training fail.

**What actually mattered.** The extra feature blocks carry essentially no signal on their own: alone they score A = .60–.64, the same as always predicting Recruitment. Adding them lowers B.

## How much the search overfits

The whole search-and-select procedure was run inside outer splits, and the winner was scored on rows the search never saw.

- **5 outer folds, no time shift.** Selection optimism is small, at most 0.006. But the selected config scored .901 on held-out rows against .913 for the fixed anchor: −0.012, paired SE 0.011.
- **Chronological outer split.** Here the search saw only trials before 2005. It picked an LDA config with no prior correction, which scored **.762** on the later rows against **.858** for the anchor (−0.096, SE 0.031). Its inner validation contained almost no shift, so it couldn't learn that a correction was needed.
- **On a third CV seed.** The selected config beats the anchor by only +0.009 in J (SE 0.010). It beats plain LR by +0.036 (SE 0.014).

## Candidate CSVs

All are in `midterm/part1/submissions/`. Each has a `.manifest.json` with its config, its scores with standard errors, and its predicted test label rates. All are trained on the 235 labeled rows, use the 1,014 unlabeled rows as the pool, and match the sample file's IDs and format.

| File | What it is |
|---|---|
| `cand1_selected_1se.csv` | **Main candidate**, selected by the search. Its predicted test label rates match the SLD estimate of the test distribution (Recruitment 0.577 vs 0.57). |
| `cand5_lr_cs_sld_robust.csv` | **Robust hedge.** The anchor plus C&S pseudo-labels, so it satisfies the rule. It matches the anchor on the nested splits. It differs from cand1 on about 15% of test rows. |
| `cand2_compliant_argmax.csv` | The best-scoring rule-compliant config. 96.5% identical to cand1. |
| `cand4_ens3_cs_sld.csv` | The research report's LR + LDA + C&S ensemble. |
| `reference/ref_LR_SLD25_expF1.csv`, `reference/ref_unconstrained_argmax.csv` | For comparison only. Neither satisfies the rule. |
| `part1_train_filled.csv` | The 1,014 train rows with their missing labels filled in. |

## What didn't reproduce, didn't work, or is uncertain

- **The research report's SLD gain (0.816 → 0.856)** holds only when SLD is paired with expected-F1 decoding. Against the stronger most-probable-combination baseline, SLD alone does little.
- **The research report's val gain (0.857 → 0.893) is an artifact.** Its smoothing put about 17% probability on the Safety+Efficacy combination. That combination is absent from the 157 training rows but appears 5 times in val. A proper SLD gives .846–.859.
- **SLD collapses when weakly damped** (a ≤ 25) on temperature-softened or LDA outputs: one rare combination absorbs most of the estimated prior. A per-label SLD, tried as a fix, made things worse.
- **A real bug was fixed.** Combinations missing from a fold's training rows must keep probability exactly zero; otherwise SLD blows them up, and val drops to 0.175.
- **Protocol B exaggerates the shift.** Recruitment falls to 0.40 there, versus an estimated 0.57 on test. The balanced-prior choice could therefore over-correct on test. The predicted test label rates suggest it doesn't, but this is unverified.
- **Funding** has only 12 labeled positives, so its F1 is noise.
- **Not done.** HDBSCAN and Sinkhorn alignment are implemented but weren't searched. There is no bootstrap bagging at refit; 5 seeds are averaged instead. BBSE wasn't tried.
- **For Gradescope.** `midterm/` is about 100 MB, mostly search logs under `results/`, which can be left out. The 0.6 GB results cache is not included in this folder; it rebuilds automatically into `.p1_cache/`.

## Decision

If only one Kaggle final can be picked, the evidence doesn't separate cand1 and cand5.

- **cand1** is what the requested 1-SE procedure selects, and it has the best val score.
- **cand5** is the safer bet, given the nested results.

Treat a public-leaderboard gap of under about 0.05 between them as a tie.
