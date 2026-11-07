# 40 · Candidate submissions

All files are in `midterm/part1/submissions/`. Each CSV has 312 rows plus a header, `ID,Predicted`, with space-separated reason names, in the exact ID order of the sample submission (checked in `submit.write_candidate`). Each has a `.manifest.json` with:

- the full config
- scores on A/B/C with SEs and per-label F1
- predicted test label rates, cardinality and combination counts
- the SLD test-prior estimate (when SLD is used)
- temperatures

**How all of them were made:** refit on all 235 labeled rows, with the 1,014 unlabeled rows as the pseudo-label pool. Transductive (test features used label-free). 5 seeds averaged; seeds change only inner folds. **Nothing was submitted to Kaggle.**

## What each file is

| File | Pipeline | Rule-compliant? | Why it exists |
|---|---|---|---|
| **`cand1_selected_1se.csv`** | Balanced LDA + C&S hard pseudo-labels (graph k=10 on PCA-64/L2, α₁ 0.8, α₂ 0.64, keep conf ≥ 0.5, λ 0.052) + temperature + exp-F1 | yes (graph fills labels) | **Main.** What the 1-SE procedure selects (`search#787`). |
| **`cand5_lr_cs_sld_robust.csv`** | Powerset LR (C=0.1) + C&S hard pseudo-labels (same graph, α₁ 0.8, α₂ 0.64, conf ≥ 0.5, λ 0.1) + SLD(a=25) + exp-F1; no temperature | yes | **Robust hedge.** The a-priori anchor made compliant; it matches the anchor on the nested outer splits. |
| `cand2_compliant_argmax.csv` | cand1 + K-means(8) soft-assignment features + kNN-vote(20) features, λ 0.063, α₂ 0.42 | yes | Highest re-evaluated J among compliant configs (`search#691`) |
| `cand4_ens3_cs_sld.csv` | Equal-weight ensemble of powerset LR, LDA + T, and C&S-direct (α₁ = α₂ = 0.5), then SLD(a=100) + exp-F1 | yes (C&S member) | The research report's recommended ensemble; a different mechanism, as a hedge |
| `reference/ref_LR_SLD25_expF1.csv` | Powerset LR + SLD25 + exp-F1 (the "anchor") | **no** (no unsupervised label filling) | For calibrating leaderboard feedback only |
| `reference/ref_unconstrained_argmax.csv` | LDA(bal) + criteria + pathways + drugmeta + kNN-vote20 + plain self-training (λ 0.22) + T + exp-F1 | **no** (plain self-training) | The best J overall (`search#1122`); comparison only |
| `part1_train_filled.csv` | The 1,171 train rows with the 1,014 masked labels filled by the cand1 pipeline. Adds `label_source` (given/filled) and `fill_confidence`. | | The assignment's "augment your dataset" deliverable |

`cand3_paired_1se` was skipped because the paired-SE rule picked the same config as cand1.

## Scores

From `candidates_summary.csv` and the manifests. R=10, CV seed 1, which is not the search seed. These are post-selection numbers, so they are optimistic for cand1 and cand2.

| Candidate | J (SE) | A | B | B2005 | C (val) | C macro-F1 | Complexity |
|---|---|---|---|---|---|---|---|
| cand1 | 0.892 (0.016) | 0.909 | 0.876 | 0.889 | **0.882** | **0.769** | 1.75 |
| cand2 | **0.899** (0.016) | 0.908 | 0.890 | 0.895 | 0.872 | 0.745 | 3.75 |
| cand4 | 0.879 (0.017) | **0.912** | 0.846 | 0.844 | 0.865 | 0.689 | 3.75 |
| cand5 | 0.889 (0.016) | 0.909 | 0.869 | 0.864 | 0.865 | 0.616 | 1.5 |
| ref anchor | 0.881 (0.016) | 0.899 | 0.863 | 0.858 | 0.853 | 0.602 | 0.5 |
| ref unconstrained argmax | 0.905 (0.014) | 0.901 | 0.910 | 0.923 | 0.885 | 0.762 | 5.75 |

**Val (C) per-label F1:**

| Candidate | Safety | Funding | Efficacy | Recruitment | Other |
|---|---|---|---|---|---|
| cand1 | 0.941 | 0.667 | 0.720 | 0.971 | 0.545 |
| cand2 | 0.941 | 0.667 | 0.583 | 0.990 | 0.545 |
| cand4 | 0.889 | 0.400 | 0.769 | 0.952 | 0.435 |
| cand5 | 0.947 | **0.000** | 0.759 | 0.952 | 0.421 |
| ref anchor | 0.824 | 0.000 | 0.769 | 0.943 | 0.476 |

Val has 4 Funding positives, so a Funding F1 of 0 vs 0.667 is a few rows. The macro-F1 gap between cand1 and cand5 comes mostly from Funding and Other.

**cand1 on all protocols, with SEs** (notebook 05, cell 4):

| Protocol | n | Sample-F1 (SE) | Macro-F1 (SE) | S / F / E / R / O |
|---|---|---|---|---|
| official val (157 → 78) | 78 | 0.882 (0.030) | 0.769 (0.070) | .941 / .667 / .720 / .971 / .545 |
| repeated CV (235, 5×10) | 235 | 0.909 (0.015) | 0.836 (0.027) | .912 / .809 / .887 / .970 / .601 |
| chrono holdouts (mean of 3) | ~98 | 0.876 (0.029) | 0.830 (0.038) | .931 / .727 / .898 / .945 / .646 |
| chrono 2005 | 98 | 0.889 (0.027) | 0.853 (0.039) | .971 / .778 / .909 / .940 / .667 |

## How the candidates differ on test

**Predicted test label rates**, against the label-free SLD estimate of the test prior from the LR anchor (manifests; [`fig_test_label_rates.png`](../midterm/part1/figures/fig_test_label_rates.png)):

| | Safety | Funding | Efficacy | Recruitment | Other | Cardinality |
|---|---|---|---|---|---|---|
| SLD-estimated test prior | 0.134 | 0.017 | 0.099 | 0.570 | 0.287 | |
| cand1 | 0.144 | 0.026 | 0.125 | **0.577** | 0.272 | 1.144 |
| cand2 | 0.144 | 0.029 | 0.128 | 0.574 | 0.260 | 1.135 |
| cand4 | 0.141 | 0.019 | 0.115 | 0.615 | 0.301 | 1.192 |
| cand5 | 0.167 | 0.013 | 0.125 | 0.603 | 0.263 | 1.170 |
| ref anchor | 0.147 | 0.013 | 0.115 | 0.606 | 0.324 | 1.205 |
| ref unconstrained | 0.141 | 0.029 | 0.135 | 0.571 | 0.340 | 1.215 |

**Exact agreement between test predictions** (fraction of the 312 rows with identical label sets; computed read-only from the CSVs):

| | cand1 | cand2 | cand4 | cand5 | ref anchor | ref unconstr. |
|---|---|---|---|---|---|---|
| cand1 | 1 | 0.965 | 0.862 | **0.846** | 0.846 | 0.865 |
| cand2 | | 1 | 0.859 | 0.856 | 0.846 | 0.862 |
| cand4 | | | 1 | 0.875 | 0.885 | 0.788 |
| cand5 | | | | 1 | 0.894 | 0.792 |

- **cand1 and cand5 disagree on about 15% of test rows** (about 48 rows). On a public split of about 100 rows, roughly 15 of those would be visible. Each row is worth about 0.01 sample-F1, so a lopsided split could move the public gap by a few hundredths in either direction, consistent with the "under 0.05 is a tie" rule.
- **cand1 and cand2 are near-duplicates** (96.5% identical), so submitting both wastes a slot.

## Nested and fixed-outer evidence

From `results/search/nested/fixed_configs_outer.csv`. cand1 and cand2 are contaminated because they were selected on these rows:

| Config | Pooled CV outer | vs anchor (paired SE) | Chrono outer | vs anchor (paired SE) |
|---|---|---|---|---|
| cand1 | 0.902 | −0.011 (0.013) | 0.889 | +0.032 (0.016) |
| cand2 | 0.909 | −0.004 (0.011) | 0.895 | +0.037 (0.021) |
| cand4 | 0.913 | 0.000 (0.007) | 0.844 | −0.014 (0.018) |
| cand5 | 0.912 | −0.001 (0.007) | 0.864 | +0.006 (0.015) |
| anchor | 0.913 | | 0.858 | |

When the *selection procedure itself* was nested, the procedure's picks scored 0.901 pooled vs the anchor's 0.913, and 0.762 vs 0.858 on the chrono split ([20_search_and_selection.md §6](20_search_and_selection.md)).

## Recommendation and rationale

**The builder's decision** (REPORT.md, "Decision"): submit both cand1 and cand5. If only one Kaggle final can be picked, the evidence doesn't separate them.

**Case for cand1:**

- It is what the pre-stated, requested procedure (1-SE selection on J) selects.
- It has the best val score among compliant candidates (0.882) and the best val macro-F1 (0.769).
- It has a clear mechanism: graph labels improve LDA's covariance.
- Its predicted Recruitment rate (0.577) matches the label-free SLD estimate (0.570), so it doesn't look over-corrected.
- It is the simplest compliant config within noise (complexity 1.75).

**Case for cand5:**

- It equals the anchor, which won the nested comparison. The anchor beat the procedure's picks by 0.012 pooled and 0.096 on the chrono split.
- It uses an *estimated* prior correction (SLD on the test batch) rather than a fixed uniform prior. That is safer if the test prior isn't what B implies.
- It has the lowest variance across seeds (A repeat sd 0.0037 vs 0.0051 for cand1; manifests, `sample_f1_rep_sd`).
- It is still rule-compliant.

**What would tip the decision (my summary of the evidence, not new analysis):**

- **The fresh-seed J gap** is +0.009 ± 0.010 in cand1's favour. **The nested outer gap** for the *procedure* is −0.012 ± 0.011 against it. Both are about 1 SE, in opposite directions.
- **cand1's weak point:** a fixed uniform decision prior could over-predict rare reasons if the true test prior is closer to the training prior. The predicted rates argue against this.
- **cand5's weak points:** SLD estimated on only 312 test rows, and LR gaining nothing from the pseudo-labels (its compliance is nominal: λ 0.1 hard C&S labels move LR J by ±0.01).

**Leaderboard policy** (from the research report, adopted by the builder):

- Treat public-LB differences under about 0.05 as ties. The public split is roughly 94-156 rows, an assumption; SE about 0.022-0.028.
- Don't tune on the LB.

**If the team wants a third submission:** cand4 tests a different mechanism (ensemble + SLD). The reference anchor would tell how much the C&S labels change LR, but it is not rule-compliant as a final.

## The filled training labels

`part1_train_filled.csv` is the 1,171 train rows (the file has many physical lines because criteria text contains newlines):

- 157 rows `label_source = given`, 1,014 `filled`
- no −100 left (notebook 05)

**Label rates of the filled rows by labeler** (`filled_label_rates_by_labeler.csv`):

| Labeler | Safety | Funding | Efficacy | Recruitment | Other |
|---|---|---|---|---|---|
| selected pipeline (cand1) | 0.081 | 0.023 | 0.127 | 0.693 | 0.230 |
| C&S (graph) | 0.085 | 0.013 | 0.106 | 0.695 | 0.135 |
| label spreading | 0.098 | 0.030 | 0.104 | 0.690 | 0.134 |
| K-means k24 → label | 0.047 | 0.027 | 0.107 | 0.709 | 0.110 |
| GMM k16 → label | 0.128 | 0.000 | 0.091 | 0.754 | 0.027 |
| SLD-EM (LR) | 0.065 | 0.000 | 0.093 | 0.712 | 0.200 |

The selected pipeline's filled Recruitment rate (0.693) is close to the SLD estimate for the pool (0.716; see [01](01_data_findings.md) §3). Exact agreement with the other labelers on these rows: C&S 0.848, SLD-EM 0.872, spreading 0.768, K-means 0.712, GMM 0.714.

**Two subtleties:**

- **These filled labels come from the final cand1 model's own predictions on the pool** (`submit.fill_pool_labels`). They are not the C&S pseudo-labels the pipeline trains on internally, which is why they differ from the C&S column.
- **The pipeline's internal C&S labels use only rows with C&S confidence ≥ 0.5.** The file fills *all* 1,014 rows. The fraction of pool rows passing the 0.5 filter isn't recorded in the results; see [60_open_questions.md](60_open_questions.md).
