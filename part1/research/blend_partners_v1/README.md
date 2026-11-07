# Complementary classifier partners

Selected `anchored90_partners10` as the second, riskier manual trial: 90% anchored, 5% tri-training and 5% SVM hard vote. It improves Train157 by .000796 and Pooled235 by .001862, ties validation and loses .013605 on chronology. Paired wins/ties/losses against current are 3/4/1 and 6/1/1 respectively: it fails the strict both-pool rule. Its chronology loss is concentrated in the 22-row 2007 window (.939394 to .878788). The 85/15 alternative slightly lowers Train157 and was not exported. Selection preceded opening the final test scores for this batch. The prepared CSV changes 4 of 312 labelsets; no score is known.

## Matched evaluation

Settings were registered in commit `885775ba` before scoring this batch. The original anchored 0.89636 recipe remains the policy parent; the submitted 75/25 blend (reported Kaggle 0.89850) is the practical comparator. All recipes used all 84 standard cases: eight five-fold grouped repeats on Train157 and Pooled235, original validation78 and three chronological windows (39/22/37). Report complete OOF sample F1 per repeat; population SD, macro and every per-label metric are in the compact tables. Saved fold-local component predictions were reused, with no new model fitting or new split selection. Existing fitting boundaries and duplicate-group purges remain those of the frozen testing policy.

| Recipe | Train157 | Pooled235 | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| anchored | 0.932059 | 0.921365 | 0.891026 | 0.884354 |
| anchored85_partners15 | 0.931794 | 0.925000 | 0.891026 | 0.877551 |
| anchored90_partners10 | 0.932856 | 0.925266 | 0.891026 | 0.870748 |
| current75_25 | 0.932059 | 0.923404 | 0.891026 | 0.884354 |
| lda_control | 0.931794 | 0.921631 | 0.905983 | 0.877551 |

Every chronological window, paired repeat delta and per-label metric is retained under `results/`. Neither exported candidate meets the strict positive-mean / at-least-five-repeat-wins rule in both pools. These are explicit manual trial exceptions; the scored leader remains unchanged.

## Experiment sequence and limitations

This is an adaptive three-batch search on reused development data: seven fixed weight/cutoff settings, then two partner-only blends, then one fixed half-step. Each batch was committed before its own scores were inspected, but later batches were informed by earlier results. All ten candidate recipes, including rejected settings, are documented. There is no untouched holdout and no guarantee of a Kaggle improvement. The previous Kaggle improvement motivated this search.

The partner mixture is recovered as `(ensemble - .7 * matched_Ani_LDA_source) / .3`; the original ensemble is exactly .7 Ani-LDA + .15 tri-training + .15 SVM hard vote. IDs, truth, score range and reconstruction are checked across all 84 cases. Anchored scores are centered from their original .3 cutoff before blending; the common decoder uses .5 and an argmax fallback for an empty labelset.

## Retained evidence and verification

Code, registered settings, all compact score tables and checks remain. Full CV/test score and prediction arrays are retained only for the two exported finalists, with exact CSVs and SHA-256 manifests. Existing source fits and controls are referenced without duplication; no failed fitted models or generated datasets were created. `retention.json` hashes current retained records. Preparation manifests are unsubmitted until a separate result record supersedes them.

Both exports reuse components fitted on all 235 verified labels plus the 1,014 originally masked training rows, under the existing inductive protocol. This final fitting population is separate from the fold-local evaluation. No test input was used for fitting and no new fit was required to change blend weights.

From the repository root, run:

```sh
.venv/bin/python part1/research/blend_followup_v1/verify.py
```

This checks all three source registrations, re-scores 8,316 retained metric values, checks CV averages/SD and paired counts, verifies selected CV arrays, independently reconstructs both exported score arrays and CSVs, and checks retention hashes. The exports differ on three test labelsets from each other. For a fresh metric replay, use a disposable checkout with only these batches' generated result directories absent, then run each study's `evaluate` command. Registration and export commands refuse overwriting existing records.

Environment: Python 3.12.13; numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.0.
