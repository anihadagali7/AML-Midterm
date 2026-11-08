# Fixed blend weights and global cutoffs

**Closed, not submitted.** The user confirmed that this candidate was skipped. It has no Kaggle score or score tag. The exact CSV is archived under `submissions/archive/unsubmitted/`; [outcome.json](outcome.json) records its path and hash.

Selected `blend50_50` as the first manual trial. It ties the current blend's Train157 sample-F1 mean, improves Pooled235 by .000177 and validation by .008547, but loses .003401 on chronology. Paired wins/ties/losses against current are 4/1/3 on Train157 and 2/3/3 on Pooled235. This does not pass the strict both-pool rule and is not a promoted leader. The cutoff reductions and other weights did not provide a clear all-check improvement. Selection preceded opening this candidate's final test scores. The prepared CSV changes 3 of 312 labelsets; no score is known.

## Matched evaluation

Settings were registered in commit `faeadb01` before scoring this batch. The original anchored 0.89636 recipe remains the policy parent; the submitted 75/25 blend (reported Kaggle 0.89850) is the practical comparator. All recipes used all 84 standard cases: eight five-fold grouped repeats on Train157 and Pooled235, original validation78 and three chronological windows (39/22/37). Report complete OOF sample F1 per repeat; population SD, macro and every per-label metric are in the compact tables. Saved fold-local component predictions were reused, with no new model fitting or new split selection. Existing fitting boundaries and duplicate-group purges remain those of the frozen testing policy.

| Recipe | Train157 | Pooled235 | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| anchored | 0.932059 | 0.921365 | 0.891026 | 0.884354 |
| blend50_50 | 0.932059 | 0.923582 | 0.899573 | 0.880952 |
| blend60_40 | 0.931529 | 0.923936 | 0.891026 | 0.880952 |
| blend70_30 | 0.931794 | 0.923759 | 0.891026 | 0.877551 |
| blend75_25_cut475 | 0.932590 | 0.922872 | 0.891026 | 0.884354 |
| blend75_25_cut525 | 0.930202 | 0.923404 | 0.891026 | 0.884354 |
| blend80_20 | 0.931529 | 0.922518 | 0.891026 | 0.884354 |
| blend90_10 | 0.932059 | 0.921986 | 0.891026 | 0.884354 |
| current75_25 | 0.932059 | 0.923404 | 0.891026 | 0.884354 |
| lda_control | 0.931794 | 0.921631 | 0.905983 | 0.877551 |

Every chronological window, paired repeat delta and per-label metric is retained under `results/`. Neither exported candidate meets the strict positive-mean / at-least-five-repeat-wins rule in both pools. These are explicit manual trial exceptions; the scored leader remains unchanged.

## Experiment sequence and limitations

This is an adaptive three-batch search on reused development data: seven fixed weight/cutoff settings, then two partner-only blends, then one fixed half-step. Each batch was committed before its own scores were inspected, but later batches were informed by earlier results. All ten candidate recipes, including rejected settings, are documented. There is no untouched holdout and no guarantee of a Kaggle improvement. The previous Kaggle improvement motivated this search.

The partner mixture is recovered as `(ensemble - .7 * matched_Ani_LDA_source) / .3`; the original ensemble is exactly .7 Ani-LDA + .15 tri-training + .15 SVM hard vote. IDs, truth, score range and reconstruction are checked across all 84 cases. Anchored scores are centered from their original .3 cutoff before blending; the common decoder uses .5 and an argmax fallback for an empty labelset.

## Retained evidence and verification

Code, registered settings, all compact score tables and checks remain. The user confirmed that both exported candidates were not submitted. Their exact small CSVs are archived under `submissions/archive/unsubmitted/`, with paths and hashes in each candidate study's `outcome.json`. After rescoring all 8,316 metrics and checking the exports, four redundant candidate CV/test prediction arrays (299,243 bytes total) were removed. Existing component/control scores still reproduce every result and both CSVs without refitting. The half-step had no detailed outputs to remove. `retention.json` hashes the currently retained records. Preparation manifests and their original file paths are historical; outcome records give the current archive paths. No new score tag is warranted.

Both exports reuse components fitted on all 235 verified labels plus the 1,014 originally masked training rows, under the existing inductive protocol. This final fitting population is separate from the fold-local evaluation. No test input was used for fitting and no new fit was required to change blend weights.

From the repository root, run:

```sh
.venv/bin/python part1/research/blend_followup_v1/verify.py
```

This checks all three source registrations, re-scores 8,316 retained metric values from existing component predictions, checks CV averages/SD and paired counts, independently reconstructs both archived CSVs, checks retention hashes and unsubmitted status, and verifies our existing 0.89850 result commit and annotated tag. Discarded candidate arrays are unnecessary for verification. The exports differ on three test labelsets from each other. For a fresh metric replay, use a disposable checkout with only these batches' generated result directories absent, then run each study's `evaluate` command. Registration and export commands refuse overwriting existing records.

Environment: Python 3.12.13; numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.0.
