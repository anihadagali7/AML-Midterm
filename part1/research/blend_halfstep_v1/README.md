# A fixed halfway blend

Not exported. This single fixed half-step combines half the current 75/25 blend with half the 90/10 partner blend. Train157 gains .000265 and Pooled235 .000709, while validation ties and chronology loses .010204. The smaller CV gain and continuing chronology regression did not justify a third submission. Against current, paired wins/ties/losses are 2/5/1 and 4/2/2. It fails the strict both-pool rule. No detailed candidate predictions or test export are retained.

## Matched evaluation

Settings were registered in commit `9fc1d060` before scoring this batch. The original anchored 0.89636 recipe remains the policy parent; the submitted 75/25 blend (reported Kaggle 0.89850) is the practical comparator. All recipes used all 84 standard cases: eight five-fold grouped repeats on Train157 and Pooled235, original validation78 and three chronological windows (39/22/37). Report complete OOF sample F1 per repeat; population SD, macro and every per-label metric are in the compact tables. Saved fold-local component predictions were reused, with no new model fitting or new split selection. Existing fitting boundaries and duplicate-group purges remain those of the frozen testing policy.

| Recipe | Train157 | Pooled235 | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| anchored | 0.932059 | 0.921365 | 0.891026 | 0.884354 |
| blend_halfstep | 0.932325 | 0.924113 | 0.891026 | 0.874150 |
| current75_25 | 0.932059 | 0.923404 | 0.891026 | 0.884354 |

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
