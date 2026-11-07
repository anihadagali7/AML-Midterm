# Part Two manual Kaggle submissions

The [latest research comparison](../experiments/stack_improvements/RESULTS.md)
benchmarks both Ani winners and all proposed variants on identical saved splits.
Read its decision before choosing another upload; original numbered files below
remain historical artifacts. Any new shortlisted files and their unknown Kaggle
scores are recorded in [stack_improvements_manifest.json](stack_improvements_manifest.json).

Lower log loss is better. Best reported reference: **0.58688**. Our 03 scored **0.59562**, then 02 improved to **0.58993**. Initial 01/04 remain unscored. **Hold 05/06; use the new fixed benchmark for future comparisons.**

| submission | validation_logloss | temporal_logloss | kaggle_score |
| --- | --- | --- | --- |
| 01_balanced_ensemble.csv | 0.55413 | 0.55258 |  |
| 02_text_heavy.csv | 0.55359 | 0.55783 | 0.58993 |
| 03_phase_specialized.csv | 0.55126 | 0.55328 | 0.59562 |
| 04_capped_year.csv | 0.55397 | 0.55387 |  |

## Historical follow-ups — on hold

Hold **05/06** after the Ani comparison. The new chronological benchmark favors Ani over our linear endpoint.

| submission | validation_logloss | temporal_logloss | kaggle_score |
| --- | --- | --- | --- |
| 05_text_later_phases.csv | 0.55860 | 0.56431 |  |
| 06_lighter_tree_blend.csv | 0.55572 | 0.56071 |  |

## Optional Ani-anchored candidates

07 blends 75% of the known 0.58688 reference with 25% of Ani's capped Phase III trend alternative. It has small favorable point differences across both benchmark tracks, but temporal intervals include zero. 08 is an Ani-heavy linear blend: it improves grouped CV but is less consistent across time. Neither has a known Kaggle score. Prefer 07 if spending another exploratory submission; retain Ani as the confirmed best. See [the benchmark report](../experiments/local_benchmark/RESULTS.md).

| submission | mean_abs_change_from_ani | kaggle_score |
| --- | --- | --- |
| 07_ani_mild_phase3_trend.csv | 0.00453 |  |
| 08_ani_linear25.csv | 0.01280 |  |

All numbered files have exactly 1,500 rows and columns `ID,label`. Index CSVs are bookkeeping and should not be uploaded. Leaderboard results are tied to file hashes in `leaderboard_results.json`; exact recipes are in the corresponding manifest JSONs.
