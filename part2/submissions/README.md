# Part Two manual Kaggle submissions

**Current best: file 10, Kaggle 0.58609.** The user reported second place behind
0.58526. The improvement over Ani's 0.58688 is 0.00079; the gap to the reported
leader is 0.00083. Use this exact file as the next experiment's reference.
The [current ledger](leaderboard_results.json) records the score against its hash.

**New candidates:** [11_file10_legacy20.csv](11_file10_legacy20.csv) is the first
upload recommendation. Files 12–14 are prepared reserves; choose the next upload
after receiving file 11's score. All four Kaggle scores are currently unknown.
See the [common comparison and SCP command](../experiments/leaderboard_sprint/RESULTS.md)
and [four-submission queue](sprint_queue.json).

The [complete common benchmark](../experiments/all_submissions/RESULTS.md) directly
compares all ten files and both Ani winners on the same fixed splits. Use its
tables for cross-recipe comparisons; the historical 01–06 temporal scores below
used a different holdout and should not be compared with the newer scores.
On the common strict benchmark, file 10 improves file 02's chronological mean
from 0.540324 to 0.535890 and longer-gap mean from 0.568506 to 0.563189, while
grouped CV worsens from 0.578048 to 0.586891. The chronological intervals include
zero; the later Kaggle score provides separate evidence of an improvement.

**Best scored file: [10_ani_fast2_neighbors_half.csv](10_ani_fast2_neighbors_half.csv).**
It beats Ani's recency model on all six chronological windows in both preprocessing
tracks, with a small grouped-CV improvement. [09_ani_replace_knn5.csv](09_ani_replace_knn5.csv)
has the larger gain on the original three windows, but a small CV regression.
File 10 scored **0.58609**; file 09 remains unscored.
The chronological uncertainty intervals remain unchanged by the Kaggle result.

| Candidate | Strict chronological mean | Strict grouped CV | Larger-gap mean | Kaggle |
| --- | ---: | ---: | ---: | --- |
| Ani recency reference | 0.537446 | 0.587411 | 0.564681 | 0.58688 |
| 09: sharper recent kNN | 0.535213 | 0.589230 | 0.563657 | Unknown |
| 10: recent-neighbor blend | 0.535890 | 0.586891 | 0.563189 | **0.58609** |

See [full findings](../experiments/ani_extensions/RESULTS.md) and the
[recipes, checksums and native bundle manifest](ani_extensions_manifest.json).
The export manifest is frozen at its pre-Kaggle state; the current ledger records
later scores. The all-submission benchmark retains its original ledger snapshot
so local fitting and selection evidence remain reproducible.
The [previous research round](../experiments/stack_improvements/RESULTS.md) and its
[empty promotion manifest](stack_improvements_manifest.json) remain historical records.

Historical results below used Ani's **0.58688** reference. Our 03 scored **0.59562**, then 02 improved to **0.58993**. File 10 is now the best scored model at **0.58609**. Initial 01/04 remain unscored. **Hold 05/06; use the fixed benchmark for future comparisons.**

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

07 blends 75% of the known 0.58688 reference with 25% of Ani's capped Phase III trend alternative. It has small favorable point differences across both benchmark tracks, but temporal intervals include zero. 08 is an Ani-heavy linear blend: it improves grouped CV but is less consistent across time. Neither has a known Kaggle score. These are historical proposals; file 10 is now the confirmed best and the current sprint above determines subsequent upload choices. See [the benchmark report](../experiments/local_benchmark/RESULTS.md).

| submission | mean_abs_change_from_ani | kaggle_score |
| --- | --- | --- |
| 07_ani_mild_phase3_trend.csv | 0.00453 |  |
| 08_ani_linear25.csv | 0.01280 |  |

All numbered files have exactly 1,500 rows and columns `ID,label`. Index CSVs are bookkeeping and should not be uploaded. Leaderboard results are tied to file hashes in `leaderboard_results.json`; exact recipes are in the corresponding manifest JSONs.
