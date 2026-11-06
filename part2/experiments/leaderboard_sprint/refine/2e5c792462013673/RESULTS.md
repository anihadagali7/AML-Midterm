# Refinement after file 11 reached 0.58474

The next reported score is **file 13: 0.58465**, a further 0.00009 improvement. The 19 new mixtures were declared before their evaluation. The recipes reuse the exact 13 fixed splits and both original preprocessing policies; no additional base models are fitted.

The stronger 30% legacy weight gave only a small Kaggle gain over 20%. **File 15** retained 20% legacy ensemble and added 10% recent-weighted tree and 10% Ani phase trend; 60% remained file 10. It had the best combined chronological mean among candidates satisfying the declared adverse-gap limit under both policies, but scored **0.58492**, worse than file 13 by 0.00027. The improved historical averages did not transfer on this upload.

The prepared **file 16** keeps the successful 30% legacy weight and replaces 20% of file 10 with the sharper-neighbor file 09, leaving 50% file 10. It improves both chronological means and the latest forward window versus file 13 in both tracks, with small regressions in grouped CV, official loss and the worst gap. It adds neither the recent-tree nor phase-trend endpoint tested by file 15. The separate nine-recipe final-slot batch also evaluated smaller interpolations and nearby tree/trend weights.

**Final decision: hold file 16; do not spend the last slot on current evidence.** The user confirmed first place with nine submissions. The small chronological gains and intervals spanning zero do not give a strong case for this upload, and a single returned score would leave no follow-up slot. The earlier exploratory upload recommendation was withdrawn; the frozen local selection remains as historical evidence, with the later decision in submission_decision.json. All local labels have been consulted repeatedly; bootstrap intervals do not correct for adaptive selection. File 13 at 0.58465 remains the best scored submission.

## Strict

| model | official | temporal | gap_mean | gap_worst | cv | latest_delta |
| --- | --- | --- | --- | --- | --- | --- |
| file10 | 0.557839 | 0.535890 | 0.563189 | 0.628816 | 0.586891 | 0.000231 |
| file11 | 0.554437 | 0.533926 | 0.561563 | 0.629638 | 0.581169 | 0.000000 |
| file13 | 0.553279 | 0.533405 | 0.561247 | 0.630614 | 0.578966 | 0.000297 |
| file14 | 0.557730 | 0.532758 | 0.561623 | 0.631722 | 0.580729 | -0.002297 |
| legacy0.2_tree0.1_trend0.1 | 0.554265 | 0.532700 | 0.560879 | 0.630974 | 0.578254 | -0.001187 |
| legacy0.3_tree0.05_trend0.1 | 0.553096 | 0.532894 | 0.560852 | 0.631094 | 0.577509 | -0.000303 |
| legacy0.3_sharp0.2 | 0.553512 | 0.533037 | 0.561154 | 0.630814 | 0.579112 | -0.000279 |

## Published

| model | official | temporal | gap_mean | gap_worst | cv | latest_delta |
| --- | --- | --- | --- | --- | --- | --- |
| file10 | 0.559173 | 0.534547 | 0.562306 | 0.626429 | 0.585453 | -0.000329 |
| file11 | 0.555292 | 0.532817 | 0.560799 | 0.627428 | 0.579885 | 0.000000 |
| file13 | 0.553946 | 0.532421 | 0.560552 | 0.628549 | 0.577793 | 0.000601 |
| file14 | 0.558820 | 0.531713 | 0.560841 | 0.629407 | 0.579541 | -0.002232 |
| legacy0.2_tree0.1_trend0.1 | 0.555009 | 0.531727 | 0.560191 | 0.628945 | 0.577093 | -0.000818 |
| legacy0.3_tree0.05_trend0.1 | 0.553703 | 0.531970 | 0.560205 | 0.629178 | 0.576386 | 0.000209 |
| legacy0.3_sharp0.2 | 0.554137 | 0.532038 | 0.560508 | 0.628980 | 0.577868 | -0.000009 |

## Paired differences from file 13

| model | suite | kind | delta | lower | upper |
| --- | --- | --- | --- | --- | --- |
| legacy0.2_tree0.1_trend0.1 | main | temporal | -0.000705 | -0.001491 | 0.000072 |
| legacy0.2_tree0.1_trend0.1 | main | group_cv | -0.000712 | -0.001198 | -0.000229 |
| legacy0.3_tree0.05_trend0.1 | main | temporal | -0.000511 | -0.001135 | 0.000095 |
| legacy0.3_tree0.05_trend0.1 | main | group_cv | -0.001457 | -0.001852 | -0.001058 |
| legacy0.3_sharp0.2 | main | temporal | -0.000368 | -0.000762 | 0.000000 |
| legacy0.3_sharp0.2 | main | group_cv | 0.000146 | -0.000205 | 0.000489 |
| legacy0.2_tree0.1_trend0.1 | gaps | temporal | -0.000368 | -0.001142 | 0.000440 |
| legacy0.3_tree0.05_trend0.1 | gaps | temporal | -0.000395 | -0.001000 | 0.000266 |
| legacy0.3_sharp0.2 | gaps | temporal | -0.000093 | -0.000445 | 0.000245 |

## Submission evidence

- `11_file10_legacy20.csv`: 0.58474 (user-reported)
- `13_file10_legacy30.csv`: 0.58465 (user-reported)
- `15_file10_legacy20_tree10_trend10.csv`: 0.58492 (user-reported)
- `16_file10_legacy30_sharp20.csv`: unknown Kaggle score

[Every recipe, strict](summary_strict.csv), [published](summary_published.csv), [every period and phase](all_scores.csv), [paired uncertainty](paired_comparisons.csv), [verification](verification.json), and [frozen recipes/source hashes](manifest.json) retain the complete record. Receipts preserve native dependency hashes and their pre-submission score state. See the live queue and ledger for later observations.

## Reproduce

```sh
.venv/bin/python part2/sprint_refine.py benchmark
.venv/bin/python part2/sprint_refine.py export --model legacy0.2_tree0.1_trend0.1 --filename 15_file10_legacy20_tree10_trend10.csv
.venv/bin/python part2/sprint_refine_report.py
```

Existing native component caches are required for export; the earlier benchmark workflows reproduce them. All supervised final components use 9,050 labeled rows. No new Kaggle upload is performed by these commands.
