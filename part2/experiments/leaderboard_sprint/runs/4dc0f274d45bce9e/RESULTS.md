# Iteration after file 10 scored 0.58609

The user has up to four further Kaggle submissions in the current window. File 10 is the scored baseline; these are adaptive local experiments. No new file has a known score until the user returns a Kaggle result.

## First candidate

File 11 uses **80% file 10 + 20% the original balanced ensemble**, with every final component trained on all 9,050 labeled rows. The 30% mixture has the best local means, but 20% limits regression on the latest forward window and worst gap. Use the first Kaggle result before increasing this weight. File 12 is a distinct reserve: 50% file 10, 25% sharper neighbors (file 09), and 25% Ani's Phase III trend control.

The first batch freezes 32 proposals: five fast-neighbor blend weights, complementary mixtures, and ten small meta-model changes. Every recipe uses the same ten original splits plus three longer-gap checks. Strict preprocessing is the primary comparison; the published-recipe sensitivity track preserves each component's original feature policy. Equal-phase log loss is lower-is-better.

## Strict

| model | official | temporal | gap_mean | gap_worst | cv |
| --- | --- | --- | --- | --- | --- |
| file10 | 0.557839 | 0.535890 | 0.563189 | 0.628816 | 0.586891 |
| mix_legacy01_0.2 | 0.554437 | 0.533926 | 0.561563 | 0.629638 | 0.581169 |
| mix_sharp_trend | 0.556909 | 0.535544 | 0.562491 | 0.627765 | 0.586105 |
| mix_legacy01_0.3 | 0.553279 | 0.533405 | 0.561247 | 0.630614 | 0.578966 |

## Published

| model | official | temporal | gap_mean | gap_worst | cv |
| --- | --- | --- | --- | --- | --- |
| file10 | 0.559173 | 0.534547 | 0.562306 | 0.626429 | 0.585453 |
| mix_legacy01_0.2 | 0.555292 | 0.532817 | 0.560799 | 0.627428 | 0.579885 |
| mix_sharp_trend | 0.558153 | 0.534112 | 0.561671 | 0.625768 | 0.584478 |
| mix_legacy01_0.3 | 0.553946 | 0.532421 | 0.560552 | 0.628549 | 0.577793 |

## Paired uncertainty

The historical evaluation labels have been reused. Intervals are descriptive and do not correct for adaptive model selection. No local point improvement guarantees a better Kaggle score.

| model | suite | kind | delta | lower | upper |
| --- | --- | --- | --- | --- | --- |
| mix_legacy01_0.2 | main | temporal | -0.001964 | -0.003555 | -0.000472 |
| mix_legacy01_0.2 | main | group_cv | -0.005722 | -0.006688 | -0.004767 |
| mix_sharp_trend | main | temporal | -0.000346 | -0.001207 | 0.000459 |
| mix_sharp_trend | main | group_cv | -0.000786 | -0.001527 | -0.000099 |
| mix_legacy01_0.3 | main | temporal | -0.002485 | -0.004832 | -0.000288 |
| mix_legacy01_0.3 | main | group_cv | -0.007925 | -0.009347 | -0.006517 |
| mix_legacy01_0.2 | gaps | temporal | -0.001626 | -0.003237 | -0.000041 |
| mix_sharp_trend | gaps | temporal | -0.000698 | -0.001554 | 0.000157 |
| mix_legacy01_0.3 | gaps | temporal | -0.001942 | -0.004314 | 0.000430 |

## Artifacts

- [All strict results](summary_strict.csv)
- [All published-recipe results](summary_published.csv)
- [Every period and phase](all_scores.csv)
- [Frozen selection](selection.json)
- [Verification](verification.json)
- [Recipe/source manifest](manifest.json)

## Reproduce

```sh
.venv/bin/python part2/sprint_benchmark.py --track strict
.venv/bin/python part2/sprint_benchmark.py --track published
.venv/bin/python part2/sprint_benchmark.py --track strict --summarize
.venv/bin/python part2/sprint_report.py
```

Native caches are ignored by Git. Export receipts record exact CSV hashes, recipes, training populations and native replay errors. Original submissions remain unchanged.
