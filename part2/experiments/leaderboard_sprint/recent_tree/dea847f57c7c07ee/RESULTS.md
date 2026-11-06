# Recent-weighted tree follow-up

One fixed recency-weighted CatBoost model is fitted per original split. Four prespecified probability blends are compared to files 10 and 11. No settings were adjusted inside this batch. This is adaptive follow-up research using previously consulted labels.

File 14 (`tree20`) is 80% file 10 plus 20% the new depth-6 tree. Its stronger standard forward mean and latest-period performance come with a regression on the worst individual longer-gap window. Compare both the average and that adverse period before interpreting the result. The Kaggle score is unknown until reported.

## Strict

| model | official | temporal | gap_mean | gap_worst | cv |
| --- | --- | --- | --- | --- | --- |
| file10 | 0.557839 | 0.535890 | 0.563189 | 0.628816 | 0.586891 |
| file11 | 0.554437 | 0.533926 | 0.561563 | 0.629638 | 0.581169 |
| recent_legacy20 | 0.555906 | 0.534089 | 0.562311 | 0.630513 | 0.582082 |
| recent_legacy30 | 0.555456 | 0.533613 | 0.562330 | 0.631897 | 0.580325 |
| recent_tree | 0.595390 | 0.549593 | 0.593329 | 0.703187 | 0.598888 |
| tree10 | 0.557358 | 0.533970 | 0.561989 | 0.629707 | 0.583275 |
| tree20 | 0.557730 | 0.532758 | 0.561623 | 0.631722 | 0.580729 |

## Published

| model | official | temporal | gap_mean | gap_worst | cv |
| --- | --- | --- | --- | --- | --- |
| file10 | 0.559173 | 0.534547 | 0.562306 | 0.626429 | 0.585453 |
| file11 | 0.555292 | 0.532817 | 0.560799 | 0.627428 | 0.579885 |
| recent_legacy20 | 0.556819 | 0.532986 | 0.561559 | 0.628300 | 0.580789 |
| recent_legacy30 | 0.556193 | 0.532636 | 0.561650 | 0.629826 | 0.579141 |
| recent_tree | 0.595390 | 0.549593 | 0.593329 | 0.703187 | 0.598888 |
| tree10 | 0.558565 | 0.532775 | 0.561153 | 0.627334 | 0.581953 |
| tree20 | 0.558820 | 0.531713 | 0.560841 | 0.629407 | 0.579541 |

## Paired uncertainty

Intervals do not correct for the adaptive research history.

| reference | suite | kind | delta | lower | upper |
| --- | --- | --- | --- | --- | --- |
| file10 | main | temporal | -0.003132 | -0.005359 | -0.000962 |
| file10 | main | group_cv | -0.006163 | -0.007596 | -0.004671 |
| file11 | main | temporal | -0.001168 | -0.002734 | 0.000398 |
| file11 | main | group_cv | -0.000441 | -0.001444 | 0.000569 |
| file10 | gaps | temporal | -0.001566 | -0.003930 | 0.000777 |
| file11 | gaps | temporal | 0.000060 | -0.001412 | 0.001612 |

[Every period](all_scores.csv), [verification](verification.json), and [frozen settings](manifest.json) retain the complete record. Final exports use all 9,050 labeled rows and preserve the 0.58609 baseline file.
