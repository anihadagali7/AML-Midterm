# Four-submission iteration

**File 11 scored 0.58474 and reached first place (user-reported).** It improves the
round's file 10 reference, 0.58609, by 0.00135 and beats the previously reported
leader's 0.58526 by 0.00052. Files 12–14 remain unscored reserves. One of the four
additional uploads was used; the original first-place objective has been achieved.
The [submission queue](../../submissions/sprint_queue.json) records returned scores
and the limit of four further uploads in the user's current window.

**The winning file 11** blends 80% file 10 with 20% of the original balanced
ensemble, refitted on all 9,050 labeled rows. Its improvements in both chronological
averages have supporting paired intervals, and it improves grouped CV. It still
worsens the worst individual gap window slightly. These are adaptively reused
historical evaluations. The later Kaggle result corroborates this candidate's
improvement; it does not make those local intervals independent or validate other weights.

All candidates use the same ten original splits and three longer-gap splits.
Preprocessing is fitted on training rows in the primary strict comparison.
The second track preserves each component's published feature policy. All four
candidates improve both chronological averages and grouped CV in both tracks.

| File | Recipe | Strict forward mean | Strict longer-gap mean | Strict worst gap | Strict grouped CV |
| --- | --- | ---: | ---: | ---: | ---: |
| 10 | Scored reference | 0.535890 | 0.563189 | 0.628816 | 0.586891 |
| 11 | 80% file 10 + 20% original ensemble | 0.533926 | 0.561563 | 0.629638 | 0.581169 |
| 12 | 50% file 10 + 25% sharp neighbors + 25% phase trend | 0.535544 | 0.562491 | 0.627765 | 0.586105 |
| 13 | 70% file 10 + 30% original ensemble | 0.533405 | 0.561247 | 0.630614 | 0.578966 |
| 14 | 80% file 10 + 20% recent-weighted tree | 0.532758 | 0.561623 | 0.631722 | 0.580729 |

Lower is better. File 13's larger ensemble weight gives stronger means but more
regression on adverse periods. File 14 has the strongest standard forward mean
and improves the latest forward period, with the largest worst-gap regression.
File 12 gives smaller gains with a favorable worst-gap result and provides a
different direction. File 11's success supports the complementary-ensemble
hypothesis, but does not establish that file 13's larger weight will help further.
Preserve file 11 and hold the three unscored reserves; no further upload is needed
to meet the reported first-place objective.

## Copy the winning submission

Run on the computer used to upload to Kaggle:

```sh
scp matthew@100.91.18.77:/Users/matthew/Desktop/michael/AML-Midterm/part2/submissions/11_file10_legacy20.csv ./part2_submission_11.csv
```

The submitted artifact is `part2_submission_11.csv`. Its SHA-256 is
`26ea655feba5094c2045657d328d8a16041f74eca5106f556f1971c7e4bab635`.

## Complete evidence

- [32-recipe blend/meta batch](runs/4dc0f274d45bce9e/RESULTS.md): 1,014 evaluations including controls, every period and paired uncertainty.
- [Four-recipe recent-tree batch](recent_tree/dea847f57c7c07ee/RESULTS.md): 182 evaluations including controls and independent blend replay.
- [Initial frozen policy](../../SPRINT_PROTOCOL.md) and [tree follow-up policy](../../SPRINT_TREE_PROTOCOL.md).
- [First-batch receipts](runs/4dc0f274d45bce9e/exports) and [tree receipt](recent_tree/dea847f57c7c07ee/exports/14_file10_recent_tree20.json): exact hashes, training populations, source versions and native replay errors.

All 16 existing regression tests passed. Each export has 1,500 unique sample-order
IDs and valid probabilities. Predictions reproduce exactly from native models
after replacing evaluation labels and identifiers. All final supervised fits use
the 9,050 labeled rows; file 10's exact CSV is preserved.

## Reproduce

From the repository root, with the existing benchmark dependencies and native
caches from the earlier runs available:

```sh
make -C part2 sprint
make -C part2 sprint-final
make -C part2 sprint-check
```

Native caches are ignored by Git. A fresh checkout must first reproduce the
all-submission benchmark and final neighbor bundle using their existing workflows.
The CSVs, held-out predictions, audits, protocols and reports are retained in Git.
