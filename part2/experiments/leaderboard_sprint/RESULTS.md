# Part Two submission sprint

**Best scored submission: `13_file10_legacy30.csv`, 0.58465.** Scores are user-reported and tied to exact CSV hashes in the [ledger](../../submissions/leaderboard_results.json). File 11 reached reported first place at 0.58474; file 13 subsequently improved to 0.58465.

3 of the original four additional uploads have reported results; 1 remain unconfirmed. The [live queue](../../submissions/sprint_queue.json) records the latest request. The user revised the active deadline to 2026-11-07 00:06:49 UTC.

## Direct comparison

All entries use the same ten original splits plus three longer-gap checks, equal-phase log loss, and pooled repeated grouped CV. Lower is better. These are the primary strict-preprocessing results; both preprocessing tracks and every period remain in the linked reports.

| File | Forward mean | Longer-gap mean | Worst gap | Grouped CV | Kaggle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.535890 | 0.563189 | 0.628816 | 0.586891 | 0.58609 |
| 11 | 0.533926 | 0.561563 | 0.629638 | 0.581169 | 0.58474 |
| 12 | 0.535544 | 0.562491 | 0.627765 | 0.586105 | Unknown |
| 13 | 0.533405 | 0.561247 | 0.630614 | 0.578966 | 0.58465 |
| 14 | 0.532758 | 0.561623 | 0.631722 | 0.580729 | Unknown |
| 15 | 0.532700 | 0.560879 | 0.630974 | 0.578254 | 0.58492 |
| 16 | 0.533037 | 0.561154 | 0.630814 | 0.579112 | Unknown |

## What the feedback changed

File 11's 20% original-ensemble contribution improved file 10 by 0.00135 on Kaggle. Increasing it to 30% in file 13 gained another 0.00009, showing a much smaller incremental benefit. File 15's recent-tree/phase-trend mixture improved local averages but scored 0.58492, losing 0.00027 to file 13. That result argues against increasing those additions solely because historical validation favors them.

**Hold the final remaining submission.** The user confirmed first place with nine total uploads. The prepared file 16 retains 30% original ensemble, replaces 20% of file 10 with sharper-neighbor file 09, and leaves 50% file 10. Both chronological averages improve versus file 13, but the gains are small, intervals span zero, and grouped CV worsens slightly. Its initial exploratory upload recommendation was withdrawn after reviewing the value of the final slot. File 16 remains an unsubmitted reserve; file 13's exact CSV remains intact.

## Experiments retained

- [Initial 32-recipe batch](runs/4dc0f274d45bce9e/RESULTS.md): 1,014 evaluations including controls.
- [Four recent-tree mixtures](recent_tree/dea847f57c7c07ee/RESULTS.md): 182 evaluations including controls.
- [19 refinements around files 11 and 13](refine/2e5c792462013673/RESULTS.md): 598 evaluations including controls and the final submission choice.
- [Nine final-slot alternatives](finish/b235582538f63f9c/RESULTS.md): 312 evaluations including controls; no candidate from this batch was submitted.

All local labels were repeatedly consulted. Paired bootstrap intervals share trial weights across phases, but do not correct for adaptive model selection. Later Kaggle results are separate observations; they do not retroactively make local comparisons independent.

## Verification and reproduction

Exports preserve sample order, 1,500 unique IDs, valid probabilities and exact native inference after evaluation labels/IDs are replaced. All supervised final components use 9,050 labeled rows. Native caches are ignored by Git; source hashes, held-out predictions, component hashes and export receipts are retained. Frozen export receipts keep their pre-submission score state; the live ledger records later scores.

```sh
make -C part2 sprint
make -C part2 sprint-final
make -C part2 sprint-check
make -C part2 sprint-refine
make -C part2 sprint-refine-check
```

A fresh checkout must first reproduce the earlier all-submission and final-neighbor native caches. None of these commands uploads to Kaggle.
