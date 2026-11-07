# AML-Midterm

## Part 1: stopping reasons

The **anchored reason classifier** scored **0.89636 sample-wise F1** on Kaggle (user-reported; higher is better), using a final fit on all 235 verified labels. Its exact submission and original Ani reference are preserved with score tags and hashes.

- [Current executed notebook](part1/part1.ipynb) and [Part 1 overview](part1/README.md)
- [Testing methodology](part1/TESTING_POLICY.md) and [research index](part1/RESEARCH_INDEX.html)
- [Scored submissions](part1/submissions/README.md) and [retention policy](part1/RETENTION.md)
- [Historical notebook archive](part1/research/historical_notebooks/README.md)

## Part 2: current best submission

**[File 13: original-ensemble blend](part2/submissions/13_file10_legacy30.csv)**
scored **0.58465 log loss** on Kaggle (user-reported first place; lower is better).
Its score tag is `part2-submission-13-kaggle-0.58465`.

The model blends 70% of the recent-neighbor file 10 with 30% of the original
ensemble, with all supervised components fitted on 9,050 labeled rows. The
[Part 2 overview](part2/README.md) links the fixed-split evaluation, full research
history, reproduction commands and exact submission hashes.

- [Trial results and submission decisions](part2/experiments/leaderboard_sprint/RESULTS.md)
- [Trial-phase commits and tagged submissions](part2/submissions/HISTORY.md)
- [Score ledger](part2/submissions/leaderboard_results.json)
- [Main-branch integration record](part2/docs/MAIN_INTEGRATION.md)

File 15 scored 0.58492 and did not improve the winner. File 16 remains unsubmitted;
the final available submission slot was held because the evidence was weak.
