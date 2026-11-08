# Part 2 hand-in preparation

The working project stays in `part2/`. The finalized reading order is in
[`submission/midterm/part2/`](../../submission/midterm/part2/README.md).

From the repository root:

```sh
python part2/packaging/build_handin.py
python part2/packaging/verify_handin.py submission/dist/part2-midterm-code.zip --output submission/part2_package_verification.json
```

The builder selects from tracked and non-ignored Part 2 files, preserving the
selected files' bytes. It includes every supplied input dataset and Python source,
all trial settings and summary tables, figures and the row predictions used by the
finalized comparisons. It excludes other intermediate arrays, repeated per-trial
prediction files/audits, logs and report drafts. The full working history stays put.
It does not copy or change Part 1. The three notebooks embed all retained Python
sources, recompute the final comparison figures, and verify the scored CSV.

`train_final.py` exposes the exact final training path without dependency on old
fitted caches. `capture_recorded.py` was used once against the historical fitted
caches to record native component probabilities and cache hashes; the hand-in
includes those small arrays so quick replay does not require the original caches.
It is never needed for fresh training. `notebooks.py` defines the finalized cells.

The builder writes generated resources under `submission/midterm/part2/resources/`
and the portable archive under `submission/dist/`. Both are ignored to avoid
committing duplicate copies of the selected research evidence. Final notebooks,
small regenerated figures/tables, and verification receipts are versioned.
