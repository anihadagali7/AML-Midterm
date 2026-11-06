# Model caches and source-checkout reproduction

For the new neighbor extensions, saved benchmark arrays and source audits are
versioned in `experiments/ani_extensions`; native models under its `cache/` are
regenerable and ignored. Rebuild the final bundle with
`.venv/bin/python part2/extension_followthrough.py final`, then run
`.venv/bin/python part2/extension_verify.py`. The final-fitting command rebuilds
the old base bundle too if needed. Replay the recommended candidate using
`.venv/bin/python part2/extension_predict.py --input part2/outcome_test.csv --output /tmp/replayed.csv`.
Full comparison commands and source are in [ani_extensions.ipynb](ani_extensions.ipynb)
and the [extension report](experiments/ani_extensions/RESULTS.md).

The repository includes the complete experiment code, executed notebooks, supplied data, frozen benchmark splits, held-out predictions, score tables, uncertainty estimates, figures, source snapshots, and submission CSVs. Regenerable `.joblib` model/feature caches are excluded from Git. They remain intact in the original working directory.

`experiments/model_cache_manifest.json` records the original local cache paths, sizes, and SHA-256 hashes. This distinguishes the cached models used for the recorded experiments from a later rebuild. Numerical differences across machines or library versions can affect those hashes; the pinned `requirements.txt`, source hashes, data hashes, and saved probabilities provide the comparison references.

## Read existing findings

Open `local_benchmark.ipynb`, `part2.ipynb`, or `experiments/local_benchmark/RESULTS.md`. Their default settings read the versioned results and do not require the omitted model caches. Model replay and inference commands require rebuilding the relevant caches first.

## Rebuild the strict benchmark models

From the repository root, with the pinned packages installed in `.venv`:

```sh
.venv/bin/python part2/local_benchmark.py --suite full --force
make -C part2 benchmark-test
make -C part2 benchmark-verify
```

`--force` recomputes every split's predictions, fitting models when their cache files are missing. Without it, a checkout containing the complete saved scores may reuse those results without rebuilding native models. The existing protocol rejects changed data; code/package changes create a different result fingerprint.

## Rebuild Ani's exact reference and final candidates

```sh
.venv/bin/python part2/reproduce_ani.py
.venv/bin/python part2/benchmark_final.py
```

The archived Ani notebooks and CSVs are included under `experiments/ani_comparison/source`. Exact reproduction regenerates its original transductive feature cache and training/train+validation models. Final candidates preserve the archived winning prediction file as their anchor and rebuild the full-data linear component if necessary. Complete saved benchmark comparisons must exist before candidate generation.

## Rebuild the original search and submission models

Open `part2.ipynb`, set `RUN_TRAINING=True`, and run all cells. This runs all original searches before rebuilding their fitted-model bundles, predictions and reports. For the later benchmark workflow, use `local_benchmark.ipynb` and its reproduction instructions.

Reported Kaggle scores are tied to submission file hashes in `submissions/leaderboard_results.json`. A rebuilt CSV with different probabilities does not inherit a known score merely because its filename is unchanged. Preserve the original versioned submission files when making new recipes.
