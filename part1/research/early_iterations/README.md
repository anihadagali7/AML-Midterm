# Earlier iterations and comparison corrections

These records complete the Part 1 research branch with work that previously existed only in the shared working folder. The simple and trusted prototypes are historical, superseded candidates. The current submitted leader remains the [anchored classifier](../../README.md), with user-reported Kaggle **0.89636**.

All 275 previously uncommitted working files were accounted for: 174 source/settings/summary records were archived, 22 files of useful leader/control evidence were retained at their original paths, and 79 generated nonwinning outputs were excluded. Local working copies were not deleted by this import. Additional source dependencies and protocol context are included where needed.

## Research history

| Work | Finding and records |
|---|---|
| Simple methodical models | Removing PCA/GMM and using three self-training passes reached .913191 on the historical Pooled235 benchmark versus the older Ani .908865 control. Later-year diagnostics did not confirm the gain. [Notebook](snapshot/part1/simple_methodical.ipynb), [settings](snapshot/part1/experiments/simple_methodical/selected.json), [all results](snapshot/part1/experiments/simple_methodical/all_results.csv). No scored submission is recorded. |
| Trusted-label prototype | Fixed gold-derived class weights, reduced pseudo-label weights, binary neighbor votes, and two passes reached .915337 against the same older control. Original validation remained weaker. [Notebook](snapshot/part1/trusted_iteration.ipynb), [settings](snapshot/part1/experiments/trusted_iteration/selected.json), [all results](snapshot/part1/experiments/trusted_iteration/all_results.csv). No scored submission is recorded. |
| Corrected Ani comparison | The earlier claims had used only Pooled235 and the older no-LDA Ani recipe. The corrected comparison used both historical pools and current Ani LDA; the trusted prototype did not establish a clear improvement across both. [Executed report](../../current_ani_pool_comparison.ipynb), [conclusion](../../experiments/current_ani_pool_audit/review.json), [scores](../../experiments/current_ani_pool_audit/summary.csv), [source](snapshot/part1/compare_ani_pools.py). |
| Anchored development | Earlier stage sources, recipes and summaries are retained here. The selected ablation stage keeps its detailed original evidence at [06_final_ablations](../../experiments/paired_improvement/06_final_ablations/), alongside the existing final candidate and common grouped comparison. [Compact earlier stages](snapshot/part1/experiments/paired_improvement/). |

These historical CV scores use the old saved ungrouped protocols. Compare current models using [TESTING_POLICY.md](../../TESTING_POLICY.md); do not rank the numbers in this history against the newer grouped benchmark.

## Evidence and reproduction

[Pre-import checks](pre_import_checks.json) independently rescored **2,246 recipe/repeat results** and **193 prototype summary rows** against the saved predictions and supplied gold labels. No fitting or tuning occurred. The [manifest](manifest.json) identifies each retained file, its original path, hash, and role. Omitted outputs are recorded as aggregate counts, without another backup archive.

```sh
python part1/research/early_iterations/verify.py
```

The command checks the retained hashes and import coverage. Detailed current-model verification remains `python part1/current_leaders_local.py verify`.

`snapshot/` mirrors original paths for reading frozen source, settings, and results. Original inputs and saved folds remain in the main Part 1 tree. This is an archive, not a standalone training directory: old scripts and notebooks may require regenerated upstream outputs and their recorded environment. Use a disposable working directory for any replay. Historical preparation statuses, recommendations, and artifact hashes describe their original checkpoints; they do not supersede the current model or submission ledger.
