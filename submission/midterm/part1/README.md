# Part 1 notebook hand-in

**Winner: anchored80 / strict-inductive five-seed Cand2_20, reported Kaggle sample-wise F1 .89935.** [Exact scored CSV](part1_best_submission.csv).

This is the Part 1 code/data hand-in for the current report. It covers the winning model, representative baseline and feature experiments, the benchmark and blend decisions, and several distinct unsuccessful directions. It does not create a notebook for every tuning run.

| Notebook | What it does |
|---|---|
| [01_final_model.ipynb](01_final_model.ipynb) | Explains both winning components, checks data boundaries, recomputes per-label validation metrics, reconstructs the exact CSV, and freshly fits both components and all five Cand2 seeds |
| [02_baselines_and_features.ipynb](02_baselines_and_features.ipynb) | Fresh historical baseline and feature validation; separate retained K-means/DBSCAN, pathways, drug and other feature screens |
| [03_benchmarks_and_blends.ipynb](03_benchmarks_and_blends.ipynb) | Recomputes all 84 held-out cases for the winner/controls, paired repeat results, report comparisons and the J ordering audit |
| [04_unsuccessful_directions.ipynb](04_unsuccessful_directions.ipynb) | Rescores matched Other, partial-label and complementary-head predictions and PySR sufficient statistics; explains the negative submission outcomes |
| [05_source_code.ipynb](05_source_code.ipynb) | Complete visible packaged Python source and original notebook-only study code, with exact hash checks |

## Run locally or in Colab

Use Python 3.12. Extract the complete ZIP, open a terminal in `midterm/part1`, and install `python -m pip install -r requirements.txt`. Open the numbered notebooks and run them in order, or individually. In Colab, upload/extract the full ZIP, change into `midterm/part1` and install the same requirements before running. Uploading only a notebook omits its required data and evidence.

Notebook 01 runs fresh validation and final235 training by default. Set `RUN_FRESH_TRAINING=False` for a quicker retained-evidence review. Fresh scores are checked against the frozen benchmark only after fitting. No stored scores enter fresh fitting. New outputs go under `generated/`; the scored CSV and frozen resources are not overwritten.

Notebook 02 deliberately preserves the **historical transductive preprocessing/pseudo-label pool** of the earlier feature experiment. It is not the final anchored/Cand2 model. Its displayed validation results must not be compared as if they used the current grouped protocol. Notebooks 03 and 04 clearly identify their use of retained predictions/statistics. They do not claim to refit all past experiments.

`resources/` includes supplied Part 1 inputs, the exact model sources and configurations, selected predictions, compact results, and a SHA-256 manifest. No external trial data or hidden labels are fetched. `results/` contains tables, figures, fresh-fit receipts and execution checks. Figure-generation code is in the notebooks. The frozen scored artifact remains on its original immutable tag, `part1-anchored80-cand2-inductive20-trainval235-kaggle-0.89935`.

The four analysis notebooks run without Git, an old worktree, or fitted-model caches. Notebook 05 exports source for review without running every original search. Historical runners and archived notebooks may require regeneration of omitted intermediates or the full research checkout; only the finalized notebooks and explicit fresh winner path are certified portable. PySR source, equations, fixed search settings and its separate Python/Julia locks are included; default execution does not install Julia or rerun 420 symbolic searches.

## Build from the working repository

Materialized resources and ZIPs are generated rather than duplicated in Git. After cloning the repository, install the scientific requirements plus iterative-stratification and RDKit, then run from its root:

```sh
python -m pip install -r part1/reporting/requirements.txt rdkit==2026.9.1
python part1/packaging/build_handin.py
```

The builder copies verified local inputs/evidence, writes and executes the five notebooks, and creates `submission/dist/part1-midterm-code.zip` with a `midterm/part1/` root. Use `--skip-execution` for drafting or `--zip-only` to repackage existing executed notebooks. Run `python part1/packaging/verify_handin.py` to test the ZIP in an isolated directory with no research checkout.

This package leaves the separate Part 2 hand-in and combined report unchanged. The older `midterm/part1_best_model.ipynb` remains a historical artifact and is not the final model entry point.
