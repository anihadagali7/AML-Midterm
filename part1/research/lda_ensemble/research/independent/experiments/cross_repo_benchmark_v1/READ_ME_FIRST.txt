COMMON PART 1 COMPARISON: SIX FROZEN RECIPES

Open confirmation8/report.html for the final eight-repeat comparison.
Open report.html for the original three-repeat comparison and chronological
windows/per-reason precision, recall and F1. Both result sets are retained.

Evaluation:
- Six recipes, identical grouped five-fold assignments within each label pool.
- Three existing frozen repeats, then five additional prespecified repeats for
  every recipe. No model settings or thresholds were tuned in this comparison.
- Eight repeats x five folds x two pools x six recipes = 480 CV pipeline fits.
  90 existing local fits were reused under matching source/configuration/fold
  cache keys and independently rescored; 390 additional CV fits were run.
- Six fresh original-validation fits. Our three local recipes reproduce their
  previous validation F1 exactly. Ani here uses training-only feature access,
  so its validation result is not the original transductive submission result.
- Eighteen chronological fits: six recipes on each of three fixed windows.
  Pool 98 distinct held-out trials for the chronological summary; retain every
  window separately. Do not treat these as independent repeat measurements.
- Three imported adapter parity checks reproduce saved source probabilities
  within 1.5e-14 and all predicted labels exactly.
- A representative pooled-fold audit for all six recipes changes held-out
  labels/features and 72 matching masked duplicates. No predictions change.
  Predicting one query at a time also agrees with batch predictions.

Frozen source branch: 74253451908166368ec6e83339b478db3cca95eb
Source notebook/artifact snapshots and hashes: sources/, source_manifest.json
Recipe configurations: registry.json, ../../cross_repo_models.py
Initial protocol and chronological row assignments: protocol.json
Extra repeat protocols: confirmation8/train/ and confirmation8/pooled/
All CV scores and OOF predictions: train_cv/, pooled_cv/, confirmation8/
Validation and chronological predictions: validation/, chronology/
Verification: verification.json, confirmation8/verification.json,
              fitting_boundary_audit.json, parity.json

Reproduction, from the AMLMidterm workspace root with its pinned environment:
  .venv/bin/python run_cross_repo_benchmark.py prepare
  .venv/bin/python run_cross_repo_benchmark.py parity
  .venv/bin/python run_cross_repo_benchmark.py train
  .venv/bin/python run_cross_repo_benchmark.py pooled
  .venv/bin/python run_cross_repo_benchmark.py validation
  .venv/bin/python run_cross_repo_benchmark.py chronology
  .venv/bin/python run_cross_repo_benchmark.py report
  .venv/bin/python run_cross_repo_benchmark.py verify
  .venv/bin/python confirm_cross_repo_cv.py train
  .venv/bin/python confirm_cross_repo_cv.py pooled
  .venv/bin/python confirm_cross_repo_cv.py report
  .venv/bin/python audit_cross_repo_boundaries.py

The extension's prepare command was run once before extension scores. Reuse its
saved protocols; it intentionally refuses to overwrite existing assignments.
Environment versions and frozen code hashes are recorded in protocol.json.

Interpretation:
Ani's training-only adaptation leads both eight-repeat CV averages. Classical
tri-training leads chronological F1. The saved hybrid still has the best
original validation score. Earlier tuning used these labels, so these are
development comparisons, not independent future-test estimates. Start year
is a temporal stress-test proxy, not the time a stopping label became known.
Gold-label sampling enriched rare reasons; chronological label proportions
cannot establish actual Kaggle test proportions. The Funding label has zero
positives in the 2007 window, so its window-specific recall is not estimable
(stored zero_division=0 scores are accompanied by support counts).

No final all-235-label submission fit, test-input fitting, new submission,
repository merge, or promotion of the saved current model was performed.
The provided data, original benchmark protocols/registries/comparison tables,
current-best configuration, and current submission retain their hashes.


POST-BENCHMARK SUBMISSION RESULT
After the benchmark, the user authorized a final 235-label classical
tri-training fit. Its exact CSV was submitted manually and the user reported
Kaggle F1 0.88012. Ani remains the best reported submission at 0.89423.
Current score record: ../../predictions/tri_training_trainval235/kaggle_result.json
The submission preparation manifest is retained as a historical snapshot.
CSV SHA256: e3155e729f7be6bebe67d04bde524edd85be7c0514474fcb3cd5c50ba0241145
No score for a 157-label tri-training submission is known. This result does not
isolate the effect of adding validation labels. Three submissions remain from
the four the user reported before this upload. Benchmark measurements and
recipes remain unchanged; the new external score is recorded separately.
