Question
Could adding the 78 validation-source labels before submission materially change
the frozen three-way model, or explain why its local advantage did not beat Ani?

Observed sensitivity
The exact 157-label and 235-label fits disagree on the complete predicted label
set for 29 of 312 test rows (9.3%), with 38 individual binary label changes.
Test labels remain unknown. This is not an accuracy comparison, and no new
submission CSV was exported.

Paired local experiment
Keep the same eight saved pooled CV repeats and five grouped folds per repeat.
For every fold, compare two fits on the same held-out rows:
  reduced: eligible gold labels from the original training file only;
  expanded: all eligible gold labels, including the validation source.
Use the same original masked rows after held-out-group exclusion. All model
settings, seeds, thresholds and ensemble weights remain unchanged. Reuse the
previously saved expanded-arm predictions with source, truth and ID checks.

Results (sample-wise F1, eight-repeat mean)
Three-way, all235 held-out rows: 0.913032 -> 0.916489, delta +0.003457,
  expanded arm wins5/8 repeats.
Three-way, original-training-source held-out rows: 0.930467 -> 0.926752.
Three-way, validation-source held-out rows: 0.877938 -> 0.895833.
Ani inductive control, all235 held-out rows: 0.902748 -> 0.909840,
  expanded arm wins5/8 with one tie.

Interpretation
Adding labels changes predictions materially; its effect need not be tiny.
The local paired check shows an average benefit, with mixed effects by source.
It does not establish whether the actual235-label fit helps or hurts on Kaggle.
Within folds the comparison is approximately125 versus188 fitting labels,
an analogue of157 versus235. Sample count and added-label composition change
together. All development labels and these folds have been reused before.

Another comparison limitation
The local Ani control excludes test inputs from fitting, whereas the original
Ani Kaggle submission used transductive fitting. The ensemble's local advantage
therefore does not compare it with the exact original submitted Ani pipeline.
The reported Kaggle scores0.89401 and0.89423 are numerically very close.

Verification
Reduced fits assert fitting IDs are from the original training file only.
Changing every validation-source label leaves representative reduced-fit
predictions unchanged. Saved predictions were rescored and protected source
data, labels, recipes and submissions stayed unchanged.

Reproduce from /Users/matthew/Desktop/AMLMidterm
.venv/bin/python diagnose_refit.py prepare
.venv/bin/python diagnose_refit.py run
.venv/bin/python diagnose_refit.py sensitivity
.venv/bin/python diagnose_refit.py report

Files
protocol.json: frozen comparison, exact40 folds, recipes, source/data hashes.
cv_results.json: every repeat score, fitting count and reference provenance.
model0.npz and model1.npz: paired out-of-fold scores, truth and IDs.
test_sensitivity.json: prediction changes with no accuracy claim.
boundary_audit.json: held-out/unused-label access checks.
summary.json, comparison.csv, report.html: verified results.
