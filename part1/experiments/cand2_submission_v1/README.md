# Cand2: exploratory submission

**Ready CSV:** [cand2_cs_features_inductive5.csv](../../submissions/candidates/cand2_cs_features_inductive5.csv). Prepared for the user's request for one distinct, promising direction to try immediately. This is the previously benchmarked **strict-inductive five-seed** Cand2 adaptation, fitted on all 235 verified labels and the 1,014 originally masked training rows. It is not Dylan's original transductive CSV. No upload or Kaggle outcome is recorded.

The model uses Correct & Smooth graph pseudo-labels, soft K-means8 features, group-safe 20-neighbor label features, a balanced label-combination LDA classifier, inner temperature calibration, and expected sample-F1 decoding. It averages combination probabilities from inner seeds 0–4 before decoding. All settings are unchanged from the prior evaluated adaptation; no new tuning or blending was done.

It changes **49 of 312** test predictions relative to anchored. This provides a materially different prediction set for an exploratory slot. Cand2 was the strongest standalone Dylan candidate on both grouped CV pools, but it remains worse than anchored on every current-policy check:

| Prior five-seed evaluation | Anchored | Cand2 |
|---|---:|---:|
| Train157 grouped CV, eight repeats | .932059 | .923301 |
| Pooled235 grouped CV, eight repeats | .921365 | .903599 |
| Original validation78 | .891026 | .871795 |
| Chronology98 | .884354 | .863946 |

These are prior measurements from audit commit `2bf7e4f`, not fresh full-CV runs during export. [Full prior comparison](prior_comparison.csv). The candidate failed the existing advancement rule; the user requested an exploratory submission direction after seeing those results. Preparing it does not promote it above anchored. Dylan's J scores from the native transductive procedure are not scores for this exported inductive adaptation. The separately tested C&S modifications of anchored also failed their screen; this export tests the complete Cand2 recipe rather than those component replacements.

The exporter freshly replayed the complete five-seed original-validation fit and matched the prior Cand2 probabilities and labels **exactly**, then performed a separate final235 fit. Final fitting uses training/validation inputs only. Test inputs are transformed and predicted without fitting preprocessing, pseudo-label generation, calibration or graph structure on them. Nested group checks passed; gold labels were preserved. The output has 312 unique IDs in sample-submission order and valid nonempty label sets.

Exact CSV SHA-256: `521f2825cfc20711f435edc633038b1ca59834e4804f06fc358fce05337ddde7`.

[Prepared-artifact record](prepared_submission.json), [frozen export protocol](protocol.json), [unchanged source provenance](provenance.json), [prediction differences](changed_predictions.csv), and final probability arrays are retained. This new worktree/branch starts from main `256fa8d`; source and benchmark evidence were copied with hashes from the audit branch. The original five-seed benchmark has already checked the adapter across all saved grouped and supplementary cases.

Use Python 3.12.13 with [requirements-lock.txt](requirements-lock.txt). From the repository root, `python part1/experiments/cand2_submission_v1/check_export.py` checks the saved export without fitting. `build.py` contains the full replay and export procedure and refuses to overwrite an existing CSV. For a fresh reproduction, use a disposable checkout, remove only this candidate's exported CSV there, and run that script. Preserve this exact file for any later reported score; record the result in a separate commit/tag after the user reports it.
