# Absolute-radius attribution controls

Two adaptive controls reuse the exact absolute DBSCAN radii of the disease/criteria scale-0.25, median-neighbor-radius candidates, but cluster on the original embeddings only. This separates a radius effect from an added-feature effect. Classifier, label rules, weights, cutoff, and fallback stay fixed.

                                        experiment                 radius_source       eps  sample_f1  macro_f1  different_trial_predictions_vs_expanded  generated_trials  clusters  noise_trials
embeddings_at_dbscan_diseases_scale0p25_q50_radius dbscan_diseases_scale0p25_q50 10.095424   0.893162  0.804324                                        4                 0         7           456
embeddings_at_dbscan_criteria_scale0p25_q50_radius dbscan_criteria_scale0p25_q50 10.137843   0.893162  0.804324                                        4                 0         8           450

No validation/test fitting, final validation-label refit, Kaggle submission, commit, or tag. All saved classifier reloads and protected-input checks passed.
