# Safety-only adjustment for later years

Use the frozen K=2 classifier without year as an input feature. Reuse the same 78 validation trials, threshold 0.5, and nonempty-label fallback. No classifier or cluster model is refitted and no labels or weights change.

| Maximum Safety probability increase | Sample-wise F1 | Macro F1 | Safety F1 | Safety TP | Safety FP | Safety FN | Changed trials |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.8568 | 0.6808 | 0.8571 | 9 | 3 | 0 | 0 |
| 0.05 | 0.8440 | 0.6547 | 0.8182 | 9 | 4 | 0 | 1 |
| 0.10 | 0.8440 | 0.6547 | 0.8182 | 9 | 4 | 0 | 1 |

## Method

`late_factor = clip((start_year - training_mean_year) / training_std_year, 0, 1)`

`adjusted_Safety = min(1, original_Safety + maximum_boost * late_factor)`

Reference statistics come from the existing 545-row K=2 training dataset: mean year **2005.0147**, standard deviation **3.2768**. Full boost starts at year **2008.2915**. The fixed grid is 0, 0.05, and 0.10: the latter two are capped increases of 5 and 10 percentage points in Safety probability. These are heuristic adjustments, not calibrated probabilities. Other reason probabilities are left exactly unchanged. The existing fallback can change an otherwise empty trial's chosen reason when Safety becomes more probable.

## Results

Selection by validation sample-wise F1 chooses maximum boost **0.00**, with F1 **0.8568**. Ties favor no adjustment, then the smaller boost.

The original K=2 model already identifies all 9 Safety-positive validation trials, with 3 false positives. There are no missed Safety cases for this adjustment to recover on validation. Added positive Safety predictions therefore cannot improve Safety recall in this set. This limits what this small validation sample can tell us about a later-year prior adjustment; it does not establish performance on the later test distribution.

Inspect `changed_trials.csv` for exactly which predictions improved or worsened. A larger Safety probability by itself is not a success; the primary criterion is overall sample-wise F1.

## Changed predictions

| Adjustment | Trial | Year | Actual | Original | Adjusted | Safety probability before | Safety probability after |
|---|---|---:|---|---|---|---:|---:|
| boost_005 | TR08868 | 2008 | Other | Other | Safety | 0.000087 | 0.045639 |
| boost_010 | TR08868 | 2008 | Other | Other | Safety | 0.000087 | 0.091192 |

In this run, both boosts change the same trial from a correct Other prediction to an incorrect Safety prediction. The adjusted Safety probabilities remain below 0.5; the change comes through the existing highest-probability fallback. This illustrates that probability adjustments can affect fallback choices even without crossing the classification threshold.

## Artifacts and controls

`comparison.csv` records overall and per-reason F1, Safety precision/recall, error counts, fallback use, and prediction changes. `validation_predictions/` records raw and adjusted probabilities, year factors, and labels for all three settings. `changed_trials.csv` records each changed prediction and its trial-wise F1 effect. `settings.json` records the exact adjustment and checksums of the reused training data and model.

Checks confirm the zero-boost control reproduces K=2's recorded validation scores, other reason probabilities are unchanged, early trials receive no boost, increases stay within their caps, and frozen model/data/submission files remain unchanged. No new training dataset is created because this experiment only changes postprocessing.

The unadjusted K=2 Kaggle score is **0.87393**, reported after manual submission. A score of 0.86111 was reported for a Safety adjustment, provisionally assigned to the recommended +0.05 file; +0.10 remains unknown. The earlier full year-feature classifier scored 0.85149; that was a separate intervention.

## Safety-adjusted Kaggle exports

The following files were prepared at Michael's request to compare with unadjusted K=2 at **0.87393**. Both nonzero variants scored 0.8440 on validation; no adjustment remains the validation-selected rule.

| Maximum boost | File | Changed test trials | Safety predictions | Kaggle status |
|---|---|---:|---:|---|
| 0.05 | `submissions/archive/submitted/kmeans_k2_safety_boost005__reported_0.86111.csv` | 7 | 64 | 0.86111 (reported; assumed +0.05) |
| 0.10 | `submissions/archive/unsubmitted/kmeans_k2_safety_boost010__157labels.csv` | 10 | 67 | pending manual submission |

The two files contain identical predictions: **False**. If identical, upload only one. Otherwise, start with the smaller adjustment when checking only one variant. `submissions.csv` and `submission_manifest.json` record file hashes and settings.

## Reported Safety-adjustment Kaggle result

Michael reported **0.86111**. This is provisionally assigned to the recommended **+0.05** submission; the boost strength was not explicitly stated. The +0.10 result remains unknown.

| Rule | Validation sample-wise F1 | Kaggle score |
|---|---:|---:|
| Unadjusted K=2 | **0.8568** | **0.87393** |
| Tested Safety adjustment (assumed +0.05) | 0.8440 | 0.86111 |

The observed Kaggle difference is **-0.01282**, approximately **-1.28 F1 points**. The validation difference is approximately **-0.01282** as well. Validation and Kaggle agree in direction and almost exactly in size for this particular adjustment. Retain unadjusted K=2 as the best observed Kaggle model.

This agreement does not establish a reliable general correlation: K=2 versus the supervised baseline and the addition of year previously produced opposite validation/Kaggle directions. Use validation as a useful but imperfect guide, with occasional Kaggle checks as agreed. This result argues against this tested Safety boost, not every possible temporal method.
