# K-means augmentation versus the supervised baseline

All scores below use the same 78 validation trials and the same 0.5 threshold with a nonempty-label fallback.

| Model | Generated trials | Training trials | Sample-wise F1 | Macro F1 | Change from baseline |
|---|---:|---:|---:|---:|---:|
| baseline_C1000 | 0 | 157 | 0.8932 | 0.8043 | +0.0000 |
| k02 | 388 | 545 | 0.8568 | 0.6808 | -0.0363 |
| k05 | 565 | 722 | 0.8547 | 0.7024 | -0.0385 |
| k08 | 591 | 748 | 0.7991 | 0.5513 | -0.0940 |
| k10 | 706 | 863 | 0.8376 | 0.5316 | -0.0556 |
| k15 | 662 | 819 | 0.8376 | 0.6415 | -0.0556 |

## Generated positive labels

| K | Safety | Funding | Efficacy | Recruitment | Other |
|---|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 0 | 388 | 0 |
| 5 | 0 | 0 | 0 | 565 | 0 |
| 8 | 0 | 0 | 0 | 591 | 0 |
| 10 | 0 | 0 | 128 | 578 | 0 |
| 15 | 0 | 0 | 31 | 631 | 0 |

The generated labels are heavily skewed toward Recruitment. None of the five runs creates a positive Safety, Funding, or Other label. These generated trials therefore add negative targets for the rarer reasons. The reduced macro F1 and several reduced per-reason F1 scores are consistent with this imbalance; because the true hidden training labels are unavailable, the audit does not establish which generated labels are incorrect.

## Per-reason validation F1

| Model | Safety | Funding | Efficacy | Recruitment | Other |
|---|---:|---:|---:|---:|---:|
| baseline_C1000 | 0.9412 | 0.6667 | 0.8125 | 0.9697 | 0.6316 |
| k02 | 0.8571 | 0.4000 | 0.7500 | 0.9524 | 0.4444 |
| k05 | 0.9412 | 0.4000 | 0.7826 | 0.9174 | 0.4706 |
| k08 | 0.6667 | 0.0000 | 0.7200 | 0.8991 | 0.4706 |
| k10 | 0.6154 | 0.0000 | 0.7647 | 0.9703 | 0.3077 |
| k15 | 0.6667 | 0.4000 | 0.8148 | 0.9259 | 0.4000 |

## Interpretation

The supervised baseline has the highest validation sample-wise and macro F1 among all six configurations. K=2 has a slightly higher reported Kaggle score, so retain both models as references. K=2 is the best K-means setting by the validation primary metric; K=5 has the best macro F1 among the K-means settings, but both remain below the supervised baseline. This result applies to the cluster-frequency labeling rule and fixed confidence/weight settings tested here; it does not show that every semi-supervised method will perform worse.

Best K-means configuration: **K=2**, sample-wise F1 **0.8568**, macro F1 **0.6808**.
Its sample-wise F1 difference from baseline is **-0.0363**.

The supervised Kaggle reference is **0.87072**, reported after Michael's manual submission. Michael reported **0.87393** for K=2 after manual submission, compared with **0.87072** for the supervised baseline. K=2 is lower on validation but higher on Kaggle by **0.00321** (0.321 F1 points). The ranking reversed; the Kaggle difference is small and does not establish a reliable advantage on future trials.

Method: cluster all 1,171 training embeddings after training-only standardization, using 150 dimensions without additional PCA. Estimate cluster reason prevalences from the 157 known labels, smoothed toward their global prevalences with strength 2. Accept complete generated label vectors only with at least 3 labeled cluster members, minimum assigned-label confidence 0.7, and at least one positive reason. Known labels receive weight 1 and generated labels weight 0.25. Fit C=1000 one-vs-rest logistic regression with a fresh scaler on each included training dataset. Only K varies across the five augmentation experiments.

These confidence values are heuristic cluster frequencies. Validation selects among five settings, and the 78-trial validation set has already informed earlier supervised tuning. This comparison does not establish performance on an independent test set.

Artifacts: `datasets/` contains the exact training rows and weights, `label_audits/` records all original training trials including excluded unknown labels, `clusters/` records support and reason frequencies, `models/` stores fitted pipelines and cluster models, and `validation_predictions/` stores probabilities, labels, and fallback use. `comparison.csv` includes per-reason F1, and `settings.json` records settings and checksums.

Verification: refactoring reproduced baseline probabilities and predictions; original known labels were preserved; validation IDs were excluded from augmented training; the saved winning K-means dataset reproduced its predictions; and an independently fitted binary classifier confirmed sample weights were routed correctly.

## K=2 Kaggle submission

Prepared `submissions/archive/submitted/kmeans_k2__kaggle_0.87393.csv` with 312 predictions from the saved K=2 classifier, trained on 157 known-label and 388 generated-label trials. Test fallback cases: 25. The prediction rule matches the supervised baseline. **Kaggle score: 0.87393 (reported by Michael after manual submission).** Baseline Kaggle reference: **0.87072**. `k02_submission.json` records the model and submission checksums.
