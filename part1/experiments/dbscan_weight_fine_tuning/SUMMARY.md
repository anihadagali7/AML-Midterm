# Fine tuning DBSCAN pseudo-label weight caps

72 weight-cap/budget combinations plus a labeled-only control were evaluated on the same 78 validation trials. Positive confidence threshold 0.60, negative threshold 0.10, clustering, 150 embeddings, training-only scalers, classifier C=1000, prediction cutoff 0.5, and nonempty fallback remain fixed.

Class budgets: 35%, 40%, 45%, 50%, 55%, 60%, 65%, 70%. Per-label caps: 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35. The two caps vary independently. Known labels keep raw weight 1, generated positive/negative total weights preserve the known class ratio, and fitting weights are normalized to total 157 per reason.

Reference sample F1: **0.910256**, macro F1: **0.811421**.
Best configuration: **cap0p25_budget0p5**; sample F1 **0.910256**, macro F1 **0.811421**; sample F1 change **+0.000000**.
The 73 configurations produced **10 distinct target/weight tables**, including the control.

## Per-label cap changes at the original 50% class budget

| Per-label cap | Class-weight budget | Generated trials | Sample F1 | Macro F1 | Change |
|---:|---:|---:|---:|---:|---:|
| 0.15 | 50% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0.175 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.2 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.225 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.25 | 50% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.275 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |

## Class-budget changes at the original 0.25 per-label cap

| Per-label cap | Class-weight budget | Generated trials | Sample F1 | Macro F1 | Change |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 35% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 40% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 45% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 50% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 55% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 60% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 65% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 70% | 133 | 0.910256 | 0.811421 | +0.000000 |

## Active constraints

Constraint counts across accepted reasons/configurations: {'positive_example_cap': 72}.
A nominal cap only changes fitting when it is the smallest bound. For the current 8 generated Efficacy positives and 17 known positives, a per-example cap of 0.25 limits the added positive weight to 2, or about 11.76% of known positive weight. Thus a class budget of 35–70% is already above that bound. The negative weights are adjusted to preserve the same added fraction for the negative class.

## All combinations

| Per-label cap | Class-weight budget | Generated trials | Sample F1 | Macro F1 | Change |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 50% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 45% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 55% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 40% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 60% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 35% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 65% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.25 | 70% | 133 | 0.910256 | 0.811421 | +0.000000 |
| 0.225 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.275 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.225 | 45% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.2 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.225 | 55% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.275 | 45% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.275 | 55% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.225 | 40% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.225 | 60% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.2 | 45% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 45% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.175 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.275 | 40% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.275 | 60% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.2 | 55% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 55% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.2 | 40% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.2 | 60% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 50% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 40% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 60% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.15 | 50% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0.175 | 45% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.225 | 35% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.225 | 65% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 45% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.175 | 55% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.275 | 35% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.275 | 65% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 55% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.225 | 70% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 45% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.15 | 45% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0.175 | 40% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.175 | 60% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.275 | 70% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.2 | 35% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.2 | 65% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 40% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 60% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 55% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 35% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 65% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.15 | 55% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0.2 | 70% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 40% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 60% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.3 | 70% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.15 | 40% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0.15 | 60% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0.175 | 35% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.175 | 65% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 35% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 65% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.15 | 35% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0.15 | 65% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0.175 | 70% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.325 | 70% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 35% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 65% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.35 | 70% | 133 | 0.897436 | 0.803884 | -0.012821 |
| 0.15 | 70% | 133 | 0.897436 | 0.801106 | -0.012821 |
| 0 | 0% | 0 | 0.893162 | 0.804324 | -0.017094 |

## Selected generated labels

| Reason | Generated positives | Generated negatives | Positive raw weight total | Negative raw weight total |
|---|---:|---:|---:|---:|
| Safety | 0 | 0 | 0.000000 | 0.000000 |
| Funding | 0 | 0 | 0.000000 | 0.000000 |
| Efficacy | 8 | 125 | 2.000000 | 16.470588 |
| Recruitment | 0 | 0 | 0.000000 | 0.000000 |
| Other | 0 | 0 | 0.000000 | 0.000000 |

## Verification

The original configuration reproduces the previous targets, weights, and probabilities. The zero-weight control reproduces the labeled-only baseline. Known targets remain unchanged, DBSCAN noise stays unlabeled, and validation never enters clustering or fitting. The selected model was reloaded and independently refitted from its saved dataset with matching validation probabilities; protected earlier artifacts retain their checksums.

Every dataset, label audit, classifier, validation prediction, configuration, and checksum is saved locally. Selection uses sample-wise F1; ties favor the original settings. Validation tuning is exploratory and does not establish independent test improvement. No Kaggle submission was used; no commit or tag was made.

The primary validation metric did not improve; retain the existing DBSCAN candidate.

## Sensitivity of the nearest settings

The nearest per-label caps, **0.225** and **0.275**, each change the predicted reason set for only **one of the 78 validation trials** relative to 0.25. In both comparisons that trial uses the nonempty fallback: small changes in the Efficacy probability change which reason has the highest probability when all probabilities are below 0.5. The current cap wins this comparison, but this one-trial difference is limited evidence for choosing a precise weight and does not establish that 0.25 will generalize better.
