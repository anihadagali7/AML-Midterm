# DBSCAN final-refit Kaggle review

Michael reported sample F1 **0.86452** for the final 235-label DBSCAN CSV. Its SHA-256 is unchanged from preparation: `19f29c7bcbda974f8ae9646d3ced83bad2953ead45412e88731e8c12e55d2661`.

| Submission | Verified labels | Reported Kaggle F1 |
|---|---:|---:|
| Supervised logistic baseline | 157 | 0.87072 |
| K-means k=2 | 157 | 0.87393 |
| K-means k=2 + year | 157 | 0.85149 |
| K-means k=2 + Safety year boost | 157 | 0.86111 (boost variant assumed in prior record) |
| DBSCAN final refit | 235 | 0.86452 |

The final DBSCAN score is lower than the baseline and K-means, but not the lowest previous score. It is 0.00620 below the supervised baseline and 0.00941 below K-means.

## What the result establishes

The submitted final fitted model did not carry over the selected model's 0.910256 validation score. That score was measured on 78 trials for a model fitted with 157 verified labels. Hundreds of settings were compared against those same 78 trials, so selection can favor settings that happen to fit that small evaluation set. The final model was fitted using all 235 verified labels and has no independent local validation score.

The assignment states that visible training labels are mostly early plus rare-reason enrichment, while test trials are latest and reason frequencies differ. This limits how well random training folds or the supplied validation set can predict Kaggle rankings. The aggregate score does not isolate the cause of this particular result.

## Refit differences, verified from saved artifacts

- Classifier scaling changed from fitting on 157 to 235 known trials; cluster scaling/clustering changed from 1,171 to 1,249 rows.
- The same q25 DBSCAN radius rule yielded eps 0.00012142366141394104 before refit and 0.0001459541408476203 afterward.
- Generated Efficacy labels changed from 8 positives + 125 negatives to 8 positives + 165 negatives. All other reasons still use verified labels only. The effective influence of generated labels also changed under the same weight-budget rule.
- Classifier weights were relearned with C=1000, LBFGS, tol=1e-4 and unchanged threshold/fallback. Earlier experiments showed sensitivity to optimizer tolerance; exact fitted weights depend on the fitting population even when settings are frozen.
- Predictions changed on 34 of 312 test trials. Before/after predicted counts: Safety 59/55, Funding 7/8, Efficacy 49/50, Recruitment 180/183, Other 63/74. These are predictions, not true reason frequencies, and do not reveal which changes were correct.
- Saved-model prediction replay, CSV IDs, row order, schema, nonempty labels, and artifact checksums passed. No submission-format error was found in these checks.

## Unresolved comparison

The original 157-label DBSCAN model has never been submitted, so **we cannot conclude that refitting hurt performance** from this score. A future comparison of its frozen existing CSV with this final-refit CSV would directly compare the two fits on the same Kaggle evaluation. Its existing archived CSV must not be mistaken for another 235-label refit.

Repeated three-fold CV did not demonstrate a DBSCAN advantage: it matched supervised predictions in all 15 folds, usually because its pseudo-labeling rule abstained with only 103–106 verified fitting trials. GMM was close but slightly lower on average. Normalization had better CV but worse supplied validation. The available evaluations do not establish a robust ranking for Kaggle.

No new model tuning, prediction changes, automatic upload, commit, or tag was performed while recording this result. The original 157-label source model and its Kaggle CSV remain unchanged and unsubmitted; the existing best-submission tag remains unchanged.

## Subsequent original-fit submission

Michael subsequently reported **0.87500** for the original 157-label DBSCAN predictions. This is **0.01048** above the 235-label refit on the same Kaggle evaluation. The original CSV/model were unchanged. This comparison establishes the observed ranking of the two fitted pipelines, without isolating individual refit changes. Ani remains higher at 0.89423.
