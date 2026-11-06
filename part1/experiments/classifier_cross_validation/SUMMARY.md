# Repeated cross-validation and three classifier experiments

Five predeclared repeats of three-fold multilabel-stratified CV use the 157 verified training trials. Each fold fits about 105 verified trials plus the 1,014 originally unlabeled trials for clustering. Held-out trials are excluded entirely from classifier scaling, cluster scaling, clustering, label mapping, pseudo-label generation, and classifier fitting. All methods share the same splits. The external 78-trial validation set is never used for fitting or CV selection.

Three full-training baseline recipes reproduce their saved targets, raw weights, and validation probabilities. Pseudo-labels are rebuilt per fold, then frozen while testing classifier changes. Weights total the number of verified fitting trials per reason. Classifier/cluster scalers are refit within each fold; DBSCAN keeps its training-derived q25 radius rule rather than a radius taken from all training trials.

## Results

| Configuration | CV sample F1 | Repeat SD | CV change vs DBSCAN | Fold wins/ties/losses | Validation sample F1 | Validation change |
|---|---:|---:|---:|---|---:|---:|
| supervised | 0.891890 | 0.011105 | +0.000000 | 0/15/0 | 0.893162 | -0.017094 |
| dbscan | 0.891890 | 0.011105 | +0.000000 | 0/15/0 | 0.910256 | +0.000000 |
| gmm | 0.889130 | 0.018334 | -0.002760 | 6/2/7 | 0.905983 | -0.004274 |
| tolerance_0p001 | 0.894013 | 0.009146 | +0.002123 | 4/8/3 | 0.897436 | -0.012821 |
| tolerance_0p0003 | 0.891677 | 0.012048 | -0.000212 | 2/9/4 | 0.897436 | -0.012821 |
| tolerance_3em05 | 0.891465 | 0.013966 | -0.000425 | 4/7/4 | 0.897436 | -0.012821 |
| tolerance_1em05 | 0.891720 | 0.007042 | -0.000170 | 7/2/6 | 0.884615 | -0.025641 |
| normalized_C1 | 0.847558 | 0.009660 | -0.044331 | 1/0/14 | 0.794872 | -0.115385 |
| normalized_C10 | 0.907856 | 0.010026 | +0.015966 | 14/0/1 | 0.829060 | -0.081197 |
| normalized_C100 | 0.909130 | 0.008676 | +0.017240 | 14/0/1 | 0.829060 | -0.081197 |
| normalized_C1000 | 0.907856 | 0.009327 | +0.015966 | 12/0/3 | 0.850427 | -0.059829 |
| normalized_C10000 | 0.907856 | 0.006117 | +0.015966 | 12/1/2 | 0.863248 | -0.047009 |
| normalized_C100000 | 0.907856 | 0.006117 | +0.015966 | 12/1/2 | 0.863248 | -0.047009 |
| blend_gmm_0p25 | 0.891465 | 0.010009 | -0.000425 | 1/13/1 | 0.910256 | +0.000000 |
| blend_gmm_0p5 | 0.894013 | 0.019044 | +0.002123 | 5/5/5 | 0.901709 | -0.008547 |
| blend_gmm_0p75 | 0.891465 | 0.020172 | -0.000425 | 5/5/5 | 0.901709 | -0.008547 |

CV sample F1 pools all 157 out-of-fold trials within each repeat and averages the five repeat scores. Repeat SD and paired fold win/tie/loss counts describe stability; repeats and folds overlap in training and do not supply independent-test confidence intervals.

## Predeclared experiments

- Baselines: supervised logistic, DBSCAN + logistic, tied-covariance GMM k=10 + logistic; C=1000, tol=1e-4.
- Tolerance: 1e-3, 3e-4, 1e-4, 3e-5, 1e-5; fixed DBSCAN data and C=1000.
- Embedding normalization: StandardScaler then unit L2 length per trial; fixed DBSCAN targets, C=1/10/100/1000/10000/100000 and tol=1e-4. These C values account for the feature-scale change.
- Blend: probability average of the original DBSCAN and GMM classifiers; GMM weights 0/0.25/0.5/0.75/1, with endpoints reused from baseline models. Threshold/fallback remain fixed.

CV selections were recorded before evaluating the new settings on the external validation set: {"blend": "blend_gmm_0p5", "normalization": "normalized_C100", "overall": "normalized_C100", "tolerance": "tolerance_0p001"}.

## Interpretation limits and retained model

The 157 training labels are early/rare-enriched. Random CV measures split stability within those rows, fitting 103-106 verified trials per fold instead of 157. It does not reproduce the supplied validation population or necessarily the Kaggle distribution; disagreement alone does not establish its cause. CV scores and the external validation score are different evaluation settings and should not be compared as if they were the same metric estimate. Both have influenced exploratory model development; neither is a fresh independent final test.

Keep the saved current DBSCAN model and prepared Kaggle file unchanged pending review. No final train+validation refit, test prediction, Kaggle submission, commit, push, or tag was performed.

## Reproduction and saved artifacts

Notebook: part1/classifier_cross_validation.ipynb. Additional dependency: iterative-stratification==0.1.9. splits/ stores all exact fold assignments and per-label balances. contexts/ stores row roles, pseudo-label targets/raw and normalized weights, scalers, cluster models, responsibilities and support. models/ and predictions/ store every context/configuration. Baselines are fully rebuilt inside folds, not borrowed from full-training generated labels. fit_audit.csv records actual binary fits, context_audit.csv records fitting populations/convergence, and pseudo_label_audit.csv records generated labels. All 256 saved candidate packets reload with matching probabilities; 26 independent classifier refits from persisted targets/normalized weights reproduce probabilities. settings.json records grids, selection, versions, checksums, and safeguards.

## Findings

- No tested change exceeds the current 0.910256 sample F1 on the supplied 78 validation trials. A 25% GMM blend ties it with identical predicted labels.
- Supervised and DBSCAN predictions match on all 15 CV folds, averaging 0.891890 sample F1. DBSCAN adds no pseudo-labels in 14 folds; the remaining fold adds 81 reason labels without changing held-out predicted labels. This is limited evidence about its full-157-label benefit: its generation rule usually abstains with only 103-106 verified fitting trials.
- GMM averages 0.889130, slightly below the supervised/DBSCAN CV controls; it wins 6 folds, ties 2, and loses 7 against DBSCAN.
- The CV-selected tolerance 1e-3 averages 0.894013, a 0.002123 gain, with 4 fold wins, 8 ties, and 3 losses. It scores 0.897436 on supplied validation; retain original tol=1e-4.
- Unit-length embeddings at C=100 average 0.909130 in CV, a 0.017240 gain, with 14 fold wins and 1 loss. They score 0.829060 on supplied validation. The best validation score among normalized variants is 0.863248 at C=10000/100000, also below the current model.
- The CV-selected 50% GMM blend averages 0.894013, with 5 fold wins, 5 ties, and 5 losses. It scores 0.901709 on supplied validation. Blending has not established a consistent improvement.

Normalization is a real CV improvement in this run, but it does not transfer to the supplied validation rows. Differences in fitting-set size and sampled populations can affect rankings. The data here do not isolate the cause. Random folds draw from the same 157 early/rare-enriched training labels; they do not recreate the supplied validation population or guarantee Kaggle rankings. Keep the current saved model while retaining the alternative fits for review.

Verification independently rescored all 256 saved prediction files and checked fitting/held-out ID separation. All protected source-model, source-data, and current Kaggle-file checksums are unchanged.
