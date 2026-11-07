# Other leading recipes on the fixed training-only benchmark

Three fixed contenders: GMM tied k=10 + logistic C=1000; K-means k=8 + logistic C=1000; DBSCAN + standardized/unit-length embeddings + logistic C=100. Compare with the unchanged training-only Ani and DBSCAN references on the exact same 40 saved splits. No parameter search.

## Results

| Recipe | Mean sample F1 | Repeat SD | Mean macro F1 | Change vs Ani | Change vs DBSCAN |
|---|---:|---:|---:|---:|---:|
| ani_train_only | 0.908865 | 0.006631 | 0.822043 | +0.000000 | +0.020550 |
| kmeans_k8 | 0.895089 | 0.010270 | 0.807697 | -0.013777 | +0.006773 |
| dbscan_normalized_C100 | 0.892624 | 0.008962 | 0.819820 | -0.016241 | +0.004309 |
| dbscan_train_only | 0.888316 | 0.003882 | 0.807876 | -0.020550 | +0.000000 |
| gmm_tied_k10 | 0.887553 | 0.014661 | 0.795465 | -0.021312 | -0.000762 |

## Paired results for the three added candidates

| Candidate | Reference | Mean change | Fold wins / ties / losses | Repeat wins / ties / losses |
|---|---|---:|---|---|
| gmm_tied_k10 | ani_train_only | -0.021312 | 11/3/26 | 1/0/7 |
| gmm_tied_k10 | dbscan_train_only | -0.000762 | 17/7/16 | 5/0/3 |
| kmeans_k8 | ani_train_only | -0.013777 | 10/4/26 | 0/0/8 |
| kmeans_k8 | dbscan_train_only | +0.006773 | 22/10/8 | 7/0/1 |
| dbscan_normalized_C100 | ani_train_only | -0.016241 | 9/7/24 | 0/0/8 |
| dbscan_normalized_C100 | dbscan_train_only | +0.004309 | 20/1/19 | 5/0/3 |

## Fixed procedure and fitting access

Use all 235 known labels in eight repeats of five folds, with the saved KFold seeds100–107 and exact existing assignments. Each fit uses 188 verified labels and 1,014 originally unlabeled training trials. Every held-out trial and every Kaggle test trial is excluded from all fitting, preprocessing, clustering, cluster-label mapping, confidence calculations, and augmentation. Classifier scaler fits the 188 verified rows; cluster scaler fits the 1,202 training-pool rows. No extra PCA for GMM or K-means.

GMM uses ten tied-covariance components,n_init=3,max_iter=500,tol=1e-3,reg_covar=1e-3,seed=42. K-means uses k=8, n_init=10, max_iter=300, tol=1e-4,seed=42. Both use the unchanged partial-label mapping and balanced pseudo-label weight-cap rules, rebuilding labels from fitting labels inside each fold. Weights normalize to 188 per reason. Logistic solvers remain LBFGS,tol1e-4,max_iter=5000; C=1000 for GMM/K-means and C=100 for normalized DBSCAN.

The normalized candidate uses exactly the corresponding saved DBSCAN fold targets/weights, followed by verified-only StandardScaler and row L2 Normalizer. The normalizer does not learn from held-out inputs. Threshold 0.5 and highest-probability fallback stay fixed for all new recipes; Ani retains its native 0.3 cutoff.

## Interpretation and reuse

Primary F1 pools 235 out-of-fold predictions per repeat and averages the 8 repeat scores. Fold/repeat win counts and SD are descriptive; repeated folds overlap and are not independent final tests. Recipe choices were made before these scores, with no post-score C/k/cutoff search. The repeatedly used labeled pool and random folds may not reproduce later Kaggle trials. Historical validation scores serve only to verify implementation, not as additional winning criteria. Keep the original Ani reference frozen while recording any candidate that improves it.

## Reproduction and saved artifacts

Notebook: part1/leading_recipes_cv.ipynb. protocol.json/PROTOCOL.md record the exact source recipes and source-fold checksum references before fitting. datasets/ saves every new fold’s ID/target/raw/normalized-weight dataset; models/ saves all 120 new classifier packets; clustering/ saves80 cluster models, responsibilities, support and reason rates. predictions/ includes 120 new and 80 reference held-out prediction files. fit_audit.csv, generated_labels.csv, role_audit.csv, repeat/subgroup scores and paired comparisons preserve details. Shared raw features and fold assignments remain at ../paired_recipe_cv/, checked by protocol hashes.

All three native 157-label controls reproduce their original probabilities and (for clustering) targets/weights. Every new model reloads with matching probabilities. All 200 saved prediction sets independently rescore. All three first-fold classifiers independently refit from persisted targets/weights. Original models, datasets, source folds and references retain their hashes. No submission CSV, upload, commit, push, or tag was created.

## Observed interpretation

Training-only Ani remains first at **0.908865** sample F1. The strongest newly tested alternative is K-means k8 at **0.895089**, followed by normalized DBSCAN C100 at **0.892624**. Original DBSCAN scores **0.888316**, and GMM tied k10 scores **0.887553**.

K-means improves on original DBSCAN by 0.006773, winning 22 folds, tying ten, and losing eight; its repeat mean is higher in seven of eight repeats. Normalization improves by 0.004309, winning 20 folds and losing 19, with one tie; its repeat mean is higher in five repeats. GMM is close on mean but lower by 0.000762 and has greater repeat variation.

None of the three added recipes beats Ani on mean sample F1. Ani wins all eight repeat averages against K-means and normalization, and seven of eight against this GMM. Keep Ani as the fixed reference for subsequent experiments. These results change the ranking relative to the old 78-trial comparison; they do not establish a new Kaggle ranking or guarantee transfer.
