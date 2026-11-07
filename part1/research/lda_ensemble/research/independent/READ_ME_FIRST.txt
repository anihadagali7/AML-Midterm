AML MIDTERM — PART 1 — CURRENT RESULTS

Best measured validation sample-wise F1: 0.91239316
Validation macro F1: 0.81044357
Earlier validation target: strictly above 0.92735. Target met: False.
Latest reported Kaggle leader: 0.93xx, exact score not provided. Local validation above 0.93: False.
This modestly exceeds the user's reported 0.9103 validation result; it does not establish a statistically reliable improvement.
Primary common grouped five-fold CV F1: 0.91047417; repeat standard deviation: 0.00737.
Same saved folds and seeds [20261126, 20261127, 20261128] for every candidate. See notebook 10 and experiments/benchmark_v1/comparison.csv.
Supplementary pooled train-plus-validation CV F1: 0.89229314; repeat standard deviation: 0.00468. This evaluates fresh fold fits of the same recipe on 235 pooled labels, not the exact saved model. See notebook 11.
Kaggle score: not measured. Validation and CV are not leaderboard scores.

CURRENT FILES
predictions/current_best/part1_submission.csv — 312 rows, ID and Predicted.
predictions/current_best/part1_train_augmented.csv — all 157 original gold labels preserved; 1,014 missing-label rows predicted.
predictions/current_best/manifest.json — exact model, metrics, source hashes and submission hash.
midterm/03_current_report.ipynb and .html — current report and figures.
midterm/04_reproduce_current.ipynb — embedded source and executed reproduction check.
midterm/05_tri_training_and_shift.ipynb — tri-training, target-aware selection, matched controls and results (when present).
midterm/06_stability_and_specialists.ipynb — bootstrap stability, feature additions and nested score specialists (when present).
midterm/07_cotraining_and_distances.ipynb — separate-view co-training, learned distances and matched controls (when present).
midterm/08_svm_refinements.ipynb — per-reason tuning, reliability weights and local SVM specialists (when present).
midterm/09_shift_graph_and_subgroups.ipynb — label-frequency correction, graph SVMs and Other subgroups (when present).
midterm/10_common_cv_benchmark.ipynb — primary common-fold model comparison with separate existing validation scores (when present).
midterm/11_pooled_cv_comparison.ipynb — supplementary train-plus-validation CV of frozen leading recipes (when present).
experiments/benchmark_v1/protocol.json and comparison.csv — saved fold assignments and primary comparison table (when present).
experiments/validation-comparisons.csv — every recorded validation comparison, including failures to improve.
experiments/trials.csv and trials.jsonl — 2293 logged CV runs, 1840 distinct configurations, 49 failed attempts retained.
Full-size CV and ensemble-only screens have separate files; they are not all counted in the main trial log.

REPRODUCE
From this directory (or the portable midterm directory):
  .venv/bin/python reproduce_current.py
For a new environment, use Python 3.12 and requirements-lock.txt.
The script reads the frozen experiments/current-best.json configuration, trains only on supplied training rows, writes predictions, and then verifies its validation score. It does not tune on validation labels.

CURRENT METHOD
A fixed per-reason combination uses the previous semi-supervised SVM for Safety, Funding and Efficacy, and classical tri-training for Recruitment and Other. This combination was selected from 32 masks using training-fold predictions and the frozen robust weighting criterion before validation evaluation. Empty combined predictions fall back to the base SVM's decoded labels.
Both components fit StandardScaler to gold and original masked training embeddings, retaining all 150 coordinates. No validation/test features enter the selected inductive fitters.
Base component: KMeans with 10 clusters, 10 initializations and seed 42, then five balanced RBF SVMs with C=1 and gamma=0.001. At each pseudo-label weight 0.003, 0.01 and 0.03, perform three label/refit iterations. Original gold labels remain fixed. If a cluster has at least two gold rows and at least 80% agreement for a binary reason, conflicting pseudo-labels receive one tenth the normal weight. Transform margins with sigmoid(2*margin), threshold at 0.55, and use the largest score if all labels would be empty.
Tri-training component: ordinary bootstrap samples initialize an RBF SVM (C=1, gamma=0.001), logistic regression (C=0.3), and ridge classifier (alpha=10). Peer agreement and the classical error/size conditions govern pseudo-label updates, with unit pseudo-label weight and no cluster gate in this component. Majority voting at threshold 0.5 decodes predictions; smoothed vote scores break ties for empty-label fallback. At most eight update rounds are used.
All 157 gold and 1,014 originally masked training rows are available for final inference. Validation remains excluded. Final hybrid scores are binary decisions; component margin/vote scores are not calibrated probabilities.
Earlier screens tested auxiliary criteria, disease, drug, target, action, pathway, structure and date features. They did not improve the earlier incumbent on validation; drug-target additions tied it. The new hybrid uses embeddings only. Exhaustive feature-subset testing of this new hybrid has not been performed.

EVIDENCE AND LIMITATIONS
89 recorded standalone validation candidate evaluations across 31 batches, including equivalent variants. Their hyperparameters and decision rules were chosen from training-side experiments before each small validation batch. Those original fits excluded validation labels and made no per-row corrections. Repeated aggregate comparisons still create selection bias: this is no longer a pristine holdout. The separately authorized pooled CV deliberately uses original validation labels only in the training portion of each fold, with all held-out groups excluded; it does not replace the saved training-only model or reinterpret its original validation score.
Five-fold iterative multilabel splits group identical embeddings and exclude matching masked duplicates from training. Repeats use three seeds. Leave-one-group-out evaluates 150 distinct gold embedding groups. All learned transformations stay inside their permitted folds. Temporal stress tests are affected by the intentional rare-label enrichment of later gold rows.
Descriptive grouped validation bootstrap interval: 0.8568–0.9569; this does not correct selection bias.
The search is broad, not an exhaustive enumeration of every possible algorithm or feature subset. No external trial data, extra labels, ID predictors or deep learning were used. Literature was consulted only for methods. Some explicitly marked research variants used query features transductively; the selected candidate does not.

HISTORY
Historical model-selection repeated CV F1: 0.91748054; leave-one-embedding-group-out development F1: 0.92866242. These historical diagnostics do not replace the primary common benchmark.
Additional fixed-policy grouped five-fold audit on seeds [20261111, 20261112, 20261113]: 0.90375088 (split standard deviation 0.00361). Different split sets give different estimates; these are development audits, not a nested replay of the full search.

The earlier delivered full-fit ensemble scored 0.850427 on validation; its 0.936306 training CV did not transfer. It is preserved under the older filenames and in notebooks 01/02, which describe that historical stage. The previous 0.910256 validation incumbent is preserved under predictions/checkpoints/pre_phase21_091026 and experiments/phase21-prior-incumbent.json. Use the current_best directory and notebooks 03/04/05 for the current candidate.
Part 2 and the combined five-page course report remain outside this Part 1 work.
