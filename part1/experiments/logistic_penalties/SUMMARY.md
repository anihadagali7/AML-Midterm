# Logistic penalty-type comparison

35 settings cross C=0.1/1/10/100/500/1000/2000 with L1 fractions 0/0.25/0.50/0.75/1. Zero is L2, one is L1, intermediate values are Elastic Net. The saved original LBFGS/L2 model remains incumbent.

## Numerical optimization

An initial SAGA attempt hit max_iter=100000 on a weakly regularized fit and is archived under saga_probe/, excluded from ranking. Final fits solve the same weighted logistic objective with full-batch L-BFGS-B. For nonzero L1 fraction, write w=u-v with nonnegative u,v and penalize sum(u+v); optimal splitting yields the ordinary L1 norm. The intercept is unpenalized. Pure L2 uses unconstrained coefficients. Loss and penalty are normalized consistently with scikit-learn: weighted loss sum plus penalty/C, divided by total fitting weight.

All 175 fits require optimizer success and normalized KKT residual <=1e-07. Finite-difference gradient check error: 1.8e-07. Seven native-solver objective checks compare against tight scikit-learn LBFGS/L2 for all reasons and converged SAGA L1/Elastic Net at C=1 for Funding. Matched full-batch L2 controls distinguish penalty changes from solver/convergence changes. Fitted coefficient dictionaries are portable; predict_model in logistic_penalties.ipynb applies the usual sigmoid decision function.

Incumbent sample F1 **0.910256**, macro F1 **0.811421**. Selected **incumbent_lbfgs_l2_C1000**: sample F1 **0.910256**, macro F1 **0.811421**.

## Best by penalty family

| Penalty | L1 fraction | C | Optimizer | Sample F1 | Macro F1 | Nonzero /750 | Change vs incumbent | Change vs matched L2 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| L2 | 0 | 1000 | LBFGS_incumbent | 0.910256 | 0.811421 | 750 | +0.000000 | +0.055556 |
| ElasticNet | 0.25 | 2000 | full_batch_L-BFGS-B | 0.858974 | 0.708182 | 511 | -0.051282 | +0.004274 |
| L1 | 1 | 1 | full_batch_L-BFGS-B | 0.831197 | 0.634762 | 111 | -0.079060 | -0.027778 |

## All configurations

| Penalty | L1 fraction | C | Optimizer | Sample F1 | Macro F1 | Nonzero /750 | Change vs incumbent | Change vs matched L2 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| L2 | 0 | 1000 | LBFGS_incumbent | 0.910256 | 0.811421 | 750 | +0.000000 | +0.055556 |
| L2 | 0 | 10 | full_batch_L-BFGS-B | 0.867521 | 0.718678 | 750 | -0.042735 | +0.000000 |
| L2 | 0 | 1 | full_batch_L-BFGS-B | 0.858974 | 0.695743 | 750 | -0.051282 | +0.000000 |
| ElasticNet | 0.25 | 2000 | full_batch_L-BFGS-B | 0.858974 | 0.708182 | 511 | -0.051282 | +0.004274 |
| ElasticNet | 0.25 | 1000 | full_batch_L-BFGS-B | 0.856838 | 0.697655 | 498 | -0.053419 | +0.002137 |
| ElasticNet | 0.25 | 500 | full_batch_L-BFGS-B | 0.856838 | 0.697655 | 490 | -0.053419 | +0.002137 |
| ElasticNet | 0.25 | 100 | full_batch_L-BFGS-B | 0.856838 | 0.697655 | 466 | -0.053419 | +0.002137 |
| L2 | 0 | 1000 | full_batch_L-BFGS-B | 0.854701 | 0.703577 | 750 | -0.055556 | +0.000000 |
| L2 | 0 | 500 | full_batch_L-BFGS-B | 0.854701 | 0.703577 | 750 | -0.055556 | +0.000000 |
| L2 | 0 | 2000 | full_batch_L-BFGS-B | 0.854701 | 0.703577 | 750 | -0.055556 | +0.000000 |
| L2 | 0 | 100 | full_batch_L-BFGS-B | 0.854701 | 0.703577 | 750 | -0.055556 | +0.000000 |
| ElasticNet | 0.25 | 10 | full_batch_L-BFGS-B | 0.844017 | 0.684094 | 419 | -0.066239 | -0.023504 |
| ElasticNet | 0.5 | 1000 | full_batch_L-BFGS-B | 0.844017 | 0.682264 | 361 | -0.066239 | -0.010684 |
| ElasticNet | 0.5 | 500 | full_batch_L-BFGS-B | 0.844017 | 0.682264 | 351 | -0.066239 | -0.010684 |
| ElasticNet | 0.5 | 2000 | full_batch_L-BFGS-B | 0.844017 | 0.682264 | 374 | -0.066239 | -0.010684 |
| ElasticNet | 0.5 | 100 | full_batch_L-BFGS-B | 0.844017 | 0.674737 | 331 | -0.066239 | -0.010684 |
| L2 | 0 | 0.1 | full_batch_L-BFGS-B | 0.841880 | 0.599954 | 750 | -0.068376 | +0.000000 |
| ElasticNet | 0.75 | 2000 | full_batch_L-BFGS-B | 0.839744 | 0.664486 | 248 | -0.070513 | -0.014957 |
| ElasticNet | 0.5 | 10 | full_batch_L-BFGS-B | 0.835470 | 0.671113 | 266 | -0.074786 | -0.032051 |
| ElasticNet | 0.75 | 1000 | full_batch_L-BFGS-B | 0.835470 | 0.662692 | 243 | -0.074786 | -0.019231 |
| ElasticNet | 0.75 | 500 | full_batch_L-BFGS-B | 0.835470 | 0.662692 | 234 | -0.074786 | -0.019231 |
| ElasticNet | 0.25 | 1 | full_batch_L-BFGS-B | 0.833333 | 0.649131 | 334 | -0.076923 | -0.025641 |
| L1 | 1 | 1 | full_batch_L-BFGS-B | 0.831197 | 0.634762 | 111 | -0.079060 | -0.027778 |
| ElasticNet | 0.75 | 100 | full_batch_L-BFGS-B | 0.829060 | 0.648406 | 220 | -0.081197 | -0.025641 |
| ElasticNet | 0.5 | 0.1 | full_batch_L-BFGS-B | 0.820513 | 0.572313 | 122 | -0.089744 | -0.021368 |
| L1 | 1 | 10 | full_batch_L-BFGS-B | 0.816239 | 0.625571 | 128 | -0.094017 | -0.051282 |
| ElasticNet | 0.75 | 0.1 | full_batch_L-BFGS-B | 0.811966 | 0.615419 | 76 | -0.098291 | -0.029915 |
| ElasticNet | 0.5 | 1 | full_batch_L-BFGS-B | 0.811966 | 0.622476 | 217 | -0.098291 | -0.047009 |
| ElasticNet | 0.75 | 10 | full_batch_L-BFGS-B | 0.811966 | 0.620159 | 192 | -0.098291 | -0.055556 |
| ElasticNet | 0.75 | 1 | full_batch_L-BFGS-B | 0.811966 | 0.624273 | 158 | -0.098291 | -0.047009 |
| L1 | 1 | 100 | full_batch_L-BFGS-B | 0.807692 | 0.624428 | 132 | -0.102564 | -0.047009 |
| L1 | 1 | 1000 | full_batch_L-BFGS-B | 0.803419 | 0.608655 | 136 | -0.106838 | -0.051282 |
| L1 | 1 | 500 | full_batch_L-BFGS-B | 0.803419 | 0.608655 | 135 | -0.106838 | -0.051282 |
| L1 | 1 | 2000 | full_batch_L-BFGS-B | 0.803419 | 0.608655 | 137 | -0.106838 | -0.051282 |
| ElasticNet | 0.25 | 0.1 | full_batch_L-BFGS-B | 0.799145 | 0.540124 | 216 | -0.111111 | -0.042735 |
| L1 | 1 | 0.1 | full_batch_L-BFGS-B | 0.782051 | 0.502378 | 43 | -0.128205 | -0.059829 |

## Controls, artifacts, and interpretation

Keep the exact 150-dimensional embedding/scaler, DBSCAN targets/raw weights, normalized fitting-weight total 157 per reason, class_weight=None, cutoff 0.5, and highest-probability fallback fixed. Penalty and C are shared across the five reason classifiers within a configuration. No validation/test labels enter fitting and no final train+validation refit is performed.

Every saved model reloads with matching probabilities and rescored sample/macro F1. The selected model independently refits from persisted data/normalized weights with matching predictions. Source targets, raw weights, and protected artifacts retain their checksums. Randomly rotated embeddings do not inherently favor sparse coordinate selection. Scores are exploratory selections on the repeatedly consulted 78 validation trials, not independent test estimates or an exhaustive per-reason penalty/C search.

datasets/ saves shared fixed data and normalized weights; reason_models/ saves all 175 coefficient fits; models/ and validation_predictions/ preserve full configurations. fit_audit.csv, optimization_attempts.csv, solver_crosschecks.csv, comparison.csv, best_by_penalty.csv, and settings.json preserve convergence, sparsity, scores, and checksums. saga_probe/ retains the incomplete first numerical attempt for transparency.

No Kaggle submission, commit, or tag was made.

No setting improves primary sample F1; retain the existing L2 model and its prepared Kaggle file.

## Native solver tolerance diagnostic

The native scikit-learn LBFGS solver at the same C=1000, but tol=1e-9, gives sample F1 0.854701 and macro F1 0.703577, matching the fully optimized L2 control's labels. The original tol=1e-4 model gives 0.910256/0.811421 and stops after 13–15 iterations; the tighter fits take 54–116. Thus stopping tolerance materially affects this model, and the incumbent is a particular fitted recipe rather than the tightly optimized L2 solution. Earlier stopping may be acting as implicit regularization; that interpretation is an inference. The current model and CSV remain unchanged. Best converged-by-penalty results are saved in best_converged_by_penalty.csv; compare those as well as the incumbent when assessing penalty effects.
