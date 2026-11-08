# Complementary heads with a supervised diversity penalty

This study tests the user's suggestion to add a second, deliberately different predictive head, then learn a readout over both heads. The exact preceding exploratory submission scored **0.86645**, recorded separately at `0394e373`; the best reported Kaggle recipe remains **0.89935** (`cde3d500`). This work starts from main `766d16d3` on `codex/part1-complementary-heads` in its own worktree.

The [frozen protocol](protocol.json) predates all benchmark scoring. Original anchored (.89636) is the architecture parent. The scored anchored80/Cand2_20 (.89935) recipe is the advancement control; cached Cand2 predictions are only used for that comparison. No actual Kaggle test inputs enter this study.

## What is being tested

- **First head:** unchanged anchored, including logistic regression, cosine neighbors, shrinkage LDA, and cluster-assisted pseudo-label generation.
- **Second head:** five binary logistic predictors over a 128-dimensional Nyström RBF feature map, optimized by gradients. The map and scaling use eligible fitting inputs; only supplied gold labels enter this head's supervised loss.
- **Diversity:** add a squared centered error-correlation penalty against crossfitted anchored residuals. Compare strengths 0 (ordinary control), .1 and .3. The supervised loss keeps its ordinary sign; it is never reversed.
- **Combination:** compare a fixed 80/20 blend in decision-aligned probability space with a regularized readout of all five logits from each head. An anchored-only calibration readout is a further control. Every learned readout uses genuine inner held-out scores from both heads.

This is a classical adaptation, not a jointly trained neural network. Anchored stays fixed and has no neural layers to remove. The Nyström map is an unsupervised kernel approximation, not a learned hidden neural representation. The learned readout uses the heads' pre-decision scores; there is no hidden-layer truncation. This follows the assignment's no-deep-learning restriction ([PDF](../../../Midterm_Project.pdf), page 7; [testing policy](../../TESTING_POLICY.md)).

The motivation relates to [Liu and Yao's negative-correlation learning](https://www.sciencedirect.com/science/article/pii/S0893608099000738); this implementation uses a different, fixed-anchor squared-correlation objective. [Gradient reversal](https://www.jmlr.org/papers/v17/15-239.html) reverses a particular adversarial objective, not the outcome-prediction loss. Penalizing agreement in raw predictions would punish two models for agreeing on correct answers. Even penalizing shared errors can harm accuracy, so reduced correlation alone cannot qualify a candidate.

Earlier experiments already included kernels, RFM mixtures, Jacobian-derived forests/boosting and regularized combiners. The distinct hypothesis here is the explicit error-correlation training penalty and its matched strength-zero control, followed by nested cross-label fusion. See [source provenance](provenance.json).

## Evaluation and reproducibility

All arms use the saved eight repeats of five grouped folds on Train157 and Pooled235, plus original validation78 and three chronology windows (98 rows). The complete diagnostic schedule was declared before scoring. All preprocessing, pseudo-label generation, kernel features, and readouts are refitted within their fitting boundaries. Three inner folds train the combiner. Each inner kernel training partition has another three-fold anchored crossfit for residual training. This gives 13 anchored fits, 12 kernel-head fits and four fusion/calibration fits per outer case, with no outer or inner held-out labels entering any upstream fit.

The original validation check retains its historical outer-boundary duplicate policy. Grouped CV, chronology, and every inner split purge masked copies of held-out groups. All labels and saved splits remain reused development evidence, not a new untouched holdout.

Run from the repository root with NumPy, pandas, SciPy, scikit-learn, iterative-stratification and threadpoolctl installed. The exact versions are in `protocol.json`.

```sh
python part1/research/complementary_heads_v1/check_complement.py
python part1/research/complementary_heads_v1/head_study.py run --output /tmp/complementary-heads-replay --workers 4
python part1/research/complementary_heads_v1/verify_heads.py /tmp/complementary-heads-replay
```

`register` was run once before scoring; do not overwrite the frozen protocol. Replay requires unchanged sources and a new output directory. The math check validates the full analytic diversity gradient against finite differences, exercises optimization on synthetic data, checks decoder equivalence and reconstructs all 1,008 nested group partitions. The independent verifier rescores sample/macro/per-label metrics and reconstructs saved fitting boundaries.

## Results

No v1 candidate passes the predeclared both-pool rule. The strongest pooled result was the ordinary fixed blend, but it lost Train157 and both supplementary checks. The learned two-head readouts all regressed.

| Recipe | Train157 | Pooled235 | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| parent | 0.935244 | 0.921631 | 0.891026 | 0.887755 |
| anchored | 0.932059 | 0.921365 | 0.891026 | 0.884354 |
| head_ordinary | 0.712049 | 0.719858 | 0.666667 | 0.408163 |
| head_diverse10 | 0.696125 | 0.704433 | 0.649573 | 0.408163 |
| head_diverse30 | 0.683386 | 0.677128 | 0.649573 | 0.408163 |
| blend_ordinary | 0.931263 | 0.924025 | 0.878205 | 0.863946 |
| blend_diverse10 | 0.932059 | 0.923848 | 0.878205 | 0.863946 |
| blend_diverse30 | 0.933121 | 0.923848 | 0.878205 | 0.863946 |
| calibration | 0.932059 | 0.918883 | 0.882479 | 0.870748 |
| fusion_ordinary | 0.931263 | 0.917553 | 0.876068 | 0.867347 |
| fusion_diverse10 | 0.930732 | 0.917553 | 0.876068 | 0.867347 |
| fusion_diverse30 | 0.930998 | 0.917730 | 0.876068 | 0.867347 |

The ordinary standalone kernel head was already weak (Pooled .719858), falling to .704433 and .677128 with the diversity penalties. Mean held-out error correlation across labels and pooled repeats fell from .5781 to .5730/.5645, but the heads made more new errors. This does not establish that a strong complementary head cannot help. It motivated the separately frozen [v2 followup](../complementary_heads_v2/README.md), with stronger kernel fitting and a lower head cutoff.

The full run completed 1,092 fresh anchored fits, 1,008 kernel fits and 336 readout fits. All outer anchored scores matched cached controls exactly. Independent verification checked 252 sample/macro metric records, 1,260 per-label records and all 1,008 nested boundaries. Some small chronology inner folds produce sklearn single-sample covariance warnings; fits finish with finite scores and every fitting fold contains all five classes.

[All repeat and per-label results](results/summary/) remain available, along with [verification](results/summary/verification.json). Retained matched head/fusion predictions preserve the ordinary versus diverse comparison and calibration/control evidence. Unsuccessful fixed-blend predictions and all per-fold caches were removed only after independent verification.

```sh
python part1/research/complementary_heads_v1/retained_evidence.py verify
```
