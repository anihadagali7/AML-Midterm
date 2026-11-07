# Anchored neighbor and blend experiments

The fixed **75% anchored / 25% LDA ensemble** blend improved Pooled235 CV by
**0.002039**, winning all eight paired repeats. It tied the parent's mean
Train157 CV, original validation sample F1, and every chronological window.
It **does not pass the registered advancement rule in both pools**. Neither
neighbor modification improved both CV pools. After the user authorized a
manual trial of the fixed blend, they reported **Kaggle 0.89850**, improving the
parent's 0.89636 by **0.00214**. The blend is now the highest recorded submitted
score; the original CV-rule outcome remains unchanged. See the
[exact-file result](../../submissions/results/anchored75-lda25-trainval235.json).

## Fixed recipes and comparison

The user requested all three proposed changes. [Protocol](protocol.json),
training code and mathematical checks were committed as `924ae04` before any
candidate scoring. The parent is the unchanged, inductive anchored recipe.
There was one setting per idea, with no blend-weight, threshold or seed search.

| Arm | Change to the parent |
|---|---|
| Distinct-group neighbors | Collapse the current neighbor bank by the benchmark's raw-embedding hash. Average member label vectors, use the first member's vector, and retrieve ten distinct groups. Each group contributes one cosine-squared vote. |
| Quarter-weight generated neighbors | Keep ten-row retrieval, but give generated-label neighbors 0.25 times the gold-row vote weight. Apply the weight to both numerator and normalization. |
| Anchored75/LDA25 | Mix 75% anchored decision scores with 25% of the frozen submitted LDA ensemble's scores, after aligning their decision cutoffs. |

Both neighbor changes apply during the two existing label-generation passes
and final prediction. They leave the fitting row population, scaling, PCA,
GMM, logistic regression, LDA, confidence settings, and final decoder fixed.
They can consequently change which generated labels are accepted later.

For the blend, transform the parent's native scores using
`expit(logit(clip(score, 1e-9, 1-1e-9)) - logit(0.3))`, then mix with the LDA
ensemble's existing centered scores. Use cutoff 0.5 plus nonempty argmax.
These are decision scores, not established calibrated probabilities. The
second component is the full existing **70/15/15 LDA ensemble**, not just the
parent's individual LDA classifier. This adds inference complexity.

## Complete results

All entries below use the same saved grouped assignments and fitting access.
CV values are mean complete-OOF sample F1 over eight five-fold repeats.

| Model | Train157 CV | Pooled235 CV | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| Unchanged anchored | .932059 | .921365 | .891026 | .884354 |
| Distinct-group neighbors | .929936 | .919149 | .895299 | .880952 |
| Quarter-weight generated neighbors | .930732 | .917730 | .886752 | .867347 |
| **Anchored75/LDA25** | **.932059** | **.923404** | **.891026** | **.884354** |
| Existing LDA ensemble control | .931794 | .921631 | .905983 | .877551 |

| Candidate | Train157 delta; wins/ties/losses | Pooled235 delta; wins/ties/losses | Passes both pools? |
|---|---:|---:|---|
| Distinct-group neighbors | −.002123; 0/4/4 | −.002216; 2/0/6 | No |
| Quarter-weight generated neighbors | −.001327; 3/1/4 | −.003635; 2/0/6 | No |
| Anchored75/LDA25 | .000000; 3/3/2 | +.002039; 8/0/0 | No |

The predeclared advancement rule requires a positive mean paired difference
and at least five wins in **each** pool. The blend meets only Pooled235's rule.
Its population SD is .004246 on Train157 and .006510 on Pooled235; the parent's
are .006369 and .006695. Repeats reuse the same labeled examples and are not
independent significance tests.

| Held-out years | Rows | Anchored | Distinct groups | Quarter generated | 75/25 blend | LDA control |
|---|---:|---:|---:|---:|---:|---:|
| 2005–2006 | 39 | .863248 | .863248 | .863248 | .863248 | .871795 |
| 2007 | 22 | .939394 | .924242 | .893939 | .939394 | .878788 |
| 2008–2010 | 37 | .873874 | .873874 | .855856 | .873874 | .882883 |

The blend makes **identical label predictions on all 98 chronological rows**.
On validation it changes one row's labels without changing that row's sample
F1; macro F1 falls from .765430 to .754973. The sample-F1 ties therefore do not
mean all validation predictions or metrics are unchanged.

Across the eight Pooled235 repeats, the blend changes 27 row-evaluations
covering 17 distinct IDs: 17 improve in sample F1, seven worsen, and three tie.
Train157 has five improving and five worsening row-evaluations. This is a small
but consistent pooled-CV signal, with no improvement on the local later-year
checks. It favored further consideration over the two tested neighbor changes,
with its added complexity explicit. The subsequent Kaggle result is recorded
separately below; it does not change the original rule or justify inferring
Kaggle gains from CV alone.

## Testing and provenance

- **252 fresh scored fits:** 84 each for unchanged anchored and the two neighbor
  variants (40 Train157 folds, 40 Pooled235 folds, validation, three chronological
  windows). One additional first-fold implementation-control fit matched the
  parent exactly.
- **84 reused LDA evaluations:** saved scores from the identical CV folds and
  supplementary fitting populations. LDA was not retrained. Every target ID,
  truth vector and decoder was checked; its four headline scores reproduce the
  common comparison. The fixed blend was evaluated on all 84 cases.
- Both pools use the saved eight-repeat assignments, source-row order and
  masked-embedding-group purges. Train157 excludes validation-source inputs.
  Original validation retains the historical no-purge check. Query inputs and
  labels are excluded from fitting; the actual test file was hashed only.
- All three arms received all supplementary checks regardless of CV outcome,
  as declared before scoring. This was a fixed diagnostic comparison.
- Fresh anchored raw CV scores matched their frozen references with **zero
  maximum error**; all supplementary predictions matched. All 29 registered
  source/data/protocol hashes stayed unchanged.
- A separate TP/FP/FN calculation checked **2,310 metric values** across all
  80 repeat records and 25 supplementary records, including every label's
  precision, recall, F1 and support. Maximum difference was `1.11e-16`.
- Synthetic checks passed duplicate-vote invariance, parity for unique groups,
  conflicting-label averaging, and generated-vote normalization.

LDA CV evidence comes from
`../../experiments/current_leaders_grouped_v1/snapshots/lda_ensemble_{train,pooled}.npz`.
Its four supplementary packets come from
`../lda_ensemble/experiments/ani_lda_revision_cv_v1/supplementary/`.
Exact source hashes and package versions are in the protocol. The frozen
case-builder is reused from `../anchored_transductive_v1/run.py`.

## Retention and reproduction

[Results](results/) retain every repeat, paired delta, population SD, label
metric, supplementary result, fit-access count, and prediction-change count.
[Retention manifest](retention.json) covers approximately **583 KiB** of results.
Full scores, predictions, IDs and truth for the promising blend and both
controls are retained in `results/useful_comparison.npz`; the two unsuccessful
neighbor arms retain compact records. Temporary prediction packets and detailed
round audits are removed after verification, without a duplicate archive.

From the repository root, with the anchored environment active:

```sh
python part1/research/anchored_neighbor_blend_v1/check_neighbor_math.py
python part1/research/anchored_neighbor_blend_v1/evidence.py verify
```

The second command checks all retained hashes and source hashes, reconstructs
the saved cases, verifies fit-access counts, scores and blend arithmetic,
independently rescores 1,386 retained comparison metric values, and recomputes
all compact aggregate tables. It performs no model fits. The earlier 2,310-value
check also covered the now-discarded unsuccessful neighbor predictions.

To rerun every fit, choose a disposable directory that does not exist:

```sh
python part1/research/anchored_neighbor_blend_v1/study.py run \
  --output /tmp/anchored-neighbor-blend-replay --workers 4
python part1/research/anchored_neighbor_blend_v1/evidence.py retain \
  --raw /tmp/anchored-neighbor-blend-replay \
  --destination /tmp/anchored-neighbor-blend-replay-retained
```

Do not rerun `register` or overwrite this frozen study. Keep the complete
experiment history: local comparisons have not always predicted Kaggle
ordering, as documented in the [prior study](../anchored_transductive_v1/README.md#cv-and-kaggle-ordering).

## Subsequent submission preparation

After reviewing these results and the strict promotion-rule distinction, the
user authorized a manual trial of the fixed 75/25 blend. The
[prepared full-235 submission](../anchored_lda_blend_submission_v1/README.md)
combines the two existing final fits, with no new fitting or tuning. It changes
five of 312 test label sets. The user subsequently reported **0.89850**, a
**0.00214** improvement. The result is associated with that exact CSV and remains
user-reported; the study's original evaluation and advancement outcome remain
unchanged. The improvement supports this fixed blend, not a guarantee that
other blend weights or CV gains will improve Kaggle.
