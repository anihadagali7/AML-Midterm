# Stronger complementary kernel heads

This is the second fixed comparison for the user's two-head idea. The [first study](../complementary_heads_v1/README.md) found that its ordinary second head was weak even without a diversity penalty: Pooled235 sample-F1 .719858. Its combined candidates failed the full advancement rule. This followup was registered after seeing those results and is adaptive development, not independent confirmation.

The [v2 protocol](protocol.json) was frozen before v2 scoring. The first head remains the exact anchored pipeline. The second uses the same Nyström RBF kernel logistic model and supervised error-correlation objective as v1, with a fixed 100-fold reduction in its nonintercept L2 coefficient (.005 to .00005). Its label cutoff is .3, matching the anchored cutoff. Compare the ordinary strength-zero model with error-correlation penalties .1 and .3; no other strength, cutoff, or regularization search is performed.

Fixed blends now use the raw arithmetic scores, `.8*anchored + .2*kernel`, decoded at .3 with a nonempty-label fallback. The learned readout uses decision-aligned logits from both heads. V1's learned readout architecture, regularization, strict nested fitting, gold-only second-head supervision, unsupervised kernel feature fitting, anchored pseudo-label generation, and all saved outer folds remain unchanged. The anchored-only calibration control must reproduce v1 exactly.

The [runner](strong_study.py) reuses v1's frozen evaluation implementation while explicitly replacing the second-head class, raw-score blend and source registry. All inherited sources and data are protected by hashes. Each version evaluates 84 outer cases: 40 saved Train157 folds, 40 saved Pooled235 folds, validation78 and three chronological windows. Each includes 1,092 fresh anchored fits, 1,008 kernel-head fits and 336 learned readouts. Inner folds never use outer query labels or embeddings; masked copies of inner held-out groups are purged. No actual Kaggle test inputs are loaded.

This is still a classical adaptation, not simultaneous neural-head training or hidden-layer truncation. Anchored stays fixed, and the learned combiner reads pre-decision logits. The assignment prohibits deep learning. The explicit diversity penalty concerns errors, not raw prediction agreement, and the label-loss gradient is never reversed. Exact objective and score-space differences are in the protocol.

## Reproduction

Run from the repository root with the dependency versions recorded in the protocol:

```sh
python part1/research/complementary_heads_v2/check_strong.py
python part1/research/complementary_heads_v2/strong_study.py run --output /tmp/complementary-heads-v2-replay --workers 4
python part1/research/complementary_heads_v2/verify_strong.py /tmp/complementary-heads-v2-replay
```

Do not rerun registration over the frozen protocol. The numerical-gradient and score-conversion checks use synthetic data. The full verifier reconstructs every nested boundary and independently recomputes metrics, including exact agreement with the retained .89935 comparison and v1 calibration control. Some small chronology inner folds contain just one example of a class; sklearn's covariance warnings do not stop those fits. This limitation affects the evidence for rare labels.

Both studies reuse development labels and folds. A smaller error correlation by itself is not evidence of a better ensemble, and a local gain would not guarantee a Kaggle improvement. The exact previous .86645 result is recorded separately at `0394e373`; .89935 remains the best user-reported submitted result.

## Results and interpretation

**No candidate qualifies against the .89935 leader.** The stronger second-head control is materially better than v1, so this is the more useful comparison for the diversity hypothesis.

| Recipe | Train157 | Pooled235 | Validation78 | Chronology98 |
|---|---:|---:|---:|---:|
| parent | 0.935244 | 0.921631 | 0.891026 | 0.887755 |
| anchored | 0.932059 | 0.921365 | 0.891026 | 0.884354 |
| head_ordinary | 0.907458 | 0.881738 | 0.831197 | 0.812925 |
| head_diverse10 | 0.902760 | 0.870301 | 0.839744 | 0.804422 |
| head_diverse30 | 0.895727 | 0.852394 | 0.833333 | 0.763605 |
| blend_ordinary | 0.931529 | 0.919681 | 0.886752 | 0.877551 |
| blend_diverse10 | 0.933386 | 0.920567 | 0.886752 | 0.874150 |
| blend_diverse30 | 0.933386 | 0.920922 | 0.886752 | 0.870748 |
| calibration | 0.932059 | 0.918883 | 0.882479 | 0.870748 |
| fusion_ordinary | 0.929671 | 0.919504 | 0.880342 | 0.870748 |
| fusion_diverse10 | 0.928079 | 0.919504 | 0.880342 | 0.874150 |
| fusion_diverse30 | 0.927813 | 0.919326 | 0.880342 | 0.874150 |

The ordinary second head improved from v1 Pooled .719858 to .881738. With penalties .1/.3, its standalone pooled score fell to .870301/.852394. Mean held-out error correlation across five labels and eight pooled repeats fell from .81535 to .76311/.71216. Training squared-error correlation also dropped. The penalty changes behavior as intended, but it costs standalone accuracy.

The fixed stronger-penalty blend improves over the **matched ordinary blend**, by .001858 on Train157 (5 wins / 2 ties / 1 loss) and .001241 on Pooled235 (6 / 1 / 1). That is a modest favorable development result for the mechanism, not a win over our best submission. Validation ties the ordinary blend; chronology worsens from .877551 to .870748.

Against the **.89935 leader**, that same diverse blend loses .001858 on Train157 (3 / 1 / 4), .000709 on Pooled235 (3 / 0 / 5), .004274 on validation and .017007 on chronology. No combined candidate passes the predeclared both-pool rule. No final235 CSV was prepared and no Kaggle submission was made.

The learned readouts do not improve the result. Strong-penalty fusion scores .927813/.919326 on the two grouped pools, versus .933386/.920922 for its fixed blend. Its pooled Other F1 changes from the established leader's .621284 to .622146, a tiny descriptive increase with no overall gain. The fixed diverse blend Other F1 is .617165. This supports caution about fitting a more flexible combiner to only 235 reused labels; it does not prove that all jointly trained neural representations would fail.

Only the classical fixed-anchor adaptation was evaluated. The user's complete simultaneous neural-head and hidden-layer-truncation proposal remains untested because it conflicts with the assignment restriction. The raw-label gradient was never reversed.

[Complete results](results/summary/) include every repeat, paired comparison, label metric, Other subgroup and diversity diagnostic. The independent [verification receipt](results/summary/verification.json) checks 252 metric records, 1,260 per-label records, all 1,008 nested partitions, unchanged original inputs and controls, and exact v1 calibration reproduction. Useful lambda0/.3 standalone, fixed-blend and fusion predictions are retained; intermediate-strength prediction caches are removed after verification.

```sh
python part1/research/complementary_heads_v2/evidence.py verify
```
