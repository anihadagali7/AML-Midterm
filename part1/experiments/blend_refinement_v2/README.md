# Follow-up to the reported 0.89935 blend

**Next exploratory CSV:** [refinement_log_pool.csv](../../submissions/candidates/refinement_log_pool.csv). It keeps the same anchored/Cand2 components and 80/20 weights, but averages log-odds instead of raw selection scores. It changes **five of 312** test label sets. No upload or score is recorded for this new file.

The user reported **0.89935** for the arithmetic 80/20 blend and requested a follow-up within ten minutes. That exact result was recorded separately at commit `cde3d50` with immutable tag `part1-anchored80-cand2-inductive20-trainval235-kaggle-0.89935`; this branch starts from current main `31f3fa9`. [Parent result](parent_reported_result.json), [exact parent CSV](parent_scored.csv).

## Selected candidate

For each label, clip both component selection scores to [1e-9,1−1e-9], then compute `sigmoid(.8*logit(anchored) + .2*logit(Cand2))`. Use the same .3 cutoff and nonempty argmax fallback. This is geometric pooling of odds from existing selection scores; neither component's scores are claimed to be calibrated marginals.

| Check | Scored 0.89935 parent | Log-odds candidate | Delta |
|---|---:|---:|---:|
| Train157, eight grouped repeats | .935244 | .935775 | +.000531 |
| Pooled235, eight grouped repeats | .921631 | .922429 | +.000798 |
| Original validation78 | .891026 | .891026 | .000000 |
| Pooled chronology98 | .887755 | .887755 | .000000 |

Paired wins/ties/losses: **3/4/1** on Train157 and **5/2/1** on Pooled235. The mean gains are small and the training-repeat win count does **not** satisfy the usual five-win rule. This is a user-requested exploratory submission, not a confirmed replacement. All three chronological windows and per-label metrics are retained in [results](results/).

SHA-256: `43d8c7ef4d090461630ed293cacaefffa4607bacee5a03e6d9ea7dd9fd614abe`. [Prepared-file record](prepared_submission.json), [changed predictions](changed_predictions.csv).

## Complete search history

All candidates were specified before their batch's metrics were computed. The [first batch](../blend_refinement_v1/protocol.json) tried Cand2 weights .10/.30/.40 and 10%/25% additions of the previous LDA ensemble to the centered parent. None improved both CV means. The best under its registered criterion was the 10% LDA addition, but it was below the parent on Train157 and chronology. It was not supplied to the user or uploaded; its unselected detailed outputs were removed after verification.

The [second batch protocol](protocol.json) recorded those negative findings and fixed four further candidates: 10%/20% native-scale LDA additions, 80/20 log-odds pooling, and a fixed per-label uncertainty-weighted Cand2 contribution. No further batch followed. The log-odds candidate was the only second-batch variant with positive mean changes in both pools.

| All nine candidates | Train157 | Pooled235 | Validation | Chronology |
|---|---:|---:|---:|---:|
| Cand2 weight .10 | .932325 | .922872 | .891026 | .880952 |
| Cand2 weight .30 | .933652 | .920479 | .891026 | .887755 |
| Cand2 weight .40 | .932325 | .918262 | .882479 | .887755 |
| Parent + centered LDA10 | .934448 | .921631 | .891026 | .884354 |
| Parent + centered LDA25 | .932325 | .922163 | .891026 | .884354 |
| Native LDA10 | .934713 | .921277 | .891026 | .884354 |
| Native LDA20 | .933917 | .921543 | .891026 | .884354 |
| **Log-odds 80/20** | **.935775** | **.922429** | **.891026** | **.887755** |
| Uncertainty-weighted Cand2 | .933917 | .921720 | .891026 | .887755 |

Selection first applied the registered supplementary-regression guardrail, then maximized the worse of the two CV mean deltas against the scored parent, breaking ties by their sum and registration order. The second batch's winner outranked the first batch's winner by the same criterion. The four checks were not collapsed into J or another overall promotion score. The exact final test predictions were not used to choose weights or infer hidden correctness. The selected candidate remains below the formal repeat-win threshold.

## Access, controls and verification

This is recombination of already fitted models, not new classifier training. Every component's CV predictions come from the identical saved grouped assignments and fitting access, with gold/query IDs and truth aligned. The original anchored, 0.89935 parent and LDA controls reproduce their existing four benchmark means. Final components were fitted on 235 verified labels plus 1,014 originally masked training rows; test inputs were used for prediction only.

All eight repeats and supplementary cases were declared for both small diagnostic batches. Sample F1 is computed over each complete OOF repeat, then averaged; chronology pools all 98 rows. Sources, fixed scripts and input hashes are frozen in each protocol. No model, weight or decoder was changed after scoring within a batch. These are reused development labels, and the second batch follows the first batch's failures, so selected gains are optimistic rather than an untouched estimate.

Independent arithmetic checked sample F1 and every label's precision/recall/F1/support across all nine candidates and controls. The final log-odds formula was independently checked through the equivalent weighted-product-of-odds formula. IDs, row count, valid nonempty labels, exact parent CSV reproduction, source hashes and output SHA-256 passed. [Verification receipt](verification.json).

Python 3.12.13 with the repository's [numerical dependencies](../../requirements-simple.txt). From the repository root, `python part1/experiments/blend_refinement_v2/verify.py` checks retained evidence without fitting. Each `run.py` replays its fixed batch and refuses to overwrite an exported file. Use disposable checkouts for replays, preserving the exact candidate supplied to the user. Detailed selected predictions remain for this useful candidate; compact source, inputs, all scores and negatives remain for both batches.
