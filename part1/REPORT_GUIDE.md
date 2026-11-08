# Part 1 report guide

The selected winner is **80% anchored / 20% strict-inductive five-seed Cand2**, user-reported Kaggle **0.89935**. This page is generated from retained measurement tables and immutable scored records by `python part1/reporting/build.py`. It does not fit models. [Complete method](MODEL.md) · [Report handoff](REPORT_HANDOFF.md) · [Source tables](reporting/benchmark_comparison.csv) · [Exact submissions](reporting/scored_submissions.csv).

## Comparable results

Every CV column below uses all eight saved grouped repeats. Validation is the original 157-label fit evaluated on 78 rows. Chronology pools 98 predictions from three fixed earlier-to-later windows. These are sample-wise F1 values. Kaggle numbers are user-reported, not independently queried.

| Model | Train157 | Pooled235 | Validation 78 | Chronology98 | Kaggle |
|---|---:|---:|---:|---:|---:|
| Anchored80/Cand2_20 (winner) | 0.935244 | 0.921631 | 0.891026 | 0.887755 | 0.89935 |
| Anchored75/LDA25 | 0.932059 | 0.923404 | 0.891026 | 0.884354 | 0.89850 |
| Anchored | 0.932059 | 0.921365 | 0.891026 | 0.884354 | 0.89636 |
| 80% prior 75/25 + 20% PySR | 0.927548 | 0.919770 | 0.884615 | 0.877551 | 0.89423 |
| LDA ensemble 70/15/15 | 0.931794 | 0.921631 | 0.905983 | 0.877551 | 0.89401 |
| Anchored80/diverse-kernel20 | 0.933386 | 0.920922 | 0.886752 | 0.870748 | 0.89316 |
| Cand2 strict five-seed | 0.923301 | 0.903599 | 0.871795 | 0.863946 | 0.89102 |
| Log-odds anchored80/Cand2_20 | 0.935775 | 0.922429 | 0.891026 | 0.887755 | 0.88888 |
| Partial-Other80/Cand2_20 | 0.920382 | 0.914982 | 0.891026 | 0.901020 | 0.86645 |

The winner improves anchored by .003185 on Train157, .000266 on Pooled235, ties validation, and improves chronology by .003401. The reported Kaggle gain is .00299 over anchored and .00085 over the previous 75/25 blend. The earlier LDA blend retains the higher Pooled235 mean.

Its paired wins/ties/losses versus anchored are **5/1/2 on Train157 and 4/1/3 on Pooled235**. It therefore failed the original requirement of at least five wins in both pools. The subsequent manual trial, explicitly authorized despite that result, provided the highest reported Kaggle score. Preserve that history instead of rewriting the advancement rule.

## Facts to keep distinct

- The final artifact uses 235 gold labels and 1,014 originally masked training rows. The original validation result belongs to the earlier 157-label fit; it is not independent validation of that final artifact.
- The winner's local blend measurements recombine previously verified held-out component scores. They are not new fits of both components for every blend. Its final export refit anchored and reused verified five-seed Cand2 final scores.
- The assignment's missing-label generation is retained. Anchored uses clustering-assisted self-training; Cand2 adds graph-generated labels and learned cluster/neighbor features. Test embeddings enter only prediction under the strict adapter.
- The supplied Cand2 manifest retains its original `transductive: true` flag as provenance. The submitted strict adapter overrides that access. Do not attach native/transductive scores to the strict submitted artifact.
- The final blend uses fixed arithmetic raw scores and cutoff .3 plus argmax. It uses neither log-odds pooling nor a learned final combiner. Standalone Cand2 used a different expected-F1 decoder.
- All labels/splits are reused development evidence. Overlapping CV repeats are not independent significance tests. The Kaggle partition and actual upload bytes were not independently inspected.
- The PySR experiment's declared parent was the earlier .89850 blend. Its reported .89423 is a separate artifact from original Ani's numerically identical .89423.
- The observed .89316 complementary-head regression does not isolate the diversity penalty's Kaggle effect; its matched ordinary-kernel control was not submitted. Joint neural training was not performed.

## Required label-level reporting

For the winner's validation result, use `experiments/cand2_followup_v1/results/label_metrics.csv`, selecting `track=current`, `kind=validation`, `recipe=blend`. The complete repeat and window metrics are in the adjacent `repeat_metrics.csv`. Original validation label predictions match anchored: per-label F1 Safety .941176, Funding .666667, Efficacy .758621, Recruitment .990099, Other .470588; macro F1 .765430 and sample-wise F1 .891026. Their supports are 9, 4, 14, 50, 10. Do not present these as hidden-test label scores.

## Experimental story and limitations

Start with the embedding-based anchored classifier, then explain why adding a small contribution from the graph/cluster/neighbor Cand2 pipeline helped even though standalone Cand2 scored worse. Keep the previous LDA blend as a strong control. Summarize later negative directions: log-odds pooling, altered pseudo-label treatment for Other, learned combiners, explicit diversity penalties, and bounded symbolic regression.

The [research index](RESEARCH_INDEX.html) links each recipe, fixed settings, full results and outcome. The original J audit is a small selected historical panel, not proof of a general-purpose ranking metric. Its ungrouped CV and chronology boundaries differ from the current grouped/earlier-only checks. Use [the matched audit](experiments/cand2_followup_v1/README.md) for those exact definitions and counts; do not mix J and grouped sample-F1 values into one ranking.

Use the selected artifact and method as the report conclusion. No additional model search or submission is queued. Historical studies keep their original preparation-time conclusions; later result records and this guide determine current status.
