# Part 1 research handoff

Read the [Part 1 README](README.md) for the current model and comparable results, then [TESTING_POLICY.md](TESTING_POLICY.md) before new experiments. The detailed experiment history is indexed separately in [RESEARCH_INDEX.html](RESEARCH_INDEX.html).

## Current position

The anchored reason classifier is the highest recorded, user-reported Kaggle result at **0.89636**. Its final fit used all 235 verified labels plus the 1,014 originally masked training rows, with inductive input access. The [scored result](submissions/results/anchored-trainval235.json) identifies its frozen recipe and exact CSV. The historical preparation manifest still says unsubmitted; the later score record supersedes that status.

On the common grouped benchmark, anchored beats current Ani LDA in both CV pools. Against the submitted LDA ensemble it is essentially tied: Train157 +.000265, Pooled235 −.000266, lower original validation, higher pooled chronology. The [complete comparison](experiments/current_leaders_grouped_v1/comparison.csv) preserves the tradeoffs. Original Ani's .89423 belongs to its original transductive artifact; no Kaggle score is established for the changed current Ani LDA CSV.

## Next experiment

Use anchored as the declared parent when trying to improve the submitted model. Start with a small, explicit hypothesis and the saved grouped splits. Complete both pools and the supplementary checks under the current policy before claiming a full comparison. Keep the LDA ensemble as a matched reference because its local results remain competitive. A negative finding against an older parent does not establish that the same change fails against anchored.

The [LDA study catalog](research/lda_ensemble/experiments/lda_research_checkpoint_v1/index.html) retains feature tests, failed versions, and selection rules. Its [temporal and decision follow-ups](research/lda_ensemble/experiments/temporal_generalization_review_v1/report.html) tested fourteen fixed variants without a confirmed ensemble upgrade. [New directions](research/new_directions/index.html) retains 136 settings across fourteen studies; none was promoted or submitted. Consult the relevant family before repeating an idea.

The [anchored transductive study](research/anchored_transductive_v1/README.md) completed all 84 standard cases for the unchanged parent and two target-input ablations. Geometry-only access slightly improved both CV means but failed the repeat-win rule and regressed on validation/chronology. Adding target-generated labels also failed the local advancement rule. Keep the inductive leader; the compact records include every repeat and supplementary result.

[CV and Kaggle have not always ranked our models alike](research/anchored_transductive_v1/README.md#cv-and-kaggle-ordering): LDA improved both CV means over the earlier ensemble but tied its Kaggle score; anchored was essentially tied with LDA on CV and improved Kaggle. Treat the transductive conclusion as insufficient local evidence to promote, not proof of a worse Kaggle result. Neither transductive variant has been submitted. The frozen evaluation and advancement rules remain unchanged.

The [three fixed neighbor/blend experiments](research/anchored_neighbor_blend_v1/README.md) also completed the full comparison. Collapsing duplicate neighbor groups and quarter-weighting generated neighbor votes hurt both CV means. A fixed 75% anchored / 25% existing LDA ensemble blend improved Pooled235 from .921365 to .923404 with eight wins, but tied Train157's mean (3 wins / 3 ties / 2 losses), validation sample F1, and every chronological window. It does not pass the both-pool advancement rule and has no Kaggle score. Keep it as a promising matched comparison, with full predictions retained; the submitted leader stays unchanged. This used 252 fresh scored fits plus matched saved LDA scores, not fresh LDA refits.

## Repository and evidence

The research is consolidated under Part 1. Use current `main` as the starting point for new work. The [canonical notebook](part1.ipynb) runs the anchored final fit; the [two former entry notebooks](research/historical_notebooks/README.md) remain archived byte for byte. Keep current model files in their existing locations; their paths are part of frozen provenance. Archived absolute paths describe removed workspaces. Use [current_leaders_local.py](current_leaders_local.py) and the replay wrappers linked from the research index to run in the current checkout.

[Verification commands](README.md#testing-and-verification) check retained hashes, predictions, and scores without fitting. Source, fixed settings, summaries, and failures remain for unsuccessful studies; detailed leader/reference predictions remain available. Some historical fold caches were discarded, so their old file hashes cannot be checked again. The [retention policy](RETENTION.md) and wrapper outputs identify this boundary.

[AGENTS.md](AGENTS.md) governs local experiment commits and immutable score tags. The [submission guide](submissions/README.md) records exact scored artifacts. [The existing commit index](research/consolidation_v1/commit_index.json) covers the research consolidation and five backfilled result checkpoints; subsequent documentation changes appear in Git history. Merging, pushing, and manual submission are separate actions.

The [archived LDA handoff](research/lda_ensemble/RESEARCH_HANDOFF.md) preserves its detailed historical context. Its leader names and submission-budget statements describe that checkpoint, not today's starting point.
