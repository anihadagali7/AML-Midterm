# Part 1 research handoff

Read the [Part 1 README](README.md) for the current model and comparable results, then [TESTING_POLICY.md](TESTING_POLICY.md) before new experiments. The detailed experiment history is indexed separately in [RESEARCH_INDEX.html](RESEARCH_INDEX.html).

## Current position

The anchored reason classifier is the highest recorded, user-reported Kaggle result at **0.89636**. Its final fit used all 235 verified labels plus the 1,014 originally masked training rows, with inductive input access. The [scored result](submissions/results/anchored-trainval235.json) identifies its frozen recipe and exact CSV. The historical preparation manifest still says unsubmitted; the later score record supersedes that status.

On the common grouped benchmark, anchored beats current Ani LDA in both CV pools. Against the submitted LDA ensemble it is essentially tied: Train157 +.000265, Pooled235 −.000266, lower original validation, higher pooled chronology. The [complete comparison](experiments/current_leaders_grouped_v1/comparison.csv) preserves the tradeoffs. Original Ani's .89423 belongs to its original transductive artifact; no Kaggle score is established for the changed current Ani LDA CSV.

## Next experiment

After the user reported **Cand2 .89102**, the [matched J and fixed-blend follow-up](experiments/cand2_followup_v1/README.md) evaluated the exact submitted inductive five-seed recipe on J's original splits. J still preferred Cand2 (.896171) to anchored (.885733), reversing their Kaggle scores. This is another counterexample to J as the primary ranker, not a general disproof. One fixed 80% anchored / 20% Cand2 blend reached .935244 Train157 / .921631 Pooled235 / .891026 validation / .887755 pooled chronology. It misses the five-win rule in Pooled235 (4/1/3 wins/ties/losses) and regresses in one chronological window. A 14-change exploratory CSV is prepared in the study; no blend Kaggle score or promotion is established.

Use anchored as the declared parent when trying to improve the submitted model. Start with a small, explicit hypothesis and the saved grouped splits. Complete both pools and the supplementary checks under the current policy before claiming a full comparison. Keep the LDA ensemble as a matched reference because its local results remain competitive. A negative finding against an older parent does not establish that the same change fails against anchored.

The [LDA study catalog](research/lda_ensemble/experiments/lda_research_checkpoint_v1/index.html) retains feature tests, failed versions, and selection rules. Its [temporal and decision follow-ups](research/lda_ensemble/experiments/temporal_generalization_review_v1/report.html) tested fourteen fixed variants without a confirmed ensemble upgrade. [New directions](research/new_directions/index.html) retains 136 settings across fourteen studies; none was promoted or submitted. Consult the relevant family before repeating an idea.

## Repository and evidence

The research is consolidated under Part 1. Use current `main` as the starting point for new work. The [canonical notebook](part1.ipynb) runs the anchored final fit; the [two former entry notebooks](research/historical_notebooks/README.md) remain archived byte for byte. Keep current model files in their existing locations; their paths are part of frozen provenance. Archived absolute paths describe removed workspaces. Use [current_leaders_local.py](current_leaders_local.py) and the replay wrappers linked from the research index to run in the current checkout.

[Verification commands](README.md#testing-and-verification) check retained hashes, predictions, and scores without fitting. Source, fixed settings, summaries, and failures remain for unsuccessful studies; detailed leader/reference predictions remain available. Some historical fold caches were discarded, so their old file hashes cannot be checked again. The [retention policy](RETENTION.md) and wrapper outputs identify this boundary.

[AGENTS.md](AGENTS.md) governs local experiment commits and immutable score tags. The [submission guide](submissions/README.md) records exact scored artifacts. [The existing commit index](research/consolidation_v1/commit_index.json) covers the research consolidation and five backfilled result checkpoints; subsequent documentation changes appear in Git history. Merging, pushing, and manual submission are separate actions.

The [archived LDA handoff](research/lda_ensemble/RESEARCH_HANDOFF.md) preserves its detailed historical context. Its leader names and submission-budget statements describe that checkpoint, not today's starting point.
