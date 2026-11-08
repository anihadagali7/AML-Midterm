# Part 1 research handoff

**Research is closed for this report phase.** The selected winner is anchored80/Cand2_20, reported Kaggle **.89935**. Start with the [model](MODEL.md), [report guide](REPORT_GUIDE.md), and [exact checkpoint](REPORT_HANDOFF.md). All completed research branches listed in the [integration record](reporting/INTEGRATION.md) are reconciled with the incoming PySR main branch.

## Outcomes to explain

| Direction | Outcome and evidence |
|---|---|
| Small graph/cluster/neighbor contribution | [Strict Cand2](experiments/cand2_submission_v1/README.md) scored .89102 alone; the fixed 20% contribution to anchored reached [the selected .89935](experiments/cand2_followup_v1/README.md). |
| Earlier LDA complement | [75/25 anchored/LDA](research/anchored_lda_blend_submission_v1/README.md) scored .89850 and retains the higher Pooled235 mean. |
| Alternative ranking measures | [Derived benchmark audit](experiments/derived_benchmark_audit_v1/README.md) and [matched Cand2 audit](experiments/cand2_followup_v1/README.md): J is a diagnostic from a small historical panel, not a reliable universal Kaggle ranking. |
| Blend weights, thresholds, partners | [Follow-ups](research/blend_followup_v1/README.md), [partners](research/blend_partners_v1/README.md), and [half-step](research/blend_halfstep_v1/README.md) closed; two exports confirmed unsubmitted. [Log-odds pooling](experiments/blend_refinement_v2/README.md) scored .88888 despite small local gains. |
| Hard labels and Other | [Graph components](experiments/anchored_graph_components_v1/README.md) and [Other specialists/combiner](research/other_combiner_v2/README.md) did not justify promotion. [Partial-label secondary experiment](research/partial_secondary_v1/README.md) scored .86645 after final export. |
| Explicit complementary head | [Diverse kernel and learned fusion](research/complementary_heads_v2/README.md): fixed blend scored .89316; no variant displaced the winner. |
| Symbolic regression | [PySR](research/pysr_v1/README.md) regressed on all four headline checks against its .89850 parent; [manual trial](research/pysr_blend_submission_v1/README.md) scored .89423. |

The [research index](RESEARCH_INDEX.html) includes earlier experiments and failures. Retained results are sufficient to describe negative findings without rerunning searches. Original preparation manifests can still say “unsubmitted”; the later [canonical score records](submissions/README.md) determine current status.

## Preserve the evidence

Use the existing [testing policy](TESTING_POLICY.md) and [retention policy](RETENTION.md). Do not rewrite frozen protocols, move scored tags, or attach a score to a different CSV. Useful matched predictions remain because they verify comparisons; unsuccessful fitted models and fold caches were pruned. The report integration changes organization and prose, not scored model recipes or data.
