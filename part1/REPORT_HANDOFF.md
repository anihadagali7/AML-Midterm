# Part 1 report handoff

The selected winner is **80% anchored / 20% strict-inductive five-seed Cand2**, user-reported Kaggle sample-wise F1 **0.89935**. Model search is closed. The complete source and evidence are now in this checkout alongside the reconciled PySR work.

## Exact winning checkpoint

| Item | Reference |
|---|---|
| Result commit | `cde3d500e1304f2afeeb6dc4c7f50e49190c6a4b` |
| Immutable annotated tag | `part1-anchored80-cand2-inductive20-trainval235-kaggle-0.89935` |
| Exact submitted CSV | [anchored80_cand2_inductive20.csv](submissions/candidates/anchored80_cand2_inductive20.csv) |
| CSV SHA-256 | `c20aa937fa8b1e5aea55bcd6564c6acd952f0d74030cc7c1a437eac54a1ee257` |
| Canonical score record | [anchored80-cand2-inductive20-trainval235.json](submissions/results/anchored80-cand2-inductive20-trainval235.json) |
| Frozen export protocol | [blend_export_protocol.json](experiments/cand2_followup_v1/blend_export_protocol.json) |
| Protocol SHA-256 | `8b4e2e9e69c2fddf026038a883411aa3c4e59cb870025d01c5fe2729a4b76cb4` |
| Preparation source commit | `ef4d7e0e`; later score record supersedes its unsubmitted status |

All links are relative to this repository. No separate research worktree is needed to read or verify the winner. Scored tags and their original commits remain intact.

## Report files and evidence

- [Complete method](MODEL.md) describes both components and the fixed final decision rule.
- [Generated report guide](REPORT_GUIDE.md) contains nine comparable models, required per-label results, paired comparisons, and limitations, with source paths in the adjacent CSV table.
- [Report source](../report.tex) and [compiled PDF](../report.pdf) now describe the selected blend and reported .89935 result. Part 2 content is preserved from the incoming main branch.
- [Verification and integration record](reporting/README.md) explains how to audit artifacts, numerical summaries, and documentation without retraining.
- [Research index](RESEARCH_INDEX.html) and [submission ledger](submissions/README.md) cover the completed alternatives, including PySR .89423, fixed diverse kernel .89316, log-odds pooling .88888, and partial-Other .86645.

The anchored notebook remains a component notebook. It does not implement the complete winning blend. Use the frozen [blend exporter](experiments/cand2_followup_v1/export_blend.py), [strict Cand2 adapter](experiments/cand2_followup_v1/strict_adapter.py), and retained final score packets for full reproduction.

## Claims that need care

The final artifact uses 235 supplied gold labels and 1,014 originally masked rows; test inputs are prediction-only. The 78-row validation result belongs to a separate 157-label fit. The winner's benchmark recombines verified held-out component scores; final export refit anchored and reused the verified five-seed Cand2 packet.

The winner did not pass the original paired-repeat advancement rule. Its exploratory manual result determined the report selection. Do not describe it as best on every local benchmark: the prior anchored/LDA blend has the higher Pooled235 mean. The reported gain is .00299 over anchored and .00085 over that prior blend.

All supplied labels and saved splits have influenced development. Kaggle scores are user-reported; uploaded bytes and leaderboard partition were not independently inspected. The repository identifies the intended exact files and immutable checkpoints. Overlapping repeats do not establish independent statistical significance.

Historical study conclusions remain frozen. In particular, PySR's parent was the .89850 anchored/LDA blend, and its .89423 is a different artifact from original Ani's equal score. The latest fixed kernel experiment was a classical complementary-head model, not joint neural training. No further experiment or submission is queued.
