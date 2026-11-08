# Notebook submission review

Reviewed against the supplied [assignment brief](midterm/Midterm_Project.pdf).
The numbered notebooks are the current hand-in; the older
`midterm/part1_best_model.ipynb` is explicitly marked as a historical baseline.

| Assignment requirement | Notebook evidence |
| --- | --- |
| Part 1 unsupervised labeling and data augmentation | Part 1 notebook 01 explains PCA/GMM and graph-based labeling, freshly fits the selected components, and exports training reason labels. |
| Fill missing labels while preserving supplied labels | The complete export contains 1,171 training rows: 157 supplied and 1,014 predicted. A separate export identifies the confidence-filtered labels accepted during self-training. |
| Compare features and explain unsuccessful choices | Part 1 notebook 02 compares embeddings, year, text, fingerprints and other feature families, with an interpretation of the sample-F1/macro-F1 trade-off. |
| Part 1 sample-wise, per-reason and macro F1 | Part 1 notebook 01 reports all three on the original validation split. Notebook 03 distinguishes grouped and chronological comparisons. |
| Part 2 phase-specific F1, ROC-AUC and average precision | Part 2 notebook 01 compares the eligibility baseline, intermediate models and final model on the same held-out validation rows. |
| Reason-error counts and false-success rates for every phase | Part 2 notebook 01 displays every category, including the unmatched group, and distinguishes supplied from inferred reason labels. |
| Improve the binary classifier and discuss trade-offs | Part 2 notebooks 01–02 explain feature/recency/neighbor/ensemble changes, before/after metrics, remaining errors and unsuccessful proposals. |
| Explain the validation-to-test gap | Part 2 notebook 01 compares strict validation loss with the recorded later-test result and states what that comparison can establish. |
| Code, supplied data, experiment settings, results and figure code | Each part includes a source appendix, materialized resources and notebook figure-generation cells. Historical research records are identified separately. |
| Code/data inside a folder named `midterm` | Both ZIPs use `midterm/part1/` or `midterm/part2/` and can be extracted into the same folder. |

All eight numbered notebooks were executed in the checkout and again from their
ZIPs in temporary directories outside the repository. The source-export checks
pass, local notebook links resolve, and both winning CSV hashes are unchanged.
Detailed current archive hashes and execution results are in
[Part 1 verification](part1_package_verification.json) and
[Part 2 verification](part2_package_verification.json).

Part 1 performs fresh winning-model validation and final training by default.
Part 2 defaults to saved-validation rescoring and recorded-component replay;
`RUN_FINAL_TRAINING=True` enables its full fresh fit. The Part 2 receipt also
retains the earlier independent fresh-fit verification. The complete historical
experiment search is not retrained by default.

The builders preserve the edited numbered notebooks. Their optional
`--refresh-notebooks` flag deliberately replaces notebook cells from the original
templates. A normal build or `--zip-only` retains the final narrative.

For Gradescope, use the code/data ZIPs under `dist/` and the separate combined
`report.pdf`. Do not upload only the notebook files: their datasets and supporting
resources are required.
