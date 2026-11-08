# Winning Part 1 model

**Selected result: anchored80/Cand2_20, reported Kaggle sample-wise F1 .89935.** The final prediction is

`score = 0.8 * anchored_score + 0.2 * Cand2_label_marginal`.

Return every label scoring at least .3, and always include the highest-scoring label. The weights and cutoff are fixed. Scores are not established calibrated probabilities. [Exact CSV](submissions/candidates/anchored80_cand2_inductive20.csv) · [score record](submissions/results/anchored80-cand2-inductive20-trainval235.json) · [benchmark evidence](experiments/cand2_followup_v1/README.md).

## Inputs and fitting population

Both components use the 150 supplied stopping-reason embedding coordinates. The final artifact uses all 235 supplied gold labels and the 1,014 originally masked training rows. The validation benchmark instead trains on the original 157 labels and evaluates the separate 78 rows. IDs align records; start year defines chronology checks. Neither is a predictive feature. Actual Kaggle test embeddings enter prediction only.

## Anchored component

[Implementation](anchored_solution.py) and [executed component notebook](part1.ipynb).

1. Fit a standard scaler on the permitted gold and masked rows. Fit PCA with 30 components and a 30-component diagonal Gaussian mixture on those fitting inputs only. Use three GMM initializations and random seed 0. PCA/GMM support missing-label generation; classifiers use all 150 standardized coordinates.
2. Fit multiclass logistic regression with `C=.1`, shrinkage LDA with equal five-class priors, and cosine 10-nearest-neighbor voting. Duplicate a gold row once for each positive reason in the multiclass fits. Logistic weights divide a multi-label row's mass across its positives and use class weights determined from the original fitting gold labels. Neighbor votes retain all positive labels.
3. Combine logistic, neighbor and LDA scores with weights .5/.3/.2. Neighbor weights are nonnegative cosine similarity squared. GMM contributes 5% only to the pseudo-label teacher; the classifier mixture contributes 95%.
4. Generate labels for masked rows whose maximum teacher score is at least .8. Use cutoff .3 plus argmax to choose their labels. Rebuild from gold plus currently accepted pseudo-labeled rows for two refreshes, with three classifier fits in total. Generated rows have logistic weight .25. LDA and neighbor components do not share that logistic-only downweighting.

On the final 235 anchored fit, 757 masked rows were accepted in the last refresh. These are model-generated labels, not additional verified labels. Gold labels are preserved.

## Cand2 component

[Frozen supplied configuration](experiments/cand2_followup_v1/cand2_manifest.json), [strict adapter](experiments/cand2_followup_v1/strict_adapter.py), and [supplied pipeline implementation](experiments/cand2_followup_v1/source/p1lib/).

Cand2 models eight fixed label combinations and converts their probabilities into five label marginals. It adds soft membership features from 8-cluster k-means and 20-neighbor label-vote features to standardized embeddings. A Correct-and-Smooth graph labeler supplies pseudo-labels; its graph uses standardized embeddings reduced to 64 PCA coordinates, L2 normalization, 10-neighbor connectivity and symmetric maximum weights. The fixed configuration has generated-label weight .06307249103049938, correction coefficient .8, smoothing coefficient .4199619766036297, hard label generation and confidence cutoff .5.

The classifier is the supplied balanced weighted shrinkage LDA implementation. Temperature calibration uses inner held-out predictions. The strict adapter refits preprocessing, graph features, generated labels and classifiers within each inner partition and purges masked embedding duplicates of held-out groups. Five seeds 0–4 are averaged. Standalone Cand2 used expected-F1 decoding; the winning blend instead consumes its label marginals and applies the shared fixed .3 cutoff.

The supplied manifest's `transductive: true` field describes the original supplied configuration. The submitted strict adapter overrides query access. The exact .89102 standalone and .89935 blend artifacts are inductive; do not substitute Dylan's native transductive benchmark values or CSVs.

## Reproduction and verification

The winning final export refit anchored and reused the hash-verified final five-seed Cand2 packet. Its [frozen exporter](experiments/cand2_followup_v1/export_blend.py) reproduces anchored's exact scored predictions and reconstructs the standalone Cand2 CSV hash before writing the blend. Its [export protocol](experiments/cand2_followup_v1/blend_export_protocol.json) and [preparation manifest](experiments/cand2_followup_v1/blend_prepared_submission.json) retain their original pre-submission status; the later score record supersedes it.

The winner's complete benchmark, source dependencies and final probability packets are now available in this checkout. Run the retained verifier from the repository root using Python 3.12 and the [recorded dependencies](experiments/cand2_followup_v1/requirements-lock.txt):

```sh
python part1/experiments/cand2_followup_v1/verify_retained.py
python part1/reporting/build.py
```

These commands check retained measurements and exact artifacts without training. Refit only in a disposable checkout using the frozen study instructions; the exporter intentionally refuses to overwrite an existing scored CSV. The current `part1.ipynb` is the anchored-component notebook. It does not implement the complete winning blend.

## Interpretation

The selected score is .00299 above standalone anchored and .00085 above the previous anchored/LDA blend. The local pooled-CV improvement over anchored is small and does not meet the original paired-win rule. The reported Kaggle outcome drove the user's final choice. [Report guide](REPORT_GUIDE.md) preserves the full comparison and limits of these reused development measurements.
