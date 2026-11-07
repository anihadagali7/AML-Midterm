# Extending Ani's outcome model: second research round

The previous round mostly changed meta-model weights/calibration and tested
slower recency. It improved random grouped CV but not forward prediction. Its
promotion rule also required near-best random CV against the unweighted model,
even though that model has worse chronological and reported Kaggle performance.
The user's request to continue opens this separate experiment; historical
results and the previous rule are preserved, not retrospectively relabeled.

## Fixed comparison and decision

Reuse all ten original saved splits, labels, trial groups, and equal-phase log
loss. Compare both Ani winners and the existing Phase III trend control. Keep
strict fold-local preprocessing and published feature-transduction separate.
Use exactly the same predictions for the reproduced controls.

The prediction task concerns later cohorts. Rank candidates primarily by their
three-window chronological mean. A shortlist candidate should improve this mean
by at least 0.001 against the confirmed recency winner, improve at least two
windows, keep worst-window loss within 0.005, and keep grouped CV within 0.003
of that same winner. The unweighted winner's stronger random-CV result remains
reported, but is no longer a mandatory veto. This correction is fixed before
scoring new recipes; it does not convert old results into new evidence.

Freeze at most two recipes before evaluating the three existing gapped forward
windows. Those labels have already been consulted; the new candidate predictions
on these splits have not. They test robustness and cannot provide an untouched
generalization estimate. Require corroborating direction across strict and
published tracks and report paired trial bootstrap intervals against both
winners. Do not pretend the bootstrap corrects the adaptive research history.

## Mechanistic proposals

1. Stronger recency (tau=1.5 and tau=2) in all supervised base learners, plus
   fixed halfway blends with Ani. Previous work only tested tau=3 or slower.
2. Recent neighbors and clinical-context similarity. Ani's kNN ignores recency,
   uses only eligibility text, fixes k=50, and shrinks to a 50% prior. Compare
   phase-specific empirical priors, recent weighting, several neighborhood
   sizes, and joint eligibility/disease/drug/ICD similarity. Label-based neighbor
   statistics are rebuilt inside every grouped inner fold.
3. More recent/nonnegative calibration and shallow nonlinear stacking. Learn
   nonlinear relationships among the base probabilities without adding external
   data. The recipe registry records every setting before the batch is scored.

Further experiments, if required by evidence, get separate named registries and
their adaptive context is documented. Do not alter candidate definitions under
an existing result directory. Efficiently reuse existing out-of-fold base
predictions and controls where fitting boundaries and source hashes agree.

## Constraints and final delivery

Follow the assignment: no deep learning, no retrieval of the original labeled
dataset, no reverse engineering test outcomes, no identifiers or Part One reason
inputs as predictors. Training-fold-only preprocessing is the primary track.
The published track is explicitly transductive and never uses evaluation labels.
Final models use the same 9,050 labeled rows as Ani, after separate held-out
evaluation. Any new Kaggle file has an unknown score until actually submitted.
Keep all failed proposals, source hashes, exact predictions, and fitting IDs.
