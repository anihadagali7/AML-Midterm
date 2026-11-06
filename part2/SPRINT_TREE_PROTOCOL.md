# Fixed recency-tree follow-up

Declared after the first 32-recipe batch, before any new tree fitting or scores.
The balanced legacy ensemble improves both time-window averages when mixed with
file 10, but larger weights regress on the latest forward period and the oldest
fitting population's gap test. Hypothesis: recent weighting in the tree component
can retain complementary nonlinear predictions with less chronological drift.

Use the original training-only feature dictionaries, TF-IDF, SVD and metadata
mask from each of the exact 13 saved splits. Fit one CatBoost model per split:
depth 6, 1,000 trees, learning rate .035, L2 6, seed 42, no year feature, no early
stopping, and exponential recency tau=3 with weights normalized to mean one.
No further tree settings are searched in this batch.

Freeze four probability blends before evaluating: 10%/20% direct recent-tree
weight against file 10; and 20%/30% weight on an endpoint that averages the
original three-model linear component with this recent tree. Compare against
file 10 and file 11 on every saved split under both existing preprocessing
policies. The tree stays training-only in both tracks. Report all periods,
grouped CV and official validation; use the sprint decision criteria. Final
training uses all 9,050 labeled rows only if a candidate warrants export.

This is adaptive research following the first batch; the same held-out labels
have been consulted before. No Kaggle test labels or individual test outcomes
are available or inferred. Slot selection remains contingent on actual returned
Kaggle scores, with at most four additional submissions in the user's window.
