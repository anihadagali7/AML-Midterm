# Direct comparison of all saved Part Two submissions

This is a comparison run, not a new recipe search. Include numbered files 01–10,
both Kaggle-scored Ani references, and Ani's unsubmitted Phase III trend control.
Freeze the existing ensemble weights, features, hyperparameters, seeds and tree
counts. No early stopping, tuning, new submission files or Kaggle uploads.

Use the ten persisted splits in `experiments/local_benchmark/protocol.json` and
the three persisted longer-gap splits in
`experiments/ani_extensions/decision/gaps/protocol.json`. Every model has exactly
the same fitting and evaluation trial IDs in each comparison. The primary metric
is the equal mean of Phase I, II and III log losses. Pool grouped-CV predictions
within each repeat before averaging the two repeats. Report supplied validation,
three forward windows, their worst period, grouped CV, and the three gap windows
separately. Do not combine these into a newly optimized scalar ranking.

The primary strict track refits all preprocessing on fitting rows only. Reuse
previous Ani/09/10 predictions only after checking original source/input hashes
and exact split membership. Rebuild legacy linear and CatBoost components on each
split: vocabulary, IDF, categorical dictionaries, SVD projections, metadata
frequency mask, recency weights and year cap all use fitting rows only.

The secondary published-recipe track preserves each original recipe's feature
policy: Ani/09/10 use the existing feature-only transductive track; 01–06 and the
linear component of 08 remain training-only as originally implemented. Their
unchanged predictions may be reused across the two tracks. This is a sensitivity
check of the original recipes, not a second common-preprocessing experiment.

Use manifests to reconstruct exact files 01–06; derive 07 as 75% Ani plus 25%
Ani trend, and 08 as 75% Ani plus 25% the original three-linear-model component.
Map 09/10 to their frozen extension recipes. Check the reconstructed official
validation predictions against saved original component probabilities and the
original test CSVs against predictions from the rebuilt official-training model.
This replay checks recipe fidelity; test labels are unavailable and never used.

Record source/data/protocol/submission hashes, package versions, training IDs,
per-row held-out probabilities, per-phase metrics, every window, and paired
trial-bootstrap differences for file 10 against each other submission. Include
both Ani references in the same comparison. Report any reproduction discrepancy.
Confidence intervals are descriptive: these labels were used during earlier
model selection and the intervals do not correct for adaptive experimentation.

The saved Kaggle scores apply only to their exact historical file hashes. Files
01–06 originally used 8,037 training rows in their final submissions; Ani and
07–10 used 9,050. Here every recipe receives the same fitting rows per split.
Thus this benchmark compares algorithms under equal local fitting populations,
without assigning a Kaggle score to a hypothetical refit of an older submission.
