# First Part Two submission result

User-reported Kaggle log loss for `03_phase_specialized.csv`: **0.59562**. The exact submitted artifact hash is preserved in `../../submissions/leaderboard_results.json`. Lower is better.

The comparison model summarized in `../../PNG image.png` reportedly scored **0.58688**. Its notebook and prediction files have not been provided. Our result is **0.00874 worse**. It is also 0.00034 worse than the original reported 0.59528 benchmark.

03's equal-phase validation log loss was 0.551255, versus the comparison image's rounded 0.556. Its training-only chronological holdout loss was 0.553285. The apparent local advantage did not transfer to the leaderboard. Neither local score should be presented as evidence that 03 beats the reference on Kaggle.

The final CSVs reproduce byte for byte from the saved model bundle. Model preprocessing is fitted on training data only; identifiers, outcome labels supplied at inference, and reason information do not change the predictions. This does not rule out modeling weaknesses. Temporal distribution shift, repeated model selection on the same local evaluations, calibration, and component-specific errors remain possible explanations; one aggregate leaderboard result cannot distinguish them.

## Next comparison

Try **02_text_heavy.csv** next. Relative to 03, its tree fractions change from 25%/50%/75% in Phases I/II/III to 25%/25%/25%. Phase I predictions are identical, making this a useful test of less tree influence in the later phases. An improvement would support exploring that direction, not prove that trees alone caused the original gap.

Wait for that result before choosing another submission. 01 is another model-balance comparison; 04 tests capped year effects and produces higher average success probabilities. Neither has an established Kaggle advantage.

The 0.58688 model's source and validation/test predictions would support a more informative feature, calibration, and ensemble comparison. Keep it as the best reported reference until another submitted model actually improves its score.

## Update: second submission

User-reported Kaggle score for `02_text_heavy.csv`: **0.58993**, an improvement of **0.00569** over 03 and still **0.00305** worse than the 0.58688 reference. Its exact file hash is recorded in the ledger. This supports further testing of reduced tree influence in Phases II/III, without establishing which phase caused the gain.

Two prospective follow-ups preserve Phase I exactly as in 02 and reuse the same fitted components. **05_text_later_phases.csv** removes trees from II/III. **06_lighter_tree_blend.csv** uses 12.5% trees there. Try 05 next; hold 06 until its result. These variants have worse local losses than 02 and are motivated by the observed leaderboard direction, not by a new independent evaluation. The original four files are unchanged.

## Superseding update: Ani source and repeatable benchmark

**Hold 05/06.** Ani's branch is now available at commit `1b8bd11c47428fb5453e53cdc275542b92a3b347`; his 0.58688 submission reproduces within 3.6e-9 probability. It uses learned eight-model stacking, exponential recency weighting, feature-only transduction, and final supervised refitting on train+validation. Our initial files used training labels only, so the gap cannot be assigned to the algorithm alone.

The completed benchmark compares ten saved trial-level splits: supplied validation, three future-year windows, and two repeats of grouped three-fold CV. Strict fold-local preprocessing and faithful published-recipe transduction are separate tracks. Our linear endpoint is worse than Ani across chronological windows; removing trees is no longer the preferred next test simply because 02 improved over 03.

A conservative 25% interpolation toward Ani's Phase III trend alternative has small favorable temporal estimates and better grouped CV in both tracks. Temporal paired intervals include zero, so it remains exploratory. Optional 07 preserves the winning file as its anchor; 08 is an Ani-heavy linear blend, less consistent across time. No new Kaggle improvement is claimed. See `../local_benchmark/RESULTS.md`, `../../LOCAL_TESTING.md`, and `../../local_benchmark.ipynb`.
