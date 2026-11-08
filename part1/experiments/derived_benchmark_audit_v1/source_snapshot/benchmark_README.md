# Benchmark: Michael's model vs ours, plus blends

- **Michael's model:** `anchored_solution.py`, copied unchanged from `AML-Midterm/part1/`. Logistic regression, LDA and nearest neighbours blended together, with GMM clustering helping fill in the missing labels.
- **Our candidates:** cand1 and cand5 from `../midterm/part1/submissions/`.
- **Splits:** the framework's exact folds (10 repeats, seed 1), the same ones as the v2 leaderboard. Michael's model only ever sees its permitted rows (fold-train labels plus the 1,014 unlabeled rows), never held-out or test rows.

## Scripts (each runs in a few seconds; use the framework's venv)

| Script | What it does | Output |
|---|---|---|
| `run.py` | Michael's model alone, compared to cand1 and cand5 | `results.json` |
| `ensemble.py` | Blends. Per-label scores are averaged (ours: combination posterior → label marginals), then decoded with Michael's rule: top label plus any label ≥ 0.3 | `ensemble_results.json` |
| `make_blend_csv.py` | Trains on all 235 labeled rows and writes test CSVs | `submissions/` |

## Results (sample F1)

- **A** = 5-fold CV × 10. **B** = chronological check (train on earlier years, test on later), averaged over the 2004/2005/2006 cuts. **C** = official val. **J** = (A+B)/2.
- The SEs on the differences are paired.

| Model | A | B | C | J | ΔJ vs cand1 |
|---|---|---|---|---|---|
| Michael alone | **.925** | .847 | **.891** | .886 | −.007 (±.013) |
| cand1 alone | .909 | **.876** | .882 | .892 | — |
| cand5 alone | .909 | .869 | .865 | .889 | −.003 (±.011) |
| **Michael + cand1, equal weight** | .919 | .873 | .885 | **.896** | **+.004 (±.006)** |
| Michael + cand1 + cand5, equal weight | .919 | .875 | .876 | .897 | +.005 (±.007) |
| Michael + cand5, equal weight | .922 | .851 | .878 | .887 | −.006 (±.012) |

- **Michael's model** wins random CV. It loses the chronological check because it has no correction for the test period's different label frequencies.
- **Michael + cand1** keeps most of both strengths. Its gain over either model alone is within noise, but it doesn't clearly lose on any check.
- **Michael's numbers are probably a bit inflated.** His model was iterated many times on these same 235 labeled rows.
- **The blend isn't tuned.** Weights and cutoffs other than equal weights and 0.3 scored up to .900, but picking one of those would be tuning on 235 rows, so the CSVs use equal weights and Michael's original 0.3 cutoff.

## Submission CSVs (`submissions/`)

All are trained on all 235 labeled rows plus the unlabeled pool. When retrained for these blends, cand1 and cand5 reproduce their original CSVs exactly.

| File | Notes |
|---|---|
| `blend_michael_cand1.csv` | **Suggested.** Matches cand1 on 95.5% of test rows |
| `blend_michael_cand1_cand5.csv` | Matches cand1 on 92.9% |
| `michael_anchored_alone.csv` | Matches cand1 on 84.3% |
