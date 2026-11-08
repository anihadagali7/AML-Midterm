# What each metric and check means, what it proves, and what it does not

The setting drives every choice here:
- only **235 labeled rows** (157 labeled-train rows plus 78 val rows), and
- a **time shift**: labeled rows are mostly 1993–2009, test rows are 2010–2016, and the label mix differs between them.

With so few labels, every number has a wide error bar. The time shift means a score on randomly held-out rows can be
optimistic for the test set.

---

## Sample-F1 (the Kaggle metric)

**What it is.** For each trial, compare the predicted reason set with the true set:
F1 = 2·|predicted ∩ true| / (|predicted| + |true|). Average over trials.

**Worked examples** (true set = {Recruitment, Other}):

| predicted | overlap | F1 |
|---|---|---|
| {Recruitment, Other} | 2 | 2·2/(2+2) = 1.00 |
| {Recruitment} | 1 | 2·1/(1+2) = 0.67 |
| {Recruitment, Other, Safety} | 2 | 2·2/(3+2) = 0.80 |
| {Safety} | 0 | 0.00 |

**Intuition.**
- Most trials have exactly one reason (cardinality 1.06 in labeled-train, 1.12 in val), so sample-F1 behaves almost
  like accuracy, with partial credit for multi-label rows.
- Predicting one correct label for a two-label trial still earns 0.67.
- An empty prediction scores 0, and every trial has at least one true reason, so our decoders always predict at least
  one label.

**Why decoding matters.** Given probabilities over the 8 observed label combinations, you can either:
- predict the single most probable combination ("MAP" in our tables), or
- pick the subset with the highest *expected* F1 ("exp-F1").

Example: if a model is split 50/50 between Safety and Efficacy, predicting {Safety} has expected F1 0.5, but
predicting {Safety, Efficacy} has 0.5·(2/3) + 0.5·(2/3) = 0.67. So exp-F1 hedges when the model is torn.
- Alone, it is roughly neutral (LR: J 0.857 MAP vs 0.860 exp-F1).
- Combined with prior correction it is clearly better (LR + SLD25: B 0.863 with exp-F1 vs 0.839 with MAP;
  `results/leaderboard.csv`).

**What it proves:** how well the predicted *sets* match per trial, which is exactly what Kaggle scores.
**What it does not:** it is dominated by Recruitment, which is 58–64% of labeled rows. Always predicting Recruitment
already scores 0.62 on val. A model can look good while being poor on rare reasons.

---

## Per-label F1 and macro-F1

**What it is.** For each reason separately, treat it as a yes/no task across trials and compute the usual binary
F1. Macro-F1 is the unweighted mean over the 5 reasons.

**Why we report it.** The assignment asks for it, and it exposes what sample-F1 hides: rare-label quality.
- Always predicting Recruitment: sample-F1 0.62, macro-F1 0.156.
- cand1 on val: sample-F1 0.882, macro-F1 0.769.

Per-label F1 for cand1:

| protocol | Safety | Funding | Efficacy | Recruitment | Other |
|---|---|---|---|---|---|
| A (CV) | 0.912 | 0.809 | 0.887 | 0.970 | 0.601 |
| B (chrono) | 0.931 | 0.727 | 0.898 | 0.945 | 0.646 |
| C (val) | 0.941 | 0.667 | 0.720 | 0.971 | 0.545 |

**Caveats.**
- **Funding has 12 positives among the 235 labeled rows and only 4 in val.** One val row changes Funding F1 by
  roughly 0.1 to 0.25, and macro-F1 by a fifth of that. Treat Funding F1 and macro-F1 on val as very noisy.
  Bootstrap SE of cand1's val macro-F1: about 0.07.
- **Other** is the genuinely hard label (about 0.60 everywhere). It overlaps Recruitment: 9 labeled rows are
  Recruitment+Other.

---

## Protocol A: repeated stratified cross-validation

**What it is.** Split the 235 labeled rows into 5 folds, stratified on the label combination. Train on 4 folds and
predict the 5th, rotating, then repeat the whole thing R times with different shuffles (R=4 inside the search, R=10
for the reported numbers, on a different seed from the search). Every row gets predicted R times. We report the mean
sample-F1.

**Why.** It uses every label for both training and testing, so it is the least noisy number available: SE about
0.015 to 0.017.

**What it proves:** how well a pipeline does on rows from the *same mix* as the labeled data.
**What it does not:** random folds contain **no time shift**. That is why SLD prior correction looks neutral or
slightly harmful on A (LR 0.902 → 0.899) while helping on B. A cannot say whether a model will survive the move to
2010–2016.

---

## Protocol B: chronological holdout, averaged over three cuts

**What it is.** Train on labeled rows that started before a cut year and predict the rows from the cut year on. We
use cuts 2004, 2005 and 2006 and average the three scores. The unlabeled pool, and in transductive mode all rows'
features, stays available, just as at submission time.

| cut | train rows | eval rows | Recruitment share, train → eval |
|---|---|---|---|
| 2004 | 126 | 109 | 0.84 → 0.41 |
| 2005 | 137 | 98 | 0.82 → 0.40 |
| 2006 | 149 | 86 | 0.77 → 0.42 |

**Why.** It is the only protocol that imitates the real problem: training on earlier trials, predicting later ones
with a different label mix.

**Why average three cuts.** A single cut has about 98 eval rows (SE about 0.03), and decisions flipped between cuts.
Averaging reduces cut-specific luck; the cuts share most rows, so it is not three independent samples. Example: for
plain LR, B2005 alone is 0.830 but the 3-cut mean is 0.812.

**What it proves:** whether a pipeline survives a large label-prior shift in the right direction (less Recruitment,
more Safety, Efficacy and Other).

**What it does not:**
- The shift is **exaggerated**. The eval rows include the 40 "rare-enriched" labeled rows, so Recruitment drops to
  about 0.40, while the test period is estimated at about 0.57. B can therefore reward over-correction.
- It is noisy: SE about 0.03, so gaps under about 0.05 are not meaningful on B alone.

---

## Protocol C: the official val split

**What it is.** Train on the 157 labeled-train rows (plus the unlabeled pool), predict the 78 val rows.

**Why.** The assignment says to select using `part1_val.csv`, and it is the reported val number.

**What it proves:** performance on one specific random sample of 78 rows.

**What it does not:**
- SE is about 0.03 to 0.04, and one row is worth 0.0128.
- The 157 labeled-train rows contain **zero Safety+Efficacy trials** while val has 5. Any model trained only on them
  can never predict that combination as a unit (exp-F1 decoding can still output both labels).
- So C was logged and used as a tie-breaker, never optimized.

---

## J: the search objective

**What it is.** J = 0.5·A + 0.5·B.

**Why.** A alone ignores the shift. B alone is noisy and exaggerates the shift. Averaging penalizes configs that
only win on one side.
- Prior correction: helps B a lot (LR: B 0.812 → 0.863), costs A almost nothing (0.902 → 0.899), so J rises
  0.857 → 0.881.
- Over-aggressive correction would be held back by A.

**What it does not prove:** it is a heuristic weighting. Nothing guarantees that the test set sits halfway between
"no shift" and "B's shift". It roughly does: estimated test Recruitment 0.57, between A's 0.64 and B's 0.40.

---

## Standard errors and paired standard errors

**SE of a score.** Per-trial F1 values have a standard deviation of about 0.27, because most trials score 0 or 1.
SE ≈ 0.27 / √n:

| set | n | SE | a gap is meaningful at about |
|---|---|---|---|
| official val (C) | 78 | 0.031–0.038 | 0.06–0.08 |
| one chrono cut | 98 | 0.027–0.035 | 0.05–0.07 |
| pooled CV (A) | 235 | 0.015–0.018 | 0.03 |

**Paired SE: the right tool when comparing two models.** Two models scored on the *same* rows mostly agree, so the
noise largely cancels. Take the per-row difference in F1 and compute its SE.
- Example: on val, cand1 vs plain LR differ on only 10 of 78 rows. The gap is +0.045 with paired SE 0.024, so it is
  meaningful at about 1.9 SE. The unpaired SEs (0.030 and 0.038) would have suggested "no difference".
- For J we combine the paired A and B differences. The 3 chrono cuts share rows, so they are counted as one cut's
  worth of rows.

**What SEs do not capture:**
- the randomness of the CV splits themselves (we also report the spread across repeats, which is small),
- optimism from having *selected* the config on the same data. That needs the nested check below.

---

## The one-standard-error (1-SE) rule

**What it is.**
1. Find the config with the best J.
2. Take every config whose J is within one SE of it.
3. Among those, pick the **simplest**: fewest active components (synthetic features, labelers, extra blocks, prior
   correction, temperature).

We also required B to be no worse than plain LR's B by more than one SE, and that the pipeline satisfies the
assignment rule. (Hastie, Tibshirani & Friedman, *ESL* §7.10.)

**Worked example from our run:**
- Best rule-compliant J = 0.899, SE about 0.016, so the threshold is 0.883. 37 configs cleared it.
- The simplest of them (complexity 1.75: balanced LDA + C&S pseudo-labels + temperature + exp-F1) has J 0.892. That
  is cand1.
- The argmax (cand2) adds K-means and kNN-vote features for +0.007, which is within noise.

**Why.**
- With 235 labels and thousands of configs, the top score is partly luck: Cawley & Talbot (2010) call this
  "overfitting in model selection".
- Re-evaluating on a fresh seed already shrank search scores by about 0.001 on average
  ([fig_selection.png](../../midterm/part1/figures/fig_selection.png)).
- Preferring the simplest config within noise trades a little expected score for much less variance.

**What it does not prove:** that the chosen config beats the anchor. The rule only says it is not detectably worse
than the best. Robustness check: the paired-SE version of the rule picks the same config.

---

## The nested search-overfitting estimate

**What it is.** It tests the *whole selection procedure*, not one config.
1. Hide some labeled rows.
2. Run the entire procedure on the rest: TPE search (about 400 trials), re-evaluation of the top configs, then the
   1-SE rule.
3. Score the winner on the hidden rows, which the procedure never saw.

We did this two ways (`results/search/nested/`):
- **5 random outer folds**, pooling the 5 held-out sets (all 235 rows).
- **1 chronological outer split**: the procedure saw only trials before 2005, with internal chrono cuts at
  2001/2002, and was scored on the 98 later trials.

**Results:**

| entry | inner estimate (A) | held-out score, 5 folds | held-out score, chrono split |
|---|---|---|---|
| procedure's 1-SE choice | 0.898 | 0.901 ± 0.018 | **0.762** |
| procedure's argmax | 0.902 | 0.896 ± 0.018 | 0.760 |
| fixed anchor (LR + SLD25 + exp-F1) | 0.891 | **0.913 ± 0.016** | **0.858** |
| fixed plain LR | 0.889 | 0.895 ± 0.019 | 0.830 |

Paired difference, 1-SE choice minus anchor, on the 5 folds: −0.012 (SE 0.011). Figure:
[fig_nested.png](../../midterm/part1/figures/fig_nested.png).

**What it proves.**
- **Selection optimism is small.** The winner's inner estimate is within about 0.006 of its held-out score.
- **Searching did not beat the fixed anchor** on rows it never saw.
- **The search can only find what its validation data shows.** The internal 2001/2002 cuts contain almost no prior
  shift: both sides are mostly Recruitment. So configs without any prior correction looked best, and they failed
  under the real shift. This is why the main search's B protocol, which does contain the shift, matters. It is also
  why we ship the anchor-like cand5 as a hedge.

**What it does not prove.**
- It evaluates the procedure, not cand1 itself. Scoring cand1 on those splits is "contaminated", because cand1 was
  chosen using all 235 rows. cand1 does score 0.890 on the chrono split, but it was selected partly on those rows.
- Each nested search had about 400 trials on fewer rows than the main search, so it is a noisier imitation of it.

---

## The leak test

**What it is.** For a fold:
1. Run the pipeline normally.
2. **Overwrite the labels of every labeled row not in that fold's training set with random labels.**
3. Run it again.

If any component read a label it should not see, the predictions on the eval rows would change.
- Results: 25 configs × 3 fold types (a CV fold, the 2005 chrono fold, the val fold). Predictions were
  **bit-identical** every time (largest change 0).
- **Positive control:** a deliberately planted leak, kNN votes computed from all labeled rows, was detected (largest
  probability change 0.16). So the test can catch a leak.
- File: `results/leak_test.txt`.

**Why it matters here.** Semi-supervised pipelines have many places to leak: cluster-to-label maps, graph clamping,
pseudo-labels, kNN votes, stacker meta-features. With 235 labels, a small leak would easily produce a fake +0.02.

**What it proves:** no component reads held-out labels, for the configs tested.

**What it does not prove:**
- It checks label leakage, not feature leakage. Using test *features* without labels is a deliberate, toggleable
  choice ("transductive"). Turning it off costs 0.012 J for cand1 (`results/exp_transd.csv`).
- It covers the configs tested, not every possible config in the search space. All component types are covered,
  though.
