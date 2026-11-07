# Final-slot alternatives

Nine fixed variations were declared while file 15's score was pending. Its **0.58492** result arrived before these candidates were evaluated and is explicitly recorded in the manifest. File 13 remains better at **0.58465**. No base models were refitted.

Local loss continues to favor stronger tree/trend mixtures, even though the tested direction worsened Kaggle. The independent sharper-neighbor file 16 was prepared from the preceding refinement batch, but the final decision is to hold the remaining slot because the evidence is weak. None of this batch's candidates was exported or submitted.

The smaller interpolations between files 13 and 15 improve local averages versus file 13, but their local averaging benefit relative to the weighted endpoint losses is only about 0.00003–0.00005 on the chronological means. This is evidence for the decision to explore a different direction after file 15's 0.00027 Kaggle loss; it is not a prediction of the unknown interpolation scores. All candidate results remain in the tables below.

## Strict

| model | official | temporal | gap_mean | gap_worst | cv | latest |
| --- | --- | --- | --- | --- | --- | --- |
| file11 | 0.554437 | 0.533926 | 0.561563 | 0.629638 | 0.581169 | 0.458133 |
| file13 | 0.553279 | 0.533405 | 0.561247 | 0.630614 | 0.578966 | 0.458429 |
| file15 | 0.554265 | 0.532700 | 0.560879 | 0.630974 | 0.578254 | 0.456945 |
| path0.25 | 0.553479 | 0.533198 | 0.561121 | 0.630657 | 0.578738 | 0.458029 |
| path0.5 | 0.553710 | 0.533011 | 0.561017 | 0.630731 | 0.578543 | 0.457649 |
| path0.75 | 0.553972 | 0.532845 | 0.560937 | 0.630837 | 0.578382 | 0.457287 |
| path1.25 | 0.554588 | 0.532576 | 0.560845 | 0.631144 | 0.578159 | 0.456623 |
| path1.5 | 0.554944 | 0.532472 | 0.560835 | 0.631347 | 0.578098 | 0.456320 |
| tree0.075 | 0.554125 | 0.532945 | 0.560897 | 0.630381 | 0.578813 | 0.457137 |
| tree0.125 | 0.554456 | 0.532498 | 0.560913 | 0.631640 | 0.577757 | 0.456788 |
| trend0.05 | 0.554458 | 0.532688 | 0.561029 | 0.631278 | 0.578382 | 0.457032 |
| trend0.15 | 0.554089 | 0.532720 | 0.560742 | 0.630693 | 0.578146 | 0.456867 |

## Published

| model | official | temporal | gap_mean | gap_worst | cv | latest |
| --- | --- | --- | --- | --- | --- | --- |
| file11 | 0.555292 | 0.532817 | 0.560799 | 0.627428 | 0.579885 | 0.455325 |
| file13 | 0.553946 | 0.532421 | 0.560552 | 0.628549 | 0.577793 | 0.455926 |
| file15 | 0.555009 | 0.531727 | 0.560191 | 0.628945 | 0.577093 | 0.454507 |
| path0.25 | 0.554164 | 0.532216 | 0.560427 | 0.628601 | 0.577567 | 0.455543 |
| path0.5 | 0.554415 | 0.532032 | 0.560326 | 0.628684 | 0.577375 | 0.455178 |
| path0.75 | 0.554696 | 0.531869 | 0.560247 | 0.628798 | 0.577217 | 0.454833 |
| path1.25 | 0.555353 | 0.531606 | 0.560158 | 0.629123 | 0.577001 | 0.454200 |
| path1.5 | 0.555729 | 0.531505 | 0.560148 | 0.629334 | 0.576944 | 0.453912 |
| tree0.075 | 0.554893 | 0.531933 | 0.560192 | 0.628321 | 0.577613 | 0.454614 |
| tree0.125 | 0.555176 | 0.531563 | 0.560241 | 0.629644 | 0.576634 | 0.454434 |
| trend0.05 | 0.555210 | 0.531724 | 0.560334 | 0.629208 | 0.577234 | 0.454577 |
| trend0.15 | 0.554824 | 0.531738 | 0.560060 | 0.628704 | 0.576970 | 0.454445 |

[Every period](all_scores.csv), [interpolation diagnostics](interpolation_diagnostics.csv), [verification](verification.json) and [frozen manifest](manifest.json). The same historical labels were reused; these are adaptive comparisons, not independent validation.

## Reproduce

```sh
.venv/bin/python part2/run.py sprint_finish.py benchmark
.venv/bin/python part2/run.py sprint_finish_report.py
```

This batch only evaluates local predictions. It neither exports nor uploads a Kaggle submission.
