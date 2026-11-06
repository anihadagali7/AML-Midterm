"""Document the common CV protocol and side-by-side existing validation results."""
from pathlib import Path
import json,nbformat as nbf

ROOT=Path(__file__).resolve().parent;B=ROOT/'experiments/benchmark_v1'
s=json.loads((B/'summary.json').read_text());p=json.loads((B/'protocol.json').read_text())
base=next(r for r in s['models'] if r['incumbent'])
nb=nbf.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python (AML Midterm)',language='python',name='aml-midterm')
nb.cells=[nbf.v4.new_markdown_cell(f'''# One common CV benchmark for model comparison

The primary comparison now uses the same saved grouped five-fold splits, repeated with seeds **20261126, 20261127 and 20261128**, for every model. The current hybrid scores **{base['cv_mean']:.6f} mean CV sample-wise F1**, with repeat standard deviation **{base['repeat_std']:.6f}**. Its previously measured validation F1 remains **{base['validation_f1']:.6f}**.

This benchmark runs no additional validation evaluation and makes no model promotion. Historical CV values are preserved but are not substitutes for this common comparison score. The Kaggle score is unmeasured.'''),
nbf.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd,numpy as np\nroot=Path.cwd();bench=root/'experiments/benchmark_v1'\nsummary=json.loads((bench/'summary.json').read_text())\ntable=pd.read_csv(bench/'comparison.csv')\ntable[['candidate','cv_mean','repeat_std','paired_cv_difference','repeats_improved','validation_f1']]"),
nbf.v4.new_markdown_cell('''## What is fixed

The protocol stores the actual fold memberships and trial IDs, not only random seeds. A SHA-256 hash identifies the training CSV and another identifies the complete fold assignments. Identical embeddings, rounded to ten decimal places, form one group; multilabel stratification balances the groups across folds. Each gold row is held out exactly once per repeat. Matching originally masked duplicates are removed from the fold's training pool.

Recipes and final decoding rules are frozen before benchmarking. Each fitted transform sees only the outer training gold and permitted unlabeled training examples. Inner calibration, feature selection or parameter tuning, when part of a recipe, must remain inside that outer training fold. The runner passes query features without label columns and does not select a threshold from outer-fold outcomes.

For each repeat, the metric is sample-wise F1 over the complete out-of-fold prediction matrix, giving equal weight to each trial. The headline is the mean of the three repeat scores. The repeat standard deviation describes split sensitivity; it is not a confidence interval. Macro F1, per-reason F1, every fold's score, raw scores and decoded OOF predictions are also saved.

All models use the same outer folds, so their repeat-by-repeat differences are paired. A positive mean difference with improvement in at least two of three repeats is a descriptive shortlist criterion, not a significance test or an automatic replacement rule.'''),
nbf.v4.new_code_cell("rows=[]\nfor r in summary['models']:\n    result=json.loads((bench/(r['result_key']+'.json')).read_text())\n    for seed,score in zip(summary['seeds'],result['repeat_scores']):\n        rows.append(dict(candidate=r['candidate'],seed=seed,sample_f1=score['sample_f1'],macro_f1=score['macro_f1']))\npd.DataFrame(rows).pivot(index='candidate',columns='seed',values='sample_f1')"),
nbf.v4.new_markdown_cell('''## Validation and honest interpretation

The comparison table copies already measured validation scores from the frozen recipe records. These recipes have not changed, so this benchmark does not refit or rescore validation. Incoming recipes should first pass through the same CV comparison; shortlist them, freeze them, and then evaluate validation in a small recorded batch.

Agreement between CV and validation is useful evidence, but these are development results. Earlier work used the training labels and repeatedly consulted validation. New split seeds do not turn those labels into independent test data. Continued search can also overfit the fixed CV folds. The later-year Kaggle population can differ from both the labeled training population and validation, so any chronological stress test should be reported separately and should not replace the primary CV metric.

The retained hybrid keeps its 0.912393 validation score. There is no final all-label refit and no transductive fitting on validation/test inputs in this main benchmark. The separately authorized pooled CV in notebook 11 uses original validation rows only within the training portion of a pooled fold; its results remain a distinct supplementary measurement.'''),
nbf.v4.new_markdown_cell('''## Adding a candidate

Save a JSON object with `name`, `config`, and `rule` for an implemented fold-local recipe, then run:

```sh
.venv/bin/python benchmark_cv.py --register candidate.json
```

An optional `validation` object can reference an already recorded evaluation. Otherwise its validation entry stays empty. New methods first need an adapter compatible with the experiment engine's training-position, query-feature and excluded-group interface; registration alone cannot implement an arbitrary external model.

Running `.venv/bin/python benchmark_cv.py` regenerates the comparison table from cached results where the protocol, recipe, rule and source hashes match. `--verify-only` checks the saved folds, training hash, group separation and duplicate-exclusion counts. Changing a recipe requires a new versioned name. Keep the protocol fixed across searches.

Source is embedded in notebook 04. `AGENTS.md` records this convention for future work in the project. For the methodological background, see [scikit-learn's cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).'''),
nbf.v4.new_code_cell("protocol=json.loads((bench/'protocol.json').read_text())\n{k:protocol[k] for k in ['version','training_sha256','folds_sha256','seeds','folds_per_repeat','metric','limitation']}"),
nbf.v4.new_code_cell("assert summary['validation_evaluations_performed']==0\nassert summary['model_promotions_performed']==0\nprint('Common comparison complete; existing validation metrics reused; current submission retained.')")]
nbf.write(nb,ROOT/'midterm/10_common_cv_benchmark.ipynb')
print(json.dumps(base,indent=2))
