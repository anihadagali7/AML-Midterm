"""Summarize the completed stability, feature and nested specialist follow-ups."""
from pathlib import Path
import json,nbformat as nbf
ROOT=Path(__file__).resolve().parent;E=ROOT/'experiments'
baseline=json.loads((E/'phase22-baseline.json').read_text())
current=json.loads((E/'current-best.json').read_text())
results=sum([json.loads((E/f'phase{n}-validation-results.json').read_text()) for n in [22,23]],[])
screen22=json.loads((E/'phase22-screen.json').read_text())
screen23=json.loads((E/'phase23-screen.json').read_text())
repeat22=json.loads((E/'phase22-repeat-results.json').read_text())
repeat23=json.loads((E/'phase23-repeat-results.json').read_text())
control=next(r for r in repeat22 if r['bucket']=='control')
summary=dict(screened_configurations=len(screen22)+len(screen23),phase22_screened=len(screen22),
    phase23_screened=len(screen23),validation_candidates=len(results),
    baseline_validation_f1=baseline['best_validation_f1'],current_validation_f1=current['best_validation_f1'],
    best_followup_validation_f1=max((r['validation']['sample_f1'] for r in results),default=None),
    previous_baseline_retained=current['candidate']['config']==baseline['candidate']['config'],
    baseline_additional_cv=control['ordinary_cv'],baseline_additional_cv_std=control['cv_std'],
    additional_cv_seeds=[20261111,20261112,20261113],validation_labels_used_for_training=0,
    final_train_plus_validation_refit_performed=False,target_met=current['best_validation_f1']>.92735)
(E/'phase22-23-summary.json').write_text(json.dumps(summary,indent=2))
refs=json.loads((E/'references.json').read_text())
for title,url,use in [
 ('Breiman: Bagging Predictors','https://statistics.berkeley.edu/tech-reports/421','Motivation for averaging fixed bootstrap runs; no best-seed selection.'),
 ('scikit-learn: StackingClassifier','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html','Meta-learners must learn from out-of-fold component predictions; custom implementation uses embedding-grouped inner folds.'),
 ('Dembczynski et al.: An Exact Algorithm for F-Measure Maximization','https://proceedings.neurips.cc/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html','Exact expected sample-F1 decoding under an estimated label-pattern distribution; estimated probabilities need not be correct.')]:
 if not any(r['url']==url for r in refs):refs.append(dict(title=title,url=url,use=use))
(E/'references.json').write_text(json.dumps(refs,indent=2))
nb=nbf.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python (AML Midterm)',language='python',name='aml-midterm')
nb.cells=[nbf.v4.new_markdown_cell(f'''# Follow-up: stability, auxiliary features and nested specialists

The starting hybrid scored **{baseline['best_validation_f1']:.6f} validation sample-wise F1**. The current best is **{current['best_validation_f1']:.6f}**. Requested target: strictly above 0.92735. These scores have not been measured on Kaggle.

This follow-up tested {len(screen22)} stability/feature/score-combination configurations and {len(screen23)} nested specialists. All 78 validation rows remain excluded from fitting. The final train-plus-validation refit remains deferred.

The full search is adaptive and the validation set has been consulted repeatedly. Fixed-candidate CV audits and small frozen validation batches reduce some leakage risks but do not make the entire model-selection process unbiased.'''),
nbf.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd,numpy as np\nimport matplotlib.pyplot as plt\nroot=Path.cwd()\nsummary=json.loads((root/'experiments/phase22-23-summary.json').read_text())\nsummary"),
nbf.v4.new_markdown_cell('''## Why these experiments

The new tri-training component previously had no auxiliary-feature trials and only one bootstrap initialization. It also combined hard labels with the SVM, discarding confidence information. The earlier broad search did not test these exact combinations.

Training-only OOF errors identified Other as the largest remaining weakness: approximately 12.3 false positives and 8.3 false negatives per complete OOF pass, averaged over the original three split seeds. This motivated a dedicated Other classifier without selecting individual validation mistakes.

We used fixed ensembles of the first three or all five predetermined bootstrap seeds, never selecting the best seed. Feature blocks were added separately at two weights: eligibility criteria (word and character TF-IDF), diseases, drugs, ICD codes, drug targets/actions/pathways, molecular fingerprints/descriptors, year and field counts. Small changes to regularization, bandwidth, pseudo-label weight, confidence gates and preprocessing were also evaluated.

The score-combination variants align the original component thresholds before averaging. Their values are heuristic scores, not calibrated probabilities. The previous per-reason mask remained fixed during the feature and parameter screens.'''),
nbf.v4.new_code_cell("screen=json.loads((root/'experiments/phase22-screen.json').read_text())\nrows=[]\nfor r in screen:\n t=r['config']['tri']\n if r['bucket']=='features': rows.append(dict(feature=t['features'][0],weight=t['feature_weight'],ordinary_cv=r['ordinary'],weighted_robust_cv=r['robust']))\npd.DataFrame(rows).sort_values('weighted_robust_cv',ascending=False)"),
nbf.v4.new_markdown_cell('''## Nested specialists

Each outer training fold contains its own 15-fold grouped inner CV. The inner models generate ten score features: five from the SVM and five from tri-training. No row's label is fitted by the model that generates its inner OOF score features. Matching masked embedding duplicates of both inner and outer held-out groups are excluded.

Regularized logistic classifiers use these scores to predict Other alone, Recruitment and Other, or all five reasons. Other-only variants also test score interactions and conservative blending. A multinomial pattern model supplies a separate decoder that chooses the label set with maximum expected sample-wise F1. Earlier calibration experiments used only the old SVM; these experiments use both current components.

The specialists' complexity settings are selected from outer training CV. This is nested feature generation, not an independent nested rerun of the entire historical hyperparameter search.'''),
nbf.v4.new_code_cell("rep22=json.loads((root/'experiments/phase22-repeat-results.json').read_text())\nrep23=json.loads((root/'experiments/phase23-repeat-results.json').read_text())\nrows=[dict(batch=batch,variant=r['bucket'],mean_cv=r['ordinary_cv'],split_std=r['cv_std'],robust_cv=r['robust_cv']) for batch,rs in [('22',rep22),('23',rep23)] for r in rs]\npd.DataFrame(rows)"),
nbf.v4.new_markdown_cell('''## Validation and interpretation

Candidate configurations and decision rules were written to frozen JSON files before loading validation labels. Only finalists meeting the prespecified training-CV screen were evaluated on validation. All results, including failures to improve, appear below.

The control's scores on the additional split seeds differ from its earlier three-seed CV. That variability is useful evidence of the uncertainty created by just 157 gold rows; do not compare scores from different split sets as if they were paired measurements.'''),
nbf.v4.new_code_cell("results=sum([json.loads((root/f'experiments/phase{n}-validation-results.json').read_text()) for n in [22,23]],[])\nvalidation=pd.DataFrame([dict(candidate=r['name'],cv=r['ordinary_cv'],validation_f1=r['validation']['sample_f1'],macro_f1=r['validation']['macro_f1']) for r in results])\nvalidation"),
nbf.v4.new_code_cell("fig,ax=plt.subplots(figsize=(7,3.5))\ny=np.arange(len(validation))\nax.scatter(validation.validation_f1,y,s=65,label='Follow-up candidate')\nax.axvline(summary['baseline_validation_f1'],color='#197144',label='Starting hybrid')\nax.axvline(.92735,color='#9b3e27',linestyle='--',label='Requested target')\nfor i,s in enumerate(validation.validation_f1):ax.annotate(f'{s:.5f}',(s,i),xytext=(5,7),textcoords='offset points')\nax.set(yticks=y,yticklabels=validation.candidate,xlim=(.8,.98),xlabel='Validation sample-wise F1')\nax.margins(y=.25);ax.invert_yaxis();ax.legend(loc='lower left',fontsize=8)\nfig.tight_layout();fig.savefig(root/'figures/current_phase22_23_comparison.png',dpi=180)\nplt.show()"),
nbf.v4.new_markdown_cell('''## Reproduction and integrity

Protocols, exact configurations, OOF scores and frozen validation batches are under `experiments/phase22*` and `experiments/phase23*`. All source is embedded in notebook 04. The phase22 core was checked against phase21 with matching defaults. The nested specialist passed an uncached held-out-label perturbation check; caches were cleared to prevent them from masking leakage. Raw inputs and gold labels remain unchanged.

Method references: [bootstrap aggregation](https://statistics.berkeley.edu/tech-reports/421), [out-of-fold stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), and [exact F-measure decoding](https://proceedings.neurips.cc/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html). No external trial records or labels were retrieved.'''),
nbf.v4.new_code_cell("json.loads((root/'experiments/phase23-integrity-checks.json').read_text())")]
nbf.write(nb,ROOT/'midterm/06_stability_and_specialists.ipynb')
print(json.dumps(summary,indent=2))
