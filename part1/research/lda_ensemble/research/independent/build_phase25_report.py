"""Document the three requested SVM refinements and matched ablations."""
from pathlib import Path
import json,nbformat as nbf,numpy as np
ROOT=Path(__file__).resolve().parent;E=ROOT/'experiments'
baseline=json.loads((E/'phase25-baseline.json').read_text());current=json.loads((E/'current-best.json').read_text())
results=json.loads((E/'phase25-validation-results.json').read_text())+json.loads((E/'phase25b-validation-results.json').read_text())
frozen=json.loads((E/'phase25-validation-frozen.json').read_text());configs=json.loads((E/'phase25-configs.json').read_text())
summary=dict(raw_configurations=len(configs),families={family:sum(c['model']==family for c in configs) for family in ['label_tuned_ssl','reliable_ssl','local_ssl']},
    uniform_reliability_controls=sum(c.get('reliability')=='uniform' for c in configs),
    inner_parameter_recipes=9,inner_folds=5,decision_policies_per_raw_configuration=16,
    repeat_seeds=[20261119,20261120,20261121],validation_candidates=len(results),
    baseline_validation_f1=baseline['best_validation_f1'],
    baseline_matched_cv=float(np.mean([s['ordinary'] for s in frozen['baseline_cv']])),
    results=[dict(name=r['name'],cv=r['ordinary_cv'],validation=r['validation']['sample_f1'],macro=r['validation']['macro_f1']) for r in results],
    best_retained=current['candidate']['config']==baseline['candidate']['config'],current_validation_f1=current['best_validation_f1'],
    reported_leader='0.93xx; exact score not provided',kaggle_score=None,validation_fit=False,final_train_plus_validation_refit_performed=False)
(E/'phase25-summary.json').write_text(json.dumps(summary,indent=2))
refs=json.loads((E/'references.json').read_text())
for title,url,use in [
 ('scikit-learn: Probability calibration','https://scikit-learn.org/stable/modules/calibration.html','OOF score calibration motivates reliability estimates for pseudo-label weighting. Our target distribution is shifted, so calibrated target-domain accuracy is not guaranteed.'),
 ('Segata and Blanzieri: Fast and Scalable Local Kernel Machines','https://jmlr.org/papers/v11/segata10a.html','Motivation for local SVM specialists; our overlapping weighted experts and global blending are a separate implementation adapted to this small dataset.'),
 ('scikit-learn: SVC','https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html','Regularization, RBF bandwidth and sample weights in the classical SVM classifiers.')]:
 if not any(r['url']==url for r in refs):refs.append(dict(title=title,url=url,use=use))
(E/'references.json').write_text(json.dumps(refs,indent=2))
nb=nbf.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python (AML Midterm)',language='python',name='aml-midterm')
nb.cells=[nbf.v4.new_markdown_cell(f'''# Per-reason SVM tuning, pseudo-label reliability and local specialists

The user requested all three remaining refinements and reported a new Kaggle leader in the **0.93xx** range. The exact leaderboard score was not supplied. Kaggle and local validation remain separate evaluations; our model's Kaggle score is unmeasured.

The starting model scored **{baseline['best_validation_f1']:.6f} validation sample-wise F1**. After these tests, the current saved best scores **{current['best_validation_f1']:.6f}**. All 78 validation rows remain excluded from fitting. The final train-plus-validation refit remains deferred.

The raw screen contained 60 configurations: six nested per-reason tuning policies, 24 reliability-weighted variants plus six uniform-weight controls, and 24 local specialist variants. Predictions from each were evaluated under four thresholds and four label scopes, using training data only.'''),
nbf.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd,numpy as np\nimport matplotlib.pyplot as plt\nroot=Path.cwd()\nsummary=json.loads((root/'experiments/phase25-summary.json').read_text())\nsummary"),
nbf.v4.new_markdown_cell('''## Per-reason SVM settings

Every outer fold runs its own five-fold grouped inner CV over nine SVM recipes. The recipe grid varies regularization C, RBF gamma and pseudo-label weight near the earlier best configuration. Each reason can select a different recipe. The screen compares independent binary-F1 selection and coordinate selection targeting sample-wise F1, with penalties of 0, 0.005 or 0.02 for departing from the common baseline.

Inner preprocessing and pseudo-label updates exclude inner held-out rows and their masked embedding duplicates. The selected recipes then fit the outer training rows. This tests a complete tuning procedure; a single set of globally optimized label parameters is not reused across outer validation folds. Original gold labels are never changed.

The final label scopes considered are all reasons, Other alone, Recruitment/Other, and Safety/Funding/Efficacy. Scoped policies preserve the previous model's decisions for the remaining reasons. Empty combined predictions fall back to that model.'''),
nbf.v4.new_markdown_cell('''## Reliability-weighted pseudo-labels

The teacher is either a supervised gold-only SVM or the earlier cluster-guided semi-supervised SVM. Five-fold inner OOF predictions train the reliability estimator. The per-reason variant calibrates each reason's signed margin to a class probability. The pooled variant predicts whether a margin's binary decision is correct, using margin magnitude, signed margin and reason indicators across all five tasks.

The full-fold teacher supplies fixed provisional labels for the masked rows. Reliability controls their training weights: max(2r−1, 0) squared, with a minimum reliability cutoff of 0.5 or 0.8. Overall pseudo-label weights are 0.03, 0.1 or 0.3. KMeans cluster agreement further downweights conflicts. Zero-weight pseudo-label rows are removed so they do not influence automatic class-frequency balancing.

The student fits gold labels plus these weighted provisional labels. The calibrator affects training weights; it is not merely a final prediction threshold adjustment. Teacher labels remain fixed during this student fit. Reliability is estimated on the labeled training distribution, so it is not guaranteed to stay calibrated under the later-trial distribution shift.'''),
nbf.v4.new_markdown_cell('''## Local SVM specialists

KMeans identifies three or six regions using only permitted training embeddings. Each expert retains all gold examples with distance-based weights and uses the 250 masked examples nearest its region center, labeled by the earlier global semi-supervised SVM. Gold weights have a floor to preserve shared information across regions.

Each expert fits five binary SVMs. Its output is blended with global scores at a maximum weight of 0.25 or 0.5, reduced further for reasons with little weighted positive or negative support. Smooth routing averages nearby experts. Regularization C of 0.3/1/3 and gamma of 0.001/0.003 are screened. No query features affect cluster fitting, neighborhood selection or model training.

This tests local decision functions and global sharing. It is distinct from the earlier experiment that changed the distance metric of one global SVM.'''),
nbf.v4.new_markdown_cell('''## Selection and controls

The initial outer CV uses seed 20261118. Ordinary and robust weighted CV nominate up to two policies from each family. Those policies are frozen and reevaluated on seeds 20261119–20261121. One repeated-CV winner per family is frozen before validation scoring.

Two matched controls keep their selected policy's scope and threshold: one removes per-reason parameter tuning, and one replaces learned reliability weights with uniform pseudo-label weights. No control-specific retuning is performed. The local expert implementation also has an exact numerical zero-blend check against its global model.

All splits group identical reason embeddings. Matching masked duplicates of held-out groups are removed at both inner and outer levels. All validation predictions are saved before labels are read for reporting. The entire historical search remains adaptive; nested tuning and leakage checks do not erase selection bias from repeated validation comparisons.'''),
nbf.v4.new_code_cell("repeats=json.loads((root/'experiments/phase25-repeat-results.json').read_text())\nrows=[dict(family=r['config']['learner']['model'],scope=r['config']['scope'],threshold=r['config']['decision_rule'],mean_cv=r['ordinary_cv'],split_std=r['cv_std'],robust_cv=r['robust_cv'],settings=json.dumps(r['config']['learner'],sort_keys=True)) for r in repeats]\npd.DataFrame(rows)"),
nbf.v4.new_code_cell("results=json.loads((root/'experiments/phase25-validation-results.json').read_text())+json.loads((root/'experiments/phase25b-validation-results.json').read_text())\ntable=pd.DataFrame([dict(candidate=r['name'],cv=r['ordinary_cv'],validation_f1=r['validation']['sample_f1'],macro_f1=r['validation']['macro_f1']) for r in results])\ntable"),
nbf.v4.new_code_cell("names=['Starting best']+table.candidate.tolist()\nscores=[summary['baseline_validation_f1']]+table.validation_f1.tolist()\nfig,ax=plt.subplots(figsize=(8,4.3))\ny=np.arange(len(names));ax.scatter(scores,y,s=60)\nfor i,s in enumerate(scores):ax.annotate(f'{s:.5f}',(s,i),xytext=(5,6),textcoords='offset points')\nax.axvline(.93,color='#9b3e27',linestyle='--',label='0.93 reference; reported leader is 0.93xx')\nax.set(yticks=y,yticklabels=names,xlim=(min(.80,min(scores)-.02),max(.98,max(scores)+.02)),xlabel='Validation sample-wise F1')\nax.margins(y=.15);ax.invert_yaxis();ax.legend(loc='lower right',fontsize=7)\nfig.tight_layout();fig.savefig(root/'figures/current_phase25_comparison.png',dpi=180)\nplt.show()"),
nbf.v4.new_markdown_cell('''## Reproduction and integrity

All three approaches passed uncached held-out-label perturbation tests, query-label/batch invariance, masked-duplicate exclusion and query/masked-output consistency checks. A zero local-expert blend reproduces the global model's scores. Source is embedded in notebook 04; protocols, exact model settings, OOF predictions, diagnostics and unsuccessful outcomes are under `experiments/phase25*`.

No external trial data, extra labels, ID predictors or deep learning were used. The frozen current-best configuration and its predictions are reproduced in notebook 04; raw file hashes and gold-label preservation are checked.

References: [calibration](https://scikit-learn.org/stable/modules/calibration.html), [local SVMs](https://jmlr.org/papers/v11/segata10a.html), and [SVC settings](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).'''),
nbf.v4.new_code_cell("json.loads((root/'experiments/phase25-integrity-checks.json').read_text())")]
nbf.write(nb,ROOT/'midterm/08_svm_refinements.ipynb')
print(json.dumps(summary,indent=2))
