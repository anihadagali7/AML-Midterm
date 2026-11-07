"""Build the record of three new classical methods and their matched controls."""
from pathlib import Path
import json
import numpy as np
import nbformat as nbf

ROOT=Path(__file__).resolve().parent;E=ROOT/'experiments'
baseline=json.loads((E/'phase26-baseline.json').read_text())
current=json.loads((E/'current-best.json').read_text())
results=json.loads((E/'phase26-validation-results.json').read_text())
frozen=json.loads((E/'phase26-validation-frozen.json').read_text())
configs=json.loads((E/'phase26-configs.json').read_text())
summary=dict(raw_configurations=len(configs),family_counts={family:sum(c['model']==family for c in configs)
    for family in ['label_shift26','lap_svm26','other_subclasses26']},
    baseline_validation_f1=baseline['best_validation_f1'],baseline_matched_cv=float(np.mean([s['ordinary'] for s in frozen['baseline_cv']])),
    repeat_seeds=[20261123,20261124,20261125],validation_comparisons=len(results),
    results=[dict(name=r['name'],matched_control=r['matched_control'],cv=r['ordinary_cv'],cv_std=r['cv_std'],
        validation=r['validation']['sample_f1'],macro=r['validation']['macro_f1']) for r in results],
    baseline_retained=current['candidate']['config']==baseline['candidate']['config'],
    current_validation_f1=current['best_validation_f1'],reported_leader='0.93xx; exact score unknown',
    kaggle_score=None,validation_fit=False,final_train_plus_validation_refit=False)
summary['matched_effects']={}
for r in results:
 if r['matched_control']:continue
 control=next(x for x in results if x['matched_control'] and x['config']['learner']['model']==r['config']['learner']['model'])
 summary['matched_effects'][r['config']['learner']['model']]=dict(
    cv_difference=r['ordinary_cv']-control['ordinary_cv'],
    validation_difference=r['validation']['sample_f1']-control['validation']['sample_f1'])
(E/'phase26-summary.json').write_text(json.dumps(summary,indent=2))
refs=json.loads((E/'references.json').read_text())
for title,url,use in [
 ('Lipton et al. (2018): Detecting and Correcting for Label Shift with Black Box Predictors','https://proceedings.mlr.press/v80/lipton18a.html','Motivates estimating changed reason frequencies from unlabeled examples. Our binary soft-confusion estimator uses ridge shrinkage; the stable conditional-distribution assumption may fail here.'),
 ('Belkin et al. (2006): Manifold Regularization','https://www.jmlr.org/papers/v7/belkin06a.html','Motivates Laplacian SVM penalties. Our experiment uses a finite 192-component Nystroem kernel approximation and fold-local graph fitting.')]:
 if not any(r['url']==url for r in refs):refs.append(dict(title=title,url=url,use=use))
(E/'references.json').write_text(json.dumps(refs,indent=2))
nb=nbf.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python (AML Midterm)',language='python',name='aml-midterm')
nb.cells=[nbf.v4.new_markdown_cell(f'''# Label-frequency correction, graph SVMs and Other subgroups

The requested follow-up tested **84 configurations**: 24 label-shift corrections, 36 graph-regularized SVMs and 24 Other-subclass models. The starting candidate had validation sample-wise F1 **{baseline['best_validation_f1']:.6f}**. The current saved candidate has **{current['best_validation_f1']:.6f}**. Validation remains excluded from fitting; the final train-plus-validation refit is deferred.

The reported Kaggle leader is in the 0.93xx range; its exact score is unknown. Our actual Kaggle score is unmeasured. Local CV and validation are separate evaluations.'''),
nbf.v4.new_code_cell("from pathlib import Path\nimport json,numpy as np,pandas as pd\nimport matplotlib.pyplot as plt\nroot=Path.cwd()\nsummary=json.loads((root/'experiments/phase26-summary.json').read_text())\nsummary"),
nbf.v4.new_markdown_cell('''## Label-frequency correction

The teacher combines the incumbent's SVM scores for Safety/Funding/Efficacy and tri-training scores for Recruitment/Other. SVM scores are aligned so their previous 0.55 threshold corresponds to 0.5. A five-fold grouped inner CV supplies teacher scores for every outer-training gold row. Logistic calibrators (C=0.3 or 3) fit those OOF scores to the original binary labels.

The full-fold teacher predicts the allowed masked training examples. Two estimators infer each reason's prevalence in those examples: regularized soft-confusion inversion (a binary BBSE adaptation) and an EM likelihood update with 20 source-prior pseudo-observations. BBSE uses ridge 0.01 toward the labeled source frequency. Estimated prevalences are clipped to [0.01,0.99]. The log-odds adjustment is capped at ±1.5 and multiplied by 0.25, 0.5 or 1.

The shift is applied either to calibrated scores or the original aligned teacher scores. The uncalibrated-score variant is a heuristic comparison. Calibration summaries and prevalence estimates are training estimates, not independent accuracy guarantees. The method assumes stable score distributions conditional on each reason; deliberate rare-label sampling, changing trial years and correlated labels can violate this assumption.

Only the 1,014 originally masked training rows, with held-out embedding duplicates removed, inform target prevalence. Validation and test features do not enter prior estimation. The matched control sets the shift strength to zero and keeps the same calibration, threshold and scope.'''),
nbf.v4.new_markdown_cell(r'''## Graph-regularized SVM

Each training fold fits StandardScaler and a 192-component Nystroem approximation of the RBF kernel. A graph joins the 10 or 25 nearest neighbors among allowed gold and masked training embeddings. Edge weights use local distance scales; the graph is symmetrized. Its combinatorial Laplacian is $L=D-W$.

Writing the finite feature matrix as $\Phi$, compute $G=\Phi^T L\Phi/N$ and scale it to mean diagonal one. For each reason, minimize

$$\tfrac12 w^T(I+\lambda G)w+C\sum_{i\in\mathrm{gold}} a_i\max(0,1-y_i(\phi_i^Tw+b)).$$

Cholesky whitening converts this objective to an ordinary balanced SVM. The graph weights tested are 0.1, 1 and 10; C is 0.3, 1 or 3; RBF gamma is 0.001 or 0.003. Only gold labels enter the hinge loss. The unsupervised graph changes the fitted classifier, whose predictions supply the missing labels. This is a finite-kernel approximation of a Laplacian SVM, not an exact infinite-dimensional solver.

The matched control uses the identical kernel approximation and settings with graph weight zero. Its purpose is to isolate the graph penalty. Query features only undergo the fitted transformations; they never alter the graph.'''),
nbf.v4.new_markdown_cell('''## Other subgroups

PCA to 25 components is fitted on permitted training embeddings. KMeans then clusters the gold Other-positive rows into two or three subgroups. If a subgroup contains fewer than three gold examples, the number of subgroups is reduced. All Other-negative gold examples remain class zero.

A regularized multiclass logistic classifier learns these latent subclass targets. Its Other score is the sum of probabilities for the positive subclasses. The representation is either standardized embeddings or a 192-component RBF kernel map. C is 0.3, 1 or 3. Aggregate Other versus non-Other receives balanced gold weights; each positive subclass is not independently rebalanced to the size of the negative class.

Masked rows whose teacher Other score is at least 0.8 or at most 0.2 receive fixed provisional targets with weight 0.03. Positive provisional rows are assigned to the nearest learned Other center. The resulting score replaces or blends equally with the incumbent's Other score. Other reason decisions are preserved by the scoped policy. Original gold labels are never overwritten.

The matched control uses one Other subgroup with identical remaining settings. This isolates the benefit of splitting the positive class.'''),
nbf.v4.new_markdown_cell('''## Selection and validation discipline

The first screen uses grouped five-fold CV seed 20261122. Four thresholds (0.4/0.5/0.6/0.7) and four label scopes are considered for shift and graph models; subclass models alter only Other. Scope choices for the first two families are all labels, Safety/Funding/Efficacy, Efficacy/Other, or Recruitment/Other. Empty scoped predictions fall back to the incumbent's decisions.

Ordinary and robust weighted CV each nominate a policy per family. Up to six policies are frozen and evaluated on three new grouped split seeds (20261123–20261125). Selection within each family maximizes mean sample-F1 minus 0.2 times its split standard deviation, with robust weighted performance breaking ties. Matched controls inherit the chosen settings, scope and threshold without retuning.

The three finalists and three controls are frozen before validation evaluation. Their validation predictions are saved before validation labels are read for reporting. The original engine also logs its best decoding rule for diagnostic purposes; phase 26 selection uses only the explicit frozen policies described here.

The repeated full history of CV and validation comparisons creates selection optimism. A nested inner calibrator does not make the entire historical search an unbiased nested-CV estimate. All duplicate reason embeddings are grouped, and matching masked duplicates of held-out groups are excluded at every fitting level.'''),
nbf.v4.new_code_cell("repeats=json.loads((root/'experiments/phase26-repeat-results.json').read_text())\npd.DataFrame([dict(family=r['config']['learner']['model'],scope=r['config']['scope'],threshold=r['config']['decision_rule'],mean_cv=r['ordinary_cv'],split_std=r['cv_std'],settings=json.dumps(r['config']['learner'],sort_keys=True)) for r in repeats])"),
nbf.v4.new_code_cell("results=json.loads((root/'experiments/phase26-validation-results.json').read_text())\ntable=pd.DataFrame([dict(candidate=r['name'],control=r['matched_control'],cv=r['ordinary_cv'],validation_f1=r['validation']['sample_f1'],macro_f1=r['validation']['macro_f1']) for r in results])\ntable"),
nbf.v4.new_markdown_cell('''## What the controls establish

None of the three finalists or controls exceeds the retained 0.912393 validation model. Label-frequency correction scores 0.871795 versus 0.876068 without the shift, so correction itself does not explain a gain. The graph penalty scores 0.880342 versus 0.858974 without that penalty, an improvement within this matched pair that still trails the incumbent. Three Other subgroups score 0.897436, exactly the same aggregate validation F1 as one subgroup.

The selected frequency-correction policy reaches 0.930644 on repeated training CV but loses substantially on validation. Its zero-shift control also has high CV and lower validation. This supports caution about transferring training-CV rankings; it does not by itself prove a particular cause of the mismatch or a Kaggle score. The incumbent is preserved rather than replacing it based on CV alone.'''),
nbf.v4.new_code_cell("names=['Starting best']+table.candidate.str.replace('phase26_','',regex=False).tolist()\nscores=[summary['baseline_validation_f1']]+table.validation_f1.tolist()\nfig,ax=plt.subplots(figsize=(9,4.8))\ny=np.arange(len(names));ax.scatter(scores,y,s=55)\nfor i,s in enumerate(scores):ax.annotate(f'{s:.5f}',(s,i),xytext=(5,5),textcoords='offset points')\nax.axvline(.93,color='#9b3e27',linestyle='--',label='0.93 reference; reported Kaggle leader is 0.93xx')\nax.set(yticks=y,yticklabels=names,xlim=(min(.80,min(scores)-.02),max(.98,max(scores)+.02)),xlabel='Validation sample-wise F1')\nax.margins(y=.13);ax.invert_yaxis();ax.legend(loc='lower right',fontsize=7)\nfig.tight_layout();fig.savefig(root/'figures/current_phase26_comparison.png',dpi=180)\nplt.show()"),
nbf.v4.new_markdown_cell('''## Integrity and reproduction

All three methods pass held-out-label perturbation, query-label invariance, query-batch invariance, masked-duplicate exclusion and consistent query/masked output tests. Additional checks confirm the graph objective's algebraic transformation, equality with an ordinary finite-kernel SVM when the graph penalty is zero, and zero-shift equality with the aligned teacher scores. Synthetic label-shift inputs verify adjustment direction.

No external trial data, added labels, ID predictors or deep learning are used. Source is embedded in notebook 04. Exact configurations, OOF outputs, selection records and diagnostics are stored under `experiments/phase26*`. Raw input hashes and preservation of all 157 original gold labels are checked when reproducing the retained candidate.

References: [label-shift estimation](https://proceedings.mlr.press/v80/lipton18a.html) and [manifold regularization](https://www.jmlr.org/papers/v7/belkin06a.html).'''),
nbf.v4.new_code_cell("json.loads((root/'experiments/phase26-integrity-checks.json').read_text())")]
nbf.write(nb,ROOT/'midterm/09_shift_graph_and_subgroups.ipynb')
print(json.dumps(summary,indent=2))
