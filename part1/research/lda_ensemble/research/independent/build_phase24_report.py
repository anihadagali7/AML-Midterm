"""Document the two requested experiments and their matched control."""
from pathlib import Path
import json,nbformat as nbf,numpy as np
ROOT=Path(__file__).resolve().parent;E=ROOT/'experiments'
baseline=json.loads((E/'phase24-baseline.json').read_text())
current=json.loads((E/'current-best.json').read_text())
results=json.loads((E/'phase24-validation-results.json').read_text())+json.loads((E/'phase24b-validation-results.json').read_text())
frozen=json.loads((E/'phase24-validation-frozen.json').read_text())
configs=json.loads((E/'phase24-configs.json').read_text())
by={r['name']:r for r in results}
summary=dict(raw_configurations=len(configs),co_training_configurations=sum(c['model']=='co_training' for c in configs),
    no_exchange_screen_controls=sum(bool(c.get('no_exchange')) for c in configs),
    metric_configurations=sum(c['model']=='metric_ssl' for c in configs),
    decision_policies_per_configuration=12,repeat_seeds=[20261115,20261116,20261117],
    validation_candidates=len(results),baseline_validation_f1=baseline['best_validation_f1'],
    baseline_matched_cv=float(np.mean([x['ordinary'] for x in frozen['baseline_cv']])),
    co_training_validation_f1=by['phase24_co_training']['validation']['sample_f1'],
    metric_validation_f1=by['phase24_metric_ssl']['validation']['sample_f1'],
    no_exchange_validation_f1=by['phase24_exact_no_exchange']['validation']['sample_f1'],
    best_retained=current['candidate']['config']==baseline['candidate']['config'],
    current_validation_f1=current['best_validation_f1'],target_met=current['best_validation_f1']>.92735,
    validation_fit=False,final_train_plus_validation_refit_performed=False)
(E/'phase24-summary.json').write_text(json.dumps(summary,indent=2))
refs=json.loads((E/'references.json').read_text())
for title,url,use in [
 ('Blum and Mitchell: Combining Labeled and Unlabeled Data with Co-Training','https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf','Motivation for separate embedding/metadata views exchanging pseudo-labels. Our quota, agreement, weighting and cluster safeguards are adaptations; the original assumptions are not established here.'),
 ('scikit-learn: NeighborhoodComponentsAnalysis','https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html','Gold-only per-reason linear metric learning, combined with original distances in an SVM kernel.')]:
 if not any(r['url']==url for r in refs):refs.append(dict(title=title,url=url,use=use))
(E/'references.json').write_text(json.dumps(refs,indent=2))
nb=nbf.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python (AML Midterm)',language='python',name='aml-midterm')
nb.cells=[nbf.v4.new_markdown_cell(f'''# Separate-view co-training and reason-specific distance learning

Both requested approaches were implemented and evaluated. Neither improved the saved **{current['best_validation_f1']:.6f} validation sample-wise F1**. Co-training scored **{summary['co_training_validation_f1']:.6f}**; distance learning scored **{summary['metric_validation_f1']:.6f}**. The target of strictly exceeding 0.92735 remains unmet. Kaggle performance has not been measured.

All 157 original gold labels and 1,014 originally masked training rows stayed in their original roles. The 78 validation rows remain outside fitting. No external trial data, extra labels, ID predictors, deep learning or train-plus-validation refit were used.'''),
nbf.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd,numpy as np\nimport matplotlib.pyplot as plt\nroot=Path.cwd()\nsummary=json.loads((root/'experiments/phase24-summary.json').read_text())\nsummary"),
nbf.v4.new_markdown_cell('''## Separate-view co-training

The first learner is an RBF SVM over standardized reason embeddings. The second is a regularized logistic classifier over a separate metadata view. Three views were tested: eligibility criteria; drugs, targets, actions and diseases; and their combination. TF-IDF vocabularies and feature normalization are learned only from permitted training rows.

Each binary reason task starts from gold labels. For up to eight rounds, each learner proposes confident positive and negative masked rows for its peer, with at most three additions per class per round and smaller quotas for rare classes. Gold labels never change. Comparisons test mutual agreement versus unrestricted cross-teaching, pseudo-label weights 0.1/0.3, and prediction from the embedding learner alone versus a small metadata vote. These confidence values are margin-based heuristics, not calibrated probabilities.

KMeans provides an unsupervised safeguard: clusters with at least two gold examples and 80% agreement downweight conflicting pseudo-labels by a factor of ten. This uses only training labels and features. The original co-training theory's view-sufficiency/independence assumptions are not established here; weak metadata controls are a reason for caution.

There were 24 exchange configurations and six no-exchange screen controls. The selected policy uses the drug/target/action/disease view, pseudo-label weight 0.3, unrestricted cross-teaching and embedding-only final scores. Its threshold is 0.65, and training CV selected it to replace Recruitment/Other decisions while retaining the incumbent for the other reasons. Empty combined outputs fall back to the incumbent.'''),
nbf.v4.new_markdown_cell('''## Reason-specific distance learning

Each binary reason learns a separate geometry from the gold rows of its current fold. Two families were evaluated: Neighborhood Components Analysis (NCA), and regularized Fisher-style contrasts between positive subtypes and negative examples. The latter uses Ledoit-Wolf covariance shrinkage and one, two or three positive subtype centers.

PCA to 25 or 50 dimensions is fitted using permitted gold and masked training embeddings. The supervised projection uses only gold labels. Learned squared distances are normalized using fitting rows, then blended with original squared distances at weights 0.15, 0.4 or 0.7. The resulting RBF kernel enters the earlier cluster-guided semi-supervised SVM pipeline, retaining its gold labels and low-weight pseudo-label schedule.

Eighteen NCA and eighteen Fisher-style configurations were screened. The selected metric uses 50 PCA components, two positive subtypes, 15% learned distance and threshold 0.55 for all five reasons. No per-reason changes were made after examining validation scores.'''),
nbf.v4.new_markdown_cell('''## Evaluation protocol

The initial screen used grouped five-fold CV with seed 20261114. Identical embeddings stay in the same fold, and matching masked duplicates of held-out groups are excluded from fitting. For each raw configuration, a training-only screen compared four thresholds and three scopes: all reasons, Other alone, or Recruitment/Other. These 792 policy comparisons reuse the 66 raw model configurations; they are not 792 independent model fits.

Ordinary and robust weighted training-CV winners from each family were frozen and reevaluated on seeds 20261115–20261117. The better repeated-CV policy in each family became its validation finalist. The exact no-exchange follow-up copies the selected co-training settings, scope and decoder, disabling only label exchange. It does not tune a new control decoder.

All validation predictions were saved before loading validation labels for scoring. This is controlled feature/model development, but the full historical search is adaptive and validation has been consulted repeatedly. Neither CV nor validation is an unbiased final estimate after that selection.'''),
nbf.v4.new_code_cell("repeat=json.loads((root/'experiments/phase24-repeat-results.json').read_text())\nrows=[dict(learner=r['config']['learner']['model'],metric=r['config']['learner'].get('metric'),scope=r['config']['scope'],threshold=r['config']['decision_rule'],mean_cv=r['ordinary_cv'],split_std=r['cv_std'],weighted_robust_cv=r['robust_cv']) for r in repeat]\npd.DataFrame(rows)"),
nbf.v4.new_code_cell("results=json.loads((root/'experiments/phase24-validation-results.json').read_text())+json.loads((root/'experiments/phase24b-validation-results.json').read_text())\nrows=[dict(candidate=r['name'],cv=r['ordinary_cv'],validation_f1=r['validation']['sample_f1'],macro_f1=r['validation']['macro_f1']) for r in results]\ntable=pd.DataFrame(rows)\ntable"),
nbf.v4.new_markdown_cell(f'''## Interpretation

Label exchange helped its matched control: validation F1 increased from **{summary['no_exchange_validation_f1']:.6f} to {summary['co_training_validation_f1']:.6f}**. That improvement did not exceed the existing hybrid's **{summary['baseline_validation_f1']:.6f}**. The distance-learning finalist also failed to transfer its training-CV improvement to validation. The saved incumbent and its submission therefore remain unchanged.

The incumbent's CV on these same repeat seeds was **{summary['baseline_matched_cv']:.6f}**. Comparisons should use matched seeds; estimates from the earlier split sets differ because the gold dataset is small.'''),
nbf.v4.new_code_cell("names=['Saved best','Co-training','Exact no-exchange control','Distance learning']\nscores=[summary[k] for k in ['baseline_validation_f1','co_training_validation_f1','no_exchange_validation_f1','metric_validation_f1']]\nfig,ax=plt.subplots(figsize=(7.4,3.6))\ny=np.arange(len(names))\nax.scatter(scores,y,s=65,color=['#197144']+['#2563a0']*3)\nfor i,s in enumerate(scores):ax.annotate(f'{s:.5f}',(s,i),xytext=(5,7),textcoords='offset points')\nax.axvline(.92735,color='#9b3e27',linestyle='--',label='Requested validation target')\nax.set(yticks=y,yticklabels=names,xlim=(.82,.98),xlabel='Validation sample-wise F1')\nax.margins(y=.18);ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8)\nfig.tight_layout();fig.savefig(root/'figures/current_phase24_comparison.png',dpi=180)\nplt.show()"),
nbf.v4.new_markdown_cell('''## Integrity and reproducibility

Both approaches passed held-out-label perturbation, query-label/batch invariance, masked-duplicate exclusion and query-versus-masked-output consistency checks. Caches were cleared before perturbation to avoid masking leakage. With learned-distance weight zero, the metric model reproduced the original SVM scores to numerical precision.

Source files: `phase24.py`, `phase24_run.py`, `phase24_control.py` and `phase24_integrity.py`, all embedded in notebook 04. Configurations, policies, update diagnostics, predictions and results are under `experiments/phase24*`. The archive includes every unsuccessful trial.

References: [Blum and Mitchell's co-training paper](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf) and [NCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html).'''),
nbf.v4.new_code_cell("json.loads((root/'experiments/phase24-integrity-checks.json').read_text())")]
nbf.write(nb,ROOT/'midterm/07_cotraining_and_distances.ipynb')
print(json.dumps(summary,indent=2))
