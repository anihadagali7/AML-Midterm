"""Document the user-requested tri-training / selection-weighting comparison."""
from pathlib import Path
import json,nbformat as nbf
ROOT=Path(__file__).resolve().parent;E=ROOT/'experiments'
results=sum([json.loads((E/f'{b}-validation-results.json').read_text()) for b in ['phase21','phase21b','phase21c','phase21d']],[])
inc=json.loads((E/'phase21-prior-incumbent.json').read_text())
current=json.loads((E/'current-best.json').read_text())
by_name={r['name']:r for r in results}
def val(name):return by_name[name]['validation']['sample_f1']
summary=dict(tri_configurations_screened=44,initial_fresh_cv_configurations=len(json.loads((E/'phase21-refit-shortlist.json').read_text())['candidates']),fresh_cv_seeds=[20261107,20261108,20261109],validation_candidates=len(results),best_tri_validation_f1=val('classical_tri_majority'),guarded_tri_validation_f1=val('tri_ordinary_cv'),target_weighted_selection_validation_f1=val('existing_target_weighted_cv + robust_weighting_cv'),initial_reason_hybrid_validation_f1=val('training_selected_reason_hybrid'),reason_hybrid_validation_f1=val('classical_tri_reason_hybrid'),exact_control_validation_f1=val('exact_no_teaching_ablation'),incumbent_validation_f1=inc['best_validation_f1'],current_validation_f1=current['best_validation_f1'],validation_improvement=current['best_validation_f1']-inc['best_validation_f1'],current_validation_macro_f1=current['candidate']['validation']['macro_f1'],selected_hybrid_ordinary_cv=current['candidate']['ordinary_cv'],selected_hybrid_leave_group_out_cv=current['leave_one_embedding_group_out_f1'],incumbent_retained=False,target_met=False,validation_used_for_training=False,final_train_plus_validation_refit_performed=False)
(E/'phase21-summary.json').write_text(json.dumps(summary,indent=2))
refs=json.loads((E/'references.json').read_text())
url='https://www.jmlr.org/papers/v8/sugiyama07a.html'
if not any(r['url']==url for r in refs):refs.append(dict(title='Sugiyama, Krauledat, Müller (2007): Covariate Shift Adaptation by Importance Weighted Cross Validation',url=url,use='Motivation for target-aware model selection. Estimated, clipped weights and label-dependent sampling here do not establish an unbiased target-risk guarantee.'))
(E/'references.json').write_text(json.dumps(refs,indent=2))
p=E/'search-gap-audit.json';audit=json.loads(p.read_text());audit['phase21_followup']=summary;p.write_text(json.dumps(audit,indent=2))
nb=nbf.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python (AML Midterm)',language='python',name='aml-midterm')
nb.cells=[nbf.v4.new_markdown_cell('''# Tri-training and model selection under sampling bias

The two requested approaches produced a modest new best: **0.912393 validation sample-wise F1**, compared with the previous **0.910256**. Macro F1 rose from **0.780000 to 0.810444**. The winner combines the previous SVM and classical tri-training, with per-reason choices made on training folds. It remains below the requested **0.92735 validation target**. This small improvement does not establish statistical significance or a better Kaggle score.

All 78 validation rows remain outside fitting. The final train-plus-validation refit is deferred.

We screened 40 initial tri-training configurations and reranked prior configurations with training-only risk weights. Eleven diverse candidates were reevaluated on three fresh grouped five-fold splits. Follow-ups tested an exact no-teaching control, classical tri-training without added safeguards (four additional configurations), and per-reason combinations. Across these batches, eight frozen candidates were evaluated on validation. Follow-ups were informed by earlier results; each candidate's settings and combination mask were fixed from training-side evidence before its validation evaluation.'''),
nbf.v4.new_code_cell("from pathlib import Path\nimport json,numpy as np,pandas as pd\nimport matplotlib.pyplot as plt\nroot=Path.cwd()\nsummary=json.loads((root/'experiments/phase21-summary.json').read_text())\nsummary"),
nbf.v4.new_markdown_cell('''## Tri-training implementation

Each reason is a binary task with three bootstrapped classical learners. A learner receives pseudo-labels from its two peers only when they agree and the peer-error / sample-size update conditions permit an update. Updates within a round are simultaneous. True gold labels stay fixed.

Four committees were tested: three RBF SVMs; SVM/logistic/ridge; three RBF bandwidths; and RBF/polynomial/linear SVMs. Comparisons included pseudo-label weights 0.03, 0.1 and 1; conservative cross-fitted error floors; confidence filtering; pseudo-label mass caps; and no-teaching controls. KMeans cluster agreement downweights inconsistent pseudo-labels. Stratified bootstrapping keeps the rare class present.

The classical follow-up used ordinary bootstrap samples, unit pseudo-label weights, no cluster gate, and a fixed majority-vote decoder. Its selected committee combines an RBF SVM (C=1, gamma=0.001), logistic regression (C=0.3), and ridge classification (alpha=10), with class balancing and at most eight update rounds. Its standalone validation sample F1 is 0.903846.

These are explicit adaptations of [Zhou and Li's tri-training](https://www.lamda.nju.edu.cn/publication/tkde05.pdf). Weighted labels, cluster gates, confidence filters, and cross-fitted error floors do not automatically inherit its theoretical guarantees. Smoothed vote scores are heuristic values, not calibrated probabilities; threshold 0.5 preserves majority voting, and margin scores break ties if the multilabel decoder would otherwise be empty.'''),
nbf.v4.new_markdown_cell('''## Selection weights

The source contains 157 gold-status rows; the target is the feature distribution of all 1,171 training rows. No validation/test rows or reason-label values are used to estimate weights.

A logistic propensity model and a tree propensity model predict gold versus masked status using embeddings and year, with identical-embedding groups kept together in cross-fitting. Estimated risk weights are proportional to P(gold) / P(gold | features). They are clipped, normalized, and shrunk halfway toward uniform weights. Kernel mean matching provides a third estimate.

These weights change the scoring of held-out training predictions and therefore the selected model/decoder. They do not fit validation labels. Strong label-dependent selection, imperfect propensity estimates, and poor overlap mean that an unbiased target-risk guarantee is not justified. The weighting models' AUCs below measure prediction of **selection status**, not trial-reason classification.'''),
nbf.v4.new_code_cell("weighting=json.loads((root/'experiments/phase21-weighting.json').read_text())\npd.DataFrame(weighting['diagnostics']).T"),
nbf.v4.new_code_cell("weights=pd.read_csv(root/'experiments/phase21-weights.csv')\ntrain=pd.read_csv(root/'data/part1_train.csv')\nplot=weights.merge(train[['ID','start_year']],on='ID',validate='one_to_one')\nfig,ax=plt.subplots(figsize=(7,3.2))\nax.scatter(plot.start_year,plot.propensity_logistic,s=25,alpha=.7,label='Logistic propensity')\nax.scatter(plot.start_year,plot.kmm,s=20,alpha=.5,label='Kernel mean matching')\nax.axhline(1,color='gray',linestyle='--',linewidth=1)\nax.set(xlabel='Start year',ylabel='Stabilized risk weight',title='Training-only weights for CV model selection')\nax.legend()\nfig.tight_layout()\n(root/'figures').mkdir(exist_ok=True)\nfig.savefig(root/'figures/current_phase21_weights.png',dpi=180)\nplt.show()"),
nbf.v4.new_markdown_cell('''## Frozen validation results

The adjusted selector favored an HDBSCAN-consensus logistic model with drug-name features. It had target-weighted training CV around 0.9276 but only 0.86966 on external validation. This is not a demonstrated generalization improvement.

The initial per-reason hybrid was selected from 32 combinations of the incumbent and tri-training predictions on fresh training folds. Training-only selection chose tri-training for Recruitment and the incumbent for the other reasons. It scored 0.90171 on validation and was rejected.

The initial bootstrap control matched the learner kinds but differed in aggregation and decoder. The exact ablation matches the initial tri-training winner's settings, including aggregation and its fixed decoder, with cross-teaching disabled. On that exact comparison, tri-training improves 0.85470 to 0.88205. The separately selected bootstrap control scored 0.88462. These initial candidates did not beat the incumbent.

Classical tri-training without the added safeguards scored 0.903846. Applying the unchanged hybrid selection rule to that candidate yielded the new best. The selector considered 32 masks on three fresh sets of grouped training folds, favored stability and robust weighted risk, and penalized extra switches. It chose **the previous SVM for Safety, Funding and Efficacy; tri-training for Recruitment and Other**. The mask and both decoders were frozen before validation. No per-reason choices came from validation errors.

The hybrid's mean repeated grouped five-fold F1 is **0.917481**. A subsequent leave-one-embedding-group-out audit of the fixed policy scored **0.928662**. Both are training-data development scores; neither is a validation or Kaggle score. The latter audit does not rerun the complete model-selection process inside each fold.'''),
nbf.v4.new_code_cell("batches=['phase21','phase21b','phase21c','phase21d']\nresults=sum([json.loads((root/f'experiments/{b}-validation-results.json').read_text()) for b in batches],[])\nrows=[dict(candidate=r['name'],ordinary_cv=r.get('ordinary_cv'),validation_f1=r['validation']['sample_f1'],validation_macro=r['validation']['macro_f1'],rule=r['rule']) for r in results]\npd.DataFrame(rows)"),
nbf.v4.new_code_cell("names=['New SVM / tri-training hybrid','Previous best','Classical tri-training','Target-weighted selection','Exact no-teaching control']\nscores=[summary[k] for k in ['current_validation_f1','incumbent_validation_f1','best_tri_validation_f1','target_weighted_selection_validation_f1','exact_control_validation_f1']]\nfig,ax=plt.subplots(figsize=(8,3.5))\nax.scatter(scores,np.arange(len(names)),s=65,color=['#197144']+['#2563a0']*4)\nfor i,s in enumerate(scores): ax.annotate(f'{s:.5f}',(s,i),xytext=(6,5),textcoords='offset points')\nax.axvline(.92735,color='#9b3e27',linestyle='--',label='Requested validation target')\nax.set(yticks=np.arange(len(names)),yticklabels=names,xlim=(.81,1),xlabel='Validation sample-wise F1')\nax.invert_yaxis();ax.legend(loc='lower right')\nfig.tight_layout();fig.savefig(root/'figures/current_phase21_comparison.png',dpi=180)\nplt.show()"),
nbf.v4.new_markdown_cell('''## Integrity and artifacts

The current prediction file and frozen configuration now contain the new hybrid. The previous candidate is preserved under `predictions/checkpoints/pre_phase21_091026`. No train-plus-validation refit was performed. Every validation comparison is logged; this repeatedly used selection set is not a pristine test set. Kaggle performance remains unmeasured. No external trial data, extra labels or deep learning were used.

Source is embedded in notebook 04: `phase21.py`, `phase21_weights.py`, `phase21_select.py`, `phase21_hybrid.py`, `phase21_plain.py`, `phase21_classical_hybrid.py`, `phase21_fullsize_current.py` and `phase21_integrity.py`. Protocol, frozen weights, fold rankings, update diagnostics, predictions and results are under `experiments/phase21*`.

The final hybrid passed held-out-label perturbation, query-batch invariance, masked-duplicate exclusion and nonempty-output checks. Reproduction verifies raw-data hashes, all original gold labels and the exact submission checksum.

The weighting approach is motivated by [importance-weighted cross-validation](https://www.jmlr.org/papers/v8/sugiyama07a.html); its assumptions and limitations are stated above.'''),
nbf.v4.new_code_cell("json.loads((root/'experiments/phase21-integrity-checks.json').read_text())")]
nbf.write(nb,ROOT/'midterm/05_tri_training_and_shift.ipynb')
print(json.dumps(summary,indent=2))
