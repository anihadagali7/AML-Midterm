"""Report fixed-recipe training-only CV, pooled CV, and original validation."""
from pathlib import Path
import json,numpy as np,pandas as pd,nbformat as nbf
ROOT=Path(__file__).resolve().parent;P=ROOT/'experiments/pooled_benchmark_v1'
s=json.loads((P/'summary.json').read_text());p=json.loads((P/'protocol.json').read_text())
t=pd.read_csv(P/'comparison.csv');current=t.loc[t.incumbent].iloc[0]
columns=['training_cv_mean','pooled_cv_mean','original_validation_f1']
leaders={c:t.loc[np.isclose(t[c],t[c].max(),atol=1e-12,rtol=0),'candidate'].tolist() for c in columns}
conclusions=dict(model_count=len(t),leaders=leaders,current={c:float(current[c]) for c in columns},
    current_ranks={c:int(current[c+'_rank']) for c in columns},
    current_leads_all_three=all(current[c]>=t[c].max()-1e-12 for c in columns),
    current_submission_changed=False,full_train_plus_validation_refit=False,
    interpretation='Point-estimate ranks on overlapping development data, not significance tests or three independent tests.')
raw_train=pd.read_csv(ROOT/'data/part1_train.csv');raw_val=pd.read_csv(ROOT/'data/part1_val.csv')
labels=['Safety','Funding','Efficacy','Recruitment','Other'];emb=[f'emb_{i}' for i in range(150)]
saved=np.load(P/(current.pooled_result_key+'.npz'),allow_pickle=True)
truth=saved['truth'];pred=saved['predictions'];ids=saved['ids']
row_f1=2*(pred*truth[None,:,:]).sum(2)/(pred.sum(2)+truth.sum(1)[None,:])
from_val=np.isin(ids,raw_val.ID.to_numpy())
gt=pd.util.hash_pandas_object(raw_train[emb].round(10),index=False)
gv=pd.util.hash_pandas_object(raw_val[emb].round(10),index=False)
masked=raw_train[labels].eq(-100).all(axis=1)
diagnostics=dict(current_pooled_oof_original_training_rows=float(row_f1[:,~from_val].mean()),
    current_pooled_oof_original_validation_rows=float(row_f1[:,from_val].mean()),
    original_validation_rows_matching_masked_training_embeddings=int(gv.isin(gt[masked]).sum()),
    original_validation_rows_matching_gold_training_embeddings=int(gv.isin(gt[~masked]).sum()),
    interpretation='Subset OOF diagnostics, not separate holdout scores. Group CV excludes matching masked rows; the original training-only full fit legitimately used all provided training rows. This difference is not established as the cause of a score change.')
(P/'population-diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
(P/'conclusions.json').write_text(json.dumps(conclusions,indent=2))
nb=nbf.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python (AML Midterm)',language='python',name='aml-midterm')
nb.cells=[nbf.v4.new_markdown_cell(f'''# Frozen-recipe comparison with pooled cross-validation

We compared **{len(t)} recipes** using three separate measurements: the primary training-only CV, supplementary pooled train-plus-validation CV, and each original training-only fit's recorded validation score. No hyperparameters, feature choices, thresholds or model combinations were retuned during this comparison.

The retained hybrid scores **{current.training_cv_mean:.6f} training-only CV**, **{current.pooled_cv_mean:.6f} pooled CV**, and **{current.original_validation_f1:.6f} original validation F1**. Its ranks by these point estimates are **{int(current.training_cv_mean_rank)}, {int(current.pooled_cv_mean_rank)}, and {int(current.original_validation_f1_rank)}**, respectively. Tied scores share a rank.

The saved model and submission remain unchanged. The original validation score describes the original model trained without validation labels. It is not a holdout score for any pooled-fold model. No final model was fitted on all 235 labels and no Kaggle score was measured.'''),
nbf.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd,numpy as np\nimport matplotlib.pyplot as plt\nroot=Path.cwd();pooled=root/'experiments/pooled_benchmark_v1'\ntable=pd.read_csv(pooled/'comparison.csv')\ntable[['candidate','training_cv_mean','training_cv_std','pooled_cv_mean','pooled_cv_std','original_validation_f1']].sort_values('pooled_cv_mean',ascending=False)"),
nbf.v4.new_markdown_cell('''## Cohort and fixed settings

The comparison cohort was frozen before the pooled results were computed. It contains the five existing common-benchmark recipes plus every distinct inductive recipe with previously recorded validation sample-wise F1 of at least 0.900. Exact configuration/decision-rule duplicates were removed. Twenty recipes remain; some similar recipes can still make identical predictions.

Clustering and classification families, feature selections, regularization settings, pseudo-labeling rules, random seeds and decoding thresholds remain unchanged. Each fold trains a fresh instance. Learned coefficients, scaler statistics, cluster locations and provisional labels can change because the training rows change. Thus CV tests the fixed training recipe, not a single saved fitted object.

Existing internal calibration steps still fit only within their outer training fold, using their frozen settings. The Other subclass algorithm can reduce its requested cluster count if its pre-existing minimum-support rule requires it; that rule is unchanged. There is no new hyperparameter search.'''),
nbf.v4.new_markdown_cell('''## The three measurements

| Measurement | Gold labels available for splitting | Fitting and scoring |
|---|---:|---|
| Primary training-only CV | 157 | Same saved grouped five-fold splits and three seeds for every recipe; score complete out-of-fold predictions. |
| Supplementary pooled CV | 235 | Combine the 157 original training labels and 78 original validation labels; create a separate saved grouped five-fold partition for each of the same three seeds. |
| Original validation F1 | 157 for fitting, 78 held out | Reuse each unchanged recipe's already recorded score from its training-only fit. |

Both CV protocols use seeds 20261126–20261128. The actual fold memberships differ because the sample pools differ. Pair model differences within a protocol; matching seed numbers do not make the training-only and pooled estimates paired evaluations of the same held-out rows.

For each CV repeat, sample-wise F1 is calculated over the complete OOF matrix, weighting trials equally. The headline is the mean of those three repeat scores. Their standard deviation describes sensitivity to the partition; it is not a confidence interval.'''),
nbf.v4.new_code_cell("protocol=json.loads((pooled/'protocol.json').read_text())\nfolds=pd.DataFrame([{k:f[k] for k in ['seed','fold','training_gold_rows','heldout_gold_rows','original_validation_rows_in_training','original_validation_rows_heldout','masked_duplicates_excluded']} for f in protocol['folds']])\nfolds"),
nbf.v4.new_markdown_cell('''## Grouping, fitting boundaries and input preservation

The combined pool contains 235 gold rows in 222 unique embedding groups, plus the same 1,014 originally masked training examples. Identical embedding groups never cross an outer fold boundary. Masked examples matching a held-out group's embedding are excluded from that training fold. The observed gold duplicates have no conflicting reason labels.

Scaling, PCA, clusters, classifiers and pseudo-labeling fit only the current outer training gold and permitted masked rows. Rows originally belonging to validation can supply labels only when assigned to the training portion of that pooled fold. Held-out inputs do not participate in fitting. Kaggle test features are not used.

The pooled script creates an isolated in-memory dataset context and restores the original engine afterward. It does not modify either original CSV, the saved current-best configuration, any current prediction file or the primary CV fold assignments. Hash checks verify those boundaries. The current-model and nested-calibration paths also pass held-out-label perturbation and query-batch invariance tests in the enlarged data context.'''),
nbf.v4.new_code_cell("conclusions=json.loads((pooled/'conclusions.json').read_text())\nconclusions"),
nbf.v4.new_code_cell("summary=json.loads((pooled/'summary.json').read_text())\nrows=[]\nfor r in summary['models']:\n    result=json.loads((pooled/(r['pooled_result_key']+'.json')).read_text())\n    for seed,m in zip(summary['seeds'],result['repeat_scores']):\n        rows.append(dict(candidate=r['candidate'],seed=seed,F1=m['sample_f1']))\npd.DataFrame(rows).pivot(index='candidate',columns='seed',values='F1')"),
nbf.v4.new_code_cell("plot=table.sort_values('pooled_cv_mean')\ny=np.arange(len(plot));fig,ax=plt.subplots(figsize=(10,8))\nfor col,label,color in [('training_cv_mean','Training-only CV','#2864aa'),('pooled_cv_mean','Pooled CV','#188977'),('original_validation_f1','Original validation','#c46b23')]:\n    ax.scatter(plot[col],y,label=label,color=color,s=35)\nax.set(yticks=y,yticklabels=plot.candidate,xlabel='Sample-wise F1',xlim=(min(.84,float(plot[['training_cv_mean','pooled_cv_mean','original_validation_f1']].min().min())-.01),1.0))\nax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=8)\nfig.tight_layout();fig.savefig(root/'figures/current_pooled_benchmark.png',dpi=180)\nplt.show()"),
nbf.v4.new_markdown_cell('''## Interpreting the rankings

These are three complementary measurements on overlapping data, not three independent tests. Pooled CV uses a larger labeled training sample in each fold and a different mixture of labeled examples. A higher pooled score therefore cannot be attributed solely to larger held-out folds, and it is not a remeasurement of the exact saved model.

Prior development already used training labels and repeatedly consulted validation. Freezing recipes and common folds improves comparability but does not erase that selection history. The later-year Kaggle population can still differ from both labeled pools. Small rank differences should not be treated as established superiority, especially with only three dependent repeats.

The cohort and all settings were frozen before this comparison. Point-estimate leaders are reported explicitly; no recipe is automatically promoted just for topping one column. The primary training-only protocol remains the standard for future comparisons. The final all-label refit remains deferred.'''),
nbf.v4.new_markdown_cell('''## Why pooled CV need not score higher

More training labels do not guarantee a higher measured score. Pooled CV changes both the training sample and the population being scored, while keeping the algorithm settings fixed. It also applies the same strict duplicate-group exclusion rule to the original validation rows: matching masked training rows are removed when their group is held out. The original standalone validation fit used all supplied training rows, which included some unlabeled examples with identical embeddings.

The diagnostics below quantify those differences for the retained recipe. The source-specific pooled OOF scores are descriptive subsets of the same CV run; neither is a new independent validation result. These observations do not establish which factor caused the score change.'''),
nbf.v4.new_code_cell("json.loads((pooled/'population-diagnostics.json').read_text())"),
nbf.v4.new_code_cell("json.loads((pooled/'integrity.json').read_text())"),
nbf.v4.new_code_cell("preservation=json.loads((pooled/'preservation.json').read_text())\n{k:v for k,v in preservation.items() if k!='protected_sha256'}"),
nbf.v4.new_markdown_cell('''## Reproduction

Run `.venv/bin/python benchmark_cv.py` for the primary comparison and `.venv/bin/python benchmark_pooled_cv.py` for the separate pooled comparison. Saved results are reused only when the protocol, recipe, decoder and source hashes match. `--verify-only` checks pooled fold and recipe consistency; `--integrity-only` checks fitting boundaries without running the full cohort.

The cohort snapshot, exact configurations, source hashes, complete OOF predictions, fold memberships and per-fold metrics are included under `experiments/benchmark_v1` and `experiments/pooled_benchmark_v1`. Notebook 04 embeds the source files. Historical validation measurements retain their source filenames in the frozen registries.''')]
nbf.write(nb,ROOT/'midterm/11_pooled_cv_comparison.ipynb')
print(json.dumps(conclusions,indent=2))
