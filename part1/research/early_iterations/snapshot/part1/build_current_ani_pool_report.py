"""Rescore and document the corrected, version-pinned two-pool comparison."""
import hashlib
import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/current_ani_pool_audit'
predictions=pd.read_csv(OUT/'predictions.csv')
summary=pd.read_csv(OUT/'summary.csv').set_index(['pool','name'])
roles=json.loads((OUT/'role_audit.json').read_text())
assert len(roles)==55
for row in roles:
    assert set(row['fit_ids']).isdisjoint(row['held_ids'])
    assert set(row['unlabeled_ids']).isdisjoint(row['fit_ids']+row['held_ids'])
    assert len(row['unlabeled_ids'])==1014 and not row['test_features_used']
rng=np.random.default_rng(20261106)
intervals=[]
for pool,expected_rows,expected_repeats in [('train157',157,5),('trainval235',235,8)]:
    sub=predictions[predictions.pool==pool].copy()
    truth=sub[[f'actual_{r}' for r in ['Safety','Funding','Efficacy','Recruitment','Other']]].to_numpy(int)
    pred=sub[[f'predicted_{r}' for r in ['Safety','Funding','Efficacy','Recruitment','Other']]].to_numpy(int)
    sub['trial_f1']=2*(truth*pred).sum(axis=1)/(truth.sum(axis=1)+pred.sum(axis=1))
    for name,g in sub.groupby('name'):
        assert g.groupby('repeat').ID.nunique().eq(expected_rows).all()
        assert len(g)==expected_rows*expected_repeats
        mean=g.groupby('repeat').trial_f1.mean().mean()
        np.testing.assert_allclose(mean,summary.loc[(pool,name),'mean_f1'],atol=1e-15)
    wide=sub[sub.name.isin(['ani_current','trusted_two_pass'])].pivot(index=['repeat','ID'],columns='name',values='trial_f1')
    differences=(wide.trusted_two_pass-wide.ani_current).groupby('ID').mean().to_numpy()
    draws=rng.integers(0,len(differences),(10000,len(differences)))
    lo,hi=np.quantile(differences[draws].mean(axis=1),[.025,.975])
    intervals.append(dict(pool=pool,delta=float(differences.mean()),lower_95=lo,upper_95=hi,
                          caveat='Descriptive trial bootstrap; not adjusted for model selection or overlapping fits.'))
pd.DataFrame(intervals).to_csv(OUT/'paired_trial_bootstrap.csv',index=False)
review=dict(status='No established improvement over current Ani across both saved label pools.',
            supersedes='Earlier recommendation based on235-label CV against the legacy no-LDA Ani recipe.',
            current_ani_commit=json.loads((OUT/'protocol.json').read_text())['ani_commit'],
            candidate_157_delta=float(summary.loc[('train157','trusted_two_pass'),'delta_current_ani']),
            candidate_235_delta=float(summary.loc[('trainval235','trusted_two_pass'),'delta_current_ani']),
            candidate_csv_unchanged=True,kaggle_score=None,
            procedure='Use the saved15folds over157labels and40folds over235labels, strict fold-local features for every model.')
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
(ROOT/'experiments/trusted_iteration/candidate/current_reference_review.json').write_text(json.dumps(review,indent=2)+'\n')
audit=dict(fit_role_checks=55,all_saved_splits_used=True,current_ani_fetched_commit=review['current_ani_commit'],
           legacy_235_probabilities_reproduced_on_all40folds=True,candidate_235_probabilities_reproduced_on_all40folds=True,
           all_pooled_scores_independently_rescored=True,no_recipe_tuning=True,no_test_feature_fitting=True,
           source_sha256=hashlib.sha256((ROOT/'compare_ani_pools.py').read_bytes()).hexdigest())
(OUT/'checks.json').write_text(json.dumps(audit,indent=2)+'\n')

cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md("""# Corrected comparison: both label pools and current Ani

**Conclusion: the selected candidate has not established a clear improvement over current Ani.** It is slightly worse on the 157-label pool and slightly better on the 235-label pool. Current Ani has higher macro F1 on both.

This corrects two omissions in the earlier report: only the 235-label CV was run, and the reference was Ani's older no-LDA/C0.3 model. The earlier phrase “full testing procedure” was too broad, and the larger claimed gain did not compare against current Ani.

## Exact existing testing procedure

- **Original training pool:** all157 visible training labels, five saved repeats of three-fold multilabel-stratified CV, seeds42–46. Each fit has103–106 labels. The78 validation trials are excluded completely from these fits.
- **Training plus validation pool:** all235 visible labels, eight saved repeats of five-fold CV, seeds100–107. Each fit has188 labels.
- Both models use the same fold assignments and the same1,014 originally unlabeled training trials. All learned preprocessing and pseudo-labeling exclude held-out/test inputs. Score sample F1 over all out-of-fold trials within each repeat, then average repeats.
- Candidate settings are frozen. No retuning, new split generation or test-feature fitting was performed for this audit.

## Current comparator and provenance

`git fetch origin ani` confirmed current commit **8a8c3fca70f13b62e989739a0bc1d97d229186e1**. The exact [Ani notebook](https://github.com/anihadagali7/AML-Midterm/blob/8a8c3fca70f13b62e989739a0bc1d97d229186e1/midterm/part1_best_model.ipynb) is archived in `experiments/current_ani_pool_audit/ani_current_source.ipynb`. Its recipe functions are executed directly, with scaling/PCA/GMM refit on the permitted fold inputs.

Current Ani blends logistic regression0.5 (C0.1), cosine neighbors0.3, and shrinkage LDA0.2, followed by its GMM-assisted two-pass self-training. The legacy model has no LDA and uses C0.3. Ani's native notebook uses all embeddings; this comparison applies the same strict feature policy to both models, as required by the repository's active testing policy. It is not a new Kaggle result or a transductive comparison.
""")
code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display
PART1=Path.cwd()
if not (PART1/'data/part1_train.csv').exists():PART1=PART1/'part1'
OUT=PART1/'experiments/current_ani_pool_audit'
summary=pd.read_csv(OUT/'summary.csv')
display(summary[['pool','name','mean_f1','repeat_sd','macro_f1','delta_current_ani','repeat_wins','fold_wins','fold_ties']])
display(summary[['pool','name','f1_Safety','f1_Funding','f1_Efficacy','f1_Recruitment','f1_Other']])
""")
md("""Our candidate wins only one of five repeat averages on157 (two ties, two losses). On235 it wins five of eight repeats and loses three. Fold comparisons are5 wins/5 ties/5 losses on157 and20 wins/7 ties/13 losses on235. These are small, mixed differences, not a consistent improvement across the two required pools. Do not compare absolute scores between the two pools: they contain different trial populations and different fitting-set sizes.

The additional diagnostic comparisons below also use the **current** Ani recipe. The candidate has higher sample F1 on the old validation split and the earliest/latest chronological checks, and ties the middle chronological check. This does not override the mixed CV evidence or Ani's higher macro F1 in both pools. All of these are reused development comparisons.
""")
code("""display(pd.read_csv(OUT/'paired_trial_bootstrap.csv'))
diagnostics=pd.read_csv(OUT/'current_diagnostic_scores.csv')
display(diagnostics[['split','name','fit_rows','unlabeled_rows','held_rows','sample_f1','macro_f1']])
display(json.loads((OUT/'checks.json').read_text()))
""")
md("""## Checks and reproducibility

Every fit records its permitted/held-out IDs. All55 split-role checks pass. On all40 folds of the235-label pool, the legacy wrapper reproduces the historical Ani probabilities and the selected candidate reproduces its frozen probabilities. The157-label results are newly fitted on the saved15folds. All pooled scores are independently recalculated from persisted predictions.

The source below is the complete comparison driver. It uses the unchanged `trusted_solution.py` and repository metric helpers; Ani's complete source notebook is bundled alongside the results. Run `python part1/compare_ani_pools.py` from the repository environment to reproduce. The pinned source check intentionally refuses to silently reuse this result folder if Ani's fetched source changes. Use a new versioned audit for a new Ani revision.
""")
code("""import sys,types
from IPython.core.magic import register_cell_magic
sys.path.insert(0,str(PART1))
@register_cell_magic
def source_module(name,source):
    module=types.ModuleType(name.strip());module.__file__=str(PART1/(name.strip()+'.py'))
    sys.modules[name.strip()]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
""")
code('%%source_module compare_ani_pools\n'+(ROOT/'compare_ani_pools.py').read_text())
code("""RERUN_AUDIT=False
if RERUN_AUDIT:
    import compare_ani_pools
    compare_ani_pools.main()
else:
    print('Showing the saved, independently rescored current-Ani comparison on both pools.')
""")
nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
nbf.write(nb,ROOT/'current_ani_pool_comparison.ipynb')
print(summary[['mean_f1','macro_f1','delta_current_ani']].to_string())
print(pd.DataFrame(intervals).to_string(index=False))
