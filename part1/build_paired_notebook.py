"""Build and execute a report with both-pool evidence and the complete study source."""
import ast
import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/paired_improvement'
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))

md("""# Part 1: a modest improvement on both required CV pools

The selected model scores **0.925265 versus current Ani's 0.921444** on the 157-label CV and **0.921543 versus 0.913741** on the 235-label CV. Mean macro F1 improves in both pools. All four additional validation/chronological sample-F1 checks improve against the same current Ani source.

This is a local development improvement, **not a demonstrated Kaggle win or the requested 0.05 leaderboard gain**. Training-pool evidence remains noisy: its descriptive paired interval includes zero. All folds and diagnostics have been used repeatedly for development; selection can overfit them.

## Exact comparison

- Comparator: Ani commit `8a8c3fca70f13b62e989739a0bc1d97d229186e1`, archived in `experiments/current_ani_pool_audit/ani_current_source.ipynb`. This is the current recipe with LDA and C=0.1, not the older no-LDA reference.
- **157 known training labels:** five saved repeats of three-fold CV, seeds42–46, 103–106 fitting labels per fold. All78 validation inputs and labels are excluded from these fits.
- **235 known train+validation labels:** eight saved repeats of five-fold CV, seeds100–107, 188 fitting labels per fold.
- Both use the same1,014 originally unlabeled training rows. Every scaler, PCA, GMM and augmented classifier is fitted inside each fold. Held-out and Kaggle test inputs are excluded from all fitting and pseudo-labeling. No new folds were generated.
- Sample F1 is computed across the pooled out-of-fold rows within each repeat, then averaged across repeats. Unequal-sized157-pool folds are not averaged with equal weights.
- The native Ani notebook uses broader feature access. This comparison applies the repository's active strict fold-local feature policy to both recipes.

The new runner reproduces Ani's labels on all55 folds with maximum score difference1.1e-16. The previous trusted candidate's labels also reproduce on all55 folds (maximum score difference1.7e-9). Two aborted roundoff-control attempts, before interpreting results, are retained in the `00_...` folders.
""")
code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display
PART1=Path.cwd()
if not (PART1/'data/part1_train.csv').exists():PART1=PART1/'part1'
OUT=PART1/'experiments/paired_improvement'
comparison=pd.read_csv(OUT/'selected_comparison.csv')
display(comparison[['pool','name','mean_f1','repeat_sd','macro_f1','delta_ani','repeat_wins','repeat_ties','fold_wins','fold_ties']].round(6))
display(comparison[['pool','name','f1_Safety','f1_Funding','f1_Efficacy','f1_Recruitment','f1_Other']].round(6))
""")
md("""The candidate wins3 of5 training-pool repeat averages, ties1 and loses1; its fold record is8 wins/1 tie/6 losses. It wins6 of8 full-pool repeat averages; its fold record is26 wins/6 ties/8 losses. The direction is favorable on average, not on every split.

Funding is unchanged on157 and improves on235. Efficacy falls on157 and is essentially unchanged on235. Safety and Recruitment improve in both pools; Other improves on235 but falls slightly on157. This is not a uniform gain across all reasons.
""")
code("""import matplotlib.pyplot as plt
repeat=pd.read_csv(OUT/'06_final_ablations/repeat_scores.csv')
fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True,constrained_layout=True)
for ax,pool,title in zip(axes,['train157','trainval235'],['157-label pool: 5 × 3 CV','235-label pool: 8 × 5 CV']):
    w=repeat[repeat.pool==pool].pivot(index='repeat',columns='name',values='sample_f1')
    delta=w.selected_control-w.ani_current
    ax.bar(np.arange(len(delta))+1,delta,color=['#287d6e' if v>0 else '#b66c48' for v in delta])
    ax.axhline(0,color='#444444',linewidth=.8)
    ax.axhline(delta.mean(),color='#2159a0',linestyle='--',label=f'Mean: {delta.mean():+.4f}')
    ax.set(title=title,xlabel='Saved repeat',xticks=np.arange(len(delta))+1)
    ax.legend(frameon=False)
axes[0].set_ylabel('Sample-F1 change versus current Ani')
fig.savefig(OUT/'paired_repeat_changes.pdf')
fig.savefig(OUT/'paired_repeat_changes.png',dpi=160)
plt.show()
display(pd.read_csv(OUT/'paired_trial_bootstrap.csv').round(6))
""")
md("""The interval above resamples trials after averaging each trial's paired difference across repeats. It is descriptive; it does not account for choosing the recipe on these same data, overlapping training folds, or the many earlier searches. Do not interpret the positive235-pool interval as a confirmatory significance test.

## Four additional checks

The original split fits157 labels and holds out78. The chronological checks fit only trials starting no later than2003,2005 or2007 and evaluate later trials. Unlabeled trials after the cutoff are excluded too. These checks are reused development diagnostics. Compared with the previous trusted candidate, the new model improves the original split and2005 check, but is lower on2003 and2007; against current Ani, it improves all four.
""")
code("""diagnostics=pd.read_csv(OUT/'selected_diagnostics.csv')
display(diagnostics[['split','name','fit_rows','unlabeled_rows','held_rows','sample_f1','macro_f1']].round(6))
""")
md("""## What changed and why

The PDF says embeddings come from the actual stopping-reason text, were reduced with PCA and randomly rotated, and should be standardized. They are direct evidence of the target. The model uses all150 embeddings and no trial identifiers, drug features, disease labels or eligibility text. PCA30 and a diagonal30-component GMM remain the unsupervised component assisting automatic label generation; classifiers retain all150 dimensions.

Keep Ani's logistic C=0.1, final mixture0.5 logistic/0.3 cosine neighbors/0.2 shrinkage LDA, ten neighbors, two pseudo-labeling passes, acceptance confidence0.8, and final cutoff0.3 with argmax fallback. Change only:

1. Compute logistic class weights once from verified labels. Generated labels cannot change the class balance.
2. Give generated rows one-quarter weight in logistic regression. LDA and neighbors retain their original unweighted fitting; quarter weighting does not apply to them.
3. Let a neighbor with two reasons vote fully for both rather than splitting its vote.
4. Reduce GMM assistance during pseudo-label selection from20% to5%. The final prediction still uses only the supervised mixture.

The compact implementation is `anchored_solution.py`. It contains no additional model stack or learned second-stage combiner. The full-data fit uses235 known rows, accepts685 generated rows after the first pass and757 after the second. Verified labels are never overwritten.

## Methodical experiment record

Each fitted recipe was evaluated on both saved pools and all four diagnostics. There are36 named fitted evaluations representing31 distinct recipes, including controls, plus two new final-cutoff alternatives reusing the same55 fold score arrays. Every negative result is retained.

1. Add LDA to the simpler trusted recipe; compare fitting LDA on verified versus augmented labels. Augmented LDA helps more on235; restricting it to verified rows does not improve both pools.
2. Check generated-row weight, LDA weight, confidence and number of passes. A small amount of GMM assistance is more balanced than eliminating it.
3. Reduce either all classifier inputs or just LDA inputs to30/60 PCs. These changes hurt the two-pool comparison, despite a strong original-validation score for one variant; reject them.
4. Check5/20 neighbors and GMM weight0.05/0.15 around the0.1 candidate. Retain ten neighbors and0.05 GMM weight.
5. Check final decision cutoff0.25/0.35. Retain0.3.
6. Check logistic C=0.05/0.2 and revert each proposed change individually. Retain C=0.1. Reverting anchored class balance loses the157-pool gain; full generated-row weight also hurts157. Fractional neighbor votes lower both means. Restoring GMM0.2 improves157 but loses most of the235 gain.

Selection favored improving both pools, with the minimum of the two mean gains used for comparison. That is a development selection rule, not an independent test.
""")
code("""all_results=pd.read_csv(OUT/'all_results.csv')
display(all_results[['stage','pool','name','mean_f1','macro_f1','delta_ani']].round(6))
display(pd.read_csv(OUT/'06_final_ablations/both_pools.csv').round(6))
""")
md("""## Independent replay and saved candidate

The compact implementation was fitted independently on all55 saved folds and all four diagnostic splits. Every predicted label matches the selected experiment; the maximum fold-score difference is below2.1e-10. All pooled metrics were recalculated. Original labels, old candidate artifacts and protected benchmark files passed checksum checks. The serialized model reproduces its predictions exactly after reload.

`experiments/paired_improvement/candidate/part1_anchored_submission.csv` contains312 checked rows in sample-submission order and has **not been uploaded**. Its manifest records source/data/artifact hashes, versions, fitting counts and the exact selected recipe. The folder also includes generated-label decisions, the augmented training CSV, prediction scores and the model. All known labels remain intact.
""")
code("""manifest=json.loads((OUT/'candidate/manifest.json').read_text())
checks=json.loads((OUT/'candidate/independent_replay_checks.json').read_text())
assert len(checks['cv'])==55 and len(checks['diagnostics'])==4
for row in checks['cv']:
    assert set(row['feature_ids']).isdisjoint(row['held_ids'])
    assert len(row['unknown_ids'])==1014 and not row['test_features_used']
display({k:manifest[k] for k in ['status','known_labels','originally_unlabeled','test_rows','final_fit_audit','independent_fold_replays','independent_diagnostic_replays','protected_originals_unchanged','kaggle_score','versions']})
print('Maximum independent replay score error:',max(r['max_score_error'] for r in checks['cv']))
""")
md("""## Reproducible source

The following cells contain the complete code used by this study, including the small unchanged shared helpers it calls. Frozen per-stage copies are also saved beside each protocol. The current-Ani reference's original notebook and its55-fold audit are archived separately, with exact commit provenance and split hashes.

From the repository root, `.venv/bin/python part1/paired_finalize.py` reruns the55-fold/four-diagnostic independent replay and local export. It does not submit to Kaggle. For an individual original experiment use `PYTHONPATH=part1 .venv/bin/python part1/experiments/paired_improvement/STAGE/paired_search.py --stage STAGE --specs SPEC_FILE --diagnostics`, using that stage's protocol specs. The frozen scripts use their adjacent model source. Current runner source intentionally rejects overwriting an existing stage with changed source.

The notebook defaults to reading saved evidence. Optional replay cells below write new experiment folders; the checked reference and old candidates stay intact.
""")
code("""import sys,types
from IPython.core.magic import register_cell_magic
sys.path.insert(0,str(PART1))
@register_cell_magic
def source_module(name,source):
    module=types.ModuleType(name.strip());module.__file__=str(PART1/(name.strip()+'.py'))
    sys.modules[name.strip()]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
""")
# Include only shared definitions actually used in this study, preserving their text.
for filename,names,imports in [('simple_models.py',{'FEATURES','REASONS','decide'},'import numpy as np\n'),
                              ('simple_benchmark.py',{'ROOT','BASE','digest','metrics'},'import hashlib\nfrom pathlib import Path\nfrom sklearn.metrics import f1_score\nfrom simple_models import REASONS\n')]:
    source=(ROOT/filename).read_text();tree=ast.parse(source);segments=[]
    for node in tree.body:
        identifiers={node.name} if isinstance(node,(ast.FunctionDef,ast.ClassDef)) else {t.id for t in node.targets if isinstance(t,ast.Name)} if isinstance(node,ast.Assign) else set()
        if identifiers&names:segments.append(ast.get_source_segment(source,node))
    code('%%source_module '+Path(filename).stem+'\n'+imports+'\n\n'.join(segments))
for filename in ['paired_models.py','paired_search.py','paired_cutoffs.py','anchored_solution.py','paired_finalize.py']:
    code('%%source_module '+Path(filename).stem+'\n'+(ROOT/filename).read_text())
specs={p.parent.name:json.loads(p.read_text())['specs'] for p in sorted(OUT.glob('*/protocol.json')) if not p.parent.name.startswith('00') and 'specs' in json.loads(p.read_text())}
code('STAGE_SPECS='+repr(specs)+"""
RERUN_EXPERIMENTS=False
if RERUN_EXPERIMENTS:
    import paired_search
    for stage,specs in STAGE_SPECS.items():
        paired_search.run(stage+'_notebook_replay',specs)
        paired_search.diagnostics(stage+'_notebook_replay')
else:
    print('All stage settings are included above; saved results displayed by default.')
""")
code("""RERUN_INDEPENDENT_REPLAY_AND_EXPORT=False
if RERUN_INDEPENDENT_REPLAY_AND_EXPORT:
    import paired_finalize
    paired_finalize.main()
""")
nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT.parent)}}).execute()
nbf.write(nb,ROOT/'paired_improvement.ipynb')
print('Saved executed paired_improvement.ipynb')
