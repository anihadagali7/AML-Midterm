"""Compile the experiment ledger, figures, executed notebooks and portable coursework folder."""
from pathlib import Path
import json, shutil, hashlib, os
import numpy as np,pandas as pd
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
rows=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()]
trial=pd.DataFrame([{**r,'config':json.dumps(r['config'],sort_keys=True)} for r in rows]);trial.to_csv('experiments/trials.csv',index=False)
ok=[r for r in rows if r['status']=='ok']
def family(r):
 c=r['config'];return 'cluster: '+c['cluster'] if c['model']=='cluster' else c['model']
summary=pd.DataFrame([dict(family=family(r),score=r['score'],default_score=r['default_score'],id=r['id'],seconds=r['seconds']) for r in ok]).groupby('family').agg(best_development_f1=('score','max'),best_default_f1=('default_score','max'),configurations=('id','nunique'),cv_runs=('id','size'),seconds=('seconds','sum')).sort_values('best_development_f1',ascending=False)
summary.to_csv('experiments/model-family-summary.csv')
# Individual feature screening selected configurations on the ORIGINAL seed only.
features=[]
base=next(r for r in ok if r['tag']=='screen' and r['config']==dict(model='svc',C=10,gamma=.001,scale='standard'))
for block in sorted({r['config']['features'][0] for r in ok if r['tag']=='features'}):
 r=max([r for r in ok if r['tag']=='features' and r['config'].get('features')==[block]],key=lambda r:r['score'])
 fresh=[x for x in ok if x['id']==r['id'] and x['tag']=='robustness']
 from experiment import metrics,Y
 freshscores=[]
 for x in fresh:
  p=np.load(f"experiments/{x['tag']}-{x['id']}-{x['seed']}.npz",allow_pickle=True)['p'];freshscores.append(metrics(Y,p,r['rule'])['sample_f1'])
 features.append(dict(feature=block,model=r['model'],weight=r['config']['feature_weight'],first_seed_f1=r['score'],rule_frozen_on_first_seed=r['rule'],fresh_seed_mean_f1=float(np.mean(freshscores)) if freshscores else np.nan,n_fresh_seeds=len(freshscores)))
pd.DataFrame(features).to_csv('experiments/feature-ablation.csv',index=False)
val=json.loads(Path('experiments/validation-once.json').read_text());final=json.loads(Path('experiments/final-candidate.json').read_text());nested=json.loads(Path('experiments/nested-audit.json').read_text());audit=json.loads(Path('experiments/data-audit.json').read_text())
readme=f'''AML MIDTERM — PART 1

CURRENT STATUS
The requested winning Kaggle score has NOT been established.
The first frozen 3-candidate holdout comparison achieved best sample-F1 {val['selected_sample_f1']:.6f} on 78 validation trials. It did not improve on the user's approximate previous validation result. Its grouped-bootstrap 95% interval is {val['bootstrap_95_interval'][0]:.4f}–{val['bootstrap_95_interval'][1]:.4f}.
The final FULL-FIT candidate was subsequently selected using TRAINING DATA ONLY. It achieves {final['selected']['fullsize_oof_f1']:.6f} leave-one-embedding-group-out development F1. This is NOT a Kaggle score or a new external validation result. No validation-error-driven retuning was performed. The conditional nested shortlist audit scored {nested['sample_f1']:.6f}; the candidate shortlist itself was informed by exploratory CV, so this is not an unbiased nested rerun of the entire search.

FILE TO TRY ON KAGGLE
predictions/part1_fullfit_candidate.csv
312 rows, columns ID and Predicted, matching sample-submission order. Awaiting a user-run Kaggle evaluation; credentials were not configured here. Do not use a development score to claim a leaderboard win.

PRIMARY READING
midterm/01_experiment_report.ipynb — executed report with tables, figures, limitations and references.
midterm/01_experiment_report.html — browser-readable version.
midterm/02_reproduce.ipynb — all experiment source embedded and exact commands.
experiments/trials.csv / trials.jsonl — {len(rows)} logged CV runs, {len({r['id'] for r in rows})} distinct configurations; {sum(r['status']=='error' for r in rows)} failed attempt(s) retained. Repeated seeds count as additional runs. Nested and 600 leave-group-out member fits are recorded separately.
experiments/protocol.txt, phase2-protocol.txt — evaluation rules and chronological changes.
experiments/final-candidate.json — final configuration, provenance and checksum.
predictions/part1_train_augmented_fullfit.csv — original 157 labeled rows preserved; 1,014 masked rows receive predicted labels, with label_source recorded.

REPRODUCE THE FINAL CANDIDATE
From the AMLMidterm project directory, or the portable midterm directory:
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements-lock.txt
  .venv/bin/python make_candidate.py
On macOS, XGBoost/LightGBM additionally require Homebrew libomp for the full model screen; the selected final model does not require these two packages at inference.
The executed notebook lists replay commands. Raw data SHA256 hashes are in experiments/data-audit.json. The first validation pipeline is retained as experiment_v1.py and uses its own recorded engine hash.

FINAL METHOD
50%: independent logistic models (C=0.3) with rotation-preserving centered global scaling of all 150 embeddings and 1024-bit Morgan fingerprints (radius 2; max across matched molecules; zero if unavailable). Fingerprint block has relative weight 0.5. HDBSCAN on 25-dimensional, train-fitted standardized PCA supplies a cluster-label teacher. Teacher and supervised predictions must agree; their mean confidence must reach 0.95. Accepted pseudo-labels enter per-label logistic fits at weight 0.05. True labels are never replaced.
25%: standardized-embedding RBF SVMs (C=10, gamma=0.001) plus word/bigram TF-IDF drug names and criteria (relative weight 0.5 each).
25%: the same SVM family plus drug-target TF-IDF (relative weight 1.0).
Each member trains on all 157 original labeled rows and legitimate masked training rows as applicable. Probabilities are combined with fixed 0.50/0.25/0.25 weights. Threshold 0.35, with argmax fallback when all five probabilities fall below threshold.

INTERPRETATION
The search was broad and structured, not an exhaustive enumeration of every possible model, feature subset or hyperparameter. Individual rotated embedding coordinates have no semantic meaning; we retained the block and compared scaling/PCA instead of claiming coordinate-level feature explanations. Source fields were added individually, with several weights, followed by selected forward combinations and an all-features stress test.
External validation was used once for a frozen 3-candidate comparison. Later work did not recheck it. All search/CV values are development estimates and can be optimistic after extensive selection. The training labeled subset is temporally biased and enriched for rare reasons; chronological stress tests are also confounded by this enrichment. The small data do not support a guaranteed score.
No new clinical-trial records, external labels, test-label reverse engineering, ID predictors, or deep learning were used. Only provided CSV inputs and classical methods. Supplied pretrained embeddings are input data, not a newly trained deep model.
Part 2 and the final combined five-page course report are not completed by this Part 1 work.
'''
Path('READ_ME_FIRST.txt').write_text(readme)
# Notebook with directly executable plotting source.
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('# AML Midterm — Part 1 experiment report\n\n**Objective not yet verified:** beat Kaggle 0.92735 without external trial records, deep learning, or validation-set overfitting. The user’s prior Kaggle score was 0.89423. Neither number is comparable directly to a training CV estimate.\n\n**Evidence:** the first frozen validation comparison scored **0.8504**. A later full-training candidate has **0.9363 grouped leave-one-out development F1**, but no new external score. The first result is preserved; the report does not present a development result as a leaderboard win.')
code('''from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
root = Path.cwd()
if not (root / 'experiments').exists():
    root = root.parent
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
figdir = root / 'figures'; figdir.mkdir(exist_ok=True)
labels = ['Safety','Funding','Efficacy','Recruitment','Other']
train = pd.read_csv(root/'data/part1_train.csv')
known = train[labels].ge(0).all(axis=1)
trials = pd.read_csv(root/'experiments/trials.csv')
val = json.loads((root/'experiments/validation-once.json').read_text())
final = json.loads((root/'experiments/final-candidate.json').read_text())
''')
md('## Data and constraints\n\n1,171 training trials: 157 labeled, 1,014 wholly masked. The 78 validation trials are separate. The 312 test trials start later, in 2010–2016. The main target is sample-wise multilabel F1; per-reason and macro F1 are also retained. There are 150 unique embedding groups among the 157 labeled rows. Exact duplicate groups remain together in CV, and their masked counterparts are excluded from fold fitting. IDs are identifiers only.\n\nFunding has only eight positive training examples. The known labels mostly come from early years, plus a selected rare-reason enrichment sample. Random-fold CV does not reproduce the future-test distribution.')
code('''display(pd.DataFrame({'positive_training_examples': train.loc[known,labels].sum()}))
fig, axes = plt.subplots(1,2,figsize=(11,3.6))
train.loc[known,labels].sum().plot.bar(ax=axes[0], color='#326b8c', rot=25)
axes[0].set(title='Visible training labels', ylabel='Positive trials')
axes[1].hist([train.loc[known,'start_year'], train.loc[~known,'start_year']], bins=range(1993,2012), label=['Labeled','Masked'], color=['#326b8c','#d79a45'])
axes[1].set(title='Training time distribution', xlabel='Start year', ylabel='Trials'); axes[1].legend()
fig.tight_layout(); fig.savefig(figdir/'data_distribution.png'); fig.savefig(figdir/'data_distribution.pdf'); plt.show()
''')
md('## Evaluation design\n\nThe broad screen used grouped multilabel-stratified five-fold CV. Scaling, TF-IDF vocabularies, PCA, clustering, label mapping, feature transforms and pseudo-labeling are fitted on training-fold labels plus permitted masked training rows. Held-out rows never participate in preprocessing or pseudo-label generation. A test verifies that changing held-out labels, query values, or IDs cannot change fitted predictions for other rows.\n\nPromising methods were repeated on three fresh split seeds. Added fields were screened individually, followed by selected combinations and an all-fields comparison. Decision policies were a small fixed threshold grid, argmax, or exact expected-F1 enumeration under independent marginal probabilities. Extensive selection makes these scores optimistic.\n\nA five-outer/three-inner conditional nested audit selected both models and decision rules inside each outer fold. It scored 0.89384. Because the shortlist was informed by earlier full-training exploration, this is not an unbiased assessment of the entire adaptive search. Chronological cutoffs add a separate stress test, but later labeled rows are intentionally enriched for rare classes.\n\nA three-candidate shortlist was frozen before the first external validation evaluation. After its disappointing result, later experiments remained training-only; no individual validation-error retuning or repeated validation scoring occurred. Leave-one-embedding-group-out CV then checked a nearly full-size training regime (150 groups, around 156 labeled rows per fit).')
md('## Model-family comparisons\n\nThe table reports each family’s **best development result**, not an unbiased ranking. Counts include successful unique configurations and CV runs; repeated seeds are separate runs. The failed full-dimensional QDA attempt is retained in the ledger; reduced-dimensional variants were subsequently evaluated.\n\nClustering includes centroid-based K-Means and bisecting K-Means, Gaussian mixtures, hierarchical agglomeration, spectral clustering, BIRCH, DBSCAN and HDBSCAN. Label propagation uses normalized graph diffusion or hard-clamped graph inference. Supervised comparisons include logistic/linear and kernel SVM, neighbors, ridge/kernel ridge, shrinkage discriminants, Gaussian NB, random/extra trees, three boosting libraries, classifier chains, powerset and soft multiclass models, Gaussian processes, NCA and PLS. Additional tests include semi-supervised EM, Laplacian-regularized kernels, prototype dictionaries and consensus pseudo-labeling.')
code('''families = pd.read_csv(root/'experiments/model-family-summary.csv')
display(families[['family','best_development_f1','configurations','cv_runs']].round(4))
fig, ax = plt.subplots(figsize=(9,9))
families.sort_values('best_development_f1').plot.barh(x='family',y='best_development_f1',ax=ax,legend=False,color='#326b8c')
ax.set(xlim=(0.55,1),xlabel='Best development sample-F1 (selection-biased)',ylabel='')
fig.tight_layout(); fig.savefig(figdir/'model_families.png'); plt.show()
''')
md('## Features and preprocessing\n\nAll 150 rotated reason coordinates are kept as a block. Standard scaling, rotation-preserving global scaling, row normalization, PCA dimensions and whitening strengths were compared. Auxiliary groups include year, eligibility word/bigram and character TF-IDF, disease names, drugs, ICD codes, matched drugs, targets, actions, pathways, broad pathway filtering, pathway-depth counts, Morgan fingerprints, molecular descriptors, missingness and field counts. Each group was initially added alone at several weights.\n\nThe first-seed result is the selection score. When available, the fresh-seed mean below uses that first-seed decision rule unchanged. A missing fresh-seed value means the configuration was screened out or another configuration was repeated; it is not evidence of a gain. Some apparent improvements disappeared across splits. Chemistry and selected drug/text blocks were retained in the final candidate because of repeated-CV and combination checks, not per-validation-row edits. More features can overwhelm a tiny labeled set, alter kernel distances and encode fragile disease/year associations. Fingerprint missingness also reflects incomplete drug matching.')
code('''ablation = pd.read_csv(root/'experiments/feature-ablation.csv')
display(ablation.round(4))
repeat = ablation.dropna(subset=['fresh_seed_mean_f1']).sort_values('fresh_seed_mean_f1')
fig, ax = plt.subplots(figsize=(9,4.5))
repeat.plot.barh(x='feature',y=['first_seed_f1','fresh_seed_mean_f1'],ax=ax,color=['#b8cdd9','#326b8c'])
ax.set(xlim=(.75,1),xlabel='Development sample-F1',ylabel=''); ax.legend(['Selected first-seed score','Fresh seeds, frozen rule'],loc='lower right')
fig.tight_layout(); fig.savefig(figdir/'feature_robustness.png'); plt.show()
''')
md('## External validation — preserve the disappointing result\n\nThese are the only three external-validation model evaluations. All model definitions and thresholds were fixed beforehand. The selected validation result has selection optimism, and 78 rows leave wide uncertainty. It is lower than the user’s approximate previous validation performance. **No leaderboard improvement is established.**')
code('''vtable = pd.DataFrame([{k:r[k] for k in ['name','sample_f1','macro_f1']+[f'f1_{l}' for l in labels]} for r in val['candidates']])
display(vtable.round(5))
print('Selected-model grouped bootstrap 95% interval:', val['bootstrap_95_interval'])
nested = json.loads((root/'experiments/nested-audit.json').read_text())
print('Conditional nested shortlist sample-F1:', nested['sample_f1'])
temporal = pd.read_json(root/'experiments/temporal-audit.json')
display(temporal[['cutoff','n_train','n_test','sample_f1','macro_f1']].round(4))
''')
md('## Final full-training candidate and augmentation\n\nThe final candidate uses all 157 true labeled rows, unlike the earlier 15-model fold-bagging prediction strategy. It is a 50% HDBSCAN-consensus pseudo-label logistic model with Morgan fingerprints, 25% RBF SVM with criteria/drug TF-IDF, and 25% RBF SVM with target TF-IDF.\n\nFor the pseudo-label component, HDBSCAN is fitted to 25 PCA coordinates of standardized reason embeddings, without labels. Cluster labels are mapped using gold fold-training rows; noise/unlabeled clusters fall back to a distance-weighted labeled-neighbor estimate. A separate logistic model supplies the second view. Only agreed binary labels with average confidence at least 0.95 are accepted into the per-label fit, each at weight 0.05. Original true labels are retained. The auxiliary fingerprint uses radius 2 and 1,024 bits, max-pooled across valid supplied molecules; unmatched/invalid molecules contribute zero.\n\nFor final predictions, member probabilities use fixed weights 0.50/0.25/0.25 and a 0.35 threshold, with highest-probability fallback for empty predictions. Every masked row receives five binary labels. Pseudo-labels are predictions, not newly verified ground truth. The final prediction recipe and SHA256 are saved in `final-candidate.json`. The final candidate was **not re-evaluated on the used validation set**. Its 0.93631 leave-group-out score remains a selection-biased development estimate, awaiting external confirmation.')
code('''display(pd.DataFrame([{'candidate':r['name'],'leave_group_out_f1':r['fullsize_oof_f1'],'rule':r['rule']} for r in json.loads((root/'experiments/fullsize-ranking.json').read_text())]))
print('Final CSV:', final['submission'])
print('Kaggle score:', final['kaggle_score'])
aug = pd.read_csv(root/'predictions/part1_train_augmented_fullfit.csv')
display(aug.groupby('label_source')[labels].sum())
print('Ground truth rows unchanged:', np.array_equal(aug.loc[known,labels].to_numpy(),train.loc[known,labels].to_numpy()))
print('Generated masked rows:', (~known).sum())
''')
md('## Limitations and next step\n\nThe requested winning score is **not achieved or verified**. Submit `predictions/part1_fullfit_candidate.csv` once and record its Kaggle result before claiming improvement. The first holdout failure is evidence against assuming that the higher CV values transfer. The final full-fit regime addresses training-size differences but does not erase that evidence.\n\nThe broad search was structured, not exhaustive. Reusing tiny training folds for many choices creates selection bias despite strict per-fit leakage controls. Rare-label estimates are unstable; the temporal sampling scheme differs from both ordinary CV and future tests. Strong uncertainty, pseudo-label error and class-distribution shifts remain. Part 2 and the final combined five-page assignment report are outside this Part 1 package.')
refs=json.loads(Path('experiments/references.json').read_text())
md('## References\n\n'+'\n\n'.join(f"- [{r['title']}]({r['url']}) — {r['use']}" for r in refs))
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'aml-midterm','display_name':'AML Midterm','language':'python'}})
nbf.write(nb,'midterm/01_experiment_report.ipynb')
# Source notebook embeds every script and every static configuration, so the source is retained in .ipynb form.
sourcecells=[nbf.v4.new_markdown_cell('# Reproduce the Part 1 experiments\n\nThis notebook embeds the complete implementation and helper scripts. It can recreate them in an isolated `reproduction` directory; original results remain unchanged. Full search is opt-in because it runs hundreds of fits. The report notebook executes directly from saved evidence. No deep learning or external trial records are used.\n\nInstall the locked Python 3.12 environment using the commands in `READ_ME_FIRST.txt`. On macOS the XGBoost/LightGBM screen needs libomp. The selected model does not. The first validation result remains separate from the later unvalidated full-fit candidate.')]
sourcecells.append(nbf.v4.new_code_cell("from pathlib import Path\nimport shutil, subprocess, sys, json\nroot = Path.cwd()\nreplica = root / 'reproduction'\nreplica.mkdir(exist_ok=True)\n(replica/'experiments').mkdir(exist_ok=True)\n(replica/'predictions').mkdir(exist_ok=True)\nshutil.copytree(root/'data', replica/'data', dirs_exist_ok=True)\n"))
for f in sorted(Path('.').glob('*.py')):
 if f.name=='build_deliverables.py':continue
 sourcecells.append(nbf.v4.new_markdown_cell('### '+f.name+'\n\nComplete source:'))
 sourcecells.append(nbf.v4.new_code_cell('source = '+repr(f.read_text())+'\n(replica / '+repr(f.name)+').write_text(source)\n'))
configs={p.name:json.loads(p.read_text()) for p in Path('experiments').glob('*-input.json')}
sourcecells.append(nbf.v4.new_code_cell('static_configs = '+repr(configs)+'\nfor name, content in static_configs.items():\n    (replica/"experiments"/name).write_text(json.dumps(content, indent=2))\n'))
sourcecells.append(nbf.v4.new_markdown_cell('## Fast exact prediction reproduction\n\nThis fits the frozen final model recipe using only training labels and the provided masked training rows. It does not read validation labels. Set `RUN_FINAL = True` to run. A fresh output checksum is compared with the observed candidate.'))
sourcecells.append(nbf.v4.new_code_cell('''RUN_FINAL = False
if RUN_FINAL:
    for name in ['fullsize-ranking.json','final-candidate.json']:
        shutil.copy2(root/'experiments'/name, replica/'experiments'/name)
    subprocess.run([sys.executable,'make_candidate.py'],cwd=replica,check=True)
    import hashlib
    observed = hashlib.sha256((root/'predictions/part1_fullfit_candidate.csv').read_bytes()).hexdigest()
    reproduced = hashlib.sha256((replica/'predictions/part1_fullfit_candidate.csv').read_bytes()).hexdigest()
    assert observed == reproduced, (observed,reproduced)
    print('Exact prediction checksum matched:', observed)
'''))
sourcecells.append(nbf.v4.new_markdown_cell('## Full search replay\n\nThe executable command list below covers all logged search stages. Repeated seeds and adaptive stage configuration files are retained in the evidence. For an exact historical replay, copy the recorded JSON input/configuration files; do not re-select model families after seeing external validation. Nested and leave-group-out audits can be run separately. `finalize.py` deliberately refuses a second external validation comparison when its result already exists.'))
commands=[['experiment.py'],['prepare_refinement.py'],['experiment.py','--stage','refine','--configs','experiments/refine-input.json'],['experiment.py','--stage','features','--configs','experiments/features-input.json'],['prepare_consensus.py'],['experiment.py','--stage','consensus','--configs','experiments/consensus-input.json'],['robustness.py'],['prepare_metric.py'],['experiment.py','--stage','metric','--configs','experiments/metric-input.json']]
for stage,cf in [('ssl_features','ssl-features'),('phase2_repeat','phase2-repeat')]:
 for seed in ([20261104,20261105,20261106] if stage=='ssl_features' else [20261105,20261106]):commands.append(['experiment.py','--stage',stage,'--configs',f'experiments/{cf}-input.json','--seed',str(seed)])
commands.extend([['experiment.py','--stage','dictionary','--configs','experiments/dictionary-input.json'],['experiment.py','--stage','phase2','--configs','experiments/phase2-input.json']])
sourcecells.append(nbf.v4.new_code_cell('RUN_SEARCH = False\ncommands = '+repr(commands)+'\nif RUN_SEARCH:\n    for command in commands:\n        subprocess.run([sys.executable,*command],cwd=replica,check=True)\n'))
sourcecells.append(nbf.v4.new_code_cell("RUN_AUDITS = False\nif RUN_AUDITS:\n    shutil.copytree(root/'experiments',replica/'experiments',dirs_exist_ok=True)\n    for script in ['verify_engine.py','audit_candidates.py','fullsize_cv.py']:\n        subprocess.run([sys.executable,script],cwd=replica,check=True)\n"))
nbf.write(nbf.v4.new_notebook(cells=sourcecells,metadata={'kernelspec':{'name':'aml-midterm','display_name':'AML Midterm','language':'python'}}),'midterm/02_reproduce.ipynb')
# Portable folder contains the notebook-visible evidence and source files, but never .venv or credentials.
for folder in ['data','experiments','predictions']:
 shutil.copytree(folder,Path('midterm')/folder,dirs_exist_ok=True)
for p in list(Path('.').glob('*.py'))+[Path('requirements-lock.txt'),Path('READ_ME_FIRST.txt')]:shutil.copy2(p,Path('midterm')/p.name)
print('Deliverables built',len(rows),'runs',trial.id.nunique(),'configurations',len(summary),'families')
