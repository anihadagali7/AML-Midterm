"""Build a current, auditable Part 1 report without rerunning model selection."""
from pathlib import Path
import json,hashlib,shutil,zipfile,re,datetime
import numpy as np,pandas as pd,nbformat as nbf
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parent
P=ROOT/'experiments'
rows=[json.loads(s) for s in (P/'trials.jsonl').read_text().splitlines()]
flat=[]
for i,r in enumerate(rows):
 z={k:v for k,v in r.items() if not isinstance(v,(list,dict))};z['log_row']=i;z['config_json']=json.dumps(r['config'],sort_keys=True);z['config']=z['config_json'];flat.append(z)
pd.DataFrame(flat).to_csv(P/'trials.csv',index=False)
validation=[]
files=[P/'validation-once.json',P/'final-validation.json',P/'phase3-reference-validation.json']+sorted(P.glob('phase*-validation-results.json'))
for path in files:
 if not path.exists():continue
 obj=json.loads(path.read_text())
 if path.name=='validation-once.json':items=obj['candidates']
 elif path.name=='phase3-reference-validation.json':items=obj['results']
 elif isinstance(obj,list):items=obj
 else:items=[obj]
 for i,r in enumerate(items):
  m=r.get('validation',r)
  score=m.get('sample_f1',r.get('validation_sample_f1'))
  if score is None:continue
  config=r.get('config',{'members_by_label':r.get('members_by_label')});name=r.get('name',r.get('family',config.get('model','candidate_'+str(i))))
  validation.append(dict(batch=path.stem,candidate=name,rule=r.get('rule'),sample_f1=score,macro_f1=m.get('macro_f1',r.get('validation_macro_f1')),config_json=json.dumps(config,sort_keys=True)))
val=pd.DataFrame(validation);val.to_csv(P/'validation-comparisons.csv',index=False)
frozen=json.loads((P/'current-best.json').read_text());candidate=frozen['candidate'];m=candidate['validation']
assert abs(val.sample_f1.max()-frozen['best_validation_f1'])<1e-12,'Update current-best.json before reporting a new winner.'
# Descriptive uncertainty only: repeated model selection invalidates a pristine-test interpretation.
v=pd.read_csv(ROOT/'data/part1_val.csv');a=pd.read_csv(ROOT/'predictions/current_best/validation_audit.csv')
g=pd.util.hash_pandas_object(v[[f'emb_{i}' for i in range(150)]].round(10),index=False).to_numpy();groups=np.unique(g);tot=np.array([a.loc[g==k,'sample_f1'].sum() for k in groups]);counts=np.array([(g==k).sum() for k in groups]);rng=np.random.default_rng(20261104);draw=rng.integers(len(groups),size=(5000,len(groups)));boot=tot[draw].sum(1)/counts[draw].sum(1)
ci=np.quantile(boot,[.025,.975]).tolist()
summary=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),logged_cv_runs=len(rows),successful_cv_runs=sum(r['status']=='ok' for r in rows),failed_attempts=sum(r['status']!='ok' for r in rows),distinct_configs=len({r['id'] for r in rows}),validation_candidate_evaluations=len(val),validation_batches=len(val.batch.unique()),best_validation_f1=m['sample_f1'],validation_macro_f1=m['macro_f1'],descriptive_group_bootstrap_95=ci,validation_embedding_groups=len(groups),loo_current_selected=frozen.get('leave_one_embedding_group_out_f1'),target=.92735,target_met=m['sample_f1']>.92735,kaggle_score=None,limitation='Repeated CV and validation selection cause optimism; bootstrap interval is descriptive and does not adjust for selection. Equivalent model variants and verification comparisons can recur in the validation ledger.')
summary['additional_fixed_policy_cv']=frozen.get('additional_fixed_policy_cv')
summary['reported_leader']='0.93xx; exact score not provided'
summary['above_reported_leader_lower_bound']=m['sample_f1']>.93
primary_cv_text=f"Historical model-selection repeated CV F1: {candidate['ordinary_cv']:.8f}."
pooled_cv_text=''
benchmark_path=P/'benchmark_v1/summary.json'
if benchmark_path.exists():
 bench=json.loads(benchmark_path.read_text())
 registry=json.loads((P/'benchmark_v1/registry.json').read_text())
 matching=[r for r in registry if r['config']==candidate['config'] and r['rule']==candidate['rule']]
 if matching:
  common=next(r for r in bench['models'] if r['candidate']==matching[0]['name'])
  summary['common_cv_benchmark']={**common,'protocol':bench['protocol'],'seeds':bench['seeds']}
  primary_cv_text=f"Primary common grouped five-fold CV F1: {common['cv_mean']:.8f}; repeat standard deviation: {common['repeat_std']:.5f}.\nSame saved folds and seeds {bench['seeds']} for every candidate. See notebook 10 and experiments/benchmark_v1/comparison.csv."
pooled_path=P/'pooled_benchmark_v1/summary.json'
if pooled_path.exists():
 pooled=json.loads(pooled_path.read_text())
 pooled_registry=json.loads((P/'pooled_benchmark_v1/registry.json').read_text())
 matching=[r for r in pooled_registry if r['config']==candidate['config'] and r['rule']==candidate['rule']]
 if matching:
  pooled_row=next(r for r in pooled['models'] if r['candidate']==matching[0]['name'])
  summary['supplementary_pooled_cv']=pooled_row
  pooled_cv_text=f"Supplementary pooled train-plus-validation CV F1: {pooled_row['pooled_cv_mean']:.8f}; repeat standard deviation: {pooled_row['pooled_cv_std']:.5f}. This evaluates fresh fold fits of the same recipe on 235 pooled labels, not the exact saved model. See notebook 11."
(P/'current-summary.json').write_text(json.dumps(summary,indent=2))
additional_cv_text=''
if frozen.get('additional_fixed_policy_cv'):
 audit=frozen['additional_fixed_policy_cv']
 additional_cv_text=f"Additional fixed-policy grouped five-fold audit on seeds {audit['seeds']}: {audit['mean_sample_f1']:.8f} (split standard deviation {audit['split_std']:.5f}). Different split sets give different estimates; these are development audits, not a nested replay of the full search.\n"
# Add technique references; these sources contain no added trial data.
refs=json.loads((P/'references.json').read_text())
add=[('Joachims (1999), Transductive Inference for Text Classification using Support Vector Machines','https://www.cs.cornell.edu/people/tj/publications/joachims_99c.pdf','Motivation for low-density semi-supervised margins. Our alternating pseudo-label procedure is a heuristic, not an implementation of the exact TSVM optimization algorithm.'),('Huang et al., Correcting Sample Selection Bias by Unlabeled Data','https://papers.nips.cc/paper_files/paper/2006/hash/a2186aa7c086b46ad4e8bf81e2a3a19b-Abstract.html','Kernel mean matching weights, estimated only against original masked training features.'),('On Mixup Regularization','https://www.jmlr.org/papers/v23/20-1385.html','Motivation for synthetic interpolation; evaluated with classical logistic/SVM learners only.'),('Nystroem kernel approximation, scikit-learn','https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html','Classical finite kernel features for comparing squared hinge, logistic and squared losses.')]
for title,url,use in add:
 if not any(r['url']==url for r in refs):refs.append(dict(title=title,url=url,use=use))
(P/'references.json').write_text(json.dumps(refs,indent=2))
text=f'''AML MIDTERM — PART 1 — CURRENT RESULTS

Best measured validation sample-wise F1: {m['sample_f1']:.8f}
Validation macro F1: {m['macro_f1']:.8f}
Earlier validation target: strictly above 0.92735. Target met: {summary['target_met']}.
Latest reported Kaggle leader: 0.93xx, exact score not provided. Local validation above 0.93: {summary['above_reported_leader_lower_bound']}.
This modestly exceeds the user's reported 0.9103 validation result; it does not establish a statistically reliable improvement.
{primary_cv_text}
{pooled_cv_text}
Kaggle score: not measured. Validation and CV are not leaderboard scores.

CURRENT FILES
predictions/current_best/part1_submission.csv — 312 rows, ID and Predicted.
predictions/current_best/part1_train_augmented.csv — all 157 original gold labels preserved; 1,014 missing-label rows predicted.
predictions/current_best/manifest.json — exact model, metrics, source hashes and submission hash.
midterm/03_current_report.ipynb and .html — current report and figures.
midterm/04_reproduce_current.ipynb — embedded source and executed reproduction check.
midterm/05_tri_training_and_shift.ipynb — tri-training, target-aware selection, matched controls and results (when present).
midterm/06_stability_and_specialists.ipynb — bootstrap stability, feature additions and nested score specialists (when present).
midterm/07_cotraining_and_distances.ipynb — separate-view co-training, learned distances and matched controls (when present).
midterm/08_svm_refinements.ipynb — per-reason tuning, reliability weights and local SVM specialists (when present).
midterm/09_shift_graph_and_subgroups.ipynb — label-frequency correction, graph SVMs and Other subgroups (when present).
midterm/10_common_cv_benchmark.ipynb — primary common-fold model comparison with separate existing validation scores (when present).
midterm/11_pooled_cv_comparison.ipynb — supplementary train-plus-validation CV of frozen leading recipes (when present).
experiments/benchmark_v1/protocol.json and comparison.csv — saved fold assignments and primary comparison table (when present).
experiments/validation-comparisons.csv — every recorded validation comparison, including failures to improve.
experiments/trials.csv and trials.jsonl — {len(rows)} logged CV runs, {summary['distinct_configs']} distinct configurations, {summary['failed_attempts']} failed attempts retained.
Full-size CV and ensemble-only screens have separate files; they are not all counted in the main trial log.

REPRODUCE
From this directory (or the portable midterm directory):
  .venv/bin/python reproduce_current.py
For a new environment, use Python 3.12 and requirements-lock.txt.
The script reads the frozen experiments/current-best.json configuration, trains only on supplied training rows, writes predictions, and then verifies its validation score. It does not tune on validation labels.

CURRENT METHOD
A fixed per-reason combination uses the previous semi-supervised SVM for Safety, Funding and Efficacy, and classical tri-training for Recruitment and Other. This combination was selected from 32 masks using training-fold predictions and the frozen robust weighting criterion before validation evaluation. Empty combined predictions fall back to the base SVM's decoded labels.
Both components fit StandardScaler to gold and original masked training embeddings, retaining all 150 coordinates. No validation/test features enter the selected inductive fitters.
Base component: KMeans with 10 clusters, 10 initializations and seed 42, then five balanced RBF SVMs with C=1 and gamma=0.001. At each pseudo-label weight 0.003, 0.01 and 0.03, perform three label/refit iterations. Original gold labels remain fixed. If a cluster has at least two gold rows and at least 80% agreement for a binary reason, conflicting pseudo-labels receive one tenth the normal weight. Transform margins with sigmoid(2*margin), threshold at 0.55, and use the largest score if all labels would be empty.
Tri-training component: ordinary bootstrap samples initialize an RBF SVM (C=1, gamma=0.001), logistic regression (C=0.3), and ridge classifier (alpha=10). Peer agreement and the classical error/size conditions govern pseudo-label updates, with unit pseudo-label weight and no cluster gate in this component. Majority voting at threshold 0.5 decodes predictions; smoothed vote scores break ties for empty-label fallback. At most eight update rounds are used.
All 157 gold and 1,014 originally masked training rows are available for final inference. Validation remains excluded. Final hybrid scores are binary decisions; component margin/vote scores are not calibrated probabilities.
Earlier screens tested auxiliary criteria, disease, drug, target, action, pathway, structure and date features. They did not improve the earlier incumbent on validation; drug-target additions tied it. The new hybrid uses embeddings only. Exhaustive feature-subset testing of this new hybrid has not been performed.

EVIDENCE AND LIMITATIONS
{len(val)} recorded standalone validation candidate evaluations across {summary['validation_batches']} batches, including equivalent variants. Their hyperparameters and decision rules were chosen from training-side experiments before each small validation batch. Those original fits excluded validation labels and made no per-row corrections. Repeated aggregate comparisons still create selection bias: this is no longer a pristine holdout. The separately authorized pooled CV deliberately uses original validation labels only in the training portion of each fold, with all held-out groups excluded; it does not replace the saved training-only model or reinterpret its original validation score.
Five-fold iterative multilabel splits group identical embeddings and exclude matching masked duplicates from training. Repeats use three seeds. Leave-one-group-out evaluates 150 distinct gold embedding groups. All learned transformations stay inside their permitted folds. Temporal stress tests are affected by the intentional rare-label enrichment of later gold rows.
Descriptive grouped validation bootstrap interval: {ci[0]:.4f}–{ci[1]:.4f}; this does not correct selection bias.
The search is broad, not an exhaustive enumeration of every possible algorithm or feature subset. No external trial data, extra labels, ID predictors or deep learning were used. Literature was consulted only for methods. Some explicitly marked research variants used query features transductively; the selected candidate does not.

HISTORY
Historical model-selection repeated CV F1: {candidate['ordinary_cv']:.8f}; leave-one-embedding-group-out development F1: {summary['loo_current_selected']:.8f}. These historical diagnostics do not replace the primary common benchmark.
{additional_cv_text}
The earlier delivered full-fit ensemble scored 0.850427 on validation; its 0.936306 training CV did not transfer. It is preserved under the older filenames and in notebooks 01/02, which describe that historical stage. The previous 0.910256 validation incumbent is preserved under predictions/checkpoints/pre_phase21_091026 and experiments/phase21-prior-incumbent.json. Use the current_best directory and notebooks 03/04/05 for the current candidate.
Part 2 and the combined five-page course report remain outside this Part 1 work.
'''
(ROOT/'READ_ME_FIRST.txt').write_text(text)
# Current report contains executable tables/figures, while full source is embedded separately.
nb=nbf.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python (AML Midterm)',language='python',name='aml-midterm')
nb.cells=[nbf.v4.new_markdown_cell('# Part 1: current independent experiments\n\n'+text.split('CURRENT FILES')[0]),
 nbf.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd, numpy as np\nimport matplotlib.pyplot as plt\nroot=Path.cwd()\nsummary=json.loads((root/'experiments/current-summary.json').read_text())\nsummary"),
 nbf.v4.new_markdown_cell('## Selected method and validation discipline\n\n'+text.split('CURRENT METHOD\n')[1].split('\nHISTORY')[0]),
 nbf.v4.new_code_cell("ledger=pd.read_csv(root/'experiments/validation-comparisons.csv')\nledger.groupby('batch',sort=False).agg(candidates=('sample_f1','size'),best_sample_f1=('sample_f1','max'),best_macro_f1=('macro_f1','max'))"),
 nbf.v4.new_markdown_cell('## Current validation performance\n\nPer-reason F1 complements the competition’s sample-wise F1. The latest reported Kaggle leader is in the 0.93xx range; its exact score is unknown. Our earlier 0.92735 target and the newer leaderboard reference both concern sample-wise F1, measured on different evaluation datasets.'),
 nbf.v4.new_code_cell("current=json.loads((root/'experiments/current-best.json').read_text())\nm=current['candidate']['validation']\nlabels=['Safety','Funding','Efficacy','Recruitment','Other']\nfig,ax=plt.subplots(figsize=(7,3.4))\nax.bar(labels,[m['f1_'+x] for x in labels],color='#2563a0')\nax.set(ylim=(0,1.04),ylabel='Validation label F1',title=f\"Sample F1 {m['sample_f1']:.5f}; macro F1 {m['macro_f1']:.5f}\")\nfig.tight_layout()\n(root/'figures').mkdir(exist_ok=True)\nfig.savefig(root/'figures/current_validation_labels.png',dpi=180)\nplt.show()"),
 nbf.v4.new_markdown_cell('## Feature additions and model coverage\n\nFeature settings below use training CV, including multiple weights. Their maxima are development scores, not independent estimates. Broad feature groups were added individually; selected groups were combined in earlier forward searches. No auxiliary group improved the earlier incumbent on validation. The new tri-training hybrid uses embeddings only; it has not undergone an exhaustive feature-subset search.'),
 nbf.v4.new_code_cell("trials=[json.loads(s) for s in (root/'experiments/trials.jsonl').read_text().splitlines()]\nfeatures=[dict(features=', '.join(r['config'].get('features',[])),weight=r['config'].get('feature_weight'),cv=r['score']) for r in trials if r['tag']=='phase6_features' and r['status']=='ok']\npd.DataFrame(features).sort_values('cv',ascending=False)"),
 nbf.v4.new_code_cell("ok=[r for r in trials if r['status']=='ok']\npd.DataFrame([dict(model=r['config']['model'],cv=r['score']) for r in ok]).groupby('model').agg(logged_runs=('cv','size'),best_development_cv=('cv','max')).sort_values('best_development_cv',ascending=False)"),
 nbf.v4.new_markdown_cell('## Reproduction and historical results\n\nRun notebook 04 to regenerate the current predictions with embedded source. All raw data hashes and original gold labels are checked. Notebooks 01/02 preserve the earlier 0.850427 validation model; their final-candidate wording is historical. The newer notebooks and current_best directory supersede that candidate.\n\nThe full experiment log retains unsuccessful trials and error records. Phase 10 initially had a non-contiguous sample-weight array error, which was corrected before valid results. The first phase 17 XGBoost run emitted an unused `lambda_` warning and used native default L2=1; the source now spells the same effective value as `reg_lambda=1`. No measured hyperparameter changed in this correction.'),
 nbf.v4.new_markdown_cell('## Technique references\n\n'+'\n\n'.join(f"[{r['title']}]({r['url']}) — {r['use']}" for r in refs))]
mid=ROOT/'midterm';mid.mkdir(exist_ok=True)
nbf.write(nb,mid/'03_current_report.ipynb')
# Reproduction notebook embeds every Python source, including search runners and this report builder.
source_files=sorted(p.name for p in ROOT.glob('*.py'))
rep=nbf.v4.new_notebook();rep.metadata.kernelspec=nb.metadata.kernelspec
rep.cells=[nbf.v4.new_markdown_cell('# Reproduce the current Part 1 candidate\n\nThis notebook embeds all experiment Python source. It creates an isolated replay folder, regenerates the frozen candidate, checks raw hashes and gold-label preservation, and compares the submission hash. It does not rerun model selection.'),nbf.v4.new_code_cell("from pathlib import Path\nimport shutil,sys,subprocess,json,hashlib\nroot=Path.cwd(); replay=root/'current_reproduction';replay.mkdir(exist_ok=True)\nshutil.copytree(root/'data',replay/'data',dirs_exist_ok=True)\n(replay/'experiments').mkdir(exist_ok=True)\nfor name in ['current-best.json','data-audit.json']:\n    shutil.copy2(root/'experiments'/name,replay/'experiments'/name)")]
for name in source_files:rep.cells.append(nbf.v4.new_code_cell('%%writefile current_reproduction/'+name+'\n'+(ROOT/name).read_text()))
rep.cells.extend([nbf.v4.new_code_cell("result=subprocess.run([sys.executable,'reproduce_current.py','--output','result'],cwd=replay,check=True,text=True,capture_output=True)\nprint(result.stdout)\nexpected=json.loads((root/'predictions/current_best/manifest.json').read_text())['submission_sha256']\nactual=hashlib.sha256((replay/'result/part1_submission.csv').read_bytes()).hexdigest()\nassert actual==expected,(actual,expected)\nprint('Exact current submission reproduced:',actual)"),nbf.v4.new_markdown_cell('To replay searches, use the phase scripts and their frozen config JSON files. `experiment.py --help` lists the original search entry point. Most later `phaseN.py` scripts run their own training-CV screen; `continued_select.py phaseN` performs documented repeat-CV selection and validation. Running these intentionally expands the experiment log and repeats validation; it is not required to reproduce the frozen candidate.')])
nbf.write(rep,mid/'04_reproduce_current.ipynb')
# Portable directory mirrors inputs and artifacts; keep historical notebooks and outputs.
for name in ['data','experiments','predictions']:
 shutil.copytree(ROOT/name,mid/name,dirs_exist_ok=True)
for file in list(ROOT.glob('*.py'))+[ROOT/'READ_ME_FIRST.txt',ROOT/'requirements-lock.txt']:
 shutil.copy2(file,mid/file.name)
if (ROOT/'AGENTS.md').exists():shutil.copy2(ROOT/'AGENTS.md',mid/'AGENTS.md')
print(json.dumps(summary,indent=2))
