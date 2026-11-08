"""Registered PySR experiment on all 84 frozen cases; no Kaggle upload."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','JULIA_NUM_THREADS','JULIA_NUM_PRECOMPILE_TASKS']:
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import importlib.metadata
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import HERE,PART1,ROOT,OLD,benchmark,LABELS,FEATURES,sha,save
sys.path.insert(0,str(HERE))
from model import SymbolicReasons,SETTINGS,SEARCH,FEATURE_NAMES,centered,decide,groups

MODES=['anchored','current75_25','anchored80_cand2_20','pysr','linear_geometry','current80_pysr20']
CANDIDATES=MODES[3:]


def sources():
    paths=[HERE/n for n in ['common.py','model.py','study.py','build_controls.py','controls.npz','control_provenance.json','best_control_result.json','requirements-lock.txt','JuliaProject.toml','JuliaManifest.toml']]
    paths += [OLD/'run.py',OLD/'models.py',PART1/'anchored_solution.py',PART1/'TESTING_POLICY.md',ROOT/'Midterm_Project.pdf']
    paths += list((PART1/'data').glob('*.csv'))
    paths += [PART1/'experiments/current_leaders_grouped_v1/snapshots'/(s+'.json') for s in ['train3','train5','pooled3','pooled5','chronology']]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def guard():
    assert sources()==__import__('json').loads((HERE/'protocol.json').read_text())['sources']


def register():
    assert not (HERE/'protocol.json').exists()
    save(HERE/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        request='User specifically requested trying PySR after closing prior blend experiments.',
        parent='Anchored 0.89636 policy parent; own submitted 0.89850 blend and current cross-agent 0.89935 blend are matched practical controls.',
        hypothesis='Short symbolic formulas can combine nonlinear interactions between neighbor support, class-centroid affinity, cluster label support and competing reasons, without searching individual randomly rotated coordinates.',
        feature_access='Only 150 supplied reason embeddings. Fit StandardScaler, normalization and KMeans(12) on eligible fitting rows. Gold-derived nearest neighbors, positive-minus-negative centroid cosine margins and smoothed cluster label rates use fitting gold only. Each training row excludes its complete embedding group from these label-derived signals; held-out queries use the frozen fitting bank. Five features per reason: its neighbor logodds, centroid margin, cluster logodds and strongest competing neighbor logodds/centroid margin. Clip rate inputs to [.01,.99]; scale derived features using fitting rows only.',
        augmentation='Generate every originally masked fitting labelset using KMeans cluster gold rates with one global-prior pseudocount, .3 threshold and argmax. Preserve observed gold exactly. Pseudo rows have total weight .10 times gold-row count. IDs serve only split/group audits.',
        settings=SETTINGS,search=SEARCH,search_seeds=[42,43,44,45,46],feature_names=FEATURE_NAMES,
        search_boundary='Each PySR search is a fresh fit wholly inside the outer training partition. Serial deterministic evolution; no wall-clock early cutoff. Formula selected using PySR best training loss/complexity rule, never held-out labels. At most 15 nodes and depth5, operators +,-,* only. Weighted logistic loss; sigmoid gives label scores. No later threshold or weight search.',
        recipes={'pysr':'Five independent PySR binary heads; sigmoid, .3 cutoff, nonempty argmax. Constant smoothed prevalence fallback only if fitting target has one class.',
                 'linear_geometry':'Diagnostic LogisticRegression(C=1) heads on precisely the same features, generated labels, weights and cutoff. Distinguishes signal quality from symbolic-search value; not another hyperparameter search.',
                 'current80_pysr20':'Fixed .8 submitted75_25 centered scores + .2 PySR scores centered from .3; .5 cutoff plus argmax. No fitting of blending weights.'},
        evaluation='All eight saved five-fold repeats in both Train157 and Pooled235, validation78 and all three chronology windows for every recipe, irrespective of screening result. 84 fresh geometry pipelines and up to420 symbolic searches plus420 linear diagnostic heads. Controls are reused exact case-matched predictions, not new fits.',
        advancement='Positive paired mean and >=5/8 repeat wins in BOTH pools versus anchored; separately report same rule against both submitted blends. Keep every supplementary regression visible; no automatic promotion or Kaggle upload.',
        boundaries='Strict inductive. Existing saved split generator enforces masked duplicate purges and chronological fitting years. Query labels never passed to fit or feature generation. Train157 excludes all validation-source rows. Actual Kaggle test inputs hashed only; final fit only if local evidence merits a candidate.',
        setup_checks='Synthetic-only API, weighted-loss, query-batch-invariance and timing checks before evaluation; no real-label pilot or tuning.',
        retention='Source, fixed settings, dependencies, all compact scores, paired deltas, equations and fit audits retained. Full predictions only for qualifying candidates/useful controls. Independently rescore before deleting failed caches. No fitted PySR models or search logs retained.',
        references=['https://pysr.ai/api','https://github.com/astroautomata/PySR','https://arxiv.org/abs/2305.01582'],
        limitations='A compact fixed PySR search budget and one feature construction, not an exhaustive test of symbolic regression. Existing validation data have influenced prior research. Label-dependent similarity features are part of the model, so these are not equations directly in original physical variables.',
        versions={n:importlib.metadata.version(n) for n in ['pysr','juliacall','numpy','pandas','scipy','scikit-learn']},julia='1.13.3',sources=sources()))


def controls():
    with np.load(HERE/'controls.npz') as z:return {k:z[k] for k in z.files}


def evaluate(c):
    start=time.monotonic();fit,q=c['fit'],c['query']
    x=fit[FEATURES].to_numpy(float);y=fit[LABELS].to_numpy(int);qx=q[FEATURES].to_numpy(float)
    with threadpool_limits(limits=1):
        model=SymbolicReasons().fit(x,y);s,l=model.predict_scores(qx)
        np.testing.assert_allclose(model.predict_scores(qx[:1])[0],s[:1],atol=1e-12,rtol=0)
    ref=controls();mask=ref['case']==c['key'];np.testing.assert_array_equal(ref['ids'][mask],q.ID);np.testing.assert_array_equal(ref['truth'][mask],q[LABELS])
    old=ref['scores'][:,mask];scores=np.concatenate([old,np.stack([centered(s),centered(l),.8*old[1]+.2*centered(s)])])
    pred=np.stack([decide(v,.5) for v in scores]);assert scores.shape==(6,len(q),5)
    packet=dict(ids=q.ID.to_numpy(str),truth=q[LABELS].to_numpy(int),scores=scores,predictions=pred)
    audit=dict(case=c['key'],pool=c['pool'],seed=c['seed'],fold=c['fold'],fit_rows=len(fit),query_rows=len(q),purged_masked=c['purged_masked'],
        fit_ids_sha256=__import__('hashlib').sha256('\n'.join(fit.ID).encode()).hexdigest(),query_batch_invariant=True,
        fitting_query_ids_disjoint=set(fit.ID).isdisjoint(q.ID),fitting_query_groups_disjoint=set(groups(x)).isdisjoint(groups(qx)),
        validation_group_overlap_allowed=c['key']=='validation',seconds=time.monotonic()-start,**model.audit)
    assert audit['fitting_query_ids_disjoint']
    if c['key']!='validation':assert audit['fitting_query_groups_disjoint']
    return c['key'],packet,audit,model.equations


def run(output,workers):
    guard();output=Path(output);(output/'predictions').mkdir(parents=True,exist_ok=True)
    jobs=benchmark.cases();pending=[]
    for c in jobs:
        files=[output/'predictions'/(c['key']+ext) for ext in ['.npz','.json']]
        if all(p.exists() for p in files):continue
        assert not any(p.exists() for p in files),'Incomplete case: inspect failure before resuming.'
        pending.append(c)
    print(f'Running {len(pending)} remaining of84 cases, workers={workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(evaluate,c) for c in pending]
        for future in as_completed(futures):
            key,packet,audit,equations=future.result()
            np.savez_compressed(output/'predictions'/(key+'.npz'),**packet)
            save(output/'predictions'/(key+'.json'),dict(audit=audit,equations=equations,protocol_sha256=sha(HERE/'protocol.json')))
            print(__import__('json').dumps(dict(completed=len(list((output/'predictions').glob('*.npz'))),total=84,case=key,seconds=round(audit['seconds'],1))),flush=True)
    guard();report(output)


def arithmetic(y,p):
    tp=(y*p).sum(0);fp=((1-y)*p).sum(0);fn=(y*(1-p)).sum(0)
    div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
    pr=div(tp,tp+fp);re=div(tp,tp+fn);f=div(2*tp,2*tp+fp+fn)
    return dict(sample_f1=float(div(2*(y*p).sum(1),y.sum(1)+p.sum(1)).mean()),macro_f1=float(f.mean()),
        **{f'{k}_{label}':float(v[j]) for k,v in [('precision',pr),('recall',re),('f1',f),('support',y.sum(0))] for j,label in enumerate(LABELS)})


def report(output,write=True):
    guard();output=Path(output);ref=controls();packets=[];audits=[];equations=[];errors=[]
    for c in benchmark.cases():
        with np.load(output/'predictions'/(c['key']+'.npz')) as z:p={k:z[k] for k in z.files}
        mask=ref['case']==c['key'];np.testing.assert_array_equal(p['ids'],ref['ids'][mask]);np.testing.assert_array_equal(p['truth'],ref['truth'][mask]);np.testing.assert_array_equal(p['scores'][:3],ref['scores'][:,mask])
        np.testing.assert_allclose(p['scores'][5],.8*p['scores'][1]+.2*p['scores'][3],atol=1e-15,rtol=0)
        np.testing.assert_array_equal(p['predictions'],np.stack([decide(s,.5) for s in p['scores']]))
        meta=__import__('json').loads((output/'predictions'/(c['key']+'.json')).read_text());assert meta['protocol_sha256']==sha(HERE/'protocol.json')
        audits.append(meta['audit']);equations += [dict(case=c['key'],**e) for e in meta['equations']];packets.append(p)
    y=np.concatenate([p['truth'] for p in packets]);pred=np.concatenate([p['predictions'] for p in packets],axis=1)
    np.testing.assert_array_equal(y,ref['truth']);rows=[];supp=[]
    masks=[(dict(pool=pool,seed=int(seed)),(ref['pool']==pool)&(ref['seed']==seed)) for pool in ['train','pooled'] for seed in np.unique(ref['seed'][ref['pool']==pool])]
    masks +=[(dict(case=case),ref['case']==case) for case in np.unique(ref['case'][ref['pool']=='supplementary'])]
    masks +=[(dict(case='chronology98'),(ref['pool']=='supplementary')&(ref['case']!='validation'))]
    for i,mode in enumerate(MODES):
        for meta,mask in masks:
            values=benchmark.metrics(y[mask],pred[i,mask]);manual=arithmetic(y[mask],pred[i,mask])
            errors.extend(abs(values[k]-manual[k]) for k in values)
            (rows if 'pool' in meta else supp).append(dict(mode=mode,**meta,**values))
    assert max(errors)<1e-12
    rows,supp=pd.DataFrame(rows),pd.DataFrame(supp)
    summary=rows.groupby(['pool','mode']).agg(mean_sample_f1=('sample_f1','mean'),population_sd=('sample_f1',lambda a:a.std(ddof=0)),mean_macro_f1=('macro_f1','mean')).reset_index()
    pairs=[];deltas=[]
    for parent in MODES[:3]:
        for pool in ['train','pooled']:
            pivot=rows[rows.pool==pool].pivot(index='seed',columns='mode',values='sample_f1')
            for mode in CANDIDATES:
                d=pivot[mode]-pivot[parent];w,t,l=int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d< -1e-12).sum())
                pairs.append(dict(baseline=parent,pool=pool,mode=mode,mean_delta=float(d.mean()),wins=w,ties=t,losses=l,meets_pool_rule=bool(d.mean()>0 and w>=5)))
                deltas +=[dict(baseline=parent,pool=pool,mode=mode,seed=int(seed),sample_f1_delta=float(v)) for seed,v in d.items()]
    tables=dict(repeat_scores=rows,supplementary_scores=supp,cv_summary=summary,paired_summary=pd.DataFrame(pairs),paired_repeat_deltas=pd.DataFrame(deltas),equations=pd.DataFrame(equations),fit_audits=pd.DataFrame([{k:(__import__('json').dumps(v) if isinstance(v,(list,dict)) else v) for k,v in a.items() if k!='search'} for a in audits]))
    check=dict(passed=True,matched_cases=84,new_geometry_fits=84,symbolic_heads=len(equations),metrics_independently_checked=len(errors),max_metric_error=max(errors),controls_reused=True,all_queries_excluded_from_fit=all(a['fitting_query_ids_disjoint'] for a in audits),all_observed_labels_preserved=all(a['observed_labels_preserved'] for a in audits),source_hashes_unchanged=True)
    if write:
        results=output/'results';results.mkdir(exist_ok=True)
        for name,frame in tables.items():frame.to_csv(results/(name+'.csv'),index=False)
        save(results/'verification.json',check)
    else:
        for name,frame in tables.items():
            saved=pd.read_csv(output/'results'/(name+'.csv'))
            pd.testing.assert_frame_equal(frame,saved,check_dtype=False,rtol=1e-12,atol=1e-12)
    print(summary.to_string(index=False));print(supp[['mode','case','sample_f1']].to_string(index=False));print(__import__('json').dumps(check,indent=2))
    return tables,check


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','report','verify']);p.add_argument('--output',type=Path,default=HERE/'run');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.command=='register':register()
    elif a.command=='run':run(a.output,a.workers)
    elif a.command=='report':report(a.output)
    else:report(a.output,write=False)
