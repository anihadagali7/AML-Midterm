"""Register and evaluate partial secondary labels on all saved grouped cases."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[v]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import warnings

HERE=Path(__file__).resolve().parent;P1=HERE.parents[1];ROOT=P1.parent
HELPER=P1/'research/anchored_transductive_v1'
sys.path[:0]=[str(HERE),str(P1),str(HELPER)]
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score,precision_recall_fscore_support
from threadpoolctl import threadpool_limits
from anchored_solution import AnchoredReasonClassifier,select_reasons
from partial_model import PartialOtherAnchored
spec=importlib.util.spec_from_file_location('_partial_saved_benchmark',HELPER/'run.py')
benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
LABELS=benchmark.LABELS;EMB=benchmark.FEATURES
CANDIDATES=['neighbor_unknown','binary_complete','binary_unknown','binary_complete_half','binary_unknown_half']
NAMES=['parent','anchored']+CANDIDATES


def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def idhash(v):return hashlib.sha256('\n'.join(v).encode()).hexdigest()
def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def cases():
    result=benchmark.cases()
    for c in result:
        for key in ['fit','query']:c[key]=c[key][['ID','start_year']+EMB+LABELS].copy()
        c['reference_key']='strict_inductive__'+('grouped_' if c['pool']!='supplementary' else '')+c['key']
    return result


def reference(c):
    z=loadnp(HERE/'parent_inputs.npz');keep=z['case']==c['reference_key']
    np.testing.assert_array_equal(z['ids'][keep],c['query'].ID)
    np.testing.assert_array_equal(z['truth'][keep],c['query'][LABELS])
    return z['anchored'][keep],z['cand2'][keep]


def sources():
    files=[HERE/n for n in ['study_run.py','partial_model.py','check_math.py','parent_inputs.npz',
        'parent_reported_result.json','provenance.json']]
    files+=[P1/'anchored_solution.py',P1/'TESTING_POLICY.md',HELPER/'run.py',HELPER/'models.py']
    files+=list((P1/'data').glob('*.csv'))
    files+=[P1/'experiments/current_leaders_grouped_v1/snapshots'/f'{name}.json'
        for name in ['train3','train5','pooled3','pooled5','chronology']]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def register():
    p=HERE/'protocol.json';assert not p.exists()
    assert len(cases())==84
    save(p,dict(version='partial-secondary-v1',registered_at=datetime.now(timezone.utc).isoformat(),
        base_main='766d16d3bc74248cb7d8f4c1e7240094ebd5a55e',
        parent='Scored .89935 arithmetic anchored80/Cand2_inductive5_20. Original anchored .89636 is secondary control.',
        hypothesis='Confident primary pseudo-labels may incorrectly imply absent Other. Keep unselected Other unknown, with gold negatives still observed. Targets the rare Recruitment+Other combination without changing supplied labels.',
        motivation='Prior full235 anchored fit generated 757 pseudo rows:756 single-label and1 two-label. Previous corrections/specialists failed the complete rule. This is a distinct hypothesis, not proof those pseudo labels are wrong.',
        candidates={
            'neighbor_unknown':'Only Other neighbor voting changes. Pseudo Other=0 is unknown and excluded from Other neighbor retrieval; gold0/1 and pseudo Other1 remain observed. Retrieve ten eligible Other neighbors. Original multiclass LR/LDA and cluster teacher unchanged. Relearn pseudo labels using altered neighbor votes each round.',
            'binary_complete':'Architectural control. Replace only the Other output of LR/LDA with binary observed-label classifiers; all pseudo Other0/1 remain complete. Neighbor and cluster mapping retain complete labels.',
            'binary_unknown':'Matched to binary_complete: exclude pseudo Other0 from binary Other LR/LDA fits, Other neighbor retrieval, and Other cluster-rate numerator/denominator/prior. Supplied negatives remain observed. Other four class outputs use original multiclass LR/LDA/votes; they can change through regenerated pseudo labels.',
            'binary_complete_half':'Fixed conservative architectural control: .4 original anchored +.4 binary_complete anchored +.2 unchanged Cand2.',
            'binary_unknown_half':'Fixed conservative partial-label variant: .4 original anchored +.4 binary_unknown anchored +.2 unchanged Cand2.'},
        binary_settings='Other LR C.1,max_iter3000,tol1e-4. Raw gold weight1,pseudo.25. Rebalance observed class-weight masses to .8 negative/.2 positive and total gold-row count each fit, preventing removal of unknown negatives from changing effective prior/regularization mass. Other LDA uses raw1/.25 weights, weighted class means/shrinkage covariance, fixed .8/.2 priors. These match the original equal-five-class .2 Other prior in intent, not mathematical identity; binary_complete is required to isolate architecture from unknown treatment.',
        unchanged='Original scaler,PCA30,GMM30,n_init3,seeds0,C.1,LR class weights, two pseudo refreshes, row acceptance max>=.8 and positive label cutoff.3+argmax. Only Other can be unknown; other four pseudo labels retain current treatment. Final80/20 blend and cutoff.3+argmax; no score normalization, thresholds or blend weights selected on outcomes.',
        schedule='All five variants on80 saved grouped folds plus validation78 and three chronology windows. Fresh unchanged anchored and three altered model fits per case; half blends reuse scores. Cached strict five-seed Cand2 reused with matching IDs/truth/plans and frozen source provenance. No new Cand2 fits needed.',
        supplementary_exception='Evaluate all variants on all four supplementary cases regardless of CV, predeclared to diagnose the hypothesis.',
        access='Inductive: only eligible fitting gold/masked inputs; exact saved order and group purges. Original validation retains historical no-outer-purge boundary. Query functions receive embeddings only. No actual Kaggle test inputs loaded during benchmarking.',
        advancement='Positive mean sample-F1 delta and>=5/8 paired wins in BOTH pools against .89935 parent. Also compare to anchored and binary_unknown versus matched binary_complete (likewise half). Keep every supplementary regression visible. No automatic promotion or upload. A qualifying variant with validation/chronology regressions no worse than .005 may receive a final235 candidate export.',
        scoring='Complete OOF sample-F1 per repeat, population SD, paired deltas/wins/ties/losses, macro F1, every label precision/recall/F1/support, Other-only and Recruitment+Other recall/false positives, each chronology window and pooled98.',
        checks='Fresh anchored matches all saved scores within1e-7 and decisions exactly; no-change subclass matches on first case. Unit-weight binary LDA equals sklearn shrinkage LDA, unknown pseudo rows cannot vote negative, known gold negatives retained, binary weighted prior masses fixed. Independently rescore outputs and verify IDs/groups/coverage/source hashes.',
        limitations='Reused development labels/splits, only nine gold Recruitment+Other examples in pooled235, unknown-label omission may add false positives or broaden neighbor radius. Binary architecture changes require its complete-label control. No outer label/threshold tuning.',
        retention='Keep source/protocol, complete compact summaries/negatives and independent verification. Keep detailed predictions only for qualifying/useful comparisons after checking all summaries; remove failed fold caches.',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),sources=sources()))
    print('Registered five fixed variants, including matched controls and half-strength pairs.',flush=True)


def guard():assert read(HERE/'protocol.json')['sources']==sources(),'Registered sources changed'


def evaluate(c,output):
    t=time.monotonic();fit,query=c['fit'],c['query']
    X,Y,Q=fit[EMB].to_numpy(float),fit[LABELS].to_numpy(int),query[EMB].to_numpy(float)
    original=Y.copy();a_ref,cc=reference(c);audits={}
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        parent=AnchoredReasonClassifier().fit(X,Y);a=parent.predict_proba(Q)
        np.testing.assert_allclose(a,a_ref,atol=1e-7,rtol=0)
        np.testing.assert_array_equal(select_reasons(a),select_reasons(a_ref))
        changed={}
        for mode in ['neighbor_unknown','binary_complete','binary_unknown']:
            m=PartialOtherAnchored(mode).fit(X,Y);changed[mode]=m.predict_proba(Q)
            audits[mode]=m.round_audit_
        control_error=None
        if c['key']=='train_20261126_0':
            b=PartialOtherAnchored('control').fit(X,Y).predict_proba(Q)
            np.testing.assert_allclose(b,a,atol=1e-10,rtol=0)
            np.testing.assert_array_equal(select_reasons(b),select_reasons(a))
            control_error=float(np.abs(b-a).max())
    np.testing.assert_array_equal(Y,original)
    scores=dict(parent=.8*a_ref+.2*cc,anchored=a_ref)
    scores.update({name:.8*s+.2*cc for name,s in changed.items()})
    for name in ['binary_complete','binary_unknown']:scores[name+'_half']=.4*a_ref+.4*changed[name]+.2*cc
    S=np.stack([scores[name] for name in NAMES]);P=np.stack([select_reasons(s) for s in S])
    assert np.isfinite(S).all() and (P.sum(2)>0).all()
    dest=Path(output)/'predictions'/(c['key']+'.npz')
    np.savez_compressed(dest,names=np.array(NAMES),ids=query.ID.to_numpy(str),truth=query[LABELS].to_numpy(int),scores=S,predictions=P)
    audit=dict(key=c['key'],pool=c['pool'],seed=c['seed'],fit_ids_sha256=idhash(fit.ID),query_ids_sha256=idhash(query.ID),
        gold=int((Y>=0).all(1).sum()),masked=int((Y<0).all(1).sum()),parent_max_error=float(np.abs(a-a_ref).max()),
        no_change_control_max_error=control_error,rounds=audits,seconds=time.monotonic()-t,sha256=sha(dest))
    save(dest.with_suffix('.json'),audit)
    return dict(key=c['key'],seconds=round(audit['seconds'],2))


def metric_rows(y,p):
    precision,recall,f1,support=precision_recall_fscore_support(y,p,zero_division=0)
    m=dict(sample_f1=float(f1_score(y,p,average='samples')),macro_f1=float(f1.mean()))
    lab=[dict(label=name,precision=float(precision[j]),recall=float(recall[j]),f1=float(f1[j]),support=int(support[j])) for j,name in enumerate(LABELS)]
    return m,lab


def collect(output):
    guard();output=Path(output);groups={};audits=[]
    for c in cases():
        p=output/'predictions'/(c['key']+'.npz');z=loadnp(p);a=read(p.with_suffix('.json'))
        assert a['sha256']==sha(p);audits.append(a)
        np.testing.assert_array_equal(z['ids'],c['query'].ID);np.testing.assert_array_equal(z['truth'],c['query'][LABELS])
        key=(c['pool'],c['seed']) if c['pool']!='supplementary' else (c['key'],-1)
        groups.setdefault(key,[]).append(z)
        if c['key'].startswith('through_'):groups.setdefault(('chronology98',-1),[]).append(z)
    rows=[];labels=[];subgroups=[]
    for (pool,seed),zs in groups.items():
        y=np.concatenate([z['truth'] for z in zs]);ids=np.concatenate([z['ids'] for z in zs])
        assert len(ids)==len(set(ids))
        if pool in ['train','pooled']:assert len(zs)==5
        for j,name in enumerate(NAMES):
            p=np.concatenate([z['predictions'][j] for z in zs]);m,ls=metric_rows(y,p)
            base=dict(pool=pool,seed=seed,recipe=name,n=len(y));rows.append(dict(**base,**m));labels.extend(dict(**base,**l) for l in ls)
            for group,mask in [('other_alone',(y[:,4]==1)&(y.sum(1)==1)),('other_with_recruitment',(y[:,4]==1)&(y[:,3]==1)),('not_other',y[:,4]==0)]:
                subgroups.append(dict(**base,subgroup=group,support=int(mask.sum()),other_predicted=int(p[mask,4].sum())))
    out=output/'summary';out.mkdir(exist_ok=True);d=pd.DataFrame(rows)
    d.to_csv(out/'repeat_scores.csv',index=False);pd.DataFrame(labels).to_csv(out/'per_label_scores.csv',index=False)
    pd.DataFrame(subgroups).to_csv(out/'other_subgroups.csv',index=False)
    summary=d.groupby(['pool','recipe']).agg(mean_sample_f1=('sample_f1','mean'),population_sd=('sample_f1',lambda x:x.std(ddof=0)),mean_macro_f1=('macro_f1','mean')).reset_index()
    summary.to_csv(out/'overview.csv',index=False);pairs=[];deltas=[];qualifies={n:True for n in CANDIDATES}
    comparisons=[(n,control) for n in CANDIDATES for control in ['parent','anchored']]+[
        ('binary_unknown','binary_complete'),('binary_unknown_half','binary_complete_half')]
    for pool in ['train','pooled']:
        pivot=d[d.pool==pool].pivot(index='seed',columns='recipe',values='sample_f1')
        for name,control in comparisons:
            v=pivot[name]-pivot[control];wins=int((v>1e-12).sum());ties=int((v.abs()<=1e-12).sum())
            passed=bool(v.mean()>1e-12 and wins>=5)
            if control=='parent':qualifies[name]&=passed
            pairs.append(dict(pool=pool,recipe=name,control=control,mean_delta=float(v.mean()),wins=wins,ties=ties,losses=8-wins-ties,passes=passed))
            deltas.extend(dict(pool=pool,recipe=name,control=control,seed=int(s),delta=float(t)) for s,t in v.items())
    pd.DataFrame(pairs).to_csv(out/'paired_summary.csv',index=False);pd.DataFrame(deltas).to_csv(out/'paired_repeat_deltas.csv',index=False)
    save(out/'qualification.json',qualifies)
    save(out/'fit_summary.json',dict(cases=len(audits),fresh_anchored_fits=84,altered_fits=252,no_change_control_fits=1,
        cached_cand2_cases=84,total_fit_seconds=sum(a['seconds'] for a in audits),parent_max_error=max(a['parent_max_error'] for a in audits),
        control_max_error=[a['no_change_control_max_error'] for a in audits if a['no_change_control_max_error'] is not None]))
    audit_rows=[]
    for a in audits:
        for mode,rounds in a['rounds'].items():
            audit_rows.extend(dict(case=a['key'],pool=a['pool'],seed=a['seed'],recipe=mode,**r) for r in rounds)
    pd.DataFrame(audit_rows).to_csv(out/'training_audits.csv',index=False)
    print(summary[summary.pool.isin(['train','pooled','validation','chronology98'])].to_string(index=False),flush=True)
    print('QUALIFIED',qualifies,flush=True)


def run(output,workers):
    guard();output=Path(output);assert not output.exists(),'Use a fresh output directory.'
    (output/'predictions').mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(evaluate,c,str(output)) for c in cases()]
        for i,f in enumerate(as_completed(futures),1):print(json.dumps(dict(done=i,total=84,**f.result())),flush=True)
    collect(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','collect']);p.add_argument('--output',type=Path,default=HERE/'results');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.command=='register':register()
    elif args.command=='run':run(args.output,args.workers)
    else:collect(args.output)
