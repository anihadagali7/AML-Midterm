"""Matched-recipe J diagnostic and one fixed complementary blend after Cand2's score."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
P1=HERE.parents[1]
sys.path[:0]=[str(HERE/'source'),str(P1)]
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits
from p1lib import cache
from p1lib.data import load, LABELS, EMB
from p1lib.head import apply_head
from anchored_solution import AnchoredReasonClassifier,select_reasons
import strict_adapter as strict


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def freeze():
    files=[HERE/'run.py',HERE/'strict_adapter.py',HERE/'cand2_manifest.json',HERE/'case_plan.json',
        HERE/'historical_comparison.csv',HERE/'prior_strict_comparison.csv',HERE/'cand2_reported_result.json',P1/'anchored_solution.py']
    files+=sorted((HERE/'source/p1lib').glob('*.py'))+sorted((P1/'data').glob('*.csv'))+sorted((HERE/'inputs').glob('*.npz'))
    hashes={str(p.relative_to(P1)):sha(p) for p in files}
    p=HERE/'protocol.json'
    if p.exists():assert read(p)['hashes']==hashes
    else:
        save(p,dict(version='cand2-followup-v1',created_utc=datetime.now(timezone.utc).isoformat(),
            base_main='31f3fa9',parent='anchored 0.89636',known_cand2_kaggle=.89102,
            question='Does fixed J rank the exact submitted inductive five-seed Cand2 correctly? Does one small fixed blend help?',
            J='Original saved A: 10 repeats x 5 ungrouped folds; B: mean cuts 2004/2005/2006; J=(A+B)/2. Original validation separate.',
            J_access='Identical eligible gold/masked IDs to native J and anchored controls. Query/test rows excluded from all model fitting. Outer J folds retain duplicate-group overlap and B retains all masked training years; this is deliberately NOT the current grouped or strict-chronology benchmark. Inner Cand2 calibration remains group-safe.',
            recipe='Exact submitted strict adapter, five inner seeds 0..4, unchanged config/head. Average float64 combo probabilities before expected-F1 decoding.',
            J_controls='Fresh anchored fit on each saved case, requiring exact score/prediction agreement with prior audit.',
            blend='0.8 * anchored native scores + 0.2 * Cand2 label marginals, .3 cutoff plus argmax; only one weight; no threshold/weight search.',
            blend_evaluation='Reuse verified predictions on all 80 grouped folds, original validation and three chronology windows. Also report J diagnostic. No new blend component fits for current-policy comparison.',
            advancement='Positive paired sample-F1 mean and >=5/8 wins in BOTH grouped pools. Supplementary checks predeclared for this fixed diagnostic even if CV fails; no automatic submission or promotion.',
            limitations='Kaggle result was known before this fixed diagnostic; no metric weights/splits/model settings selected after results. One new model is not multiple independent observations. Native J result must not be attached to the inductive submitted artifact.',hashes=hashes))


def context(D,c):
    pos={v:i for i,v in enumerate(D.df.ID)}
    tr,qu,po,fit=[np.array([pos[v] for v in c[k]]) for k in ['train_gold','query_ids','masked_ids','fit_ids']]
    keep=np.union1d(fit,qu);remap={v:i for i,v in enumerate(keep)}
    tr,qu,po=[np.array([remap[v] for v in x]) for x in [tr,qu,po]]
    idx=dict(train=np.array([i for i,v in enumerate(keep) if v in set(D.idx['train'])]),
        lab=tr,unl=po,val=np.array([],int),test=np.array([],int),labval=np.r_[tr,qu])
    sub=replace(D,df=D.df.iloc[keep].reset_index(drop=True),emb=D.emb[keep],Y=D.Y[keep],y=D.y[keep],idx=idx,year=D.year[keep])
    ctx=strict.context(sub,tr,qu,po,seed=0)
    assert ctx.D.df.iloc[ctx.unsup].ID.tolist()==c['fit_ids']
    assert ctx.D.df.iloc[qu].ID.tolist()==c['query_ids']
    assert (ctx.D.Y[qu]==-1).all() and not np.intersect1d(ctx.unsup,qu).size
    return ctx


def blend(anchor,candidate):return .8*anchor+.2*candidate


def fit_j(c,output):
    start=time.monotonic()
    D=load(str(P1/'data'));ctx=context(D,c)
    cfg=read(HERE/'cand2_manifest.json')['config']
    with np.load(HERE/'inputs'/(c['key']+'.npz')) as z:ref={k:z[k].copy() for k in z.files}
    assert ref['ids'].tolist()==c['query_ids']
    Y=ctx.D.Y[ctx.unsup].copy();Y[Y<0]=-100
    with threadpool_limits(limits=1):
        parent=AnchoredReasonClassifier().fit(ctx.D.emb[ctx.unsup],Y)
        a=parent.predict_proba(ctx.D.emb[ctx.eval])
        error=float(np.abs(a-ref['anchored_scores']).max())
        assert error<1e-10,error
        np.testing.assert_array_equal(select_reasons(a),ref['anchored_predictions'])
        probabilities=[];priors=[];temperatures=[];audits=[]
        for seed in range(5):
            cache.clear();strict.AUDIT.clear()
            cs=replace(ctx,seed=seed)
            out=strict.run_model(cfg['model'],cs,np.r_[cs.eval,cs.pool])
            probabilities.append(out['P'][:len(cs.eval)]);priors.append(out['prior']);temperatures.append(float(out['T']))
            assert not any(v['gold_group_overlap'] or v['masked_group_overlap'] for v in strict.AUDIT)
            audits.extend(strict.AUDIT)
        yp,Q,_=apply_head(np.mean(probabilities,axis=0),np.mean(priors,axis=0),ctx.D.CM,cfg['head'])
    candidate=Q@ctx.D.CM;b=blend(a,candidate)
    arrays=dict(ids=ref['ids'],truth=ref['truth'],anchored_scores=a,anchored_predictions=select_reasons(a),
        cand2_scores=candidate,cand2_predictions=yp,blend_scores=b,blend_predictions=select_reasons(b))
    for name in ['anchored','cand2','blend']:
        assert np.isfinite(arrays[name+'_scores']).all() and (arrays[name+'_predictions'].sum(1)>0).all()
    dest=Path(output)/(c['key']+'.npz');np.savez_compressed(dest,**arrays)
    meta=dict(key=c['key'],kind=c['kind'],repeat=c.get('repeat'),cut=c.get('cut'),
        fresh_parent_max_error=error,gold=len(ctx.train),masked=len(ctx.pool),query=len(ctx.eval),
        inner_contexts=len(audits),inner_group_overlaps=0,temperatures=temperatures,
        outer_gold_group_overlap=c['gold_group_overlap'],outer_masked_group_overlap=c['masked_group_overlap'],
        outer_future_masked=c['future_masked'],sha256=sha(dest),seconds=time.monotonic()-start)
    save(dest.with_suffix('.json'),meta)
    return c['key'],round(meta['seconds'],2)


def collect(output):
    grouped={};metrics=[];labels=[];audits=[]
    for c in read(HERE/'case_plan.json'):
        if c['access']=='strict_inductive':
            with np.load(HERE/'inputs'/(c['key']+'.npz')) as z:a={k:z[k].copy() for k in z.files}
            a['blend_scores']=blend(a['anchored_scores'],a['cand2_scores']);a['blend_predictions']=select_reasons(a['blend_scores'])
            track='current'
        else:
            path=Path(output)/(c['key']+'.npz');meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256'];audits.append(meta)
            with np.load(path) as z:a={k:z[k].copy() for k in z.files}
            track='J'
        kind=c['kind'];rep=c.get('repeat')
        if kind in ['train','pooled']:
            # Saved canonical plan names its repeat by seed.
            rep=c['seed'] if 'seed' in c else rep
        if kind in ['B','chronology']:rep=c.get('cut',c['key'])
        if rep is None:rep=c['key']
        grouped.setdefault((track,kind,str(rep)),[]).append(a)
    for (track,kind,rep),parts in grouped.items():
        truth=np.concatenate([a['truth'] for a in parts]);ids=np.concatenate([a['ids'] for a in parts])
        assert len(set(ids))==len(ids)
        for name in ['anchored','cand2','blend']:
            pred=np.concatenate([a[name+'_predictions'] for a in parts])
            row=dict(track=track,kind=kind,repeat=rep,recipe=name,n=len(truth),sample_f1=f1_score(truth,pred,average='samples'),macro_f1=f1_score(truth,pred,average='macro',zero_division=0))
            metrics.append(row)
            precision,recall,f,support=precision_recall_fscore_support(truth,pred,zero_division=0)
            for j,label in enumerate(LABELS):labels.append(dict(track=track,kind=kind,repeat=rep,recipe=name,label=label,precision=precision[j],recall=recall[j],f1=f[j],support=int(support[j])))
    return pd.DataFrame(metrics),pd.DataFrame(labels),audits,grouped


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();freeze();args.output.mkdir(parents=True,exist_ok=True)
    cases=[c for c in read(HERE/'case_plan.json') if c['access']=='native_dylan']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[]
        for c in cases:
            file=args.output/(c['key']+'.npz')
            if file.exists() and file.with_suffix('.json').exists():continue
            jobs.append(pool.submit(fit_j,c,args.output))
        for i,f in enumerate(as_completed(jobs),1):print(i,len(jobs),f.result(),flush=True)
    metrics,labels,audits,_=collect(args.output)
    metrics.to_csv(args.output/'repeat_metrics.csv',index=False);labels.to_csv(args.output/'label_metrics.csv',index=False)
    save(args.output/'fit_audits.json',audits)
    print(metrics.groupby(['track','kind','recipe']).sample_f1.mean().to_string(),flush=True)
    freeze()


if __name__=='__main__':main()
