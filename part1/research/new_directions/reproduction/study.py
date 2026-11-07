"""Fixed-fold research runner, independent of historical model engines."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits
import models

ROOT=Path(__file__).resolve().parent
LABELS=['Safety','Funding','Efficacy','Recruitment','Other']
EMB=[f'emb_{i}' for i in range(150)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
def metric(y,pred):
    pr,re,f,s=precision_recall_fscore_support(y,pred,zero_division=0)
    return dict(sample_f1=float(f1_score(y,pred,average='samples',zero_division=0)),
                macro_f1=float(f.mean()),per_label={l:dict(precision=pr[i],recall=re[i],f1=f[i],support=int(s[i])) for i,l in enumerate(LABELS)})
def data(pool):
    train=pd.read_csv(ROOT/'data/part1_train.csv')
    full=pd.concat([train,pd.read_csv(ROOT/'data/part1_val.csv')],ignore_index=True) if pool=='pooled' else train
    gold=full[full[LABELS].ge(0).all(axis=1)].reset_index(drop=True)
    masked=train[train[LABELS].eq(-100).all(axis=1)].reset_index(drop=True)
    return gold,masked
def groups(df):return pd.util.hash_pandas_object(df[EMB].round(10),index=False).to_numpy()
def protocol(pool):
    a=json.loads((ROOT/'protocols'/f'{pool}3.json').read_text())
    b=json.loads((ROOT/'protocols'/f'{pool}5.json').read_text())
    assert a['gold_ids']==b['gold_ids']
    return dict(gold_ids=a['gold_ids'],folds=a['folds']+b['folds'],seeds=a['seeds']+b['seeds'])
def baseline(pool):
    parts=[np.load(ROOT/'references'/f'ani_{pool}_{s}.npz',allow_pickle=True) for s in ['first','extra']]
    scores=np.concatenate([x['scores'] for x in parts]);ids=parts[0]['ids'];truth=parts[0]['truth']
    assert np.array_equal(ids,parts[1]['ids']) and np.array_equal(truth,parts[1]['truth'])
    predictions=np.array([models.threshold(p,.3) for p in scores])
    return dict(scores=scores,predictions=predictions,truth=truth,ids=ids,
                repeat_f1=np.array([metric(truth,p)['sample_f1'] for p in predictions]))
def configs():
    out=[]
    for k,t in [(10,1.),(10,2.),(30,1.),(30,2.)]:
        out.append(dict(name=f'poisson_k{k}_t{t:g}',family='poisson',neighbors=k,temperature=t))
    for dim,mass in [(30,0.),(30,.5),(80,0.),(80,.5)]:
        out.append(dict(name=f'compositional_d{dim}_u{mass:g}',family='compositional',dimensions=dim,unlabeled_mass=mass))
    for k,w,second in [(8,.5,False),(8,1.,True),(24,.5,False),(24,1.,True)]:
        out.append(dict(name=f'fisher_k{k}_w{w:g}_s{int(second)}',family='fisher',clusters=k,residual_weight=w,second_order=second))
    return out
def freeze():
    path=ROOT/'registry_v1.json';assert not path.exists()
    sources={p.name:sha(p) for p in [ROOT/'models.py',ROOT/'study.py']}
    save(path,dict(version='new-directions-v1.1-numerical-fix',frozen_at=time.time(),recipes=configs(),sources=sources,
        raw_data={p.name:sha(p) for p in (ROOT/'data').glob('*.csv')},
        protocols={p.name:sha(p) for p in (ROOT/'protocols').glob('*.json')},
        selection='Screen first 3 repeats in BOTH pools against fixed Ani inductive control. Advance positive means and >=2/3 wins in each. Confirm on 5 added repeats, require positive means and >=5/8 wins in each. Compare submitted ensemble as secondary reference. Evaluate original validation and chronology only after shortlist.',
        no_kaggle_inputs=True,access='inductive',hypotheses={
          'poisson':'Sources/sinks may avoid oversmoothing ordinary label propagation; normalized embedding graph, independent prediction extension.',
          'compositional':'Share semantic prototype estimates across all 31 reason combinations, infer missing patterns with anchored EM and decode joint expected F1.',
          'fisher':'Mixture residual directions may preserve distinctions erased by plain cluster probabilities. Signed-root and L2 normalization.'}))
    print('Frozen 12 recipes, source and protocol hashes before any fitting.',flush=True)
def registry():
    d=json.loads((ROOT/'registry_v1.json').read_text())
    for p,h in d['sources'].items():assert sha(ROOT/p)==h,f'Source changed: {p}'
    for p,h in d['raw_data'].items():assert sha(ROOT/'data'/p)==h
    for p,h in d['protocols'].items():assert sha(ROOT/'protocols'/p)==h
    return d
def evaluate(c,pool,repeats):
    reg=registry();gold,u=data(pool);p=protocol(pool);base=baseline(pool)
    assert gold.ID.tolist()==p['gold_ids']==base['ids'].tolist()
    y=gold[LABELS].to_numpy(int);np.testing.assert_array_equal(y,base['truth'])
    gg,ug=groups(gold),groups(u)
    out=ROOT/'results'/c['name']/pool;out.mkdir(parents=True,exist_ok=True)
    for ri,seed in enumerate(p['seeds'][:repeats]):
        dest=out/f'{seed}.npz'
        if dest.exists():continue
        scores=np.full(y.shape,np.nan);pred=np.full(y.shape,-1);coverage=np.zeros(len(y),int);audits=[];start=time.time()
        for f in [f for f in p['folds'] if f['seed']==seed]:
            a=np.array(f['train_gold_positions']);b=np.array(f['heldout_gold_positions'])
            assert not set(a)&set(b) and len(a)+len(b)==len(y)
            assert not set(gg[a])&set(gg[b])
            assert gold.iloc[b].ID.tolist()==f['heldout_ids']
            keep=~np.isin(ug,gg[b]);assert (~keep).sum()==f['masked_duplicates_excluded']
            pu,pred_u,audit=models.predict(gold.iloc[a],u[keep],gold.iloc[b].drop(columns=LABELS),c,LABELS,EMB)
            scores[b]=pu;pred[b]=pred_u;coverage[b]+=1
            audits.append(dict(fold=f['fold'],gold_ids=gold.iloc[a].ID.tolist(),masked_ids=u[keep].ID.tolist(),
                               query_ids=gold.iloc[b].ID.tolist(),masked_purged=int((~keep).sum()),**audit))
        assert (coverage==1).all() and np.isfinite(scores).all()
        np.savez_compressed(dest,scores=scores,predictions=pred,truth=y,ids=gold.ID.to_numpy(str))
        m=metric(y,pred)
        save(dest.with_suffix('.json'),dict(recipe=c,recipe_hash=digest(c),sources=reg['sources'],
             raw_data=reg['raw_data'],protocol_hashes=reg['protocols'],seed=seed,pool=pool,access='inductive',
             metrics=m,ani_delta=m['sample_f1']-base['repeat_f1'][ri],audits=audits,seconds=time.time()-start))
        print(json.dumps(dict(recipe=c['name'],pool=pool,seed=seed,f1=round(m['sample_f1'],6),
              delta=round(m['sample_f1']-base['repeat_f1'][ri],6),seconds=round(time.time()-start,1))),flush=True)
def summarize(repeats=3):
    reg=registry();rows=[]
    for c in reg['recipes']:
        row={'recipe':c['name'],'family':c['family']};complete=True;passes=True
        for pool in ['train','pooled']:
            p=protocol(pool);paths=[ROOT/'results'/c['name']/pool/f'{s}.npz' for s in p['seeds'][:repeats]]
            if not all(x.exists() for x in paths):complete=False;break
            metrics=[]
            for path in paths:
                z=np.load(path);m=metric(z['truth'],z['predictions']);metrics.append(m)
                saved=json.loads(path.with_suffix('.json').read_text())['metrics']
                assert abs(m['sample_f1']-saved['sample_f1'])<1e-12
            values=np.array([m['sample_f1'] for m in metrics]);diff=values-baseline(pool)['repeat_f1'][:repeats]
            wins=int((diff>1e-12).sum());passes &= diff.mean()>1e-12 and wins >= (2 if repeats==3 else 5)
            row.update({pool+'_mean':values.mean(),pool+'_sd':values.std(),pool+'_delta':diff.mean(),pool+'_wins':wins,
                        pool+'_macro':np.mean([m['macro_f1'] for m in metrics])})
        if complete:row['qualifies']=bool(passes);rows.append(row)
    df=pd.DataFrame(rows)
    if len(df):df=df.sort_values(['train_mean','pooled_mean'],ascending=False)
    df.to_csv(ROOT/f'comparison_{repeats}.csv',index=False)
    print(df.to_string(index=False),flush=True)
    return df
def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','screen','confirm','report'])
    parser.add_argument('--names',nargs='*');parser.add_argument('--repeats',type=int,default=3);args=parser.parse_args()
    if args.stage=='freeze':freeze();return
    reg=registry();cs=[c for c in reg['recipes'] if not args.names or c['name'] in args.names]
    with threadpool_limits(limits=1):
        if args.stage=='screen':
            for c in cs:
                for pool in ['train','pooled']:evaluate(c,pool,3)
            summarize(3)
        elif args.stage=='confirm':
            selected=summarize(3).query('qualifies').recipe.tolist()
            for c in cs:
                if c['name'] in selected:
                    for pool in ['train','pooled']:evaluate(c,pool,8)
            summarize(8)
        else:summarize(args.repeats)
if __name__=='__main__':
    try:main()
    except Exception:
        with (ROOT/'failures.log').open('a') as f:f.write(traceback.format_exc())
        raise
