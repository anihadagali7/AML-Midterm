"""Confirmation and supplementary checks for the v11-v13 iteration."""
import sys,json,time,importlib
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import study as e
BASE=Path(__file__).resolve().parent;PLAN=BASE/'next_diagnostics/followup_plan.json'
def freeze():
    assert not PLAN.exists()
    e.save(PLAN,dict(frozen_at=time.time(),source_sha256=e.sha(__file__),studies=['v11','v12','v13'],
       rule='Only initial qualifiers receive five additional repeats. Only eight-repeat qualifiers receive validation78 and fixed chronology98. No diagnostic exceptions.',
       comparator='Current LDA 70/15/15 ensemble',no_test_predictions=True,no_automatic_promotion=True))
def supplementary(module,model,c):
    reg=module.registry();folder=module.OUT/'supplementary'/c['name'];folder.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(BASE/'data/part1_train.csv');val=pd.read_csv(BASE/'data/part1_val.csv');full=pd.concat([train,val],ignore_index=True).set_index('ID',drop=False)
    cases=[('validation',train,val)]
    for w in json.loads((BASE/'protocols/chronology.json').read_text())['chronological_splits']:cases.append((w['name'],full.loc[w['training_ids']],full.loc[w['heldout_ids']]))
    results=[];chunks=[]
    for case,fit,q in cases:
        dest=folder/f'{case}.npz';g=fit[fit[e.LABELS].ge(0).all(1)];u=fit[fit[e.LABELS].eq(-100).all(1)]
        if not dest.exists():
            p,pred,audit=model.predict(g,u,q.drop(columns=e.LABELS),c,e.LABELS,e.EMB)
            np.savez_compressed(dest,scores=p,predictions=pred,truth=q[e.LABELS].to_numpy(int),ids=q.ID.to_numpy(str))
            e.save(dest.with_suffix('.json'),dict(recipe=c,case=case,metrics=e.metric(q[e.LABELS].to_numpy(int),pred),sources=reg['sources'],raw_data=reg['raw_data'],protocol_hashes=reg['protocols'],gold_ids=g.ID.tolist(),masked_ids=u.ID.tolist(),query_ids=q.ID.tolist(),audit=audit,plan_sha256=e.sha(PLAN),access='inductive'))
        z=np.load(dest);m=e.metric(z['truth'],z['predictions']);results.append(dict(case=case,**m))
        if case!='validation':chunks.append({k:z[k] for k in z.files})
    joined={k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]};np.savez_compressed(folder/'chronological98.npz',**joined)
    e.save(folder/'summary.json',dict(recipe=c['name'],validation=results[0],windows=results[1:],chronology=e.metric(joined['truth'],joined['predictions'])))
def run():
    plan=json.loads(PLAN.read_text());assert plan['source_sha256']==e.sha(__file__)
    with threadpool_limits(1):
        for version in plan['studies']:
            mod=importlib.import_module('study_'+version);mod.configure();cs=mod.registry()['recipes'];first=e.summarize(3)
            qualified=first.query('qualifies').recipe.tolist()
            for c in cs:
                if c['name'] in qualified:
                    for pool in ['train','pooled']:e.evaluate(c,pool,8)
            if qualified:
                confirmed=e.summarize(8).query('qualifies').recipe.tolist()
                for c in cs:
                    if c['name'] in confirmed:supplementary(mod,importlib.import_module('models_'+version),c)
            print(version,'initial qualifiers:',qualified,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run()
