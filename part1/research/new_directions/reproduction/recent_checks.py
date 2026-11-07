"""Supplementary checks for recent-method finalists and the declared v8 diagnostic."""
import json,time,sys,importlib
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import study as e
ROOT=Path(__file__).resolve().parent
def freeze():
    dest=ROOT/'recent_diagnostic_plan.json';assert not dest.exists()
    e.save(dest,dict(frozen_at=time.time(),source_sha256=e.sha(__file__),
      diagnostic=[dict(study='v8',recipe='hybrid_leaf_nested')],
      rationale='The fixed v8 blend gains both eight-repeat means and wins 5/8 train, 4/8 pooled; it failed the five-win-in-both rule. Run the fixed supplementary populations as a disclosed diagnostic. No recipe change or promotion.',
      finalists='Also run these checks for any v7-v10 recipe that passes its frozen eight-repeat gate.',
      unchanged_gate=True,no_test_fit=True,no_promotion=True,
      evidence_before_diagnostic=e.sha(ROOT/'studies/v8/comparison_8.csv')))
def supplementary(version,name):
    module=importlib.import_module('study_'+version);model=importlib.import_module('models_'+version);reg=module.registry()
    c=next(c for c in reg['recipes'] if c['name']==name);out=module.OUT/'supplementary'/name;out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'data/part1_train.csv');val=pd.read_csv(ROOT/'data/part1_val.csv')
    gold=train[train[e.LABELS].ge(0).all(axis=1)];masked=train[train[e.LABELS].eq(-100).all(axis=1)]
    full=pd.concat([train,val],ignore_index=True).set_index('ID',drop=False);cases=[('validation',gold,masked,val)]
    for w in json.loads((ROOT/'protocols/chronology.json').read_text())['chronological_splits']:
        fit=full.loc[w['training_ids']];g=fit[fit[e.LABELS].ge(0).all(axis=1)];u=fit[fit[e.LABELS].eq(-100).all(axis=1)];q=full.loc[w['heldout_ids']]
        assert len(g)==w['gold_rows'] and len(u)==w['masked_rows'];assert not set(e.groups(fit))&set(e.groups(q))
        cases.append((w['name'],g,u,q))
    chunks=[];rows=[]
    for case,g,u,q in cases:
        dest=out/f'{case}.npz'
        if not dest.exists():
            p,pred,audit=model.predict(g,u,q.drop(columns=e.LABELS),c,e.LABELS,e.EMB)
            np.savez_compressed(dest,scores=p,predictions=pred,truth=q[e.LABELS].to_numpy(int),ids=q.ID.to_numpy(str))
            e.save(dest.with_suffix('.json'),dict(name=case,recipe=c,metrics=e.metric(q[e.LABELS].to_numpy(int),pred),sources=reg['sources'],
                raw_data=reg['raw_data'],protocol_hashes=reg['protocols'],gold_ids=g.ID.tolist(),masked_ids=u.ID.tolist(),query_ids=q.ID.tolist(),audit=audit,
                plan_sha256=e.sha(ROOT/'recent_diagnostic_plan.json'),access='inductive'))
        z=np.load(dest);m=e.metric(z['truth'],z['predictions']);rows.append(dict(case=case,**m))
        if case!='validation':chunks.append({k:z[k] for k in z.files})
        print(json.dumps(dict(study=version,recipe=name,case=case,f1=m['sample_f1'])),flush=True)
    pooled={k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]};np.savez_compressed(out/'chronological98.npz',**pooled)
    e.save(out/'summary.json',dict(recipe=name,validation=rows[0],windows=rows[1:],chronology=e.metric(pooled['truth'],pooled['predictions'])))
def run():
    plan=json.loads((ROOT/'recent_diagnostic_plan.json').read_text());assert e.sha(__file__)==plan['source_sha256']
    cs=plan['diagnostic'].copy()
    for version in ['v7','v8','v9','v10']:
        path=ROOT/'studies'/version/'comparison_8.csv'
        if not path.exists():continue
        for _,r in pd.read_csv(path).query('qualifies').iterrows():
            item=dict(study=version,recipe=r.recipe)
            if item not in cs:cs.append(item)
    with threadpool_limits(limits=1):
        for item in cs:supplementary(item['study'],item['recipe'])
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run()
