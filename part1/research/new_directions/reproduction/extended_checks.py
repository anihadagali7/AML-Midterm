"""Predeclared focused diagnostics, including original validation and chronology."""
import json,types,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import study as e,study_v2,study_v3,models,models_v2,models_v3
ROOT=Path(__file__).resolve().parent
ORIGINAL_REG=e.registry
MODEL_BY_STUDY={'v1':models.predict,'v2':models_v2.predict,'v3':models_v3.predict}
def configure(version):
    if version=='v1':e.ROOT=ROOT;e.registry=ORIGINAL_REG
    elif version=='v2':e.ROOT=study_v2.OUT;e.registry=study_v2.registry
    else:e.ROOT=study_v3.OUT;e.registry=study_v3.registry
    e.models=types.SimpleNamespace(predict=MODEL_BY_STUDY[version],threshold=models.threshold)
    return e.registry()
def supplementary(version,c):
    reg=configure(version);out=ROOT/'supplementary'/c['name'];out.mkdir(exist_ok=True,parents=True)
    train=pd.read_csv(ROOT/'data/part1_train.csv');val=pd.read_csv(ROOT/'data/part1_val.csv')
    gold=train[train[e.LABELS].ge(0).all(1)];masked=train[train[e.LABELS].eq(-100).all(1)]
    full=pd.concat([train,val],ignore_index=True).set_index('ID',drop=False)
    cases=[('validation',gold,masked,val)]
    for window in json.loads((ROOT/'protocols/chronology.json').read_text())['chronological_splits']:
        fit=full.loc[window['training_ids']]
        g=fit[fit[e.LABELS].ge(0).all(1)];u=fit[fit[e.LABELS].eq(-100).all(1)];q=full.loc[window['heldout_ids']]
        assert len(g)==window['gold_rows'] and len(u)==window['masked_rows']
        assert not set(e.groups(fit))&set(e.groups(q))
        cases.append((window['name'],g,u,q))
    ys=[];preds=[];ps=[];ids=[];rows=[]
    for name,g,u,q in cases:
        dest=out/f'{name}.npz'
        if not dest.exists():
            p,pred,audit=MODEL_BY_STUDY[version](g,u,q.drop(columns=e.LABELS),c,e.LABELS,e.EMB)
            np.savez_compressed(dest,scores=p,predictions=pred,truth=q[e.LABELS].to_numpy(int),ids=q.ID.to_numpy(str))
            e.save(dest.with_suffix('.json'),dict(name=name,recipe=c,metrics=e.metric(q[e.LABELS].to_numpy(int),pred),
                   sources=reg['sources'],gold_ids=g.ID.tolist(),masked_ids=u.ID.tolist(),query_ids=q.ID.tolist(),audit=audit))
        z=np.load(dest);m=e.metric(z['truth'],z['predictions']);rows.append(dict(case=name,**m))
        if name!='validation':ys.append(z['truth']);preds.append(z['predictions']);ps.append(z['scores']);ids.extend(z['ids'])
        print(json.dumps(dict(recipe=c['name'],case=name,f1=m['sample_f1'])),flush=True)
    pooled=e.metric(np.concatenate(ys),np.concatenate(preds))
    np.savez_compressed(out/'chronological98.npz',truth=np.concatenate(ys),predictions=np.concatenate(preds),scores=np.concatenate(ps),ids=np.array(ids))
    e.save(out/'summary.json',dict(recipe=c['name'],validation=rows[0],windows=rows[1:],chronology=pooled))
def main():
    plan=json.loads((ROOT/'diagnostic_extension.json').read_text());allrows=[]
    with threadpool_limits(limits=1):
        for item in plan['candidates']:
            version=item['study'];reg=configure(version);c=next(c for c in reg['recipes'] if c['name']==item['name'])
            for pool in ['train','pooled']:e.evaluate(c,pool,8)
            table=e.summarize(8)
            row=table[table.recipe==c['name']].iloc[0].to_dict();row['study']=version
            supplementary(version,c)
            sup=json.loads((ROOT/'supplementary'/c['name']/'summary.json').read_text())
            row['validation']=sup['validation']['sample_f1'];row['chronology']=sup['chronology']['sample_f1'];allrows.append(row)
        pd.DataFrame(allrows).to_csv(ROOT/'diagnostic_comparison8.csv',index=False)
if __name__=='__main__':main()
