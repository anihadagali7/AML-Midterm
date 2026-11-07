"""Matched untuned/unweighted controls of the frozen training-selected policies."""
from pathlib import Path
import json,copy,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import phase25 as m,phase25_run as runner,experiment as e

def main():
    frozen=json.loads(Path('experiments/phase25-validation-frozen.json').read_text())
    controls=[]
    for r in frozen['candidates']:
        family=r['config']['learner']['model']
        if family not in ['label_tuned_ssl','reliable_ssl']:continue
        c=copy.deepcopy(r['config'])
        if family=='label_tuned_ssl':c['learner']=m.BASE.copy()
        else:c['learner']['reliability']='uniform'
        ss=[]
        for seed in [20261119,20261120,20261121]:
            p=runner.load_predictions(c,seed,'phase25b_control');ss.append(runner.score(p))
        controls.append(dict(name='phase25_control_'+family,config=c,rule='t0.5',
            ordinary_cv=float(np.mean([s['ordinary'] for s in ss])),
            robust_cv=float(np.mean([s['robust'] for s in ss])),
            cv_std=float(np.std([s['ordinary'] for s in ss])),per_repeat=ss))
    e.save_json('experiments/phase25b-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=controls,purpose='Fixed matched controls; no control-specific threshold or scope tuning.'))
    V=e.pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
    with threadpool_limits(limits=1):ps=[e.predict(np.arange(len(e.Y)),V,r['config']) for r in controls]
    np.savez_compressed('experiments/phase25b-validation-predictions.npz',p=ps)
    y=e.pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
    for r,p in zip(controls,ps):
        out={**r,'validation':e.metrics(y,p,r['rule'])};results.append(out);print('VALIDATION',json.dumps(out),flush=True)
    e.save_json('experiments/phase25b-validation-results.json',results)

if __name__=='__main__':main()
