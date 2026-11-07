"""Exact no-exchange ablation of the training-selected co-training policy."""
from pathlib import Path
import copy,json,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import phase24,experiment as e
import phase24_run as runner

def run():
    frozen=json.loads(Path('experiments/phase24-validation-frozen.json').read_text())
    selected=next(r for r in frozen['candidates'] if r['config']['learner']['model']=='co_training')
    c=copy.deepcopy(selected['config']);c['learner']['no_exchange']=True
    ss=[]
    for seed in [20261115,20261116,20261117]:
        p=runner.load_predictions(c,seed,'phase24b_control');ss.append(runner.score(p))
    candidate=dict(name='phase24_exact_no_exchange',config=c,rule='t0.5',
        ordinary_cv=float(np.mean([s['ordinary'] for s in ss])),
        robust_cv=float(np.mean([s['robust'] for s in ss])),
        cv_std=float(np.std([s['ordinary'] for s in ss])),per_repeat=ss)
    e.save_json('experiments/phase24b-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=[candidate],purpose='Matched control of the training-selected co-training recipe; no tuning of scope or decoder.'))
    V=e.pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
    with threadpool_limits(limits=1):p=e.predict(np.arange(len(e.Y)),V,c)
    np.savez_compressed('experiments/phase24b-validation-predictions.npz',p=[p])
    y=e.pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy()
    result={**candidate,'validation':e.metrics(y,p,'t0.5')}
    e.save_json('experiments/phase24b-validation-results.json',[result]);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
