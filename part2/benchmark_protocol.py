"""Persistent trial-level split definitions; test outcomes are never accessed."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from experiments import ROOT,load_data

VERSION='part2-local-v1'
DIRECTORY=ROOT/'experiments/local_benchmark'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    data=load_data();universe=pd.concat([data['train'],data['val']],ignore_index=True)
    fingerprints={s:digest(ROOT/f'outcome_{s}.csv') for s in ['train','val','test']}
    path=DIRECTORY/'protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text())
        assert protocol['version']==VERSION and protocol['data_sha256']==fingerprints,'Data or protocol changed. Create a new protocol version rather than mixing results.'
        return universe,protocol
    splits=[]
    def add(name,kind,tr,ev,repeat=None):
        a,b=universe.iloc[tr],universe.iloc[ev]
        assert not set(a.trial_id)&set(b.trial_id)
        assert not set(a.ID)&set(b.ID)
        assert set(a.phase)==set(b.phase)=={'I','II','III'}
        assert b.groupby('phase').label.nunique().eq(2).all()
        splits.append(dict(name=name,kind=kind,repeat=repeat,fit_ids=a.ID.tolist(),eval_ids=b.ID.tolist(),
            fit_rows=len(a),eval_rows=len(b),eval_phase_counts=b.phase.value_counts().to_dict(),
            fit_year_max=float(a.start_year.max()),eval_year_min=float(b.start_year.min()),eval_year_max=float(b.start_year.max())))
    add('official','official',np.arange(len(data['train'])),np.arange(len(data['train']),len(universe)))
    years=universe.start_year.fillna(2007).to_numpy()
    for cut,end in [(2007,2009),(2009,2011),(2011,2013)]:
        add(f'time_{cut}_to_{cut+1}_{end}','temporal',np.flatnonzero(years<=cut),np.flatnonzero((years>cut)&(years<=end)))
    strata=universe.phase+'_'+universe.label.astype(str)
    for repeat,seed in enumerate([42,43]):
        for fold,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(universe)),strata,groups=universe.trial_id)):
            add(f'group_repeat{repeat}_fold{fold}','group_cv',a,b,repeat)
    protocol=dict(version=VERSION,data_sha256=fingerprints,primary_metric='Equal mean of phase I/II/III log losses',
        fitting_universe='outcome_train + outcome_val, with evaluation trials excluded separately in every split',
        preprocessing='Outer and stacking-inner training rows only; no held-out or Kaggle-test text in vocabularies/IDF',
        temporal_selection='Prefer consistent chronological improvement; inspect the worst period and group-CV agreement. No scalar local score guarantees Kaggle performance.',
        note='Official validation and some historical data were consulted in earlier searches. New scores remain model-selection data, not independent evidence of test performance.',
        splits=splits)
    path.write_text(json.dumps(protocol,indent=2))
    return universe,protocol


def resolve(universe,split):
    indexed=universe.set_index('ID',drop=False)
    tr=indexed.loc[split['fit_ids']].reset_index(drop=True)
    ev=indexed.loc[split['eval_ids']].reset_index(drop=True)
    assert not set(tr.trial_id)&set(ev.trial_id)
    if split['kind']=='temporal':assert tr.start_year.max()<ev.start_year.min()
    return tr,ev


if __name__=='__main__':
    universe,protocol=initialize()
    print(pd.DataFrame([{k:v for k,v in s.items() if k not in ['fit_ids','eval_ids']} for s in protocol['splits']]).to_string(index=False))
