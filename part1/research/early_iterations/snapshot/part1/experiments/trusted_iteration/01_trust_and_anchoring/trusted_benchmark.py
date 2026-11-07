"""Continue on the existing benchmark, preserving earlier experiments unchanged."""
import argparse
import json
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from simple_models import FEATURES, REASONS
from simple_benchmark import ROOT, BASE, digest, load_data, report, metrics
from trusted_models import TrustedReasonClassifier, fit_and_predict

OUT = ROOT / 'experiments/trusted_iteration'


def run(stage_name, specs):
    stage = OUT / stage_name
    stage.mkdir(parents=True, exist_ok=True)
    known, unknown = load_data()
    X, Y, U = known[FEATURES].to_numpy(), known[REASONS].to_numpy(int), unknown[FEATURES].to_numpy()
    paths = [BASE / 'splits' / f'repeat{r}.csv' for r in range(8)]
    folds = np.array([pd.read_csv(p).set_index('ID').loc[known.ID, 'fold'].to_numpy() for p in paths])
    protocol = dict(stage=stage_name, specs=specs,
                    question='Effect of pseudo-label trust, anchored class balancing and targeted Other decisions.',
                    fixed_splits={p.name:digest(p) for p in paths},
                    inputs={p.name:digest(p) for p in [ROOT/'data/part1_train.csv', ROOT/'data/part1_val.csv']},
                    fitting='188 known + 1014 originally unlabeled; all 47 held and all test inputs excluded from every fit.',
                    inner_tuning='When enabled: 3 stratified inner folds, seed8128; fit excludes inner held inputs; select Other cutoff by pooled sample F1, fixed tie preference.',
                    interpretation='Reused development benchmark. Later-year diagnostics have also been seen in prior work; no untouched final test claim.')
    path = stage/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        path.write_text(json.dumps(protocol, indent=2)+'\n')
    for filename in ['trusted_models.py', 'trusted_benchmark.py']:
        target=stage/filename
        if not target.exists():
            shutil.copyfile(ROOT/filename, target)
    predictions = {s['name']:np.full((8,235,5), -1, np.int8) for s in specs}
    probabilities = {s['name']:np.full((8,235,5), np.nan) for s in specs}
    audits=[]
    start=time.time()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for repeat in range(8):
            for fold in range(5):
                held=folds[repeat]==fold
                assert held.sum()==47 and (~held).sum()==188
                for s in specs:
                    model,P,pred,audit=fit_and_predict(X[~held],Y[~held],U,X[held],s)
                    assert np.isfinite(P).all() and np.isin(pred,[0,1]).all() and pred.sum(axis=1).min()>=1
                    probabilities[s['name']][repeat,held]=P
                    predictions[s['name']][repeat,held]=pred
                    audit.update(repeat=repeat, fold=fold, name=s['name'], fit_ids=known.loc[~held,'ID'].tolist(), held_ids=known.loc[held,'ID'].tolist())
                    audits.append(audit)
            print(f'{stage_name}: repeat {repeat+1}/8, {time.time()-start:.1f}s', flush=True)
    np.savez_compressed(stage/'predictions.npz', ids=known.ID.to_numpy(str), truth=Y, folds=folds, **predictions)
    np.savez_compressed(stage/'probabilities.npz', **probabilities)
    (stage/'fit_audits.json').write_text(json.dumps(audits, separators=(',',':'))+'\n')
    with np.load(stage/'predictions.npz') as disk:
        saved={s['name']:disk[s['name']] for s in specs}
    report(stage,known,folds,saved)
    return stage


def diagnostics(stage_name):
    stage=OUT/stage_name
    specs=json.loads((stage/'protocol.json').read_text())['specs']
    known,unknown=load_data()
    reference=pd.read_csv(ROOT/'experiments/simple_methodical/verification/diagnostic_predictions.csv')
    reference=reference[reference.name=='ani_train_only']
    settings=dict(cutoffs=[2003,2005,2007], originally_unlabeled_policy='Exclude years after cutoff, including their inputs.',
                  old_validation='157 original known fit; 78 former-validation held, all1014 unknown allowed.',
                  caveat='Diagnostics reused during development, not new independent confirmation.')
    (stage/'diagnostic_protocol.json').write_text(json.dumps(settings,indent=2)+'\n')
    splits=[('original_validation', np.arange(len(known))<157, np.ones(len(unknown),bool))]
    for year in [2003,2005,2007]:
        splits.append((f'after_{year}',known.start_year.to_numpy()<=year,unknown.start_year.to_numpy()<=year))
    rows, outputs, audits=[],[],[]
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for label,fit,unlab in splits:
            X,Y,U=known.loc[fit,FEATURES].to_numpy(),known.loc[fit,REASONS].to_numpy(int),unknown.loc[unlab,FEATURES].to_numpy()
            Q,truth=known.loc[~fit,FEATURES].to_numpy(),known.loc[~fit,REASONS].to_numpy(int)
            ref=reference[reference.split==label].set_index('ID').loc[known.loc[~fit,'ID']]
            np.testing.assert_array_equal(ref[[f'actual_{r}' for r in REASONS]],truth)
            refpred=ref[[f'predicted_{r}' for r in REASONS]].to_numpy(int)
            rows.append(dict(split=label,name='ani_train_only',verified_fit=len(X),unlabeled_fit=len(U),held=len(Q),**metrics(truth,refpred)))
            for s in specs:
                model,P,pred,audit=fit_and_predict(X,Y,U,Q,s)
                rows.append(dict(split=label,name=s['name'],verified_fit=len(X),unlabeled_fit=len(U),held=len(Q),**metrics(truth,pred)))
                df=pd.DataFrame(dict(ID=known.loc[~fit,'ID'],split=label,name=s['name']))
                for j,r in enumerate(REASONS):
                    df[f'actual_{r}']=truth[:,j];df[f'predicted_{r}']=pred[:,j];df[f'probability_{r}']=P[:,j]
                outputs.append(df)
                audit.update(split=label,name=s['name']);audits.append(audit)
            print(label,{r['name']:round(r['sample_f1'],6) for r in rows if r['split']==label},flush=True)
    pd.DataFrame(rows).to_csv(stage/'diagnostic_scores.csv',index=False)
    pd.concat(outputs).to_csv(stage/'diagnostic_predictions.csv',index=False)
    (stage/'diagnostic_audits.json').write_text(json.dumps(audits,separators=(',',':'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True)
    p.add_argument('--specs',type=Path)
    p.add_argument('--diagnostics',action='store_true')
    a=p.parse_args()
    if a.specs:
        run(a.stage,json.loads(a.specs.read_text()))
    if a.diagnostics:
        diagnostics(a.stage)
