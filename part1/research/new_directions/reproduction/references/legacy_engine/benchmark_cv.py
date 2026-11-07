"""Frozen, common-fold Part 1 comparisons. Never reads validation/test data.

Run: .venv/bin/python benchmark_cv.py
Register a future frozen recipe: --register candidate.json
The candidate needs name, config and rule; an existing validation metric is optional.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,re,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits
import phase26,experiment as e

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/benchmark_v1'
SEEDS=[20261126,20261127,20261128]

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_hash(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2))

def validate_protocol(p):
    assert p['training_sha256']==file_hash(ROOT/'data/part1_train.csv'),'Training data changed; create a new protocol version.'
    assert p['gold_ids']==e.DATA.iloc[e.LAB].ID.tolist()
    assert p['label_order']==e.LABELS and p['seeds']==SEEDS
    assert p['folds_sha256']==digest(p['folds'])
    for seed in SEEDS:
        folds=[f for f in p['folds'] if f['seed']==seed]
        assert len(folds)==5
        all_held=np.concatenate([f['heldout_gold_positions'] for f in folds])
        assert np.array_equal(np.sort(all_held),np.arange(len(e.Y)))
        for f in folds:
            a=np.array(f['train_gold_positions']);b=np.array(f['heldout_gold_positions'])
            assert not set(a)&set(b)
            assert set(a)|set(b)==set(range(len(e.Y)))
            assert not set(e.GROUP[e.LAB[a]])&set(e.GROUP[e.LAB[b]])
            assert f['heldout_ids']==e.DATA.iloc[e.LAB[b]].ID.tolist()
            excluded=np.isin(e.GROUP[e.UNLAB],e.GROUP[e.LAB[b]])
            assert f['masked_duplicates_excluded']==int(excluded.sum())

def protocol():
    OUT.mkdir(exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        p=json.loads(path.read_text());validate_protocol(p);return p
    folds=[]
    for seed in SEEDS:
        for i,(a,b) in enumerate(e.make_splits(seed=seed,n_splits=5)):
            folds.append(dict(seed=seed,fold=i,train_gold_positions=a.tolist(),heldout_gold_positions=b.tolist(),
                heldout_ids=e.DATA.iloc[e.LAB[b]].ID.tolist(),
                masked_duplicates_excluded=int(np.isin(e.GROUP[e.UNLAB],e.GROUP[e.LAB[b]]).sum())))
    p=dict(version='part1-common-cv-v1',created_at=now(),training_sha256=file_hash(ROOT/'data/part1_train.csv'),
        gold_ids=e.DATA.iloc[e.LAB].ID.tolist(),label_order=e.LABELS,seeds=SEEDS,folds_per_repeat=5,
        grouping='Hash all 150 reason embedding coordinates rounded to 10 decimal places; iterative multilabel stratification over groups.',
        metric='Mean of three complete-OOF sample-wise F1 scores; equal weight per gold trial within each repeat.',
        dispersion='Population standard deviation of three repeat scores, not a confidence interval.',
        fitting='Only current outer-training gold and original masked training rows; remove masked duplicates of held-out groups. No held-out inputs participate in fitting.',
        tuning='Freeze recipe and final decoder before benchmarking. Any adaptive tuning/calibration/feature selection must occur inside the outer training fold.',
        validation='Benchmark reads no validation or test CSV. Copy existing validation metrics with provenance; evaluate new validation candidates only after CV shortlisting and freezing.',
        shortlist='Positive mean paired CV difference versus the fixed incumbent and improvement in at least 2 of 3 repeats; descriptive screen, not a significance test or automatic promotion.',
        limitation='Historical tuning used these training labels and repeatedly consulted validation. This is a common development benchmark, not an independent test. Fixed folds can also be overfit through repeated search.',
        chronology='Any chronological stress test is supplementary and separately labeled; never substitute a favorable split for this primary benchmark.',
        folds=folds,folds_sha256=digest(folds))
    validate_protocol(p);save(path,p);return p

def transductive(x):
    if isinstance(x,dict):return bool(x.get('transductive',False)) or any(transductive(v) for v in x.values())
    if isinstance(x,list):return any(transductive(v) for v in x)
    return False

def entry(r,name,source,incumbent=False):
    return dict(name=name,config=r['config'],rule=r['rule'],incumbent=incumbent,
        recorded_validation=r.get('validation'),validation_source=source,
        validation_status='Previously measured; not rerun by this benchmark',registered_at=now())

def registry(add=None):
    path=OUT/'registry.json'
    if path.exists():rs=json.loads(path.read_text())
    else:
        current=json.loads((ROOT/'experiments/current-best.json').read_text())['candidate']
        old=json.loads((ROOT/'experiments/phase21-prior-incumbent.json').read_text())['candidate']
        rs=[entry(current,'Current best hybrid','experiments/current-best.json',True),
            entry(old,'Previous cluster-guided SVM','experiments/phase21-prior-incumbent.json')]
        names={'label_shift26':'Label-frequency correction','lap_svm26':'Graph-regularized SVM','other_subclasses26':'Other subgroups'}
        for r in json.loads((ROOT/'experiments/phase26-validation-results.json').read_text()):
            if not r['matched_control']:rs.append(entry(r,names[r['config']['learner']['model']], 'experiments/phase26-validation-results.json'))
    if add:
        r=json.loads(Path(add).read_text())
        assert all(k in r for k in ['name','config','rule'])
        assert not any(x['name']==r['name'] for x in rs),'Use a unique versioned candidate name.'
        rs.append(entry(r,r['name'],str(add)))
    for r in rs:
        assert not transductive(r['config']),'Transductive recipes belong in a separate benchmark.'
        if r.get('recorded_validation') is None:r['validation_status']='Not evaluated'
    assert sum(r.get('incumbent',False) for r in rs)==1
    save(path,rs);return rs

def evaluate(r,p):
    source_files=[ROOT/'experiment.py',Path(__file__)]+[f for f in ROOT.glob('phase*.py') if re.fullmatch(r'phase\d+\.py',f.name)]
    source_files += sorted((ROOT/'experiments').glob('phase*-baseline.json'))
    sources={str(f.relative_to(ROOT)):file_hash(f) for f in source_files}
    key=digest(dict(config=r['config'],rule=r['rule'],protocol=digest(p),sources=sources))[:20]
    path=OUT/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    started=time.time();all_scores=[];all_pred=[];fold_scores=[]
    try:
        with threadpool_limits(limits=1):
            for seed in p['seeds']:
                scores=np.full(e.Y.shape,np.nan)
                for f in [x for x in p['folds'] if x['seed']==seed]:
                    a=np.array(f['train_gold_positions']);b=np.array(f['heldout_gold_positions'])
                    query=e.DATA.iloc[e.LAB[b]].drop(columns=e.LABELS)
                    pred=e.predict(a,query,r['config'],excluded_groups=e.GROUP[e.LAB[b]])
                    assert pred.shape==(len(b),len(e.LABELS)) and np.isfinite(pred).all()
                    scores[b]=pred
                    # Use only the frozen rule. Never choose a rule on outer OOF labels.
                    fold_scores.append(dict(seed=seed,fold=f['fold'],rows=len(b),**e.metrics(e.Y[b],pred,r['rule'])))
                assert np.isfinite(scores).all()
                all_scores.append(scores);all_pred.append(e.decisions(scores,r['rule']))
                print(json.dumps(dict(candidate=r['name'],seed=seed,**e.metrics(e.Y,scores,r['rule']))),flush=True)
        repeat_scores=[e.metrics(e.Y,s,r['rule']) for s in all_scores]
        values=[v['sample_f1'] for v in repeat_scores]
        result=dict(name=r['name'],status='ok',key=key,config=r['config'],rule=r['rule'],protocol_sha256=digest(p),
            source_sha256=sources,cv_mean=float(np.mean(values)),repeat_std=float(np.std(values)),
            cv_macro=float(np.mean([v['macro_f1'] for v in repeat_scores])),repeat_scores=repeat_scores,
            fold_scores=fold_scores,seconds=round(time.time()-started,3),finished_at=now())
        np.savez_compressed(OUT/(key+'.npz'),scores=all_scores,predictions=all_pred,truth=e.Y,ids=e.DATA.iloc[e.LAB].ID.to_numpy())
        save(path,result);return result
    except Exception as error:
        result=dict(name=r['name'],status='error',error=str(error),traceback=traceback.format_exc(),finished_at=now())
        with (OUT/'errors.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();p=protocol()
    if args.verify_only:print('Verified frozen folds, training hash, group separation and masked duplicate counts.');return
    rs=registry(args.register);outputs=[]
    for r in rs:outputs.append(evaluate(r,p))
    base=outputs[next(i for i,r in enumerate(rs) if r['incumbent'])]
    rows=[]
    for r,out in zip(rs,outputs):
        deltas=np.array([x['sample_f1'] for x in out['repeat_scores']])-np.array([x['sample_f1'] for x in base['repeat_scores']])
        row=dict(candidate=r['name'],incumbent=r['incumbent'],cv_mean=out['cv_mean'],repeat_std=out['repeat_std'],
            cv_macro=out['cv_macro'],paired_cv_difference=float(deltas.mean()),repeats_improved=int(np.sum(deltas>1e-12)),
            cv_shortlist=bool(deltas.mean()>1e-12 and np.sum(deltas>1e-12)>=2),
            validation_f1=(r.get('recorded_validation') or {}).get('sample_f1'),validation_status=r['validation_status'],
            validation_source=r['validation_source'],result_key=out['key'])
        rows.append(row)
    e.pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    save(OUT/'summary.json',dict(protocol='part1-common-cv-v1',protocol_sha256=digest(p),folds_sha256=p['folds_sha256'],
        updated_at=now(),seeds=SEEDS,models=rows,validation_evaluations_performed=0,model_promotions_performed=0))
    print(e.pd.DataFrame(rows)[['candidate','cv_mean','repeat_std','paired_cv_difference','validation_f1']].to_string(index=False))

if __name__=='__main__':main()
