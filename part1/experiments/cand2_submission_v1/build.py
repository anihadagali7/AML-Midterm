"""Export the previously evaluated strict-inductive, five-seed Cand2 recipe."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[1]
sys.path.insert(0, str(HERE / 'source'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from p1lib import cache
from p1lib.data import load, LABELS, write_submission
from p1lib.head import apply_head
import strict_adapter as strict

CSV = P1 / 'submissions/candidates/cand2_cs_features_inductive5.csv'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj): Path(p).write_text(json.dumps(obj, indent=2) + '\n')


def freeze():
    files = [HERE / 'build.py', HERE / 'strict_adapter.py', HERE / 'original_manifest.json',
             HERE / 'validation_reference.npz', HERE / 'validation_case.json', HERE / 'prior_comparison.csv']
    files += sorted((HERE / 'source/p1lib').glob('*.py')) + sorted((P1 / 'data').glob('*.csv'))
    hashes = {str(p.relative_to(P1)): sha(p) for p in files}
    path = HERE / 'protocol.json'
    if path.exists():
        assert read(path)['source_and_data_sha256'] == hashes
    else:
        save(path, dict(created_utc=datetime.now(timezone.utc).isoformat(),
            base_main='256fa8dabb2cd243f1162a0356e2c55b7a369a59', source_study_commit='2bf7e4f',
            recipe='cand2_compliant_argmax, strict-inductive adapter, five-inner-seed average',
            parent='Anchored 0.89636', status='Exploratory final fit authorized by user; failed prior CV advancement rule.',
            settings='Identical model/head to original manifest; transductive outer flag replaced by explicit strict context.',
            seeds=[0,1,2,3,4], aggregation='Average float64 combination posteriors and priors before expected-F1 decoder.',
            fitting='235 verified train/validation labels + 1014 originally masked training rows. No test inputs in fitting.',
            nested='Grouped inner calibration; complete inner refit; masked duplicate group purge; group-safe synthetic KNN features.',
            checks='Fresh five-seed validation replay must match retained benchmark before exporting final235 predictions.',
            selection='Selected for a distinct graph/feature/decision-rule direction, not a claimed local benchmark improvement.',
            no_upload=True, source_and_data_sha256=hashes))


def prepare(D, validation):
    if not validation:
        return strict.context(D, D.idx['labval'], D.idx['test'], D.idx['unl'], seed=0), D.idx['test']
    case = read(HERE / 'validation_case.json')
    pos = {v: i for i, v in enumerate(D.df.ID)}
    train, query, pool, fit = [np.array([pos[v] for v in case[key]]) for key in ['train_gold','query_ids','masked_ids','fit_ids']]
    keep = np.union1d(fit, query)
    remap = {v: i for i, v in enumerate(keep)}
    tr, qu, po = [np.array([remap[v] for v in rows]) for rows in [train, query, pool]]
    idx = dict(train=np.array([i for i,v in enumerate(keep) if v in set(D.idx['train'])]),
        lab=np.array([i for i,v in enumerate(keep) if v in set(D.idx['lab'])]),
        val=np.array([i for i,v in enumerate(keep) if v in set(D.idx['val'])]),
        unl=po, test=np.array([],int), labval=np.r_[tr,qu])
    sub = replace(D, df=D.df.iloc[keep].reset_index(drop=True), emb=D.emb[keep],
                  Y=D.Y[keep], y=D.y[keep], idx=idx, year=D.year[keep])
    return strict.context(sub,tr,qu,po,seed=0), qu


def one_seed(arg):
    validation, seed = arg
    D = load(str(P1 / 'data'))
    ctx, query = prepare(D, validation)
    ctx = replace(ctx, seed=seed)
    assert not np.intersect1d(ctx.unsup, query).size
    assert (ctx.D.Y[query] == -1).all()
    cfg = read(HERE / 'original_manifest.json')['config']
    cache.clear(); strict.AUDIT.clear()
    with threadpool_limits(limits=1):
        out = strict.run_model(cfg['model'],ctx,np.r_[query,ctx.pool])
    assert not any(a['gold_group_overlap'] or a['masked_group_overlap'] for a in strict.AUDIT)
    print(('Validation' if validation else 'Final235'), 'seed', seed, 'complete', flush=True)
    return dict(P=out['P'][:len(query)], prior=out['prior'], temperature=out['T'],
                nested_contexts=len(strict.AUDIT), fitting_gold=len(ctx.train), fitting_masked=len(ctx.pool))


def average(outputs, CM):
    P = np.mean([v['P'] for v in outputs],axis=0)
    prior = np.mean([v['prior'] for v in outputs],axis=0)
    assert np.isfinite(P).all() and np.allclose(P.sum(1),1)
    yp,Q,_ = apply_head(P,prior,CM,read(HERE/'original_manifest.json')['config']['head'])
    return yp,Q,P


def main():
    freeze()
    assert not CSV.exists(), 'Preserve exact exported CSV; use a new output version for another export.'
    D=load(str(P1/'data'))
    with ProcessPoolExecutor(max_workers=5) as pool:
        val=list(pool.map(one_seed,[(True,s) for s in range(5)]))
        vp,vq,vp0=average(val,D.CM)
        with np.load(HERE/'validation_reference.npz') as ref:
            np.testing.assert_allclose(vp0,ref['cand2_P'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(vq,ref['cand2_Q'],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(vp,ref['cand2_predictions'])
            replay_error=float(abs(vq-ref['cand2_Q']).max())
        print('Five-seed validation benchmark exactly reproduced.',flush=True)
        final=list(pool.map(one_seed,[(False,s) for s in range(5)]))
    yp,Q,P=average(final,D.CM)
    assert yp.shape==(312,5) and (yp.sum(1)>0).all()
    CSV.parent.mkdir(parents=True,exist_ok=True)
    sub=write_submission(D,yp,str(CSV))
    sample=pd.read_csv(P1/'data/part1_sample_submission.csv')
    assert list(sub.columns)==list(sample.columns) and sub.ID.tolist()==sample.ID.tolist()
    assert sub.ID.is_unique and len(sub)==312 and sub.Predicted.str.len().gt(0).all()
    allowed=set(LABELS)
    assert all(set(v.split()).issubset(allowed) for v in sub.Predicted)
    np.savez_compressed(HERE/'final_predictions.npz',ids=sub.ID.to_numpy(str),P=P,Q=Q,CM=D.CM,predictions=yp)
    anchor=pd.read_csv(P1/'submissions/archive/submitted/anchored_reason_trainval235__kaggle_0.89636.csv').set_index('ID')
    diff=sub.Predicted.to_numpy()!=anchor.loc[sub.ID,'Predicted'].to_numpy()
    changed=sub.loc[diff].copy();changed['anchored']=anchor.loc[changed.ID,'Predicted'].to_numpy()
    changed.to_csv(HERE/'changed_predictions.csv',index=False)
    meta=dict(status='prepared; not uploaded; Kaggle score unknown',csv=str(CSV.relative_to(P1)),
        sha256=sha(CSV),rows=312,verified_gold=235,originally_masked=1014,
        test_features_used_for_fit=False,prior_cv_passed=False,
        changed_vs_anchored=int(diff.sum()),label_counts=dict(zip(LABELS,map(int,yp.sum(0)))),
        mean_label_count=float(yp.sum(1).mean()),validation_replay_max_error=replay_error,
        validation_predictions_identical=True,seeds=[0,1,2,3,4],
        temperatures=[float(v['temperature']) for v in final],
        nested_contexts=sum(v['nested_contexts'] for v in final),
        prediction_sha256=sha(HERE/'final_predictions.npz'),protocol_sha256=sha(HERE/'protocol.json'))
    save(HERE/'prepared_submission.json',meta)
    freeze()
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
