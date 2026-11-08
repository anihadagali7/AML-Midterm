"""Fixed supplementary check of the proposed five-inner-seed submission procedure.

Freeze this check separately from the already completed single-seed benchmark.
Average raw combo posteriors and source priors before the head, as submit.final_fit.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import run as study  # Sets numerical thread limits before NumPy is imported.
import numpy as np
from threadpoolctl import threadpool_limits
from p1lib import cache
from p1lib.pipeline import run_model
from p1lib.head import apply_head
import strict_adapter as strict

HERE = Path(__file__).resolve().parent
OUT = HERE / 'five_seed'


def freeze():
    inputs = [Path(__file__), HERE/'run.py', HERE/'strict_adapter.py', HERE/'protocol.json', HERE/'case_plan.json']
    hashes = {str(p.relative_to(study.REPO)):study.sha(p) for p in inputs}
    p = OUT/'protocol.json'
    if p.exists():
        assert study.read(p)['source_hashes'] == hashes
    else:
        study.save(p, dict(created_utc=datetime.now(timezone.utc).isoformat(),
             purpose='Check the five-inner-seed procedure used to build the proposed CSVs, in addition to reproducing the published single-seed benchmark.',
             parent='Anchored 0.89636; exact same fixed seven candidates/blends and all138 contexts.',
             seeds=[0,1,2,3,4], aggregation='Average raw float64 combo posteriors and source priors BEFORE prior correction and decoding, matching submit.final_fit. Use the corresponding query batch (native) or eligible masked training pool (strict) for SLD.',
             access='Identical native and strict boundaries to the primary protocol. Five seeds alter inner OOF/calibration, not outer split assignments.',
             preexisting_results='Single-seed results were known before this supplementary check. No candidate selection or parameter changes; all seven repeated.',
             controls='Reuse the freshly fitted deterministic anchored score arrays from the primary study; do not count these as fresh anchored fits.',
             no_submission_generated=True, source_hashes=hashes))


def worker(case):
    path = OUT/'results'/case['key']
    if path.with_suffix('.json').exists():
        assert study.sha(path.with_suffix('.npz')) == study.read(path.with_suffix('.json'))['sha256']
        return dict(key=case['key'],cached=True)
    started=time.time()
    try:
        D, base, _, query = study.prepare(case)
        with np.load(HERE/'results'/(case['key']+'.npz')) as original:
            arrays = {k:original[k].copy() for k in ['ids','truth','combo_matrix','anchored_scores','anchored_predictions']}
        accum = {name:dict(P=[],prior=[],T=[],info=[]) for name in study.CANDIDATES}
        audits=[]
        with threadpool_limits(limits=1):
            for seed in range(5):
                cache.clear();strict.AUDIT.clear()
                ctx=replace(base,seed=seed)
                rows=query if case['access']=='native_dylan' else np.r_[query,ctx.pool]
                runner=run_model if case['access']=='native_dylan' else strict.run_model
                for name,stem in study.CANDIDATES.items():
                    cfg=study.read(HERE/'manifests'/(stem+'.manifest.json'))['config']
                    o=runner(cfg['model'],ctx,rows)
                    a=accum[name];a['P'].append(o['P']);a['prior'].append(o['prior']);a['T'].append(o['T']);a['info'].append(o.get('info',{}))
                audits.extend(strict.AUDIT)
            metadata={}
            for name,stem in study.CANDIDATES.items():
                cfg=study.read(HERE/'manifests'/(stem+'.manifest.json'))['config']
                a=accum[name];P=np.mean(a['P'],axis=0);source=np.mean(a['prior'],axis=0)
                target=None if case['access']=='native_dylan' else P[len(query):]
                yp,Q,prior=apply_head(P[:len(query)],source,D.CM,cfg['head'],target)
                arrays[name+'_P']=P[:len(query)];arrays[name+'_Q']=Q
                arrays[name+'_scores']=Q@D.CM;arrays[name+'_predictions']=yp
                metadata[name]=dict(temperature=a['T'],source_prior=source.tolist(),
                                   target_prior=None if prior is None else prior.tolist(),info=a['info'])
            for name,members in study.BLENDS.items():
                s=sum(arrays[m+'_scores'] for m in members)/len(members)
                yp=(s>=.3).astype(int);yp[np.arange(len(s)),s.argmax(1)]=1
                arrays[name+'_scores']=s;arrays[name+'_predictions']=yp
        for name in study.MODELS:
            assert np.isfinite(arrays[name+'_scores']).all() and (arrays[name+'_predictions'].sum(1)>0).all()
        np.savez_compressed(path.with_suffix('.npz'),**arrays)
        meta=dict(key=case['key'],seconds=time.time()-started,models=metadata,seeds=[0,1,2,3,4],
                  nested_context_count=len(audits),nested_gold_group_overlaps=sum(x['gold_group_overlap'] for x in audits),
                  nested_masked_group_overlaps=sum(x['masked_group_overlap'] for x in audits),
                  nested_audit_sha256=hashlib.sha256(json.dumps(audits,sort_keys=True).encode()).hexdigest(),
                  sha256=study.sha(path.with_suffix('.npz')),anchor_reused_from_primary=True)
        study.save(path.with_suffix('.json'),meta)
        return dict(key=case['key'],seconds=round(meta['seconds'],2))
    except Exception:
        study.save(path.with_suffix('.failure.json'),dict(traceback=traceback.format_exc()));raise


def main():
    (OUT/'results').mkdir(parents=True,exist_ok=True)
    freeze()
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(worker,c) for c in study.cases()]
        for i,f in enumerate(as_completed(tasks),1):print(i,len(tasks),json.dumps(f.result()),flush=True)


if __name__=='__main__':main()
