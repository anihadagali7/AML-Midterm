"""Uncached label/query isolation checks for both new learning approaches."""
from pathlib import Path
import json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
import phase24 as m,experiment as e

def clear():
    m.METRIC_CACHE.clear();m.VIEW_CACHE.clear();m.BASE_CACHE.clear()

def run():
    configs=[dict(model='co_training',views=['drugs','drug_targets','drug_actions','diseases_cleaned'],agreement=True,pseudo_weight=.1),
             dict(model='metric_ssl',metric='nca',metric_pca=25,components=5,metric_weight=.4)]
    a,b=e.make_splits(seed=20261114)[0];q=e.DATA.iloc[e.LAB[b]].copy();ex=e.GROUP[e.LAB[b]]
    original=e.Y.copy();checks=[]
    with threadpool_limits(limits=1):
        for c in configs:
            clear();p=e.predict(a,q,c,ex)
            try:
                e.Y[b]=1-e.Y[b];changed=q.copy();changed.loc[:,e.LABELS]=1-changed[e.LABELS].to_numpy()
                clear();p2=e.predict(a,changed,c,ex)
            finally:e.Y[:]=original
            single=e.predict(a,q.iloc[:1],c,ex)
            p3,pu,ix=e.predict(a,q,c,ex,return_unlabeled=True)
            delta=float(abs(p-p2).max());batch=float(abs(p[:1]-single).max())
            assert delta<1e-10 and batch<1e-10
            assert np.allclose(p,p3,atol=1e-10,rtol=0)
            assert not np.isin(e.GROUP[ix],ex).any()
            assert np.array_equal(ix,e.UNLAB[~np.isin(e.GROUP[e.UNLAB],ex)])
            assert pu.shape==(len(ix),5) and np.isfinite(pu).all()
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            checks.append(dict(model=c['model'],heldout_label_invariance=True,query_label_invariance=True,
                query_batch_invariance=True,return_unlabeled_query_invariance=True,
                masked_duplicates_excluded=True,max_label_difference=delta,max_batch_difference=batch))
    result=dict(checks=checks,caches_cleared_for_label_perturbation=True,validation_fit=False,
                source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['phase24.py','phase24_run.py']})
    e.save_json('experiments/phase24-integrity-checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
