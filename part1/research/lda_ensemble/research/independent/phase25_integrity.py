"""Uncached perturbation tests for all three requested techniques."""
from pathlib import Path
import json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
import phase25 as m,experiment as e

def clear():
    m.PRED_CACHE.clear();m.INNER_CACHE.clear()

def main():
    cs=[dict(model='label_tuned_ssl',selection='sample',penalty=.005,inner_splits=5),
        dict(model='reliable_ssl',teacher='ssl',reliability='pooled',pseudo_weight=.1,min_reliability=.8),
        dict(model='local_ssl',regions=3,C=1,gamma=.001,local_mix=.5)]
    a,b=e.make_splits(seed=20261118)[0];q=e.DATA.iloc[e.LAB[b]].copy();ex=e.GROUP[e.LAB[b]];original=e.Y.copy();results=[]
    with threadpool_limits(limits=1):
        for c in cs:
            clear();p=e.predict(a,q,c,ex)
            try:
                e.Y[b]=1-e.Y[b];altered=q.copy();altered.loc[:,e.LABELS]=1-altered[e.LABELS].to_numpy()
                clear();p2=e.predict(a,altered,c,ex)
            finally:e.Y[:]=original
            one=e.predict(a,q.iloc[:1],c,ex)
            p3,pu,ix=e.predict(a,q,c,ex,return_unlabeled=True)
            assert np.allclose(p,p2,atol=1e-10,rtol=0)
            assert np.allclose(p[:1],one,atol=1e-10,rtol=0)
            assert np.allclose(p,p3,atol=1e-10,rtol=0)
            assert np.array_equal(ix,e.UNLAB[~np.isin(e.GROUP[e.UNLAB],ex)])
            assert pu.shape==(len(ix),5) and np.isfinite(pu).all()
            assert np.isfinite(p).all() and ((p>=-1e-12)&(p<=1+1e-12)).all()
            results.append(dict(model=c['model'],heldout_label_invariance=True,query_label_invariance=True,
                query_batch_invariance=True,unlabeled_output_consistency=True,masked_duplicates_excluded=True,
                max_label_difference=float(abs(p-p2).max()),max_batch_difference=float(abs(p[:1]-one).max())))
        control=e.predict(a,q,{**cs[-1],'local_mix':0},ex)
        base=m.legacy_predict(a,q,m.BASE,ex)
        assert np.allclose(control,base,atol=1e-10,rtol=0)
    result=dict(checks=results,local_zero_mix_matches_global=True,caches_cleared_before_perturbation=True,
                validation_fit=False,source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['phase25.py','phase25_run.py']})
    e.save_json('experiments/phase25-integrity-checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
