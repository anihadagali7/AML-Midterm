"""Leakage, inference consistency, and objective/control checks for phase 26."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from sklearn.svm import SVC
from scipy.special import expit
from threadpoolctl import threadpool_limits
import phase26 as m,experiment as e

def clear():
    for module in [m,m.prior]:
        for name in ['PRED_CACHE','INNER_CACHE','GEOMETRY_CACHE']:
            if hasattr(module,name):getattr(module,name).clear()

def main():
    configs=[dict(model='label_shift26',method='em',strength=.5,output='raw',calibration_C=.3),
        dict(model='lap_svm26',C=1,gamma=.001,neighbors=10,graph_weight=1),
        dict(model='other_subclasses26',subgroups=3,representation='kernel',C=1,mix=.5,pseudo_weight=.03)]
    a,b=e.make_splits(seed=20261122)[0];q=e.DATA.iloc[e.LAB[b]].copy();ex=e.GROUP[e.LAB[b]];original=e.Y.copy();results=[]
    with threadpool_limits(limits=1):
        for c in configs:
            clear();p=e.predict(a,q,c,ex)
            try:
                e.Y[b]=1-e.Y[b];altered=q.copy();altered.loc[:,e.LABELS]=1-altered[e.LABELS].to_numpy()
                clear();p2=e.predict(a,altered,c,ex)
            finally:e.Y[:]=original
            one=e.predict(a,q.iloc[:1],c,ex);p3,pu,ix=e.predict(a,q,c,ex,True)
            assert np.allclose(p,p2,atol=1e-9,rtol=0)
            assert np.allclose(p[:1],one,atol=1e-9,rtol=0)
            assert np.allclose(p,p3,atol=1e-9,rtol=0)
            assert np.array_equal(ix,e.UNLAB[~np.isin(e.GROUP[e.UNLAB],ex)])
            assert pu.shape==(len(ix),5) and np.isfinite(pu).all()
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            results.append(dict(model=c['model'],heldout_label_invariance=True,query_batch_invariance=True,
                masked_duplicate_exclusion=True,query_label_invariance=True,unlabeled_output_consistency=True))
        F,Q,G,u,diag=m.geometry(a,q,configs[1],ex)
        R=cholesky(np.eye(len(G))+G,lower=True);beta=np.random.default_rng(26).normal(size=len(G))
        w=solve_triangular(R.T,beta,lower=False)
        assert np.allclose(w@(np.eye(len(G))+G)@w,beta@beta)
        zero=e.predict(a,q,{**configs[1],'graph_weight':0},ex)
        manual=[]
        for j in range(5):
            model=SVC(C=1,kernel='precomputed',class_weight='balanced').fit(F[:len(a)]@F[:len(a)].T,e.Y[a,j])
            manual.append(expit(2*model.decision_function(Q@F[:len(a)].T)))
        assert np.allclose(zero,np.array(manual).T,atol=1e-9)
        zero_shift=e.predict(a,q,{**configs[0],'strength':0},ex)
        raw=m.soft_incumbent(a,q,ex)
        assert np.allclose(zero_shift,m.clip(raw),atol=1e-9)
        for method in ['em','bbse']:
            sy=np.r_[np.zeros(60),np.ones(40)];sp=np.where(sy, .9,.1)
            target,_=m.estimate_prevalence(sp,sy,np.r_[np.full(800,.1),np.full(200,.9)],method)
            assert .1<target<.4
    result=dict(passed=True,checks=results,graph_objective_equivalence=True,zero_graph_matches_finite_kernel_svm=True,
        zero_shift_matches_raw_scores=True,synthetic_prior_shift_direction=True,caches_cleared=True,validation_fit=False,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['phase26.py','phase26_run.py']})
    e.save_json('experiments/phase26-integrity-checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
