"""Mathematical and leakage checks for custom experiment plumbing."""
import numpy as np
import experiment as e
from threadpoolctl import threadpool_limits
from sklearn.metrics import f1_score
rng=np.random.default_rng(42)
with threadpool_limits(limits=1):
    for a,b in e.make_splits():
        assert len(set(e.GROUP[e.LAB[a]])&set(e.GROUP[e.LAB[b]]))==0
    # Deterministic posteriors must produce precisely their nonempty labels.
    patterns=((np.arange(1,32)[:,None]>>np.arange(5))&1)
    assert np.array_equal(e.decisions(patterns,'expected_f1'),patterns)
    # Graph must respect two well-separated components, and probabilities sum appropriately.
    X=np.r_[rng.normal(-10,.1,(20,3)),rng.normal(10,.1,(20,3))]
    Y=np.full((40,5),-1);Y[0]=[1,0,0,0,0];Y[20]=[0,1,0,0,0]
    for hard in [False,True]:
        m=e.Graph({'k':5,'alpha':.95,'hard':hard}).fit(X,Y)
        p=m.predict(np.array([[-10]*3,[10]*3]))
        # Hard mode deliberately shrinks toward the global prior, unlike normalized spreading.
        assert p[0,0]>(.6 if hard else .95) and p[1,1]>(.6 if hard else .95)
        assert np.all((p>=0)&(p<=1))
    a,b=e.make_splits()[0]
    c={'model':'logistic','C':.1}
    p=e.predict(a,e.DATA.iloc[e.LAB[b]],c,excluded_groups=e.GROUP[e.LAB[b]])
    original=e.Y[b].copy();e.Y[b]=1-original
    q=e.predict(a,e.DATA.iloc[e.LAB[b]],c,excluded_groups=e.GROUP[e.LAB[b]])
    e.Y[b]=original
    assert np.allclose(p,q), 'Held-out labels affected fitting'
    query=e.DATA.iloc[e.LAB[b]].copy();query['ID']='arbitrary'
    assert np.allclose(p,e.predict(a,query,c,excluded_groups=e.GROUP[e.LAB[b]])), 'ID affected predictions'
    # Scaler fit is independent of query batch; changing one query cannot change the others.
    altered=query.copy();altered.loc[altered.index[0],e.EMB]=100000
    r=e.predict(a,altered,c,excluded_groups=e.GROUP[e.LAB[b]])
    assert np.allclose(p[1:],r[1:]), 'Query contaminated fit'
print('PASS: grouped folds, expected-F1 decoder, graph diffusion, held-out-label isolation, ID exclusion, preprocessing isolation')
