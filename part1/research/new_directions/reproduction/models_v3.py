"""RFM follow-ups selected from v2 evidence; all adaptation stays within each fit."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.preprocessing import StandardScaler,normalize
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import models,models_v2 as v2

BASE=dict(family='rfm',steps=2,ridge=.01,power=.5,shrink=.25,bandwidth=1.,augment=False)

def grouped_splits(gold,labels,emb,n_splits=3,seed=731):
    import pandas as pd
    group=pd.util.hash_pandas_object(gold[emb].round(10),index=False).to_numpy()
    keys,inv=np.unique(group,return_inverse=True)
    y=gold[labels].to_numpy(int)
    gy=np.array([y[inv==i].max(0) for i in range(len(keys))])
    for a,b in MultilabelStratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(gy,gy):
        yield np.flatnonzero(np.isin(inv,a)),np.flatnonzero(np.isin(inv,b))

def core(a,y,q,c):
    n,d=len(y),a.shape[1]
    if c.get('joint'):
        code=y@(2**np.arange(5))-1;target=np.eye(31)[code]
    else:target=y/y.sum(1)[:,None]
    prior=target.mean(0);centered=target-prior;M=np.eye(d)
    gradient_points=np.unique(a.round(10),axis=0) if c.get('unique_gradient') else a
    for step in range(c['steps']+1):
        K,_=v2.kernel(a[:n],a[:n],M,c['bandwidth'])
        alpha=cho_solve(cho_factor(K+c['ridge']*np.eye(n),lower=True),centered)
        if step==c['steps']:break
        G=v2.gradients(gradient_points,a[:n],M,c['bandwidth'],alpha)
        G/=np.sqrt(np.maximum(prior,.02))[None,:,None]
        agop=np.einsum('ncd,nce->de',G,G)/len(G)
        ev,U=np.linalg.eigh((agop+agop.T)/2);ev=np.maximum(ev,0)**c['power']
        agop=(U*ev)@U.T;agop*=d/max(np.trace(agop),1e-12)
        M=(1-c['shrink'])*agop+c['shrink']*np.eye(d)
    raw=np.clip(prior+v2.kernel(q,a[:n],M,c['bandwidth'])[0]@alpha,0,1)
    raw/=np.maximum(raw.sum(1,keepdims=True),1e-12)
    if c.get('joint'):return raw@models.PATTERNS,models.PATTERNS[(raw@models.UTILITY.T).argmax(1)]
    return raw,models.threshold(raw)

def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query.columns for l in labels)
    assert not set(gold.ID)&set(query.ID) and not set(masked.ID)&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    if c['family']=='rfm_fisher':
        p,_,_=v2.predict(gold,masked,query,BASE,labels,emb)
        f,_,_=models.predict(gold,masked,query,dict(family='fisher',clusters=24,residual_weight=1.,second_order=True),labels,emb)
        score=c['rfm_weight']*p+(1-c['rfm_weight'])*f
        return score,models.threshold(score),dict(rfm_weight=c['rfm_weight'])
    if c['family']=='rfm_bag':
        p,_,_=v2.predict(gold,masked,query,BASE,labels,emb)
        bag=[]
        for a,b in grouped_splits(gold,labels,emb,n_splits=5,seed=881):
            pp,_,_=v2.predict(gold.iloc[a],masked,query,BASE,labels,emb);bag.append(pp)
        score=.5*p+.5*np.mean(bag,axis=0)
        return score,models.threshold(score),dict(members=6,full_weight=.5)
    if c['family']=='rfm_nested':
        import pandas as pd
        candidates=[{**BASE,'bandwidth':bw,'ridge':ridge} for bw in [1.,.5,2.] for ridge in [.01,.1]]
        y=gold[labels].to_numpy(int);ug=pd.util.hash_pandas_object(masked[emb].round(10),index=False).to_numpy()
        inner=[np.full(y.shape,np.nan) for _ in candidates]
        audit=[]
        for a,b in grouped_splits(gold,labels,emb):
            bg=pd.util.hash_pandas_object(gold.iloc[b][emb].round(10),index=False).to_numpy()
            u=masked[~np.isin(ug,bg)]
            for i,config in enumerate(candidates):
                inner[i][b]=v2.predict(gold.iloc[a],u,gold.iloc[b].drop(columns=labels),config,labels,emb)[0]
            audit.append(dict(gold_ids=gold.iloc[a].ID.tolist(),heldout_ids=gold.iloc[b].ID.tolist(),masked_ids=u.ID.tolist()))
        from sklearn.metrics import f1_score
        choices=[]
        for i,p in enumerate(inner):
            assert np.isfinite(p).all()
            for threshold in [.3,.25,.4,.5]:
                f1=f1_score(y,models.threshold(p,threshold),average='samples',zero_division=0)
                choices.append((float(f1),-abs(threshold-.3),-i,i,threshold))
        f1,_,_,i,t=max(choices)
        p,_,_=v2.predict(gold,masked,query,candidates[i],labels,emb)
        return p,models.threshold(p,t),dict(selected=candidates[i],cutoff=t,inner_f1=f1,inner_folds=audit)
    config={**BASE,**c}
    x=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()]
    sc=StandardScaler().fit(x);a=normalize(sc.transform(x));q=normalize(sc.transform(query[emb].to_numpy()))
    p,pred=core(a,gold[labels].to_numpy(int),q,config)
    return p,pred,dict(joint=c.get('joint',False),unique_gradient=c.get('unique_gradient',False))
