"""xRFM (ICLR 2026) leaf algorithm in NumPy/SciPy; no neural training.

The sample count is below the paper's leaf-size limit, so one full leaf suffices.
Lp,q kernels, raw AGOP/max normalization and adaptive bandwidth follow the paper
and author code. Multilabel decoding and optional LDP augmentation are adaptations.
"""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import models,models_v3,models_v5,models_v7
CACHE={};SELECT={}

def root_matrix(M):
    ev,U=np.linalg.eigh((M+M.T)/2)
    return (U*np.sqrt(np.maximum(ev,0)))@U.T

def kernel(a,b,S,band,p,q):
    distance=cdist(a@S,b@S,metric='minkowski',p=p)
    return np.exp(-(distance/band)**q)

def gradients(a,x,S,band,p,q,alpha):
    aa,xx=a@S,x@S
    diff=aa[:,None,:]-xx[None,:,:]
    moment=np.sum(np.abs(diff)**p,axis=2)
    k=np.exp(-(np.maximum(moment,0)**(1/p)/band)**q)
    factor=np.zeros_like(moment);valid=moment>1e-20
    factor[valid]=-q/(band**q)*k[valid]*moment[valid]**(q/p-1)
    component=np.sign(diff)*np.abs(diff)**(p-1)
    derivative=factor[:,:,None]*component
    G=np.einsum('ijd,jc->icd',derivative,alpha,optimize=True)
    return G@S.T

def leaf_path(a,y,c,steps=4):
    target=y.astype(float) if c.get('binary') else y/y.sum(1)[:,None]
    M=np.eye(a.shape[1]);states=[]
    for step in range(steps+1):
        S=root_matrix(M);D=cdist(a@S,a@S,metric='minkowski',p=c['p'])
        vals=D[np.triu_indices(len(a),1)];band=c.get('band_multiplier',1.)*max(float(np.median(vals)),1e-8)
        K=np.exp(-(D/band)**c['q'])
        alpha=cho_solve(cho_factor(K+c['ridge']*np.eye(len(a)),lower=True),target)
        states.append(dict(S=S,alpha=alpha,band=band,M=M.copy(),step=step))
        if step==steps:break
        G=gradients(a,a,S,band,c['p'],c['q'],alpha)
        M=np.einsum('ncd,nce->de',G,G,optimize=True)/len(a)
        if c.get('diagonal'):M=np.diag(np.diag(M))
        M/=max(float(M.max()),1e-30)
        assert np.isfinite(M).all()
    return states

def inference(q,x,state,c):
    p=np.clip(kernel(q,x,state['S'],state['band'],c['p'],c['q'])@state['alpha'],0,1)
    if not c.get('binary'):
        mass=p.sum(1);p=np.divide(p,mass[:,None],out=np.full_like(p,.2),where=mass[:,None]>1e-12)
    return p

def representation(gold,masked,query,c,emb):
    raw=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()];sc=StandardScaler().fit(raw)
    z=sc.transform(raw);q=sc.transform(query[emb].to_numpy())
    if c.get('pca'):
        pc=PCA(n_components=min(150,len(z),z.shape[1]),svd_solver='full').fit(z);z,q=pc.transform(z),pc.transform(q)
    return z,q

def candidates():
    return [dict(p=2.,q=1.,ridge=.01),dict(p=2.,q=.8,ridge=.01),dict(p=1.,q=1.,ridge=.01,pca=True),
            dict(p=1.,q=.8,ridge=.01,pca=True),dict(p=2.,q=1.,ridge=.01,binary=True),
            dict(p=1.,q=1.,ridge=.1,pca=True),dict(p=1.,q=1.,ridge=.01,pca=True,diagonal=True)]

def select(gold,masked,labels,emb):
    key=models_v5.cache_key(gold,masked,gold[emb],{'family':'leaf_nested_v8'})
    if key in SELECT:return SELECT[key]
    configs=candidates();y=gold[labels].to_numpy(int);pred=[np.full((5,len(y),5),np.nan) for _ in configs];audit=[]
    ug=pd.util.hash_pandas_object(masked[emb].round(10),index=False).to_numpy()
    for a,b in models_v3.grouped_splits(gold,labels,emb,n_splits=3,seed=8901):
        g=gold.iloc[a];held=gold.iloc[b];bg=pd.util.hash_pandas_object(held[emb].round(10),index=False).to_numpy()
        u=masked[~np.isin(ug,bg)]
        for ci,c in enumerate(configs):
            z,q=representation(g,u,held.drop(columns=labels),c,emb);states=leaf_path(z[:len(g)],g[labels].to_numpy(int),c)
            for step,state in enumerate(states):pred[ci][step,b]=inference(q,z[:len(g)],state,c)
        audit.append(dict(gold_ids=g.ID.tolist(),heldout_ids=held.ID.tolist(),masked_ids=u.ID.tolist()))
    scores=[]
    for ci,c in enumerate(configs):
        for step in range(5):
            pp=models.threshold(pred[ci][step],.5 if c.get('binary') else .3)
            value=float(np.mean(2*(pp*y).sum(1)/(pp.sum(1)+y.sum(1))))
            scores.append((value,-step,-ci,ci,step))
    value,_,_,ci,step=max(scores);choice={**configs[ci],'steps':step}
    SELECT[key]=(choice,dict(inner_folds=audit,inner_best_f1=value,selected=choice,inner_choices=35))
    return SELECT[key]

def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query for l in labels) and not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    cfg={k:v for k,v in c.items() if k not in ['name','blend']};key=models_v5.cache_key(gold,masked,query,cfg)
    if key not in CACHE:
        if c.get('nested'):chosen,audit=select(gold,masked,labels,emb)
        else:chosen={k:c[k] for k in ['p','q','ridge','steps','pca','diagonal','binary'] if k in c};audit={}
        z,q=representation(gold,masked,query,chosen,emb);y=gold[labels].to_numpy(int);n=len(y)
        states=leaf_path(z[:n],y,chosen,chosen['steps']);state=states[-1];x=z[:n]
        audit={**audit,'selected':chosen,'bandwidths':[s['band'] for s in states],'metric_top_eigenvalues':np.linalg.eigvalsh(state['M'])[-5:].tolist()}
        if c.get('pseudo_mass'):
            ldp_cfg=dict(neighbors=30,binary=chosen.get('binary',False))
            _,pu,mass,la=models_v7.ldp(z@state['S'],y,q@state['S'],ldp_cfg)
            if chosen.get('binary'):certainty=2*np.abs(pu-.5).mean(1)
            else:certainty=1+np.sum(pu*np.log(np.maximum(pu,1e-12)),1)/np.log(5)
            keep=(mass>0)&(certainty>.25)
            if keep.any():
                w=mass[keep]*np.clip(certainty[keep],1e-8,1);w*=c['pseudo_mass']*n/w.sum()
                target=y.astype(float) if chosen.get('binary') else y/y.sum(1)[:,None]
                x=np.r_[x,z[n:][keep]];t=np.r_[target,pu[keep]];weights=np.r_[np.ones(n),w]
                K=kernel(x,x,state['S'],state['band'],chosen['p'],chosen['q'])
                state={**state,'alpha':cho_solve(cho_factor(K+np.diag(chosen['ridge']/weights),lower=True),t)}
            audit.update(ldp=la,pseudo_rows_used=int(keep.sum()),pseudo_total_weight=c['pseudo_mass']*n)
        else:audit['pseudo_rows_used']=0
        p=inference(q,x,state,chosen);cutoff=.5 if chosen.get('binary') else .3
        CACHE[key]=(p,cutoff,audit)
    p,cutoff,audit=CACHE[key]
    if c.get('blend'):
        aligned=p if cutoff==.5 else models_v5.centered(p)
        p=(1-c['blend'])*models_v5.cached_old(gold,masked,query,labels,emb)+c['blend']*aligned;cutoff=.5
    return p,models.threshold(p,cutoff),dict(**audit,decoder=cutoff,method='xRFM leaf; NumPy implementation')
