"""2025 kNN-LDP paper, Algorithm 1, with explicit multilabel/inductive adaptations.

Independent NumPy implementation. See references/recent_literature source URLs.
Unknown probability mass remains explicit during propagation; gold never changes.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.linear_model import LogisticRegression
import models,models_v5
CACHE={}

def propagate(neighbors,known,evidence):
    n,k=neighbors.shape
    values=evidence.copy();mass=known.astype(float);pending=~known.copy()
    reverse=[[] for _ in range(n)]
    for i in range(n):
        for j in neighbors[i]:reverse[j].append(i)
    available=mass[neighbors].mean(1);order=[]
    while pending.any():
        scores=np.where(pending,available,-1.);i=int(scores.argmax())
        if scores[i]<=0:break
        values[i]=values[neighbors[i]].mean(0);mass[i]=mass[neighbors[i]].mean();pending[i]=False;order.append(i)
        for j in reverse[i]:
            if pending[j]:available[j]=mass[neighbors[j]].mean()
    np.testing.assert_array_equal(values[known],evidence[known])
    assert np.all(mass>=0) and np.all(mass<=1+1e-12)
    return values,mass,order

def ldp(a,y,q,c):
    nodes,inv=np.unique(a.round(10),axis=0,return_inverse=True);n=len(y)
    target=y.astype(float) if c.get('binary') else y/y.sum(1)[:,None]
    counts=np.bincount(inv[:n],minlength=len(nodes));known=counts>0
    evidence=np.zeros((len(nodes),5));np.add.at(evidence,inv[:n],target)
    evidence[known]/=counts[known,None]
    k=min(c['neighbors'],len(nodes)-1)
    distance=cdist(nodes,nodes);np.fill_diagonal(distance,np.inf)
    neighbors=np.argsort(distance,axis=1,kind='stable')[:,:k]
    raw,mass,order=propagate(neighbors,known,evidence)
    def infer(v):
        d=cdist(v,nodes);ix=np.argsort(d,axis=1,kind='stable')[:,:k]
        p=raw[ix].mean(1);m=mass[ix].mean(1)
        exact=d[np.arange(len(v)),ix[:,0]]<1e-8
        p[exact]=raw[ix[exact,0]];m[exact]=mass[ix[exact,0]]
        p=np.divide(p,m[:,None],out=np.zeros_like(p),where=m[:,None]>0)
        fallback=m<=1e-12
        if fallback.any():
            gdist=cdist(v[fallback],a[:n]);gi=np.argsort(gdist,axis=1,kind='stable')[:,:min(k,n)]
            p[fallback]=target[gi].mean(1)
        return p,m,int(fallback.sum())
    p,m,f=infer(q)
    pu=np.divide(raw[inv[n:]],mass[inv[n:],None],out=np.zeros((len(a)-n,5)),where=mass[inv[n:],None]>0)
    return p,pu,mass[inv[n:]],dict(nodes=len(nodes),collapsed=len(a)-len(nodes),propagated=len(order),unreached=int(((~known)&(mass==0)).sum()),
           query_fallback=f,unknown_mass_mean=float((1-mass[~known]).mean()),generated=models.threshold(pu,.5 if c.get('binary') else .3).sum(0).tolist())

def weighted_softmax(a,t,weights,q,n_gold):
    rows,cls=np.where(t>0);w=t[rows,cls]*weights[rows]
    counts=np.bincount(cls,weights=w,minlength=5)+1e-9;w*=counts.mean()/counts[cls];w*=n_gold/w.sum()
    model=LogisticRegression(C=.3,max_iter=3000).fit(a[rows],cls,sample_weight=w)
    p=np.zeros((len(q),5));p[:,model.classes_]=model.predict_proba(q)
    return p

def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query for l in labels) and not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    cfg={k:v for k,v in c.items() if k not in ['name','blend']}
    key=models_v5.cache_key(gold,masked,query,cfg)
    if key not in CACHE:
        x=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()];sc=StandardScaler().fit(x)
        z=sc.transform(x);zq=sc.transform(query[emb].to_numpy());a=normalize(z);q=normalize(zq);y=gold[labels].to_numpy(int);n=len(y)
        p,pu,confidence,audit=ldp(a,y,q,c)
        if c.get('student'):
            target=y/y.sum(1)[:,None];weights=np.ones(n);train=z[:n];t=target
            if c.get('pseudo_mass',0):
                keep=(confidence>0)&(pu.sum(1)>0);soft=pu[keep]/pu[keep].sum(1)[:,None]
                trust=confidence[keep]*(1+np.sum(soft*np.log(np.maximum(soft,1e-12)),axis=1)/np.log(5))
                trust=np.maximum(trust,1e-8);trust*=c['pseudo_mass']*n/trust.sum()
                train=np.r_[train,z[n:][keep]];t=np.r_[t,soft];weights=np.r_[weights,trust]
                audit['student_pseudo_rows']=int(keep.sum())
            lr=weighted_softmax(train,t,weights,zq,n)
            if not c.get('pseudo_mass',0):
                sim=q@a[:n].T;ix=np.argsort(-sim,axis=1)[:,:10];w=np.take_along_axis(sim,ix,axis=1).clip(0)**2
                p=(w[:,:,None]*target[ix]).sum(1)/(w.sum(1)[:,None]+1e-9)
            p=.7*lr+.3*p
        cutoff=.5 if c.get('binary') else .3
        CACHE[key]=(p,audit,cutoff)
    p,audit,cutoff=CACHE[key]
    if c.get('blend'):
        p=(1-c['blend'])*models_v5.cached_old(gold,masked,query,labels,emb)+c['blend']*models_v5.centered(p);cutoff=.5
    return p,models.threshold(p,cutoff),dict(**audit,decoder=cutoff,method='kNN-LDP distribution propagation',binary_targets=c.get('binary',False))
