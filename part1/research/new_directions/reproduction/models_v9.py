"""Classical simplex/Dirichlet inference inspired by Martin et al., CVPR 2024.

No CLIP or neural model is loaded. Probability coordinates are generated from
provided embeddings, using fitting-only prototypes or unsupervised anchors.
The Dirichlet M-step implements equation 14 and Algorithm 1 (MM-quadratic).
"""
import numpy as np
from scipy.special import gammaln,digamma,softmax
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.cluster import KMeans
import models,models_v5
CACHE={}

def log_density(logx,alpha):
    return logx@(alpha-1).T+gammaln(alpha.sum(1))[None,:]-gammaln(alpha).sum(1)[None,:]

def fit_dirichlet(meanlog,initial,iterations=300):
    a=initial.copy();before=gammaln(a.sum(1))-gammaln(a).sum(1)+((a-1)*meanlog).sum(1)
    for step in range(iterations):
        curvature=2*(-gammaln(a+1)+digamma(a+1)*a)/a**2
        curvature=np.where(a<1e-4,np.pi**2/6,curvature)
        b=digamma(a+1)-digamma(a.sum(1))[:,None]-curvature*a-meanlog
        disc=np.sqrt(b*b+4*curvature)
        new=np.where(b>=0,2/np.maximum(b+disc,1e-30),(-b+disc)/(2*curvature))
        new=np.clip(new,1e-5,1e5)
        change=np.max(np.abs(new-a)/np.maximum(a,1e-8));a=new
        if change<1e-6:break
    after=gammaln(a.sum(1))-gammaln(a).sum(1)+((a-1)*meanlog).sum(1)
    assert (after>=before-1e-8*(1+abs(before))).all(),(before,after)
    return a,dict(iterations=step+1,relative_change=float(change),minimum_objective_gain=float((after-before).min()))

def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query for l in labels) and not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    config={k:v for k,v in c.items() if k not in ['name','blend']};key=models_v5.cache_key(gold,masked,query,config)
    if key not in CACHE:
        raw=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()];sc=StandardScaler().fit(raw)
        a=normalize(sc.transform(raw));q=normalize(sc.transform(query[emb].to_numpy()));n=len(gold)
        y=gold[labels].to_numpy(int);target=y/y.sum(1)[:,None]
        if c['representation']=='prototypes':centers=normalize(target.T@a[:n])
        else:centers=normalize(KMeans(n_clusters=16,n_init=10,random_state=2901).fit(a).cluster_centers_)
        z=np.maximum(softmax(10*a@centers.T,axis=1),1e-6);z/=z.sum(1)[:,None]
        v=np.maximum(softmax(10*q@centers.T,axis=1),1e-6);v/=v.sum(1)[:,None]
        if c.get('control'):
            p=v;audit={'representation':c['representation'],'decoder':.3,'generated':models.threshold(z[n:]).sum(0).tolist()}
        else:
            logz=np.log(z);alpha=np.ones((5,z.shape[1]));R=np.r_[target,np.full((len(a)-n,5),.2)]
            weights=np.r_[np.ones(n),np.full(len(a)-n,c['unlabeled_mass']*n/max(len(a)-n,1))]
            prior=(target.sum(0)+.5)/(n+2.5);history=[]
            for step in range(20 if c['unlabeled_mass'] else 1):
                weighted=R*weights[:,None];counts=weighted.sum(0)
                # Two global pseudo-observations stabilize rare-reason densities.
                meanlog=(weighted.T@logz+2*logz.mean(0)[None,:])/(counts[:,None]+2)
                alpha,details=fit_dirichlet(meanlog,alpha)
                prior=(counts+.5)/(counts.sum()+2.5)
                offset=c.get('prior_power',0.)*np.log(prior)
                posterior=softmax(log_density(logz,alpha)+offset,axis=1)
                R[n:]=posterior[n:];np.testing.assert_array_equal(R[:n],target)
                history.append(details)
            p=softmax(log_density(np.log(v),alpha)+c.get('prior_power',0.)*np.log(prior),axis=1)
            audit=dict(representation=c['representation'],em_steps=len(history),mm_steps=history,prior=prior.tolist(),
                       alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),generated=models.threshold(R[n:]).sum(0).tolist(),decoder=.3)
        CACHE[key]=(p,audit)
    p,audit=CACHE[key];cutoff=.3
    if c.get('blend'):
        p=(1-c['blend'])*models_v5.cached_old(gold,masked,query,labels,emb)+c['blend']*models_v5.centered(p);cutoff=.5
    assert np.isfinite(p).all()
    return p,models.threshold(p,cutoff),dict(**audit,final_cutoff=cutoff,method='Dirichlet probability-space mixture; no neural encoder')
