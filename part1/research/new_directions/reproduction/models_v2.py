"""Versioned follow-up: duplicate-aware Poisson and non-neural learned kernels."""
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import models


def poisson_unique(a,y,q,c):
    z,inv=np.unique(a.round(10),axis=0,return_inverse=True)
    targets=y/y.sum(1)[:,None]
    gold_counts=np.bincount(inv[:len(y)],minlength=len(z))
    evidence=np.zeros((len(z),5));np.add.at(evidence,inv[:len(y)],targets)
    known=gold_counts>0;evidence[known]/=gold_counts[known,None]
    counts=evidence.sum(0)
    sources=evidence/np.maximum(counts,1e-8)
    sources[known]-=(counts>0)[None,:]/known.sum()
    k=c['neighbors'];nn=NearestNeighbors(n_neighbors=k+1,n_jobs=1).fit(z)
    d,ix=nn.kneighbors(z);d,ix=d[:,1:],ix[:,1:]
    sig=np.maximum(d[:,-1],1e-5)
    w=np.exp(-d*d/(sig[:,None]*sig[ix]))
    W=sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(len(z)),k),ix.ravel())),shape=(len(z),len(z)))
    W=W.maximum(W.T);degree=np.asarray(W.sum(1)).ravel()
    ncomp,components=connected_components(W)
    # If a future fold has disconnected components, solve centered sources in each.
    for comp in range(ncomp):
        rows=components==comp;sources[rows]-=sources[rows].mean(0)
    potential=spsolve((sparse.diags(degree)-W+sparse.eye(len(z))*1e-6).tocsc(),sources)
    potential-=np.average(potential,weights=degree,axis=0)
    scale=max(np.sqrt(np.mean(potential**2)),1e-10)
    potential/=scale
    def infer(v):
        dist,ind=nn.kneighbors(v,n_neighbors=k)
        logw=-dist**2/(np.maximum(dist[:,-1,None],1e-6)*sig[ind])
        weights=np.exp(logw-logw.max(1,keepdims=True))
        logits=np.einsum('ij,ijk->ik',weights,potential[ind])/weights.sum(1)[:,None]
        exact=dist[:,0]<1e-8;logits[exact]=potential[ind[exact,0]]
        logits[:,counts==0]=-1e3
        return softmax(logits/c['temperature'],axis=1)
    p=infer(q);pu=infer(a[len(y):])
    return p,models.threshold(p),dict(nodes=len(z),collapsed=len(a)-len(z),components=ncomp,
                                      generated=models.threshold(pu).sum(0).tolist())


def kernel(a,b,M,bandwidth):
    am=a@M;bm=b@M
    d2=np.maximum(np.sum(am*a,1)[:,None]+np.sum(bm*b,1)[None,:]-2*am@b.T,0)
    distance=np.sqrt(d2)
    return np.exp(-distance/bandwidth),distance


def gradients(q,x,M,bandwidth,alpha):
    k,d=kernel(q,x,M,bandwidth)
    factor=np.divide(k,d*bandwidth,out=np.zeros_like(k),where=d>1e-7)
    out=[]
    for j in range(alpha.shape[1]):
        v=factor*alpha[:,j][None,:]
        out.append((v@x-q*v.sum(1)[:,None])@M)
    return np.stack(out,axis=1)


def rfm(a,y,q,c):
    n,d=len(y),a.shape[1];target=y/y.sum(1)[:,None]
    prior=target.mean(0);centered=target-prior
    M=np.eye(d);history=[]
    x=a[:n];band=c['bandwidth'];reg=c['ridge']
    for step in range(c['steps']+1):
        K,_=kernel(x,x,M,band)
        alpha=cho_solve(cho_factor(K+reg*np.eye(n),lower=True),centered)
        if step==c['steps']:break
        # Gradients on permitted fitting inputs only. This does not train a network.
        G=gradients(a,x,M,band,alpha)
        G/=np.sqrt(np.maximum(prior,.02))[None,:,None]
        agop=np.einsum('ncd,nce->de',G,G)/len(a)
        ev,U=np.linalg.eigh((agop+agop.T)/2)
        ev=np.maximum(ev,0)**c.get('power',.5)
        agop=(U*ev)@U.T
        agop*=d/max(np.trace(agop),1e-12)
        M=(1-c['shrink'])*agop+c['shrink']*np.eye(d)
        history.append(dict(top_eigenvalues=np.linalg.eigvalsh(M)[-5:].tolist()))
    def infer(z):
        Kq,_=kernel(z,x,M,band)
        p=np.clip(prior+Kq@alpha,0,1)
        return p/np.maximum(p.sum(1,keepdims=True),1e-12)
    p=infer(q);pu=infer(a[n:]);pseudo=0
    if c.get('augment'):
        # Explicit unsupervised missing-label generation in the learned metric.
        ev,U=np.linalg.eigh(M);z=a@(U*np.sqrt(np.maximum(ev,0)))
        km=KMeans(n_clusters=24,n_init=5,random_state=193).fit(z)
        counts=np.bincount(km.labels_[:n],minlength=24)
        votes=np.zeros((24,5));np.add.at(votes,km.labels_[:n],target)
        rates=(votes+prior)/(counts[:,None]+1)
        cp=rates[km.labels_[n:]]
        accept=(pu.max(1)>=.85)&(cp.max(1)>=.65)&(pu.argmax(1)==cp.argmax(1))&(counts[km.labels_[n:]]>=3)
        pseudo=int(accept.sum())
        if pseudo:
            xx=np.r_[x,a[n:][accept]];yy=np.r_[target,models.threshold(cp[accept])]
            yy=yy/yy.sum(1,keepdims=True)
            weights=np.r_[np.ones(n),np.full(pseudo,min(.1,.3*n/pseudo))]
            K,_=kernel(xx,xx,M,band)
            aa=cho_solve(cho_factor(K+np.diag(reg/weights),lower=True),yy-prior)
            p=np.clip(prior+kernel(q,xx,M,band)[0]@aa,0,1);p/=np.maximum(p.sum(1,keepdims=True),1e-12)
    return p,models.threshold(p),dict(metric_iterations=history,pseudo_rows_used=pseudo,
                                      generated=models.threshold(pu).sum(0).tolist())


def predict(gold,masked,query,c,labels,emb):
    assert not set(gold.ID)&set(query.ID) and not set(masked.ID)&set(query.ID)
    assert not any(k in query.columns for k in labels)
    assert masked[labels].eq(-100).all().all()
    y=gold[labels].to_numpy(int);assert np.isin(y,[0,1]).all()
    x=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()]
    sc=StandardScaler().fit(x);a=normalize(sc.transform(x));q=normalize(sc.transform(query[emb].to_numpy()))
    p,pred,audit=globals()[c['family']](a,y,q,c)
    assert np.isfinite(p).all() and np.isin(pred,[0,1]).all() and (pred.sum(1)>0).all()
    return p,pred,audit
