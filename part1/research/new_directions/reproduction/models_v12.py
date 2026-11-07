"""Classical adaptation of VAST (David & Weinshall, September 2026 preprint).

Uses supplied embeddings, graph evidence propagation, and regularized linear
softmax. No encoder, neural classifier, downloaded training data or weights.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler,normalize
import models,models_v5,current_reference
CACHE={};TEACHERS={}

def connected_kernel(a):
    n=len(a);D=cdist(a,a,'sqeuclidean');np.fill_diagonal(D,np.inf)
    order=np.argsort(D,axis=1,kind='stable')
    def graph(k):
        g=np.zeros((n,n),bool);g[np.arange(n)[:,None],order[:,:k]]=True
        return g|g.T
    lo,hi=1,n-1
    while lo<hi:
        mid=(lo+hi)//2
        if connected_components(csr_matrix(graph(mid)),directed=False,return_labels=False)==1:hi=mid
        else:lo=mid+1
    mask=graph(lo);K=np.exp(-D)*mask;degree=K.sum(1)
    correction=np.sqrt(np.log1p(degree[:,None]*degree[None,:]))
    K=np.divide(K,correction,out=np.zeros_like(K),where=correction>0)
    np.fill_diagonal(K,0.)
    assert np.isfinite(K).all() and np.allclose(K,K.T)
    return csr_matrix(K),lo

def propagate(K,known,target,enabled=True):
    # Seeding and gold-node frozen roles follow paper Eq. 2 / Algorithm 1.
    # Gold evidence rows may aggregate neighbors; actual gold training labels
    # are never replaced. Fractional multi-reason seeds are our adaptation.
    V=np.full(target.shape,1/target.shape[1])+K@target
    mass=V/V.sum(1,keepdims=True);conf=mass.max(1)
    frozen=known|(conf>=.75);history=[int(frozen.sum())]
    if enabled:
        for _ in range(len(known)+1):
            receive=~frozen;send=conf>1/target.shape[1]+1e-12
            new=V.copy();new[receive]+=(K@(V*send[:,None]))[receive]
            assert np.array_equal(new[frozen],V[frozen])
            V=new;mass=V/V.sum(1,keepdims=True);conf=mass.max(1)
            after=known|(conf>=.75);assert np.all(after[frozen])
            history.append(int(after.sum()))
            if np.array_equal(after,frozen):break
            frozen=after
        else:raise AssertionError('Propagation exceeded finite-freezing bound')
    return mass,dict(frozen_counts=history,rounds=len(history)-1,uninformed=int((conf<=.2+1e-12).sum()))

def adaptive_gate(p):
    conf=p.max(1)
    if not len(p):return np.zeros(0,bool),dict(base_threshold=.75,bimodal=False)
    H,E=np.histogram(conf,bins=100,range=(float(conf.min()),float(conf.max())))
    H=np.convolve(H,np.ones(3)/3,mode='same')
    peaks=[i for i in range(100) if H[i]>0 and H[i]>=(H[i-1] if i else -1) and H[i]>=(H[i+1] if i<99 else -1)]
    peaks=sorted(peaks,key=lambda i:(-H[i],i));first=peaks[0]
    second=next((p for p in peaks[1:] if abs(p-first)>=5 and H[p]>=.15*H[first]),None)
    bimodal=second is not None and conf.max()>conf.min()
    tau=float(E[max(first,second)]) if bimodal else .75
    means=p.mean(0);thresholds=np.maximum(tau*means/means.max(),.2)
    keep=conf>thresholds[p.argmax(1)]
    return keep,dict(base_threshold=tau,bimodal=bimodal,class_thresholds=thresholds.tolist(),accepted=int(keep.sum()))

def teacher(z,y,enabled):
    a=normalize(z);nodes,inv=np.unique(a.round(10),axis=0,return_inverse=True);n=len(y)
    counts=np.bincount(inv[:n],minlength=len(nodes));known=counts>0
    target=np.zeros((len(nodes),5));np.add.at(target,inv[:n],y/y.sum(1)[:,None])
    target[known]/=counts[known,None]
    K,k=connected_kernel(nodes);p,audit=propagate(K,known,target,enabled)
    pu=p[inv[n:]];keep,gate=adaptive_gate(pu)
    return pu,keep,dict(**audit,**gate,connectivity_k=k,nodes=len(nodes),collapsed=len(a)-len(nodes),propagation=enabled)

def objective(flat,a,y,n,soft,trust,strength,reg,direction):
    W=flat.reshape(a.shape[1],5);logp=a@W;logp-=logsumexp(logp,axis=1,keepdims=True);p=np.exp(logp)
    loss=-np.sum(y*logp[:n])/n+.5*reg*np.sum(W[:-1]**2)
    G=np.zeros_like(p);G[:n]=(p[:n]-y)/n
    if len(soft) and strength:
        w=strength*trust/len(soft)
        if direction=='forward':
            loss-=np.sum(w[:,None]*soft*logp[n:]);G[n:]=w[:,None]*(p[n:]-soft)
        else:
            diff=logp[n:]-np.log(np.maximum(soft,1e-12));kl=(p[n:]*diff).sum(1)
            loss+=np.dot(w,kl);G[n:]=w[:,None]*p[n:]*(diff-kl[:,None])
    grad=a.T@G;grad[:-1]+=reg*W[:-1]
    return float(loss),grad.ravel()

def student(z,y,pu,keep,q,c):
    n=len(y);a=np.column_stack([np.r_[z[:n],z[n:][keep]],np.ones(n+keep.sum())])
    target=y/y.sum(1)[:,None];soft=pu[keep];trust=soft.max(1) if len(soft) else np.empty(0)
    args=(a,target,n,soft,trust,c['strength'],1/(.3*n),c['direction'])
    first=minimize(objective,np.zeros(a.shape[1]*5),args=(*args[:5],0.,*args[6:]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-7})
    fitted=first if c['strength']==0 else minimize(objective,first.x,args=args,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-7,'maxls':50})
    assert fitted.success or np.max(np.abs(fitted.jac))<1e-4,(fitted.message,np.max(np.abs(fitted.jac)))
    p=softmax(np.column_stack([q,np.ones(len(q))])@fitted.x.reshape(a.shape[1],5),axis=1)
    return p,dict(optimizer_iterations=int(fitted.nit),gradient_inf=float(np.max(np.abs(fitted.jac))),pseudo_rows_used=int(keep.sum()))

def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query for l in labels) and not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    cfg={k:v for k,v in c.items() if k not in ['name','blend']};key=models_v5.cache_key(gold,masked,query,cfg)
    if key not in CACHE:
        raw=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()];sc=StandardScaler().fit(raw);z=sc.transform(raw)
        q=sc.transform(query[emb].to_numpy());y=gold[labels].to_numpy(int)
        tkey=models_v5.cache_key(gold,masked,gold[emb],{'propagate':c['propagate']})
        if tkey not in TEACHERS:TEACHERS[tkey]=teacher(z,y,c['propagate'])
        pu,keep,audit=TEACHERS[tkey]
        if c['strength']==0:keep=np.zeros(len(masked),bool)
        p,sa=student(z,y,pu,keep,q,c);CACHE[key]=(p,dict(**audit,**sa))
    p,audit=CACHE[key];cutoff=.3
    if c.get('blend'):
        old,_=current_reference.predict(gold,masked,query,labels,emb)
        p=(1-c['blend'])*old+c['blend']*models_v5.centered(p);cutoff=.5
    return p,models.threshold(p,cutoff),dict(**audit,decoder=cutoff,method='VAST-inspired classical linear student',direction=c['direction'],strength=c['strength'])
