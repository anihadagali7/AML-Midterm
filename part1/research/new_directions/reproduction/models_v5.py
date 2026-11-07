"""New semi-supervised objectives and selective use of complementary learned geometry."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax,logsumexp,expit,logit
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import Nystroem
import models,models_v2
ROOT=Path(__file__).resolve().parent
CACHE={}

def centered(p):return expit(logit(np.clip(p,1e-9,1-1e-9))-logit(.3))

def old_prediction(gold,masked,query,labels,emb):
    """Faithful frozen three-way reference; fitting data are rebuilt without query rows."""
    engine=ROOT/'references/legacy_engine'
    if str(engine) not in sys.path:sys.path.append(str(engine))
    import run_cross_repo_benchmark as legacy
    # Original row order matters to deterministic clustering and bootstrapping.
    sources=pd.concat([pd.read_csv(ROOT/'data/part1_train.csv'),pd.read_csv(ROOT/'data/part1_val.csv')],ignore_index=True)
    order={v:i for i,v in enumerate(sources.ID)}
    frame=pd.concat([gold,masked],ignore_index=True)
    frame=frame.iloc[np.argsort([order[i] for i in frame.ID])].reset_index(drop=True)
    assert not set(frame.ID)&set(query.ID)
    with legacy.data_context(frame):
        e=legacy.e;pos=np.arange(len(gold));registry=legacy.registry()
        outputs={}
        for short,name in [('ani','Ani training-only adaptation'),('tri','Classical tri-training'),('svm','Local SVM specialist hybrid')]:
            c=next(r['config'] for r in registry if r['name']==name)
            outputs[short]=e.predict(pos,query,c)
    return .7*centered(outputs['ani'])+.15*outputs['tri']+.15*outputs['svm']

def cache_key(gold,masked,query,c):
    h=hashlib.sha256(json.dumps(c,sort_keys=True).encode())
    for df in [gold,masked,query]:h.update(pd.util.hash_pandas_object(df,index=False).to_numpy().tobytes())
    return h.hexdigest()

def cached_old(gold,masked,query,labels,emb):
    key=cache_key(gold,masked,query,{'family':'threeway'})
    if key not in CACHE:CACHE[key]=old_prediction(gold,masked,query,labels,emb)
    return CACHE[key]

def entropy_objective(flat,a,y,n,q0,reg,strength,mass_strength):
    W=flat.reshape(a.shape[1],5);logp=a@W;logp-=logsumexp(logp,axis=1,keepdims=True);p=np.exp(logp)
    rowmass=y.sum(1);total=rowmass.sum()
    loss=-np.sum(y*logp[:n])/total+.5*reg*np.sum(W[:-1]**2)
    grad_logits=np.zeros_like(p);grad_logits[:n]=(p[:n]*rowmass[:,None]-y)/total
    if strength:
        pu=p[n:];lp=logp[n:];h=-np.sum(pu*lp,1)
        loss+=strength*h.mean()
        grad_logits[n:]+=-strength*pu*(lp+h[:,None])/len(pu)
        mean=pu.mean(0);ratio=np.log(np.maximum(mean,1e-12)/q0)
        loss+=mass_strength*np.sum(mean*ratio)
        grad_logits[n:]+=mass_strength*pu*(ratio[None,:]-(pu@ratio)[:,None])/len(pu)
    grad=a.T@grad_logits;grad[:-1]+=reg*W[:-1]
    return float(loss),grad.ravel()

def entropy(a,y,q,c):
    n=len(y);target=y/y.sum(1)[:,None]
    counts=target.sum(0);balanced=target*counts.mean()/np.maximum(counts,1e-8)
    if c['representation']=='kernel':
        mapping=Nystroem(kernel='rbf',gamma=.003,n_components=128,random_state=413).fit(a)
        a=mapping.transform(a)*np.sqrt(150);q=mapping.transform(q)*np.sqrt(150)
    a=np.column_stack([a,np.ones(len(a))]);q=np.column_stack([q,np.ones(len(q))])
    reg=1/(.3*balanced.sum());W=np.zeros((a.shape[1],5))
    neutral=np.full(5,.2);history=[]
    strengths=[0.] if c['strength']==0 else [0.,c['strength']*.25,c['strength']*.5,c['strength']]
    q0=neutral
    for strength in strengths:
        result=minimize(entropy_objective,W.ravel(),args=(a,balanced,n,q0,reg,strength,1.),jac=True,
                        method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10,'gtol':1e-6,'maxls':50})
        assert result.success or np.max(np.abs(result.jac))<1e-4,(result.message,np.max(np.abs(result.jac)))
        W=result.x.reshape(a.shape[1],5)
        if not strength:q0=np.clip(softmax(a[n:]@W,axis=1).mean(0),1e-8,1);q0/=q0.sum()
        history.append(dict(strength=strength,iterations=int(result.nit),objective=float(result.fun),grad_inf=float(np.max(np.abs(result.jac)))))
    p=softmax(q@W,axis=1);pu=softmax(a[n:]@W,axis=1)
    return p,models.threshold(p),dict(optimizer=history,generated_labels=models.threshold(pu).sum(0).tolist(),unlabeled_mean=pu.mean(0).tolist(),initial_mean=q0.tolist())

def multimixture(a,y,q,c):
    n,d=len(y),a.shape[1];target=y/y.sum(1)[:,None]
    means=[];classes=[]
    for j in range(5):
        positive=np.flatnonzero(y[:,j]);k=min(c['components'],max(1,len(positive)//3))
        if not len(positive):continue
        km=KMeans(k,n_init=5,random_state=421+j).fit(a[positive]);means.extend(km.cluster_centers_);classes.extend([j]*k)
    means=np.array(means);classes=np.array(classes);initial=means.copy();K=len(means)
    cov0=np.cov(a.T);cov=cov0.copy();prior=np.array([1/np.sum(classes==j) for j in classes]);prior/=prior.sum()
    weights=np.r_[np.ones(n),np.full(len(a)-n,c['unlabeled_mass']*n/max(len(a)-n,1))]
    history=[]
    def post(x,balanced=False):
        cf=cho_factor(cov+np.eye(d)*1e-5,lower=True);im=cho_solve(cf,means.T).T
        logprior=np.log(prior)
        if balanced:
            masses=np.bincount(classes,weights=prior,minlength=5)
            logprior-=np.log(masses[classes])
        logits=x@im.T-.5*np.sum(means*im,1)[None,:]+logprior
        return softmax(logits,axis=1),logits
    for step in range(30):
        R,logits=post(a)
        for j in range(5):
            mask=classes==j
            if mask.any():R[:n,mask]=softmax(logits[:n,mask],axis=1)*target[:,j,None]
        R*=weights[:,None];counts=R.sum(0)+1e-10
        means=(R.T@a+2*initial)/(counts[:,None]+2)
        second=(a.T*R.sum(1))@a;cross=a.T@R@means
        residual=(second-cross-cross.T+(means.T*counts)@means)/R.sum()
        cov=.5*residual+.5*cov0;cov=(cov+cov.T)/2
        prior=(counts+.5)/(counts.sum()+.5*K)
        history.append(float(np.trace(cov)))
    mapping=np.eye(5)[classes];p=post(q,balanced=True)[0]@mapping;pu=post(a[n:],balanced=True)[0]@mapping
    return p,models.threshold(p),dict(components=K,residual_trace=history,generated_labels=models.threshold(pu).sum(0).tolist())

def predict(gold,masked,query,c,labels,emb):
    assert not set(gold.ID)&set(query.ID) and not set(masked.ID)&set(query.ID)
    assert not any(l in query.columns for l in labels) and masked[labels].eq(-100).all().all()
    if c['family']=='complement':
        base=cached_old(gold,masked,query,labels,emb)
        rfm_cfg=dict(family='rfm',steps=2,ridge=.01,power=.5,shrink=.25,bandwidth=1.,augment=False)
        key=cache_key(gold,masked,query,rfm_cfg)
        if key not in CACHE:CACHE[key]=models_v2.predict(gold,masked,query,rfm_cfg,labels,emb)[0]
        new=centered(CACHE[key]);weights=np.array(c['weights'])
        score=base*(1-weights)+new*weights
        return score,models.threshold(score,.5),dict(weights=c['weights'],base='frozen threeway',replacement='classical learned kernel',decoder=.5)
    x=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()];sc=StandardScaler().fit(x)
    a,q=sc.transform(x),sc.transform(query[emb].to_numpy());y=gold[labels].to_numpy(int)
    if c['family']=='multimixture':
        pc=PCA(n_components=30,svd_solver='full').fit(a);a,q=pc.transform(a),pc.transform(q)
        return multimixture(a,y,q,c)
    return entropy(a,y,q,c)
