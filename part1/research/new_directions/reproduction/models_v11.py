"""Reason-specific nonlinear metrics, with the current LDA ensemble as parent."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import models,models_v5,models_v8,current_reference
CACHE={}

def separate(a,y,q,c):
    result=[];history=[];n,d=a.shape
    for j in range(y.shape[1]):
        target=y[:,j:j+1].astype(float);M=np.eye(d);bands=[]
        w=np.ones(n)
        if c.get('balance'):
            for label in [0,1]:
                selected=y[:,j]==label
                if selected.any():w[selected]=.5/selected.mean()
        for step in range(c['steps']+1):
            S=models_v8.root_matrix(M);D=cdist(a@S,a@S)
            band=max(float(np.median(D[np.triu_indices(n,1)])),1e-8);bands.append(band)
            K=np.exp(-D/band);alpha=cho_solve(cho_factor(K+.01*np.eye(n),lower=True),target)
            if step==c['steps']:break
            G=models_v8.gradients(a,a,S,band,2.,1.,alpha)[:,0,:]
            M=(G.T*w)@G/w.sum();M/=max(float(np.trace(M))/d,1e-30)
            M=.9*M+.1*np.eye(d)
        result.append(np.clip(models_v8.kernel(q,a,S,band,2.,1.)@alpha,0,1)[:,0])
        history.append(dict(reason=j,bandwidths=bands,metric_eigenvalues=np.linalg.eigvalsh(M)[-5:].tolist()))
    return np.column_stack(result),history

def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query for l in labels) and not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    old,oa=current_reference.predict(gold,masked,query,labels,emb)
    if c['kind']=='control':return old,models.threshold(old,.5),dict(**oa,decoder=.5)
    config={k:v for k,v in c.items() if k not in ['name','weights']};key=models_v5.cache_key(gold,masked,query,config)
    if key not in CACHE:
        raw=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()];sc=StandardScaler().fit(raw)
        a=sc.transform(gold[emb].to_numpy());q=sc.transform(query[emb].to_numpy());y=gold[labels].to_numpy(int)
        if c['kind']=='shared':
            cfg=dict(p=2.,q=1.,ridge=.01,binary=c.get('binary',False))
            states=models_v8.leaf_path(a,y,cfg,steps=1);new=models_v8.inference(q,a,states[-1],cfg)
            if not cfg['binary']:new=models_v5.centered(new)
            audit=dict(shared_metric=True,binary_targets=cfg['binary'])
        else:new,h=separate(a,y,q,c);audit=dict(shared_metric=False,reason_metrics=h)
        CACHE[key]=(new,audit)
    new,audit=CACHE[key];w=np.array(c['weights']);p=(1-w)*old+w*new
    return p,models.threshold(p,.5),dict(**oa,**audit,decoder=.5,weights=c['weights'],generated_labels_used=True)
