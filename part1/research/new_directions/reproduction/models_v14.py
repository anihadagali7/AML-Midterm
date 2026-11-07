"""Pair-specific discriminants coupled into a normalized reason distribution."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import models,models_v5,current_reference
from references.lda_current import functions as lda
CACHE={};PACKETS={}

def couple(r):
    # Wu, Lin & Weng (2004), method 2: minimize sum of squared pairwise
    # consistency residuals subject to probabilities summing to one.
    n,k,_=r.shape;Q=np.zeros((n,k,k))
    for i in range(k):
        for j in range(k):
            if i!=j:Q[:,i,i]+=r[:,j,i]**2;Q[:,i,j]=-r[:,j,i]*r[:,i,j]
    system=np.zeros((n,k+1,k+1));system[:,:k,:k]=Q;system[:,:k,k]=1;system[:,k,:k]=1
    rhs=np.zeros((n,k+1,1));rhs[:,k,0]=1
    answer=np.linalg.solve(system,rhs)[:,:,0];p=answer[:,:k]
    assert p.min()>-1e-8 and np.isfinite(p).all()
    residual=np.max(np.abs(np.einsum('nij,nj->ni',system,answer)-rhs[:,:,0]))
    assert residual<1e-7
    p=np.maximum(p,0);p/=p.sum(1,keepdims=True)
    return p

def fit_pairs(x,y,family):
    heads=[]
    for i in range(5):
        for j in range(i+1,5):
            ai=np.flatnonzero(y[:,i]);aj=np.flatnonzero(y[:,j]);rows=np.r_[ai,aj];target=np.r_[np.zeros(len(ai),int),np.ones(len(aj),int)]
            if not len(ai) or not len(aj):model=float(bool(len(aj)))
            elif family=='lda':model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]).fit(x[rows],target)
            else:model=LogisticRegression(C=.1,class_weight='balanced',max_iter=3000).fit(x[rows],target)
            heads.append((i,j,model))
    return heads

def pair_predict(heads,q):
    r=np.zeros((len(q),5,5))
    for i,j,model in heads:
        pj=np.full(len(q),model) if isinstance(model,float) else model.predict_proba(q)[:,1]
        pj=np.clip(pj,1e-7,1-1e-7);r[:,i,j]=1-pj;r[:,j,i]=pj
    return couple(r)

def source_predict(base,heads,q,weight):
    lr=np.zeros((len(q),5));lr[:,base['lr'].classes_]=base['lr'].predict_proba(q)
    return lda.predict_base(base,q)-weight*lr+weight*pair_predict(heads,q)

def packet(gold,masked,labels,emb):
    key=models_v5.cache_key(gold,masked,gold[emb],{'family':'pairwise_packet'})
    if key not in PACKETS:
        frame=current_reference.fitting(gold,masked);sc=StandardScaler().fit(frame[emb].to_numpy());known=np.flatnonzero(frame[labels].ge(0).all(1))
        data=dict(z=sc.transform(frame[emb].to_numpy()),gold=known,y=frame.iloc[known][labels].to_numpy(int))
        native=lda.fitted_variant(data,dict(C=.1,teacher_lda=.2,final_lda=.2))
        PACKETS[key]=(frame,sc,data,native)
    return PACKETS[key]

def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query for l in labels) and not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    config={k:v for k,v in c.items() if k!='name'};key=models_v5.cache_key(gold,masked,query,config)
    if key in CACHE:return CACHE[key]
    frame,sc,data,native=packet(gold,masked,labels,emb);z=data['z'];known=data['gold'];y=data['y'];unknown=np.setdiff1d(np.arange(len(z)),known)
    q=sc.transform(query[emb].to_numpy());w=c['weight']
    def fitted(rows,target):
        base=lda.fit_base(z[rows],target,.1,0.)
        heads=fit_pairs(z[known] if c['population']=='gold' else z[rows],y if c['population']=='gold' else target,c['head'])
        return base,heads
    if c['scope']=='final':rows,target=native['rows'],native['targets'];history=native['history']
    else:
        rows,target=known.copy(),y.copy();resp=lda.geometry(data)['resp'];history=[]
        for _ in range(2):
            base,heads=fitted(rows,target);t=target/target.sum(1,keepdims=True)
            rates=(resp[rows].T@t+t.mean(0))/(resp[rows].sum(0)[:,None]+1)
            p=.8*source_predict(base,heads,z[unknown],w)+.2*resp[unknown]@rates;keep=p.max(1)>=.8
            rows=np.r_[known,unknown[keep]];target=np.r_[y,models.threshold(p,.3)[keep]]
            history.append(dict(accepted=int(keep.sum()),reason_counts=target[len(y):].sum(0).tolist()))
    base,heads=fitted(rows,target);new=models_v5.centered(source_predict(base,heads,q,w))
    oldsource=models_v5.centered(lda.predict_base(native['base'],q));old,oa=current_reference.predict(gold,masked,query,labels,emb)
    score=old+.7*(new-oldsource);assert score.min()>-1e-10 and score.max()<1+1e-10
    audit=dict(**oa,decoder=.5,pairwise_heads=10,head=c['head'],population=c['population'],scope=c['scope'],pair_weight=w,
       pair_training_ids=frame.iloc[known if c['population']=='gold' else rows].ID.tolist(),pseudo_ids=frame.iloc[rows[len(known):]].ID.tolist(),new_pseudo_rounds=history,
       coupling='Wu-Lin-Weng method 2, constrained quadratic linear system')
    CACHE[key]=(score,models.threshold(score,.5),audit);return CACHE[key]
