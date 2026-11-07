"""Fold-local cluster ensembles with matched self-training and classifier controls."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from references.ani_adapter import fit_ani_base, ani_base_predict
import models, models_v5

ROOT=Path(__file__).resolve().parent
PREP={}; OUTPUT={}

def hard_membership(assignments):
    values=np.unique(assignments[assignments>=0])
    return (assignments[:,None]==values[None,:]).astype(float)

def prepare(gold,masked,query,labels,emb):
    key=models_v5.cache_key(gold,masked,query,{'family':'cluster_geometry_v6'})
    if key in PREP:return key,PREP[key]
    # IDs determine original row order only; never enter predictors or distances.
    sources=pd.concat([pd.read_csv(ROOT/'data/part1_train.csv'),pd.read_csv(ROOT/'data/part1_val.csv')],ignore_index=True)
    order={v:i for i,v in enumerate(sources.ID)}
    frame=pd.concat([gold,masked],ignore_index=True)
    frame=frame.iloc[np.argsort([order[i] for i in frame.ID])].reset_index(drop=True)
    known=frame[labels].ge(0).all(axis=1).to_numpy()
    fit=np.flatnonzero(known);unlab=np.flatnonzero(~known)
    y=frame.loc[known,labels].to_numpy(int)
    sc=StandardScaler().fit(frame[emb].to_numpy());z=sc.transform(frame[emb].to_numpy());q=sc.transform(query[emb].to_numpy())
    pc=PCA(30,random_state=0).fit(z);a=pc.transform(z);unit=normalize(a)
    gm=GaussianMixture(30,covariance_type='diag',n_init=3,random_state=0).fit(a)
    assert gm.converged_
    R={'gmm':gm.predict_proba(a),
       'kmeans':hard_membership(KMeans(30,n_init=10,random_state=0).fit_predict(unit)),
       'ward':hard_membership(AgglomerativeClustering(n_clusters=30,linkage='ward').fit_predict(unit)),
       'density':hard_membership(HDBSCAN(min_cluster_size=8,min_samples=3,cluster_selection_method='leaf',n_jobs=1,copy=False).fit_predict(unit))}
    # Label-free co-association: fraction of partitions putting a pair together.
    # Noise does not become one giant shared cluster. Soft GMM is hardened here.
    co=np.zeros((len(z),len(z)))
    for name,r in R.items():
        h=hard_membership(r.argmax(1)) if name=='gmm' else r
        co+=h@h.T/4
    distance=np.clip(1-co,0,1);np.fill_diagonal(distance,0)
    R['coassoc']=hard_membership(AgglomerativeClustering(n_clusters=30,metric='precomputed',linkage='average').fit_predict(distance))
    audit={name:dict(clusters=r.shape[1],covered_rows=int((r.sum(1)>0).sum())) for name,r in R.items()}
    PREP[key]=(z,q,y,fit,unlab,R,audit)
    return key,PREP[key]

def cluster_prob(r,augmented,target,unlab):
    t=target/target.sum(axis=1,keepdims=True)
    rates=(r[augmented].T@t+t.mean(0))/(r[augmented].sum(0)[:,None]+1)
    return r[unlab]@rates

def fit_one(key,prepared,members,weight,agreement=False):
    cache=(key,tuple(members),weight,agreement)
    if cache in OUTPUT:return OUTPUT[cache]
    z,q,y,fit,unlab,R,geometry=prepared
    augmented=fit.copy();target=y.copy();history=[]
    for _ in range(2):
        base=fit_ani_base(z[augmented],target);supervised=ani_base_predict(base,z[unlab])
        if members:
            probs=np.array([cluster_prob(R[name],augmented,target,unlab) for name in members])
            active=np.array([R[name][unlab].sum(1)>0 for name in members])
            votes=active.sum(0)
            mean=np.divide(probs.sum(0),votes[:,None],out=supervised.copy(),where=votes[:,None]>0)
            probability=(1-weight)*supervised+weight*mean
            if agreement:
                # Require every available cluster teacher to agree on top reason.
                winner=mean.argmax(1)
                agrees=np.all((probs.argmax(2)==winner[None,:])|(~active),axis=0)&(votes>=2)
            else:agrees=np.ones(len(unlab),bool)
        else:probability=supervised;agrees=np.ones(len(unlab),bool)
        keep=(probability.max(1)>=.8)&agrees
        generated=models.threshold(probability,.3)[keep]
        augmented=np.r_[fit,unlab[keep]];target=np.vstack([y,generated])
        history.append(dict(rows=int(keep.sum()),positive_counts=generated.sum(0).tolist(),agreement_rows=int(agrees.sum())))
    base=fit_ani_base(z[augmented],target);p=ani_base_predict(base,q)
    OUTPUT[cache]=(p,dict(members=members,weight=weight,agreement=agreement,rounds=history))
    return OUTPUT[cache]

def predict(gold,masked,query,c,labels,emb):
    assert not set(gold.ID)&set(query.ID) and not set(masked.ID)&set(query.ID)
    assert not any(l in query for l in labels) and masked[labels].eq(-100).all().all()
    key,prepared=prepare(gold,masked,query,labels,emb)
    if c.get('output_average'):
        parts=[fit_one(key,prepared,[m],c['weight']) for m in c['members']]
        p=np.mean([x[0] for x in parts],axis=0);audit=dict(component_fits=[x[1] for x in parts])
    else:p,audit=fit_one(key,prepared,c['members'],c['weight'],c.get('agreement',False))
    if c.get('replace_weight'):
        control=fit_one(key,prepared,['gmm'],.2)[0]
        p=models_v5.cached_old(gold,masked,query,labels,emb)+c['replace_weight']*(models_v5.centered(p)-models_v5.centered(control))
        assert p.min()>=-1e-8 and p.max()<=1+1e-8
        cutoff=.5
    else:cutoff=.3
    assert np.isfinite(p).all() and p.shape==(len(query),5)
    return p,models.threshold(p,cutoff),dict(cluster_geometry=prepared[-1],self_training=audit,decoder=cutoff,
                                          replace_weight=c.get('replace_weight',0),generated_labels_used=True)
