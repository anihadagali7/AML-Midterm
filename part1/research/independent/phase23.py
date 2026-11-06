"""Nested out-of-fold score specialists and a joint sample-F1 decoder.
Only original training labels are used. Each outer fold owns its inner fits.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import phase22 as p22,experiment as e

legacy_predict=e.predict
INNER_CACHE={}

def component_scores(pos,query,c,excluded,return_unlabeled=False):
    b=p22.cached(pos,query,c['base'],excluded,return_unlabeled)
    t=p22.cached(pos,query,c['tri'],excluded,return_unlabeled)
    if return_unlabeled:
        assert np.array_equal(b[2],t[2])
        return np.c_[b[0],t[0]],np.c_[b[1],t[1]],b[2]
    return np.c_[b,t]

def meta_predict(p,y,q,c):
    X=logit(np.clip(p,1e-5,1-1e-5));Q=logit(np.clip(q,1e-5,1-1e-5))
    sc=StandardScaler();X=sc.fit_transform(X);Q=sc.transform(Q)
    if c.get('interactions'):
        poly=PolynomialFeatures(2,interaction_only=True,include_bias=False)
        X=poly.fit_transform(X);Q=poly.transform(Q)
    kind=c['meta'];C=c.get('meta_C',.3)
    if kind=='patterns':
        patterns,target=np.unique(y,axis=0,return_inverse=True)
        if len(patterns)==1:return np.tile(patterns[0],(len(q),1)).astype(float)
        model=LogisticRegression(C=C,max_iter=2000).fit(X,target)
        pp=model.predict_proba(Q);patterns=patterns[model.classes_]
        choices=((np.arange(1,32)[:,None]>>np.arange(5))&1)
        utility=2*(choices@patterns.T)/(choices.sum(1)[:,None]+patterns.sum(1)[None,:])
        return choices[(pp@utility.T).argmax(1)].astype(float)
    # Preserve the previous hybrid's original label decisions outside the specialist.
    bp=q[:,:5];tp=q[:,5:]
    hb=e.decisions(bp,c['base_rule']);ht=e.decisions(tp,c['tri_rule'])
    h=np.where(np.array(c['use_tri'],bool)[None,:],ht,hb)
    empty=h.sum(1)==0;h[empty]=hb[empty]
    fallback=np.where(np.array(c['use_tri'],bool)[None,:],tp,
                      expit(logit(np.clip(bp,1e-7,1-1e-7))-logit(float(c['base_rule'][1:]))))
    indices={'other':[4],'pair':[3,4],'all':list(range(5))}[kind]
    for j in indices:
        if len(np.unique(y[:,j]))==1:prob=np.full(len(q),y[0,j])
        else:
            model=LogisticRegression(C=C,max_iter=2000).fit(X,y[:,j])
            prob=model.predict_proba(Q)[:,1]
        strength=c.get('strength',1.)
        prob=strength*prob+(1-strength)*fallback[:,j]
        h[:,j]=(prob>=.5);fallback[:,j]=prob
    empty=h.sum(1)==0;h[np.flatnonzero(empty),fallback[empty].argmax(1)]=1
    return h.astype(float)

def predict23(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    if c['model']!='joint_specialist':
        return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
    pos=np.asarray(train_pos);excluded=np.asarray([] if excluded_groups is None else excluded_groups)
    inner=c.get('inner_splits',15)
    components={k:c[k] for k in ['base','tri']}
    key=(tuple(pos),e.Y[pos].tobytes(),tuple(sorted(excluded)),e.config_id(components),inner)
    if key not in INNER_CACHE:
        # No validation/test inputs are needed to construct these training features.
        p=np.zeros((len(pos),10))
        for a,b in e.make_splits(e.Y[pos],groups=e.GROUP[e.LAB[pos]],seed=230117,n_splits=inner):
            exclusions=np.r_[excluded,e.GROUP[e.LAB[pos[b]]]]
            p[b]=component_scores(pos[a],e.DATA.iloc[e.LAB[pos[b]]],c,exclusions)
        INNER_CACHE[key]=p
    p=INNER_CACHE[key]
    q=component_scores(pos,query_frame,c,excluded_groups,return_unlabeled)
    if return_unlabeled:
        query,unlabeled,ix=q
        out=meta_predict(p,e.Y[pos],np.r_[query,unlabeled],c)
        return out[:len(query)],out[len(query):],ix
    return meta_predict(p,e.Y[pos],q,c)

e.predict=predict23
