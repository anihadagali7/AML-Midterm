"""Nested per-label tuning, calibrated pseudo-label reliability, local SVM experts.
All fit transformations and label use are confined to the current training fold.
"""
import json
import numpy as np
from scipy.special import expit,logit
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,pairwise_distances
from sklearn.svm import SVC
import phase24,experiment as e

legacy_predict=e.predict
BASE=dict(model='lowdensity_v2',cluster_gate=True,clusters=10,C=1,gamma=.001,pseudo_weight=.03)
INNER_CACHE={};PRED_CACHE={};LAST_DIAGNOSTICS={}

def parameter_grid():
    options=[BASE.copy()]
    for key,values in [('C',[.3,3]),('gamma',[.0005,.002]),('pseudo_weight',[.01,.1])]:
        options += [{**BASE,key:v} for v in values]
    options += [{**BASE,'C':3,'gamma':.0005},{**BASE,'C':.3,'gamma':.002}]
    return options

def cached_predict(pos,query,c,excluded=None,return_unlabeled=False):
    key=(tuple(pos),e.Y[pos].tobytes(),phase24.query_key(query),e.config_id(c),
         tuple(sorted([] if excluded is None else excluded)),return_unlabeled)
    if key not in PRED_CACHE:
        PRED_CACHE[key]=predict25(pos,query,c,excluded,return_unlabeled)
    return PRED_CACHE[key]

def inner_predictions(pos,config,excluded,folds=5):
    exclusion=[] if excluded is None else list(excluded)
    key=(tuple(pos),e.Y[pos].tobytes(),tuple(sorted(exclusion)),e.config_id(config),folds)
    if key not in INNER_CACHE:
        p=np.zeros((len(pos),5))
        for a,b in e.make_splits(e.Y[pos],groups=e.GROUP[e.LAB[pos]],seed=250117,n_splits=folds):
            p[b]=cached_predict(pos[a],e.DATA.iloc[e.LAB[pos[b]]],config,
                np.r_[exclusion,e.GROUP[e.LAB[pos[b]]]])
        INNER_CACHE[key]=p
    return INNER_CACHE[key]

def data(pos,query,excluded,return_unlabeled):
    idx,u=phase24.indices(pos,excluded);A,B=e.features(idx,query,dict(scale='standard'))
    Q=np.r_[B,A[len(pos):]] if return_unlabeled else B
    K=np.exp(-.001*pairwise_distances(A,metric='sqeuclidean'))
    Kq=np.exp(-.001*pairwise_distances(Q,A,metric='sqeuclidean'))
    return idx,u,A,B,Q,K,Kq

def gold_svm(pos,query,c,excluded=None,return_unlabeled=False):
    _,u,A,B,Q,K,Kq=data(pos,query,excluded,return_unlabeled);n=len(pos);out=[]
    for j in range(5):
        y=e.Y[pos,j]
        if len(np.unique(y))<2:out.append(np.full(len(Q),y[0]));continue
        model=SVC(C=1,kernel='precomputed',class_weight='balanced').fit(K[:n,:n],y)
        out.append(expit(2*model.decision_function(Kq[:,:n])))
    p=np.array(out).T
    return (p[:len(B)],p[len(B):],u) if return_unlabeled else p

def tuned(pos,query,c,excluded=None,return_unlabeled=False):
    global LAST_DIAGNOSTICS
    options=parameter_grid();ps=np.array([inner_predictions(pos,o,excluded,c.get('inner_splits',5)) for o in options])
    y=e.Y[pos];chosen=np.zeros(5,int);penalty=c.get('penalty',.005)
    def matrix(selection):return np.column_stack([ps[selection[j],:,j] for j in range(5)])
    def objective(selection):
        h=e.decisions(matrix(selection),'t0.55')
        value=f1_score(y,h,average='samples',zero_division=0)
        return value-penalty*np.count_nonzero(selection)/5
    if c.get('selection','sample')=='binary':
        for j in range(5):
            vals=[f1_score(y[:,j],ps[k,:,j]>=.55,zero_division=0)-penalty*(k!=0) for k in range(len(options))]
            chosen[j]=int(np.argmax(vals))
    else:
        for _ in range(3):
            previous=chosen.copy()
            for j in range(5):
                candidates=[]
                for k in range(len(options)):
                    trial=chosen.copy();trial[j]=k
                    candidates.append((objective(trial),k==0,-k,k))
                chosen[j]=max(candidates)[-1]
            if np.array_equal(previous,chosen):break
    outputs={k:cached_predict(pos,query,options[k],excluded,return_unlabeled) for k in np.unique(chosen)}
    LAST_DIAGNOSTICS=dict(selected_by_label={e.LABELS[j]:options[chosen[j]] for j in range(5)},inner_rows=len(pos))
    if return_unlabeled:
        indices=[x[2] for x in outputs.values()];assert all(np.array_equal(ix,indices[0]) for ix in indices)
        return np.column_stack([outputs[chosen[j]][0][:,j] for j in range(5)]),np.column_stack([outputs[chosen[j]][1][:,j] for j in range(5)]),indices[0]
    return np.column_stack([outputs[chosen[j]][:,j] for j in range(5)])

def cluster_prior(A,Y):
    n=len(Y);z=KMeans(n_clusters=10,n_init=10,random_state=42).fit_predict(A)
    prior=np.full((len(A)-n,5),.5)
    for group in np.unique(z):
        gold=z[:n]==group
        if gold.sum()>=2:prior[z[n:]==group]=Y[gold].mean(0)
    return prior

def reliability(pos,query,c,excluded=None,return_unlabeled=False):
    global LAST_DIAGNOSTICS
    teacher=BASE if c.get('teacher','ssl')=='ssl' else dict(model='gold_svm25')
    pquery,pu,ix=cached_predict(pos,query,teacher,excluded,True)
    _,u,A,B,Q,K,Kq=data(pos,query,excluded,return_unlabeled)
    assert np.array_equal(ix,u);n=len(pos);Y=e.Y[pos]
    d=logit(np.clip(pu,1e-7,1-1e-7))/2;pseudo=(d>=0).astype(int)
    kind=c.get('reliability','per_label');quality=np.ones_like(pu)
    if kind!='uniform':
        po=inner_predictions(pos,teacher,excluded,c.get('inner_splits',5))
        do=logit(np.clip(po,1e-7,1-1e-7))/2
        if kind=='per_label':
            for j in range(5):
                if len(np.unique(Y[:,j]))<2:quality[:,j]=1;continue
                model=LogisticRegression(C=1,max_iter=1000).fit(do[:,j:j+1],Y[:,j])
                pr=model.predict_proba(d[:,j:j+1])[:,1]
                quality[:,j]=np.where(pseudo[:,j],pr,1-pr)
        else:
            def features(margins):
                v=margins.ravel();labels=np.tile(np.arange(5),len(margins))
                return np.c_[np.abs(v),v,np.eye(5)[labels]]
            correct=((do>=0)==Y).ravel().astype(int)
            if len(np.unique(correct))<2:quality[:]=float(correct[0])
            else:
                model=LogisticRegression(C=1,max_iter=1000).fit(features(do),correct)
                quality=model.predict_proba(features(d))[:,1].reshape(d.shape)
        quality=np.where(quality>=c.get('min_reliability',.5),np.clip(2*quality-1,0,1)**2,0)
    priors=cluster_prior(A,Y);out=[];diagnostics=[]
    for j in range(5):
        y=Y[:,j]
        if len(np.unique(y))<2:out.append(np.full(len(Q),y[0]));continue
        sw=c.get('pseudo_weight',.1)*quality[:,j]
        sw[((priors[:,j]>=.5)!=pseudo[:,j])&(np.maximum(priors[:,j],1-priors[:,j])>=.8)]*=.1
        # Zero-weight rows are removed so they do not affect automatic class counts.
        kept=np.flatnonzero(sw>1e-9);rows=np.r_[np.arange(n),n+kept]
        model=SVC(C=1,kernel='precomputed',class_weight='balanced').fit(K[np.ix_(rows,rows)],np.r_[y,pseudo[kept,j]],sample_weight=np.r_[np.ones(n),sw[kept]])
        out.append(expit(2*model.decision_function(Kq[:,rows])))
        diagnostics.append(dict(label=e.LABELS[j],accepted=len(kept),pseudo_mass=float(sw.sum())))
    LAST_DIAGNOSTICS=dict(reliability=kind,teacher=c.get('teacher','ssl'),labels=diagnostics)
    p=np.array(out).T
    return (p[:len(B)],p[len(B):],u) if return_unlabeled else p

def local_experts(pos,query,c,excluded=None,return_unlabeled=False):
    global LAST_DIAGNOSTICS
    gq,gu,ix=cached_predict(pos,query,BASE,excluded,True)
    idx,u,A,B,Q,K,Kq=data(pos,query,excluded,return_unlabeled);n=len(pos);Y=e.Y[pos]
    assert np.array_equal(ix,u)
    global_scores=np.r_[gq,gu] if return_unlabeled else gq
    km=KMeans(n_clusters=c.get('regions',3),n_init=10,random_state=250117).fit(A)
    distances=km.transform(A)**2;query_distances=km.transform(Q)**2
    width=max(float(np.median(np.min(distances,axis=1))),1e-6)*c.get('bandwidth',1.)
    routing=np.exp(-(query_distances-query_distances.min(1)[:,None])/(2*width))
    routing/=routing.sum(1)[:,None]
    gamma=c.get('gamma',.001)
    if gamma!=.001:
        K=np.exp(-gamma*pairwise_distances(A,metric='sqeuclidean'))
        Kq=np.exp(-gamma*pairwise_distances(Q,A,metric='sqeuclidean'))
    output=np.zeros_like(global_scores);diagnostics=[]
    for region in range(km.n_clusters):
        gold_w=np.exp(-(distances[:n,region]-distances[:n,region].min())/(2*width))
        gold_w=np.maximum(gold_w,.05);gold_w/=gold_w.mean()
        chosen=np.argsort(distances[n:,region])[:min(c.get('masked_neighbors',250),len(u))]
        rows=np.r_[np.arange(n),n+chosen];local=[]
        for j in range(5):
            y=Y[:,j]
            if len(np.unique(y))<2:local.append(np.full(len(Q),y[0]));continue
            pseudo=(gu[chosen,j]>=.5).astype(int)
            confidence=np.clip(np.abs(logit(np.clip(gu[chosen,j],1e-7,1-1e-7))/2),.1,1)
            sw=np.r_[gold_w,.03*confidence]
            model=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(K[np.ix_(rows,rows)],np.r_[y,pseudo],sample_weight=sw)
            pred=expit(2*model.decision_function(Kq[:,rows]))
            support=min(gold_w[y==0].sum(),gold_w[y==1].sum())
            mix=c.get('local_mix',.5)*min(1.,support/5)
            local.append((1-mix)*global_scores[:,j]+mix*pred)
        output+=routing[:,region,None]*np.array(local).T
        diagnostics.append(dict(region=region,masked_neighbors=len(chosen),gold_weight_range=[float(gold_w.min()),float(gold_w.max())]))
    LAST_DIAGNOSTICS=dict(regions=diagnostics)
    return (output[:len(B)],output[len(B):],u) if return_unlabeled else output

def apply_policy(p,baseline,rule,scope):
    if scope!='rare':return phase24.apply_policy(p,baseline,rule,scope)
    original=e.decisions(baseline,'t0.5');out=original.copy()
    out[:,:3]=e.decisions(p,rule)[:,:3]
    empty=out.sum(1)==0;out[empty]=original[empty]
    return out.astype(float)

def predict25(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    pos=np.asarray(train_pos)
    methods={'gold_svm25':gold_svm,'label_tuned_ssl':tuned,'reliable_ssl':reliability,'local_ssl':local_experts}
    if c['model'] in methods:return methods[c['model']](pos,query_frame,c,excluded_groups,return_unlabeled)
    if c['model']=='policy25':
        pred=cached_predict(pos,query_frame,c['learner'],excluded_groups,return_unlabeled)
        base=cached_predict(pos,query_frame,c['baseline'],excluded_groups,return_unlabeled)
        if return_unlabeled:
            assert np.array_equal(pred[2],base[2])
            return apply_policy(pred[0],base[0],c['decision_rule'],c['scope']),apply_policy(pred[1],base[1],c['decision_rule'],c['scope']),pred[2]
        return apply_policy(pred,base,c['decision_rule'],c['scope'])
    return legacy_predict(pos,query_frame,c,excluded_groups,return_unlabeled)

e.predict=predict25
