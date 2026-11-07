"""Label-shift correction, finite-kernel LapSVMs, and latent Other subclasses.

Queries never participate in fitting, calibration, graph construction or prior
estimation. Every learned object belongs to the current training fold.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import expit, logit
from scipy.sparse import csr_matrix, diags
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
import phase25 as prior, experiment as e

legacy_predict=e.predict
INC=json.loads((Path(__file__).resolve().parent/'experiments/phase26-baseline.json').read_text())['candidate']['config']
PRED_CACHE={}; INNER_CACHE={}; GEOMETRY_CACHE={}; LAST_DIAGNOSTICS={}

def clip(p):return np.clip(p,1e-6,1-1e-6)

def cached(pos,q,c,excluded=None,return_unlabeled=False):
    key=(tuple(pos),e.Y[pos].tobytes(),prior.phase24.query_key(q),e.config_id(c),
         tuple(sorted([] if excluded is None else excluded)),return_unlabeled)
    if key not in PRED_CACHE:PRED_CACHE[key]=predict26(pos,q,c,excluded,return_unlabeled)
    return PRED_CACHE[key]

def soft_incumbent(pos,q,excluded=None,return_unlabeled=False):
    b=cached(pos,q,INC['base'],excluded,return_unlabeled)
    t=cached(pos,q,INC['tri'],excluded,return_unlabeled)
    def combine(bp,tp):
        bp=expit(logit(clip(bp))-logit(float(INC['base_rule'][1:])))
        return np.where(np.array(INC['use_tri'],bool)[None,:],tp,bp)
    if return_unlabeled:
        assert np.array_equal(b[2],t[2])
        return combine(b[0],t[0]),combine(b[1],t[1]),b[2]
    return combine(b,t)

def inner(pos,excluded):
    ex=[] if excluded is None else list(excluded)
    key=(tuple(pos),e.Y[pos].tobytes(),tuple(sorted(ex)))
    if key not in INNER_CACHE:
        p=np.zeros((len(pos),5))
        for a,b in e.make_splits(e.Y[pos],groups=e.GROUP[e.LAB[pos]],seed=260117,n_splits=5):
            p[b]=soft_incumbent(pos[a],e.DATA.iloc[e.LAB[pos[b]]],np.r_[ex,e.GROUP[e.LAB[pos[b]]]])
        INNER_CACHE[key]=p
    return INNER_CACHE[key]

def estimate_prevalence(po,y,pu,method):
    source=float(y.mean());source=float(np.clip(source,.01,.99))
    if method=='bbse':
        neg=float(po[y==0].mean());slope=float(po[y==1].mean()-neg)
        # Ridge toward source prevalence; weakly informative confusion estimates
        # cannot produce an arbitrarily large adjustment.
        target=(slope*(float(pu.mean())-neg)+.01*source)/(slope*slope+.01)
        diag=dict(conditional_score_separation=slope)
    else:
        target=source
        for iteration in range(200):
            r=expit(logit(clip(pu))+logit(target)-logit(source))
            new=float(np.clip((r.sum()+20*source)/(len(r)+20),.01,.99))
            if abs(new-target)<1e-8:target=new;break
            target=new
        diag=dict(em_iterations=iteration+1)
    return float(np.clip(target,.01,.99)),diag

def label_shift(pos,q,c,excluded=None,return_unlabeled=False):
    global LAST_DIAGNOSTICS
    po=inner(pos,excluded);pq,pu,u=soft_incumbent(pos,q,excluded,True)
    combined=np.r_[pq,pu] if return_unlabeled else pq
    y=e.Y[pos];out=[];diagnostics=[]
    for j in range(5):
        if len(np.unique(y[:,j]))<2:out.append(np.full(len(combined),y[0,j]));continue
        # Calibration uses OOF teacher scores, never predictions from fitting the
        # same row's label. Source confusion summaries use these training rows.
        cal=LogisticRegression(C=c.get('calibration_C',1),max_iter=1000)
        cal.fit(logit(clip(po[:,j:j+1])),y[:,j])
        source_scores=cal.predict_proba(logit(clip(po[:,j:j+1])))[:,1]
        masked_scores=cal.predict_proba(logit(clip(pu[:,j:j+1])))[:,1]
        target,diag=estimate_prevalence(source_scores,y[:,j],masked_scores,c['method'])
        source=float(np.clip(y[:,j].mean(),.01,.99))
        delta=c.get('strength',.5)*np.clip(logit(target)-logit(source),-1.5,1.5)
        raw=combined[:,j]
        if c.get('output','calibrated')=='calibrated':raw=cal.predict_proba(logit(clip(combined[:,j:j+1])))[:,1]
        out.append(expit(logit(clip(raw))+delta))
        diagnostics.append(dict(label=e.LABELS[j],source_prevalence=source,
            estimated_masked_prevalence=target,applied_log_odds_shift=float(delta),**diag))
    LAST_DIAGNOSTICS=dict(labels=diagnostics,assumption='Stable feature distribution conditional on each binary reason; not guaranteed for these data.')
    p=np.array(out).T
    return (p[:len(q)],p[len(q):],u) if return_unlabeled else p

def geometry(pos,q,c,excluded):
    idx,u=prior.phase24.indices(pos,excluded)
    key=(tuple(idx),c.get('gamma',.001),c.get('neighbors',15),c.get('components',192))
    if key not in GEOMETRY_CACHE:
        A,_=e.features(idx,e.DATA.iloc[idx[:1]],dict(scale='standard'))
        # Preserve the exact fold-local scaling for arbitrary future queries.
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler().fit(e.DATA.iloc[idx][e.EMB].to_numpy())
        A=sc.transform(e.DATA.iloc[idx][e.EMB].to_numpy())
        ny=Nystroem(kernel='rbf',gamma=c.get('gamma',.001),n_components=min(c.get('components',192),len(A)),random_state=260117)
        F=ny.fit_transform(A)
        D=pairwise_distances(A,metric='sqeuclidean');np.fill_diagonal(D,np.inf)
        k=min(c.get('neighbors',15),len(A)-1);nn=np.argsort(D,axis=1)[:,:k]
        bandwidth=np.maximum(np.sqrt(np.take_along_axis(D,nn,axis=1)[:,-1]),1e-6)
        distances=np.take_along_axis(D,nn,axis=1)
        weights=np.exp(-distances/np.maximum(bandwidth[:,None]*bandwidth[nn],1e-12))
        W=csr_matrix((weights.ravel(),(np.repeat(np.arange(len(A)),k),nn.ravel())),shape=(len(A),len(A)))
        W=W.maximum(W.T);L=diags(np.asarray(W.sum(1)).ravel())-W
        G=F.T@(L@F)/len(A);G=(G+G.T)/2
        scale=max(float(np.trace(G)/len(G)),1e-12);G/=scale
        GEOMETRY_CACHE[key]=(sc,ny,F,G,dict(nodes=len(A),undirected_edges=int(W.nnz//2),graph_penalty_scale=scale))
    sc,ny,F,G,diag=GEOMETRY_CACHE[key]
    Q=ny.transform(sc.transform(q[e.EMB].to_numpy()))
    return F,Q,G,u,diag

def lap_svm(pos,q,c,excluded=None,return_unlabeled=False):
    global LAST_DIAGNOSTICS
    F,Q,G,u,diag=geometry(pos,q,c,excluded);n=len(pos)
    # Objective: .5*w'(I+lambda*G)w + C*sum_i weighted_hinge(y_i,f_i).
    # Cholesky whitening converts it to an ordinary finite-kernel SVM.
    R=cholesky(np.eye(len(G))+c.get('graph_weight',1)*G,lower=True)
    Z=solve_triangular(R,F.T,lower=True).T
    query=solve_triangular(R,Q.T,lower=True).T
    if return_unlabeled:query=np.r_[query,Z[n:]]
    K=Z[:n]@Z[:n].T;Kq=query@Z[:n].T;out=[]
    for j in range(5):
        y=e.Y[pos,j]
        if len(np.unique(y))<2:out.append(np.full(len(query),y[0]));continue
        model=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced')
        model.fit(K,y);out.append(expit(2*model.decision_function(Kq)))
    LAST_DIAGNOSTICS=dict(**diag,approximation='192-dimensional Nystroem RBF map',graph_weight=c.get('graph_weight',1))
    p=np.array(out).T
    return (p[:len(q)],p[len(q):],u) if return_unlabeled else p

def other_subclasses(pos,q,c,excluded=None,return_unlabeled=False):
    global LAST_DIAGNOSTICS
    idx,u=prior.phase24.indices(pos,excluded);n=len(pos);y=e.Y[pos,4]
    base,pu,iu=soft_incumbent(pos,q,excluded,True);assert np.array_equal(u,iu)
    scores=np.r_[base,pu].copy() if return_unlabeled else base.copy()
    if len(np.unique(y))<2:scores[:,4]=y[0];return (scores[:len(q)],scores[len(q):],u) if return_unlabeled else scores
    A,B=e.features(idx,q,dict(scale='standard'));query=np.r_[B,A[n:]] if return_unlabeled else B
    pc=PCA(n_components=min(25,A.shape[1],len(A)-1),svd_solver='full').fit(A)
    gold=pc.transform(A[:n]);masked=pc.transform(A[n:]);positive=gold[y==1]
    k=min(c.get('subgroups',2),max(1,len(positive)//4))
    while True:
        km=KMeans(n_clusters=k,n_init=20,random_state=260117).fit(positive)
        if k==1 or np.bincount(km.labels_,minlength=k).min()>=3:break
        k-=1
    target=np.zeros(n,int);target[y==1]=km.labels_+1
    # Negative examples stay a single class; positive subclasses map back to Other.
    selected=np.flatnonzero((pu[:,4]>=.8)|(pu[:,4]<=.2)) if c.get('pseudo_weight',.03)>0 else np.array([],int)
    yp=np.zeros(len(selected),int);positive_u=pu[selected,4]>=.8
    yp[positive_u]=km.predict(masked[selected[positive_u]])+1
    rows=np.r_[np.arange(n),n+selected];yy=np.r_[target,yp]
    # Balance aggregate Other versus non-Other, not each small subtype equally.
    balance=np.where(yy>0,n/(2*np.sum(y==1)),n/(2*np.sum(y==0)))
    sw=balance*np.r_[np.ones(n),np.full(len(selected),c.get('pseudo_weight',.03))]
    if c.get('representation','kernel')=='kernel':
        ny=Nystroem(kernel='rbf',gamma=.001,n_components=min(192,len(A)),random_state=260117)
        A=ny.fit_transform(A);query=ny.transform(query)
    model=LogisticRegression(C=c.get('C',1),max_iter=2000).fit(A[rows],yy,sample_weight=sw)
    p=model.predict_proba(query)[:,model.classes_>0].sum(1)
    mix=c.get('mix',1);scores[:,4]=(1-mix)*scores[:,4]+mix*p
    LAST_DIAGNOSTICS=dict(requested_subgroups=c.get('subgroups',2),fitted_subgroups=k,
        gold_subgroup_sizes=np.bincount(km.labels_,minlength=k).tolist(),pseudo_rows=len(selected),
        pseudo_other_rows=int(positive_u.sum()),iterations=model.n_iter_.tolist())
    return (scores[:len(q)],scores[len(q):],u) if return_unlabeled else scores

def apply_policy(p,base,rule,scope):
    if scope!='efficacy_other':return prior.apply_policy(p,base,rule,scope)
    original=e.decisions(base,'t0.5');h=original.copy();h[:,[2,4]]=e.decisions(p,rule)[:,[2,4]]
    empty=h.sum(1)==0;h[empty]=original[empty]
    return h.astype(float)

def predict26(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    pos=np.asarray(train_pos)
    methods={'label_shift26':label_shift,'lap_svm26':lap_svm,'other_subclasses26':other_subclasses}
    if c['model'] in methods:return methods[c['model']](pos,query_frame,c,excluded_groups,return_unlabeled)
    if c['model']=='policy26':
        p=cached(pos,query_frame,c['learner'],excluded_groups,return_unlabeled)
        b=cached(pos,query_frame,c['baseline'],excluded_groups,return_unlabeled)
        if return_unlabeled:
            assert np.array_equal(p[2],b[2])
            return apply_policy(p[0],b[0],c['decision_rule'],c['scope']),apply_policy(p[1],b[1],c['decision_rule'],c['scope']),p[2]
        return apply_policy(p,b,c['decision_rule'],c['scope'])
    return legacy_predict(pos,query_frame,c,excluded_groups,return_unlabeled)

e.predict=predict26
