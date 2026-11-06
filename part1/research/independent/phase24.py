"""Two-view co-training and reason-specific metric learning with classical SVMs.
All model fitting uses supplied training rows only. Queries are transformed only.
"""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import phase23,experiment as e

legacy_predict=e.predict
METRIC_CACHE={}
VIEW_CACHE={}
BASE_CACHE={}
LAST_DIAGNOSTICS=[]

def indices(pos,excluded_groups):
    u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
    return np.r_[e.LAB[pos],u],u

def query_key(frame):
    return e.pd.util.hash_pandas_object(frame.drop(columns=e.LABELS,errors='ignore'),index=False).to_numpy().tobytes()

def metadata_view(idx,query,blocks):
    # Cache only label-free features, with exact fit rows and query content keys.
    key=(tuple(idx),tuple(blocks),query_key(query))
    if key not in VIEW_CACHE:
        aa=[];bb=[]
        for block in blocks:
            a,b=e.block_features(e.DATA.iloc[idx],query,block)
            aa.append(sparse.csr_matrix(a));bb.append(sparse.csr_matrix(b))
        A=normalize(sparse.hstack(aa,format='csr'))*np.sqrt(150)
        B=normalize(sparse.hstack(bb,format='csr'))*np.sqrt(150)
        if len(VIEW_CACHE)>=16:VIEW_CACHE.clear()
        VIEW_CACHE[key]=(A,B)
    return VIEW_CACHE[key]

def co_training(pos,query,c,excluded_groups=None,return_unlabeled=False):
    global LAST_DIAGNOSTICS
    idx,u=indices(pos,excluded_groups);n=len(pos);Y=e.Y[pos]
    A,B=e.features(idx,query,dict(scale='standard'))
    M,N=metadata_view(idx,query,c['views'])
    Q=np.r_[B,A[n:]] if return_unlabeled else B
    MQ=sparse.vstack([N,M[n:]],format='csr') if return_unlabeled else N
    K=np.exp(-.001*pairwise_distances(A,metric='sqeuclidean'))
    Kq=np.exp(-.001*pairwise_distances(Q,A,metric='sqeuclidean'))
    groups=KMeans(n_clusters=10,n_init=10,random_state=42).fit_predict(A)
    cluster=np.full((len(u),5),.5)
    for group in np.unique(groups):
        gold=groups[:n]==group
        if gold.sum()>=2:cluster[groups[n:]==group]=Y[gold].mean(0)
    outputs=[];diagnostics=[]
    for j in range(5):
        y=Y[:,j]
        if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
        used=[{},{}]
        def fit(view):
            extra=np.array(sorted(used[view]),int)
            rows=np.r_[np.arange(n),n+extra]
            yy=np.r_[y,[used[view][int(i)] for i in extra]].astype(int)
            weight=np.r_[np.ones(n),np.full(len(extra),c.get('pseudo_weight',.1))]
            if len(extra):
                cp=cluster[extra,j];disagree=(cp>=.5)!=yy[n:]
                weight[n:][disagree&(np.maximum(cp,1-cp)>=.8)]*=.1
            if view==0:
                model=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced')
                model.fit(K[np.ix_(rows,rows)],yy,sample_weight=weight)
            else:
                model=LogisticRegression(C=c.get('meta_C',.3),class_weight='balanced',max_iter=1500)
                model.fit(M[rows],yy,sample_weight=weight)
            return model,rows
        models=[fit(0),fit(1)]
        for round_number in range(0 if c.get('no_exchange') else c.get('rounds',8)):
            emb=expit(2*models[0][0].decision_function(K[n:][:,models[0][1]]))
            meta=models[1][0].predict_proba(M[n:])[:,1]
            predictions=[emb,meta];planned=[]
            for student in [0,1]:
                peer=1-student;p=predictions[peer]
                already=np.array(sorted(used[student]),int)
                eligible=np.ones(len(u),bool);eligible[already]=False
                if c.get('agreement',False):
                    eligible&=(predictions[0]>=.5)==(predictions[1]>=.5)
                for label in [0,1]:
                    confidence=p if label else 1-p
                    pool=np.flatnonzero(eligible&(confidence>=c.get('confidence',.8))&((p>=.5)==label))
                    # A bounded quota per class prevents majority pseudo-label avalanches.
                    quota=min(c.get('quota',3),max(1,int(np.sum(y==label)/3)))
                    chosen=pool[np.argsort(-confidence[pool],kind='stable')[:quota]]
                    planned.extend((student,int(i),label) for i in chosen)
            if not planned:break
            for student,i,label in planned:used[student][i]=label
            models=[fit(0),fit(1)]
            diagnostics.append(dict(label=e.LABELS[j],round=round_number,
                embedding_received=len(used[0]),metadata_received=len(used[1]),
                metadata_to_embedding_positive=sum(used[0].values())))
        pe=expit(2*models[0][0].decision_function(Kq[:,models[0][1]]))
        pm=models[1][0].predict_proba(MQ)[:,1]
        weight=c.get('metadata_vote',0.)
        outputs.append((1-weight)*pe+weight*pm)
    LAST_DIAGNOSTICS=diagnostics;p=np.array(outputs).T
    if return_unlabeled:return p[:len(B)],p[len(B):],u
    return p

def learned_projection(A,pos,j,c,idx):
    n=len(pos);y=e.Y[pos,j]
    params={k:c.get(k) for k in ['metric','metric_pca','components','subtypes']}
    key=(tuple(idx),tuple(pos),y.tobytes(),j,e.config_id(params))
    if key not in METRIC_CACHE:
        pc=PCA(n_components=min(c.get('metric_pca',30),A.shape[1],len(A)-1),svd_solver='full')
        Z=pc.fit_transform(A);gold=Z[:n]
        if c['metric']=='nca':
            model=NeighborhoodComponentsAnalysis(n_components=c.get('components',5),init='pca',
                max_iter=100,tol=1e-5,random_state=240117)
            model.fit(gold,y);projection=model.components_.T
            diag=dict(metric='nca',iterations=int(model.n_iter_))
        else:
            # Shrink within-class covariance; one or several positive subtype contrasts.
            residual=gold.copy()
            for v in [0,1]:residual[y==v]-=gold[y==v].mean(0)
            covariance=LedoitWolf().fit(residual).covariance_
            k=min(c.get('subtypes',1),max(1,int((y==1).sum()/3)))
            positive=gold[y==1];negative=gold[y==0].mean(0)
            centers=positive.mean(0)[None,:] if k==1 else KMeans(n_clusters=k,n_init=10,random_state=240117).fit(positive).cluster_centers_
            projection=np.linalg.solve(covariance+1e-5*np.eye(len(covariance)),(centers-negative).T)
            projection/=np.maximum(np.linalg.norm(projection,axis=0),1e-12)
            diag=dict(metric='fisher',subtypes=k)
        METRIC_CACHE[key]=(pc,projection,diag)
    return METRIC_CACHE[key]

def metric_ssl(pos,query,c,excluded_groups=None,return_unlabeled=False):
    global LAST_DIAGNOSTICS
    idx,u=indices(pos,excluded_groups);n=len(pos);Y=e.Y[pos]
    A,B=e.features(idx,query,dict(scale='standard'));Q=np.r_[B,A[n:]] if return_unlabeled else B
    D=pairwise_distances(A,metric='sqeuclidean');Dq=pairwise_distances(Q,A,metric='sqeuclidean')
    # KMeans remains the unsupervised step of this augmented-label pipeline.
    groups=KMeans(n_clusters=10,n_init=10,random_state=42).fit_predict(A)
    cluster=np.full((len(u),5),.5)
    for group in np.unique(groups):
        gold=groups[:n]==group
        if gold.sum()>=2:cluster[groups[n:]==group]=Y[gold].mean(0)
    outputs=[];diagnostics=[]
    for j in range(5):
        y=Y[:,j]
        if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
        blend=c.get('metric_weight',.3)
        if blend:
            pc,projection,diag=learned_projection(A,pos,j,c,idx)
            Z=pc.transform(A)@projection;Zq=pc.transform(Q)@projection
            learned=pairwise_distances(Z,metric='sqeuclidean')
            dq=pairwise_distances(Zq,Z,metric='sqeuclidean')
            # Scale learned geometry using fitting rows only, never query statistics.
            scale=float(np.mean(D)/max(np.mean(learned),1e-12))
            DK=(1-blend)*D+blend*scale*learned
            DQ=(1-blend)*Dq+blend*scale*dq
            diagnostics.append(dict(label=e.LABELS[j],scale=scale,**diag))
        else:DK=D;DQ=Dq
        K=np.exp(-c.get('gamma',.001)*DK);Kq=np.exp(-c.get('gamma',.001)*DQ)
        model=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(K[:n,:n],y)
        for fraction in [.1,1/3,1]:
            for _ in range(3):
                pseudo=(model.decision_function(K[n:,:model.n_features_in_])>=0).astype(int)
                sw=np.full(len(u),c.get('pseudo_weight',.03)*fraction)
                sw[((cluster[:,j]>=.5)!=pseudo)&(np.maximum(cluster[:,j],1-cluster[:,j])>=.8)]*=.1
                model=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(K,np.r_[y,pseudo],sample_weight=np.r_[np.ones(n),sw])
        outputs.append(expit(2*model.decision_function(Kq)))
    LAST_DIAGNOSTICS=diagnostics;p=np.array(outputs).T
    if return_unlabeled:return p[:len(B)],p[len(B):],u
    return p

def apply_policy(p,baseline,rule,scope):
    h=e.decisions(p,rule)
    if scope=='all':return h.astype(float)
    original=e.decisions(baseline,'t0.5')
    out=original.copy();cols=[4] if scope=='other' else [3,4]
    out[:,cols]=h[:,cols]
    empty=out.sum(1)==0;out[empty]=original[empty]
    return out.astype(float)

def predict24(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    pos=np.asarray(train_pos)
    if c['model']=='co_training':return co_training(pos,query_frame,c,excluded_groups,return_unlabeled)
    if c['model']=='metric_ssl':return metric_ssl(pos,query_frame,c,excluded_groups,return_unlabeled)
    if c['model']=='policy24':
        p=predict24(pos,query_frame,c['learner'],excluded_groups,return_unlabeled)
        key=(tuple(pos),e.Y[pos].tobytes(),query_key(query_frame),tuple(sorted([] if excluded_groups is None else excluded_groups)),return_unlabeled,e.config_id(c['baseline']))
        if key not in BASE_CACHE:BASE_CACHE[key]=legacy_predict(pos,query_frame,c['baseline'],excluded_groups,return_unlabeled)
        b=BASE_CACHE[key]
        if return_unlabeled:
            assert np.array_equal(p[2],b[2])
            return apply_policy(p[0],b[0],c['decision_rule'],c['scope']),apply_policy(p[1],b[1],c['decision_rule'],c['scope']),p[2]
        return apply_policy(p,b,c['decision_rule'],c['scope'])
    return legacy_predict(pos,query_frame,c,excluded_groups,return_unlabeled)

e.predict=predict24
