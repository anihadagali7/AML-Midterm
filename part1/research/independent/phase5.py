"""Sparse semantic reconstruction and Bayesian multilabel neighbors."""
import phase4
import experiment as e
import numpy as np,json
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.decomposition import sparse_encode
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import nnls
legacy_predict=e.predict

def predict5(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    if c['model'] not in ['sparse_examples','sparse_prototypes','mlknn']:
        return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
    pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
    A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
    kind=c['model'];X=A[:n]
    if kind=='mlknn':
        k=min(c.get('k',10),n-1);s=c.get('smoothing',1.)
        neigh=NearestNeighbors(n_neighbors=k+1).fit(X)
        ix=neigh.kneighbors(X)[1]
        ix=np.array([row[row!=i][:k] for i,row in enumerate(ix)])
        count=Y[ix].sum(1)
        cp=np.zeros((5,k+1));cn=np.zeros_like(cp)
        prior=(Y.sum(0)+s)/(n+2*s)
        for j in range(5):
            cp[j]=(np.bincount(count[Y[:,j]==1,j],minlength=k+1)+s)/(Y[:,j].sum()+s*(k+1))
            cn[j]=(np.bincount(count[Y[:,j]==0,j],minlength=k+1)+s)/((n-Y[:,j].sum())+s*(k+1))
        qi=neigh.kneighbors(Q,n_neighbors=k)[1];z=Y[qi].sum(1)
        pp=np.column_stack([prior[j]*cp[j,z[:,j]] for j in range(5)])
        pn=np.column_stack([(1-prior[j])*cn[j,z[:,j]] for j in range(5)])
        p=pp/np.maximum(pp+pn,1e-12)
    else:
        if kind=='sparse_prototypes':
            D=[];targets=[]
            for j in range(5):
                mask=Y[:,j]>0
                if c.get('singletons',True) and np.sum(mask&(Y.sum(1)==1))>=2:mask&=Y.sum(1)==1
                if not mask.any():continue
                count=min(c.get('clusters',2),mask.sum())
                centers=KMeans(n_clusters=count,n_init=10,random_state=42).fit(X[mask]).cluster_centers_
                D.extend(centers);targets.extend(np.eye(5)[[j]*count])
            D=np.array(D);targets=np.array(targets)
        else:D=X;targets=Y
        if c.get('normalize_dictionary',True):D=normalize(D);Q=normalize(Q)
        alpha=c.get('alpha',.05)
        if alpha>0:coeff=sparse_encode(Q,D,algorithm='lasso_lars',alpha=alpha,positive=True,max_iter=1000,n_jobs=1)
        else:coeff=np.array([nnls(D.T,q)[0] for q in Q])
        p=coeff@targets/np.maximum(coeff.sum(1)[:,None],1e-12)
        missing=coeff.sum(1)<1e-10
        if missing.any():p[missing]=Y[NearestNeighbors(n_neighbors=1).fit(X).kneighbors(Q[missing])[1][:,0]]
    p=np.clip(p,0,1)
    if return_unlabeled:return p[:len(B)],p[len(B):],u
    return p

e.predict=predict5

def configs():
    cs=[]
    for pc in [0,25,75]:
        for alpha in [.005,.02,.05,.1,.2]:cs.append(dict(model='sparse_examples',pca=pc,alpha=alpha,scale='global'))
    for pc in [0,25]:
        for k in [1,2,4]:
            for alpha in [0,.05,.1]:cs.append(dict(model='sparse_prototypes',pca=pc,clusters=k,alpha=alpha,scale='global'))
    for pc in [0,25,75]:
        for k in [3,5,10,20]:
            for s in [.1,1]:cs.append(dict(model='mlknn',pca=pc,k=k,smoothing=s))
    return cs

if __name__=='__main__':
    cs=configs();e.save_json('experiments/phase5-configs.json',cs)
    for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase5')
