"""Classical manifold-SVM and synthetic embedding composition; no neural models."""
import phase3 as p3
import experiment as e
import numpy as np,json
from pathlib import Path
from scipy.special import expit
from scipy.linalg import solve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
legacy_predict=e.predict

def predict4(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    if c['model'] not in ['manifold_svm','lowdensity_svm','composition']:
        return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
    pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
    idx=np.r_[e.LAB[pos],u];A,B=e.features(idx,query_frame,c);Y=e.Y[pos]
    Q=np.r_[B,A[n:]] if return_unlabeled else B
    kind=c['model'];rng=np.random.default_rng(42)
    if kind=='manifold_svm':
        gamma=c.get('gamma',.003);K=rbf_kernel(A,gamma=gamma);Kq=rbf_kernel(Q,A,gamma=gamma)
        d,ix=NearestNeighbors(n_neighbors=11).fit(A).kneighbors(A);d=d[:,1:];ix=ix[:,1:]
        W=np.zeros_like(K);np.put_along_axis(W,ix,np.exp(-d*d/np.median(d[:,-1])**2),axis=1);W=(W+W.T)*.5
        D=np.maximum(W.sum(1),1e-12);L=np.eye(len(A))-W/np.sqrt(D[:,None]*D[None,:])
        E=np.zeros((len(A),n));E[:n]=np.eye(n)
        V=solve(np.eye(len(A))+c.get('mu',1)*(L@K),E,assume_a='gen')
        Ktrain=(K@V)[:n];Ktrain=(Ktrain+Ktrain.T)*.5;Kquery=Kq@V;out=[]
        for j in range(5):
            if len(np.unique(Y[:,j]))<2:out.append(np.full(len(Q),Y[0,j]));continue
            m=SVC(C=c.get('C',10),kernel='precomputed',class_weight=c.get('balanced'),probability=c.get('proba',True),random_state=42).fit(Ktrain,Y[:,j])
            out.append(m.predict_proba(Kquery)[:,1] if c.get('proba',True) else expit(2*m.decision_function(Kquery)))
        p=np.array(out).T
    elif kind=='lowdensity_svm':
        out=[]
        for j in range(5):
            y=Y[:,j]
            if len(np.unique(y))<2:out.append(np.full(len(Q),y[0]));continue
            m=SVC(C=c.get('C',3),gamma=c.get('gamma',.003),class_weight=c.get('balanced')).fit(A[:n],y)
            for w in [c.get('pseudo_weight',.1)/10,c.get('pseudo_weight',.1)/3,c.get('pseudo_weight',.1)]:
                for _ in range(3):
                    labels=(m.decision_function(A[n:])>=0).astype(int)
                    sw=np.r_[np.ones(n),np.full(len(u),w)]
                    m=SVC(C=c.get('C',3),gamma=c.get('gamma',.003),class_weight=c.get('balanced')).fit(A,np.r_[y,labels],sample_weight=sw)
            out.append(expit(2*m.decision_function(Q)))
        p=np.array(out).T
    else:
        # Vicinal augmentation: random convex combinations, soft mixup, same-label SMOTE,
        # or semantic multi-reason union. All parents come from this training fold only.
        count=c.get('synthetic',500);left=rng.integers(n,size=count);right=rng.integers(n,size=count)
        if c.get('mix','union')=='same':
            for i in range(count):
                matches=np.flatnonzero((Y==Y[left[i]]).all(1));right[i]=rng.choice(matches)
        alpha=c.get('mix_alpha',1.);lam=rng.beta(alpha,alpha,size=count)[:,None]
        if c.get('mix','union')=='union':lam=np.clip(lam,.25,.75)
        Xnew=lam*A[left]+(1-lam)*A[right]
        if c.get('mix','union')=='soft':ynew=lam*Y[left]+(1-lam)*Y[right]
        elif c.get('mix','union')=='same':ynew=Y[left]
        else:ynew=np.maximum(Y[left],Y[right])
        xx=np.r_[A[:n],Xnew];yy=np.r_[Y,ynew];sw=np.r_[np.ones(n),np.full(count,c.get('synthetic_weight',.1))]
        # Optional cluster-supported masked training rows ensure real semi-supervised augmentation.
        if c.get('teacher'):
            t=c['teacher'];T,_=e.preprocess(e.DATA.iloc[idx][e.EMB],query_frame[e.EMB],t);ys=np.full((len(A),5),-1);ys[:n]=Y
            pt=e.Cluster(t).fit(T,ys).predict(T[n:]);sel=np.min(np.maximum(pt,1-pt),axis=1)>=.95
            xx=np.r_[xx,A[n:][sel]];yy=np.r_[yy,(pt[sel]>=.5).astype(int)];sw=np.r_[sw,np.full(sel.sum(),.05)]
        out=[]
        for j in range(5):
            if len(np.unique(Y[:,j]))<2:out.append(np.full(len(Q),Y[0,j]));continue
            if c.get('base','logistic')=='logistic':m=LogisticRegression(C=c.get('C',.3),max_iter=2000,class_weight=c.get('balanced'))
            else:m=SVC(C=c.get('C',10),gamma=c.get('gamma',.001),probability=True,random_state=42,class_weight=c.get('balanced'))
            if c.get('mix')=='soft':
                # Binary weighted likelihood representation of fractional targets, no neural model.
                xfit=np.r_[xx,xx];yfit=np.r_[np.ones(len(xx)),np.zeros(len(xx))];wfit=np.r_[sw*yy[:,j],sw*(1-yy[:,j])]
                good=wfit>1e-8;m.fit(xfit[good],yfit[good],sample_weight=wfit[good])
            else:m.fit(xx,yy[:,j].astype(int),sample_weight=sw)
            out.append(m.predict_proba(Q)[:,1])
        p=np.array(out).T
    p=np.clip(p,0,1)
    if return_unlabeled:return p[:len(B)],p[len(B):],u
    return p

e.predict=predict4

def configs():
    out=[]
    for gamma in [.001,.003,.01]:
        for C in [1,10]:
            for mu in [.1,1,10]:out.append(dict(model='manifold_svm',gamma=gamma,C=C,mu=mu))
    for gamma in [.001,.003]:
        for C in [1,10]:
            for w in [.03,.1,.3]:out.append(dict(model='lowdensity_svm',gamma=gamma,C=C,pseudo_weight=w,balanced='balanced'))
    for base in ['logistic','svc']:
        for mix in ['union','soft','same']:
            for C in ([.1,.3,1] if base=='logistic' else [1,10]):
                for weight in [.03,.1,.3]:out.append(dict(model='composition',mix=mix,base=base,C=C,gamma=.001,synthetic_weight=weight))
    return out

if __name__=='__main__':
    cs=configs();e.save_json('experiments/phase4-configs.json',cs)
    for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase4')
