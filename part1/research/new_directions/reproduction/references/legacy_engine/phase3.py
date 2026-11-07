"""Classical extensions for phase 3. Keeps the earlier engine and its provenance intact."""
import experiment as e
import numpy as np,json
from pathlib import Path
from scipy.linalg import cho_factor,cho_solve
from scipy.special import expit,softmax
from sklearn.metrics.pairwise import euclidean_distances,rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading
from threadpoolctl import threadpool_limits
legacy_predict=e.predict

def predict3(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    new={'dense_graph','kernel_neighbors','margin_svm','weighted_supervised','spectral_logistic','iterative_ssl'}
    if c['model'] not in new:return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
    pos=np.asarray(train_pos);n=len(pos);actual=e.LAB[pos]
    u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
    idx=np.r_[actual,u]
    A,B=e.features(idx,query_frame,c);y=e.Y[pos]
    Q=np.r_[B,A[n:]] if return_unlabeled else B
    kind=c['model']
    if kind in ['dense_graph','kernel_neighbors']:
        scale=np.sqrt(np.mean(np.sum(A*A,axis=1)));A=A/scale;Q=Q/scale
        gamma=c.get('bandwidth',10.)
        if kind=='kernel_neighbors':
            d=euclidean_distances(Q,A[:n],squared=True)
            if c.get('k',0):
                ix=np.argsort(d,axis=1)[:,:c['k']];mask=np.ones_like(d,bool);np.put_along_axis(mask,ix,False,axis=1);d[mask]=np.inf
            w=np.exp(-gamma*(d-d.min(1)[:,None]));p=w@y/w.sum(1)[:,None]
        else:
            W=rbf_kernel(A,gamma=gamma);np.fill_diagonal(W,0)
            degree=np.maximum(W.sum(1),1e-30);S=W/np.sqrt(degree[:,None]*degree[None,:])
            Z=np.zeros((len(A),6));Z[:n,:5]=y;Z[:n,5]=1
            F=cho_solve(cho_factor(np.eye(len(A))-c.get('alpha',.2)*S,lower=True),Z)
            Wq=rbf_kernel(Q,A,gamma=gamma)
            if c.get('message_mass',False):p=(Wq@F[:,:5])/np.maximum(Wq@F[:,5,None],1e-30)
            else:
                F=np.divide(F[:,:5],F[:,5,None],out=np.tile(y.mean(0),(len(A),1)),where=F[:,5,None]>1e-30)
                if c.get('clamp',False):F[:n]=y
                p=Wq@F/np.maximum(Wq.sum(1)[:,None],1e-30)
    elif kind=='spectral_logistic':
        # Smooth covariance reweighting rather than discarding small-eigenvalue components.
        from sklearn.decomposition import PCA
        pca=PCA(n_components=c.get('components',150),svd_solver='full')
        A=pca.fit_transform(A);Q=pca.transform(Q)
        shrink=c.get('cov_ridge',.1)*pca.explained_variance_.mean()
        scale=(pca.explained_variance_+shrink)**(c.get('power',.5)/2)
        A/=scale;Q/=scale
        p=e.fit_supervised(A[:n],y,Q,{**c,'model':'logistic'})
    elif kind=='margin_svm':
        out=[]
        for j in range(5):
            if len(np.unique(y[:,j]))<2:out.append(np.full(len(Q),y[0,j]));continue
            if c.get('kernel','rbf')=='linear':m=LinearSVC(C=c.get('C',1),class_weight=c.get('balanced'),max_iter=10000)
            else:m=SVC(C=c.get('C',1),gamma=c.get('gamma',.001),kernel=c.get('kernel','rbf'),class_weight=c.get('balanced'),degree=2)
            m.fit(A[:n],y[:,j]);out.append(expit(c.get('temperature',2)*m.decision_function(Q)))
        p=np.array(out).T
    elif kind=='weighted_supervised':
        years=e.DATA.iloc[actual].start_year.to_numpy()
        if c.get('late_weight'):w=np.where(years>2002,c['late_weight'],1.)
        else:w=np.clip(np.exp((years-years.max())/c.get('time_scale',5)),.1,1)
        w=w/w.mean()
        p=e.fit_supervised(A[:n],y,Q,{**c,'model':c.get('base','logistic')},w)
    else:
        # Iterate weighted pseudo-labels; optionally require independent cluster agreement.
        teacher=c.get('teacher')
        teacherp=None
        if teacher:
            T,_=e.preprocess(e.DATA.iloc[idx][e.EMB],query_frame[e.EMB],teacher)
            yy=np.full((len(A),5),-1);yy[:n]=y
            teacherp=e.Cluster(teacher).fit(T,yy).predict(T[n:])
        outs=[]
        for j in range(5):
            gold=y[:,j];sel=np.zeros(len(u),bool);pseudo=np.zeros(len(u),int)
            for iteration in range(c.get('iterations',5)+1):
                xx=np.r_[A[:n],A[n:][sel]];yy=np.r_[gold,pseudo[sel]]
                pw=np.full(sel.sum(),c.get('pseudo_weight',.2))
                if c.get('cap_mass',False):
                    for v in [0,1]:
                        m=pseudo[sel]==v;pw[m]*=min(1.,c.get('mass_fraction',.5)*np.sum(gold==v)/max(pw[m].sum(),1e-12))
                weights=np.r_[np.ones(n),pw]
                m=LogisticRegression(C=c.get('C',.3),max_iter=2000,class_weight=c.get('balanced'))
                if len(np.unique(yy))<2:break
                m.fit(xx,yy,sample_weight=weights)
                if iteration==c.get('iterations',5):break
                pu=m.predict_proba(A[n:])[:,1];new=(np.maximum(pu,1-pu)>=c.get('confidence',.9))
                if teacherp is not None:new&=(pu>=.5)==(teacherp[:,j]>=.5)
                # Monotonic inclusion; labels can update only within already confident assignments.
                new|=sel
                if np.array_equal(new,sel):break
                sel=new;pseudo=(pu>=.5).astype(int)
            outs.append(m.predict_proba(Q)[:,1] if len(np.unique(gold))>1 else np.full(len(Q),gold[0]))
        p=np.array(outs).T
    p=np.clip(p,0,1)
    if return_unlabeled:return p[:len(B)],p[len(B):],u
    return p

e.predict=predict3

def configs():
    cs=[]
    for pc in [0,25,75]:
        for bandwidth in [2,5,10,20,50]:
            for alpha in [.2,.8]:
                cs.append(dict(model='dense_graph',pca=pc,bandwidth=bandwidth,alpha=alpha))
    for bandwidth in [5,20,50]:
        for alpha in [.2,.8,.99]:cs.append(dict(model='dense_graph',bandwidth=bandwidth,alpha=alpha,message_mass=True))
    for pc in [0,25]:
        for bandwidth in [2,5,10,20,50]:cs.append(dict(model='kernel_neighbors',pca=pc,bandwidth=bandwidth))
    for kernel in ['linear','rbf']:
        for C in [.03,.1,.3,1,3,10]:
            for balanced in [None,'balanced']:cs.append(dict(model='margin_svm',kernel=kernel,C=C,balanced=balanced,gamma=.001))
    for base in ['logistic','svc']:
        for weight in [2,4,8]:
            for C in ([.03,.1,.3,1] if base=='logistic' else [1,10]):cs.append(dict(model='weighted_supervised',base=base,late_weight=weight,C=C,gamma=.001))
    for power in [-1,-.5,.5,1]:
        for C in [.03,.1,.3,1]:cs.append(dict(model='spectral_logistic',power=power,C=C,cov_ridge=.1))
    for C in [.1,.3,1]:
        for confidence in [.8,.9,.95]:
            for cap in [True,False]:cs.append(dict(model='iterative_ssl',C=C,confidence=confidence,cap_mass=cap,pseudo_weight=.2))
    teacher=dict(model='cluster',cluster='hdbscan',min_cluster=5,pca=25)
    for C in [.1,.3,1]:
        for conf in [.8,.95]:cs.append(dict(model='iterative_ssl',C=C,confidence=conf,cap_mass=True,pseudo_weight=.2,teacher=teacher))
    return cs

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--configs');parser.add_argument('--tag',default='phase3');parser.add_argument('--seed',type=int,default=20261104);args=parser.parse_args()
    cs=json.loads(Path(args.configs).read_text()) if args.configs else configs()
    e.save_json('experiments/'+args.tag+'-configs.json',cs)
    for i,c in enumerate(cs):
        print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,seed=args.seed,tag=args.tag)
