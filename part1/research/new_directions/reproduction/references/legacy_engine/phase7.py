"""Low-density SVM convergence, annealing, and explicitly transductive inference."""
import phase6,experiment as e
import numpy as np,json
from pathlib import Path
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.cluster import KMeans
legacy_predict=e.predict
LAST_HISTORY=[]

def predict7(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    global LAST_HISTORY
    if c['model']!='lowdensity_v3':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
    pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
    A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos]
    Q=np.r_[B,A[n:]] if return_unlabeled else B
    if c.get('transductive',False):A=np.r_[A,B]
    K=rbf_kernel(A,gamma=c.get('gamma',.001));Kq=rbf_kernel(Q,A,gamma=c.get('gamma',.001));outputs=[];history=[]
    cluster=None
    if c.get('cluster_gate'):
        z=KMeans(n_clusters=c.get('clusters',10),n_init=10,random_state=42).fit_predict(A)
        cluster=np.full((len(A)-n,5),.5)
        for group in np.unique(z):
            gold=z[:n]==group
            if gold.sum()>=2:cluster[z[n:]==group]=Y[gold].mean(0)
    for j in range(5):
        y=Y[:,j]
        if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
        m=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(K[:n,:n],y)
        initial=(m.decision_function(K[n:,:n])>=0).astype(int)
        source=c.get('balance_source','joint')
        counts=np.bincount(y if source=='gold' else np.r_[y,initial],minlength=2)
        fixed={v:(counts.sum()/(2*counts[v]))**c.get('balance_power',1.) for v in [0,1]}
        schedule=([.003,.01,.03,.1,.3,1.] if c.get('gradual',False) else [.1,1/3,1.])
        for fraction in schedule:
            w=c.get('pseudo_weight',.03)*fraction;previous=None
            for iteration in range(c.get('iterations',10)):
                d=m.decision_function(K[n:,:m.n_features_in_]);labels=(d>=0).astype(int)
                changes=len(labels) if previous is None else np.sum(previous!=labels)
                history.append(dict(label=e.LABELS[j],weight=w,iteration=iteration,changed=int(changes),positive=int(labels.sum())))
                if previous is not None and changes==0:break
                previous=labels.copy();puw=np.full(len(labels),w)
                if cluster is not None:
                    disagree=(cluster[:,j]>=.5)!=labels;confident=np.maximum(cluster[:,j],1-cluster[:,j])>=.8
                    puw[confident&disagree]*=c.get('gate_weight',.1)
                if source=='joint':
                    count=np.bincount(np.r_[y,labels],minlength=2);balance={v:(count.sum()/(2*count[v]))**c.get('balance_power',1.) for v in [0,1]}
                else:balance=fixed
                m=SVC(C=c.get('C',1),kernel='precomputed',class_weight=balance).fit(K,np.r_[y,labels],sample_weight=np.r_[np.ones(n),puw])
        outputs.append(expit(2*m.decision_function(Kq)))
    LAST_HISTORY=history;p=np.array(outputs).T
    if return_unlabeled:return p[:len(B)],p[len(B):],u
    return p

e.predict=predict7

def configs():
    cs=[]
    for transductive in [False,True]:
        for source in ['joint','initial','gold']:
            for gradual in [False,True]:cs.append(dict(model='lowdensity_v3',C=1,gamma=.001,pseudo_weight=.03,transductive=transductive,balance_source=source,gradual=gradual,iterations=15))
        for C in [.6,1,2]:
            for weight in [.01,.03,.08]:cs.append(dict(model='lowdensity_v3',C=C,gamma=.001,pseudo_weight=weight,transductive=transductive,gradual=True,iterations=15))
        for gate in [0,.1,.5]:cs.append(dict(model='lowdensity_v3',C=1,gamma=.001,pseudo_weight=.03,transductive=transductive,gradual=True,iterations=15,cluster_gate=True,clusters=10,gate_weight=gate))
    return list({e.config_id(c):c for c in cs}.values())

if __name__=='__main__':
    cs=configs();e.save_json('experiments/phase7-configs.json',cs)
    for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase7')
