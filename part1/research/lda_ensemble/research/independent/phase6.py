"""Efficient low-density SVM refinement; parameters selected without validation labels."""
import phase5,experiment as e
import numpy as np,json
from pathlib import Path
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.cluster import KMeans
legacy_predict=e.predict

def predict6(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    if c['model']!='lowdensity_v2':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
    pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
    A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
    K=rbf_kernel(A,gamma=c.get('gamma',.001));Kq=rbf_kernel(Q,A,gamma=c.get('gamma',.001));outputs=[]
    cluster=None
    if c.get('cluster_gate'):
        # Clusters fit without labels; gold label agreements control unreliable pseudo-label weight.
        z=KMeans(n_clusters=c.get('clusters',20),n_init=10,random_state=42).fit_predict(A)
        cluster=np.full((len(u),5),.5)
        for group in np.unique(z):
            gold=(z[:n]==group)
            if gold.sum()>=2:cluster[z[n:]==group]=Y[gold].mean(0)
    for j in range(5):
        y=Y[:,j]
        if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
        m=SVC(C=c.get('C',1),kernel='precomputed',class_weight=c.get('balanced','balanced'),tol=c.get('tol',.001)).fit(K[:n,:n],y)
        previous=None
        for w in [c.get('pseudo_weight',.03)/10,c.get('pseudo_weight',.03)/3,c.get('pseudo_weight',.03)]:
            for iteration in range(c.get('iterations',3)):
                d=m.decision_function(K[n:,:m.n_features_in_]);labels=(d>=0).astype(int)
                if c.get('early_stop',False) and previous is not None and np.array_equal(labels,previous):break
                previous=labels.copy();puw=np.full(len(u),w)
                if c.get('confidence_weight',False):puw*=np.clip(np.abs(d),.1,1)
                if cluster is not None:
                    confident=np.maximum(cluster[:,j],1-cluster[:,j])>=.8
                    disagree=(cluster[:,j]>=.5)!=labels
                    puw[confident&disagree]*=.1
                balance=c.get('balanced','balanced')
                if c.get('balance_power') is not None:
                    counts=np.bincount(np.r_[y,labels],minlength=2);power=c['balance_power']
                    balance={j:(len(A)/(2*counts[j]))**power for j in [0,1]}
                m=SVC(C=c.get('C',1),kernel='precomputed',class_weight=balance,tol=c.get('tol',.001)).fit(K,np.r_[y,labels],sample_weight=np.r_[np.ones(n),puw])
        outputs.append(expit(c.get('temperature',2)*m.decision_function(Kq[:,:m.n_features_in_])))
    p=np.array(outputs).T
    if return_unlabeled:return p[:len(B)],p[len(B):],u
    return p

e.predict=predict6

def configs():
    cs=[]
    for C in [.3,.6,1,2,3]:
        for gamma in [.0003,.0006,.001,.002]:
            for w in [.01,.03,.08]:cs.append(dict(model='lowdensity_v2',C=C,gamma=gamma,pseudo_weight=w))
    for pc in [50,75,100,125]:
        for C in [.6,1,2]:cs.append(dict(model='lowdensity_v2',pca=pc,C=C,gamma=.001,pseudo_weight=.03))
    for power in [.25,.5,.75,1.25]:
        for C in [.6,1,2]:cs.append(dict(model='lowdensity_v2',balance_power=power,C=C,gamma=.001,pseudo_weight=.03))
    for C in [.6,1,2]:
        for count in [10,20,40]:cs.append(dict(model='lowdensity_v2',cluster_gate=True,clusters=count,C=C,gamma=.001,pseudo_weight=.03))
    for C in [.6,1,2]:cs.append(dict(model='lowdensity_v2',confidence_weight=True,C=C,gamma=.001,pseudo_weight=.03))
    return cs

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--configs');parser.add_argument('--tag',default='phase6');parser.add_argument('--seed',type=int,default=20261104);args=parser.parse_args()
    cs=json.loads(Path(args.configs).read_text()) if args.configs else configs();e.save_json('experiments/'+args.tag+'-configs.json',cs)
    for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,seed=args.seed,tag=args.tag)
