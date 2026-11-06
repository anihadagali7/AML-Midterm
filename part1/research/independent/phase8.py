"""Training-only hierarchy for mutually exclusive rare reasons plus Recruitment.
Uses no validation labels; every fitted component is scoped to the outer fold.
"""
import phase7,experiment as e
import numpy as np,json
from scipy.special import expit,softmax
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
legacy_predict=e.predict
RARE=[0,1,2,4]

def binary(K,ki,y,ui,Kq,c,weight=None):
    if len(np.unique(y))<2:return np.full(len(K),2*y[0]-1.),np.full(len(Kq),2*y[0]-1.)
    ix=np.r_[ki,ui];n=len(ki)
    m=SVC(C=c.get('C',1),kernel='precomputed',class_weight=c.get('balanced','balanced')).fit(K[np.ix_(ki,ki)],y)
    active=ki
    if c.get('pseudo_weight',.03)>0 and len(ui):
        for fraction in [.1,1/3,1]:
            for _ in range(c.get('iterations',3)):
                pseudo=(m.decision_function(K[np.ix_(ui,active)])>=0).astype(int)
                uw=np.ones(len(ui)) if weight is None else weight
                m=SVC(C=c.get('C',1),kernel='precomputed',class_weight=c.get('balanced','balanced')).fit(K[np.ix_(ix,ix)],np.r_[y,pseudo],sample_weight=np.r_[np.ones(n),uw*c.get('pseudo_weight',.03)*fraction]);active=ix
    return m.decision_function(K[:,active]),m.decision_function(Kq[:,active])

def predict8(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    if c['model'] not in ['hierarchical_svm','multiclass_ssl']:return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
    pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
    A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
    K=rbf_kernel(A,gamma=c.get('gamma',.001));Kq=rbf_kernel(Q,A,gamma=c.get('gamma',.001));ki=np.arange(n);ui=np.arange(n,len(A))
    rec,recq=binary(K,ki,Y[:,3],ui,Kq,c);p=np.zeros((len(Q),5));p[:,3]=expit(2*recq)
    rare=Y[:,RARE].any(axis=1)
    if c['model']=='hierarchical_svm':
        gate,gateq=binary(K,ki,rare.astype(int),ui,Kq,c)
        sel=ui[expit(2*gate[ui])>=c.get('gate_threshold',.5)];ri=ki[rare]
        conditional=[]
        cc={**c,'C':c.get('conditional_C',c.get('C',1)),'pseudo_weight':c.get('conditional_weight',c.get('pseudo_weight',.03))}
        for j in RARE:
            _,q=binary(K,ri,Y[rare,j],sel,Kq,cc);conditional.append(q)
        d=np.array(conditional).T
        if c.get('conditional_mode','softmax')=='softmax':r=softmax(d*c.get('temperature',2),axis=1)
        else:r=expit(2*d)
        p[:,RARE]=r*expit(2*gateq)[:,None]**c.get('gate_power',1.)
    else:
        # Each gold row has at most one rare reason; retain Recruitment independently.
        assert (Y[:,RARE].sum(1)<=1).all()
        y=np.where(rare,np.argmax(Y[:,RARE],axis=1),4)
        m=SVC(C=c.get('C',1),kernel='precomputed',class_weight=c.get('balanced','balanced'),decision_function_shape='ovr').fit(K[:n,:n],y);active=ki
        cluster=None
        if c.get('cluster_gate'):
            z=KMeans(n_clusters=c.get('clusters',10),n_init=10,random_state=42).fit_predict(A);cluster=np.full(len(u),-1)
            for k in np.unique(z):
                known=z[:n]==k
                if known.sum()>=2:
                    count=np.bincount(y[known],minlength=5)
                    if count.max()/count.sum()>=.8:cluster[z[n:]==k]=count.argmax()
        if c.get('pseudo_weight',.03)>0:
            for fraction in [.1,1/3,1]:
                for _ in range(c.get('iterations',3)):
                    pseudo=m.predict(K[n:,active]);uw=np.full(len(u),c.get('pseudo_weight',.03)*fraction)
                    if cluster is not None:uw[(cluster>=0)&(cluster!=pseudo)]*=.1
                    m=SVC(C=c.get('C',1),kernel='precomputed',class_weight=c.get('balanced','balanced'),decision_function_shape='ovr').fit(K,np.r_[y,pseudo],sample_weight=np.r_[np.ones(n),uw]);active=np.arange(len(A))
        d=m.decision_function(Kq[:,active]);r=softmax(d*c.get('temperature',2),axis=1)
        for j,label in enumerate(m.classes_):
            if label<4:p[:,RARE[label]]=r[:,j]
        if c.get('recruitment_mode')=='multiclass':
            p[:,3]=r[:,np.flatnonzero(m.classes_==4)[0]]
    if return_unlabeled:return p[:len(B)],p[len(B):],u
    return p

e.predict=predict8

def configs():
 cs=[]
 for C in [.3,1,3]:
  for gamma in [.0003,.001,.003]:
   for w in [0,.03]:
    cs.append(dict(model='hierarchical_svm',C=C,gamma=gamma,pseudo_weight=w))
    cs.append(dict(model='multiclass_ssl',C=C,gamma=gamma,pseudo_weight=w))
 for mode in ['softmax','sigmoid']:
  for gate in [0,.5,1.]:
   for C in [1,3]:cs.append(dict(model='hierarchical_svm',C=1,conditional_C=C,gamma=.001,pseudo_weight=.03,conditional_mode=mode,gate_power=gate))
 for weight in [.01,.1]:
  for C in [1,3]:cs.append(dict(model='hierarchical_svm',C=C,gamma=.001,pseudo_weight=.03,conditional_weight=weight,gate_threshold=.7))
 for C in [.3,1,3]:
  for w in [.01,.1]:cs.append(dict(model='multiclass_ssl',C=C,gamma=.001,pseudo_weight=w,cluster_gate=True))
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase8-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase8')
