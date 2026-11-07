"""Classical random-subspace SSL ensembles and constrained pseudo-label proportions."""
import phase10,experiment as e
import numpy as np,json
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
legacy_predict=e.predict

def one(A,Q,Y,c,seed):
 rng=np.random.default_rng(seed);n=len(Y);U=np.arange(n,len(A));features=np.arange(A.shape[1])
 if c.get('feature_fraction',1)<1:features=np.sort(rng.choice(features,size=max(5,int(len(features)*c['feature_fraction'])),replace=False))
 if c.get('unlabeled_fraction',1)<1:U=np.sort(rng.choice(U,size=max(10,int(len(U)*c['unlabeled_fraction'])),replace=False))
 ix=np.r_[np.arange(n),U];A=A[ix][:,features];Q=Q[:,features];gamma=c.get('gamma',.001)*150/len(features)
 K=rbf_kernel(A,gamma=gamma);Kq=rbf_kernel(Q,A,gamma=gamma);outputs=[]
 for j in range(5):
  y=Y[:,j]
  if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
  m=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(K[:n,:n],y)
  initial=m.decision_function(K[n:,:n]);initial_prior=np.mean(initial>=0)
  prior=(1-c.get('prior_mix',0))*initial_prior+c.get('prior_mix',0)*y.mean()
  for fraction in [.1,1/3,1]:
   for _ in range(c.get('iterations',3)):
    d=m.decision_function(K[n:,:m.n_features_in_]);labels=(d>=0).astype(int);w=np.full(len(d),c.get('pseudo_weight',.03)*fraction)
    if c.get('constrain_prior'):
     labels[:]=0;k=max(1,min(len(d)-1,round(prior*len(d))));labels[np.argsort(d)[-k:]]=1
    if c.get('margin_min'):w[np.abs(d)<c['margin_min']]=0
    m=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(K,np.r_[y,labels],sample_weight=np.r_[np.ones(n),w])
  outputs.append(expit(2*m.decision_function(Kq)))
 return np.array(outputs).T

def predict11(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
 if c['model']!='subspace_ssl':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
 pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
 A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
 p=np.mean([one(A,Q,Y,c,1000+i) for i in range(c.get('members',1))],axis=0)
 if return_unlabeled:return p[:len(B)],p[len(B):],u
 return p

e.predict=predict11

def configs():
 cs=[]
 for ff in [.5,.75,.9,1.]:
  for uf in [.5,.8,1.]:
   if ff==1. and uf==1.:continue
   cs.append(dict(model='subspace_ssl',C=1,gamma=.001,feature_fraction=ff,unlabeled_fraction=uf,members=9))
 for mix in [0,.25,.5,.75,1.]:
  for w in [.01,.03,.1]:cs.append(dict(model='subspace_ssl',C=1,gamma=.001,constrain_prior=True,prior_mix=mix,pseudo_weight=w))
 for margin in [.1,.3,.6,.9,1.2]:
  for w in [.03,.1]:cs.append(dict(model='subspace_ssl',C=1,gamma=.001,margin_min=margin,pseudo_weight=w))
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase11-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase11')
