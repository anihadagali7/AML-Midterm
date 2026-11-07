"""Joint multilabel pseudo-label updates with training-derived structural constraints."""
import phase9,experiment as e
import numpy as np,json
from scipy.special import expit
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
legacy_predict=e.predict

def assign(d,Y,c):
 mode=c.get('assignment','nonempty');p=expit(2*d)
 if mode=='nonempty':return e.decisions(p,'t0.5')
 if mode=='expected_f1':return e.decisions(p,'expected_f1')
 if mode=='exclusive_rare':
  out=(d>=0).astype(int);rare=[0,1,2,4];r=d[:,rare].argmax(1);out[:,rare]=0
  for i,j in enumerate(r):out[i,rare[j]]=int(d[i,rare[j]]>=0)
  empty=out.sum(1)==0;out[np.flatnonzero(empty),d[empty].argmax(1)]=1
  return out
 if mode=='patterns':
  patterns,counts=np.unique(Y,axis=0,return_counts=True)
  score=d@patterns.T*2+c.get('prior_weight',0.)*np.log(counts/counts.sum())[None,:]
  return patterns[score.argmax(1)]
 raise ValueError(mode)

def predict10(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
 if c['model']!='structured_ssl':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
 pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
 A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
 K=rbf_kernel(A,gamma=c.get('gamma',.001));Kq=rbf_kernel(Q,A,gamma=c.get('gamma',.001));cluster=None
 if c.get('cluster_gate'):
  z=KMeans(n_clusters=c.get('clusters',10),n_init=10,random_state=42).fit_predict(A);cluster=np.full((len(u),5),.5)
  for k in np.unique(z):
   known=z[:n]==k
   if known.sum()>=2:cluster[z[n:]==k]=Y[known].mean(0)
 def fit(y,w,kernel):
  return [SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(kernel,y[:,j],sample_weight=np.ascontiguousarray(w[:,j])) if len(np.unique(y[:,j]))>1 else float(y[0,j]*2-1) for j in range(5)]
 def decision(models,kernel):return np.column_stack([m.decision_function(kernel) if hasattr(m,'predict') else np.full(len(kernel),m) for m in models])
 models=fit(Y,np.ones(Y.shape),K[:n,:n]);cols=n
 for fraction in [.1,1/3,1]:
  for _ in range(c.get('iterations',3)):
   d=decision(models,K[n:,:cols]);pseudo=assign(d,Y,c);w=np.full((len(u),5),c.get('pseudo_weight',.03)*fraction)
   if cluster is not None:w[((cluster>=.5)!=pseudo)&(np.maximum(cluster,1-cluster)>=.8)]*=.1
   models=fit(np.r_[Y,pseudo],np.r_[np.ones(Y.shape),w],K);cols=len(A)
 p=expit(2*decision(models,Kq))
 if c.get('decode_patterns'):
  patterns=np.unique(Y,axis=0);d=np.log(p.clip(1e-8,1-1e-8)/(1-p).clip(1e-8,1));weights=np.exp(2*(d@patterns.T-np.max(d@patterns.T,axis=1)[:,None]));weights/=weights.sum(1)[:,None];p=weights@patterns
 if return_unlabeled:return p[:len(B)],p[len(B):],u
 return p

e.predict=predict10

def configs():
 cs=[]
 for mode in ['nonempty','exclusive_rare','patterns','expected_f1']:
  for C in [.3,1,3]:
   for w in [.01,.03,.1]:cs.append(dict(model='structured_ssl',assignment=mode,C=C,gamma=.001,pseudo_weight=w))
 for prior in [.25,.5,1]:
  for C in [1,3]:cs.append(dict(model='structured_ssl',assignment='patterns',prior_weight=prior,C=C,gamma=.001,pseudo_weight=.03))
 for mode in ['nonempty','exclusive_rare','patterns']:
  for C in [1,3]:cs.append(dict(model='structured_ssl',assignment=mode,C=C,gamma=.001,pseudo_weight=.03,cluster_gate=True))
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase10-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase10')
