"""Audited search gaps: separate feature kernels and effective pseudo-label weighting.
All transforms, kernel weights, clustering and pseudo-labels are learned within folds.
"""
import phase18,experiment as e
import numpy as np,json
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.cluster import KMeans
legacy_predict=e.predict

def centered(K):return K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean()

def feature_kernel(fitframe,query,block):
 if block=='fingerprint':
  A=e.chemical(fitframe,block);B=e.chemical(query,block)
  def tan(a,b):
   inter=a@b.T;den=a.sum(1)[:,None]+b.sum(1)[None,:]-inter
   return np.divide(inter,den,out=np.zeros_like(inter),where=den>0)
  return tan(A,A),tan(B,A),(A.sum(1)>0).astype(float),(B.sum(1)>0).astype(float)
 A,B=e.block_features(fitframe,query,block);A=normalize(A);B=normalize(B)
 return A@A.T,B@A.T,np.sum(A*A,axis=1),np.sum(B*B,axis=1)

def predict20(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
 if c['model']!='audited_ssl':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
 pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
 idx=np.r_[e.LAB[pos],u];A,B=e.features(idx,query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
 Kbase=rbf_kernel(A,gamma=c.get('gamma',.001));Kqb=rbf_kernel(Q,A,gamma=c.get('gamma',.001))
 query=e.pd.concat([query_frame,e.DATA.iloc[u]],ignore_index=True) if return_unlabeled else query_frame
 blocks=c.get('aux_blocks',[]);kernels=[feature_kernel(e.DATA.iloc[idx],query,block) for block in blocks]
 if c.get('deduplicate'):
  _,inv,counts=np.unique(e.GROUP[idx],return_inverse=True,return_counts=True);rowweight=1/counts[inv].astype(float)
  rowweight[:n]/=rowweight[:n].mean();rowweight[n:]/=rowweight[n:].mean()
 else:rowweight=np.ones(len(A))
 cluster=None
 if c.get('cluster_gate',True):
  z=KMeans(n_clusters=10,n_init=10,random_state=42).fit_predict(A);cluster=np.full((len(u),5),.5)
  for k in np.unique(z):
   gold=z[:n]==k
   if gold.sum()>=2:cluster[z[n:]==k]=Y[gold].mean(0)
 outputs=[]
 for j in range(5):
  y=Y[:,j]
  if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
  if kernels:
   weights=np.ones(len(kernels))/len(kernels)
   if c.get('alignment'):
    target=centered(np.outer(y*2-1,y*2-1));align=[]
    for kk,_,_,_ in kernels:
     h=centered(kk[:n,:n]);align.append(max(0,float(np.sum(h*target)/(np.linalg.norm(h)*np.linalg.norm(target)+1e-15))))
    weights=np.array(align)+1e-8;weights/=weights.sum()
   fraction=c.get('aux_weight',.2);K=(1-fraction)*Kbase;Kq=(1-fraction)*Kqb;diag=np.full(len(A),1-fraction);dq=np.full(len(Q),1-fraction)
   for w,(kk,kq,di,dqi) in zip(weights,kernels):K+=fraction*w*kk;Kq+=fraction*w*kq;diag+=fraction*w*di;dq+=fraction*w*dqi
   K/=np.sqrt(diag[:,None]*diag[None,:]);Kq/=np.sqrt(dq[:,None]*diag[None,:])
  else:K=Kbase;Kq=Kqb
  def fit(k,labels,sw):
   if c.get('balance','raw')=='effective':
    counts=np.bincount(labels,weights=sw,minlength=2);bal={v:sw.sum()/(2*counts[v]) for v in [0,1]}
   else:bal='balanced'
   return SVC(C=c.get('C',1),kernel='precomputed',class_weight=bal).fit(k,labels,sample_weight=np.ascontiguousarray(sw))
  m=fit(K[:n,:n],y,rowweight[:n])
  for fraction in [.1,1/3,1]:
   for _ in range(3):
    pseudo=(m.decision_function(K[n:,:m.n_features_in_])>=0).astype(int);w=rowweight[n:]*c.get('pseudo_weight',.03)*fraction
    if cluster is not None:w[((cluster[:,j]>=.5)!=pseudo)&(np.maximum(cluster[:,j],1-cluster[:,j])>=.8)]*=.1
    m=fit(K,np.r_[y,pseudo],np.r_[rowweight[:n],w])
  outputs.append(expit(2*m.decision_function(Kq)))
 p=np.array(outputs).T
 if return_unlabeled:return p[:len(B)],p[len(B):],u
 return p

e.predict=predict20

def configs():
 cs=[]
 for block in ['criteria','drugs','drug_targets','fingerprint']:
  for w in [.1,.3]:cs.append(dict(model='audited_ssl',aux_blocks=[block],aux_weight=w))
 for pair in [['criteria','drugs'],['criteria','fingerprint'],['diseases_cleaned','drug_targets'],['drugs','drug_targets'],['fingerprint','drug_targets']]:
  for C in [.5,1,2]:cs.append(dict(model='audited_ssl',aux_blocks=pair,aux_weight=.3,C=C))
 for w in [.2,.5]:cs.append(dict(model='audited_ssl',aux_blocks=['criteria','drugs','drug_targets','fingerprint'],aux_weight=w,alignment=True))
 for C in [.3,1,3]:
  for w in [.01,.03,.1]:cs.append(dict(model='audited_ssl',balance='effective',C=C,pseudo_weight=w))
 for balance in ['raw','effective']:cs.append(dict(model='audited_ssl',balance=balance,deduplicate=True))
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase20-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase20')
