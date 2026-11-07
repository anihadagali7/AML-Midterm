"""Alternative classical kernels and training-only embedding geometry for SSL SVM."""
import phase8,experiment as e
import numpy as np,json
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel,laplacian_kernel,polynomial_kernel,linear_kernel,pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.cluster import KMeans
legacy_predict=e.predict

def predict9(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
 if c['model']!='kernel_lowdensity':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
 pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
 A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos]
 if c.get('remove_top'):
  pc=PCA(n_components=c['remove_top'],svd_solver='full').fit(A)
  A=A-pc.inverse_transform(pc.transform(A));B=B-pc.inverse_transform(pc.transform(B))
 if c.get('postnorm'):A=normalize(A)*np.sqrt(150);B=normalize(B)*np.sqrt(150)
 Q=np.r_[B,A[n:]] if return_unlabeled else B
 kind=c.get('kernel','rbf');gamma=c.get('gamma',.001)
 def kernel(a,b):
  if kind=='rbf':return rbf_kernel(a,b,gamma=gamma)
  if kind=='laplacian':return laplacian_kernel(a,b,gamma=gamma)
  if kind=='poly':return polynomial_kernel(a,b,gamma=gamma,degree=c.get('degree',2),coef0=c.get('coef0',1.))
  if kind=='linear':return linear_kernel(a,b)/150
  if kind=='rq':return (1+gamma*pairwise_distances(a,b,metric='sqeuclidean')/c.get('alpha',1.))**(-c.get('alpha',1.))
  raise ValueError(kind)
 K=kernel(A,A);Kq=kernel(Q,A);cluster=None
 if c.get('cluster_gate'):
  z=KMeans(n_clusters=c.get('clusters',10),n_init=10,random_state=42).fit_predict(A);cluster=np.full((len(u),5),.5)
  for k in np.unique(z):
   known=z[:n]==k
   if known.sum()>=2:cluster[z[n:]==k]=Y[known].mean(0)
 outputs=[]
 for j in range(5):
  y=Y[:,j]
  if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
  m=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(K[:n,:n],y)
  for fraction in [.1,1/3,1]:
   for _ in range(c.get('iterations',3)):
    d=m.decision_function(K[n:,:m.n_features_in_]);pseudo=(d>=0).astype(int);w=np.full(len(u),c.get('pseudo_weight',.03)*fraction)
    if cluster is not None:w[((cluster[:,j]>=.5)!=pseudo)&(np.maximum(cluster[:,j],1-cluster[:,j])>=.8)]*=.1
    m=SVC(C=c.get('C',1),kernel='precomputed',class_weight='balanced').fit(K,np.r_[y,pseudo],sample_weight=np.r_[np.ones(n),w])
  outputs.append(expit(2*m.decision_function(Kq)))
 p=np.array(outputs).T
 if return_unlabeled:return p[:len(B)],p[len(B):],u
 return p

e.predict=predict9

def configs():
 cs=[]
 for scale in ['global','raw','standard']:
  for norm in [False,True]:
   for remove in [0,1,3,5]:cs.append(dict(model='kernel_lowdensity',C=1,gamma=.001,scale=scale,postnorm=norm,remove_top=remove))
 for white in [.25,.5,1.]:
  for dim in [25,75,125,150]:
   for C in [.3,1,3]:cs.append(dict(model='kernel_lowdensity',C=C,gamma=.001,pca=dim,white=white))
 for kind,gs in [('laplacian',[.001,.003,.01]),('poly',[.001,.003,.01]),('rq',[.001,.003,.01]),('linear',[.001])]:
  for C in [.3,1,3]:
   for gamma in gs:cs.append(dict(model='kernel_lowdensity',kernel=kind,C=C,gamma=gamma))
 for degree in [3,4]:
  for C in [.3,1,3]:cs.append(dict(model='kernel_lowdensity',kernel='poly',degree=degree,C=C,gamma=.003))
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase9-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase9')
