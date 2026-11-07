"""Fold-local covariate-shift weights for gold labels in cluster-guided SSL."""
import phase17,experiment as e
import numpy as np,json
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
legacy_predict=e.predict

def predict18(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
 if c['model']!='weighted_ssl':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
 pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
 A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
 K=rbf_kernel(A,gamma=.001);Kq=rbf_kernel(Q,A,gamma=.001);kind=c['weighting'];years=e.DATA.iloc[e.LAB[pos]].start_year.to_numpy();gold=np.ones(n)
 if kind=='year':gold=np.exp(c['slope']*(years-2002)/8)
 if kind=='late':gold=np.where(years>2002,c['late_weight'],1.)
 if kind=='domain':
  Z=PCA(n_components=25,svd_solver='full').fit_transform(A);yy=np.r_[np.zeros(n),np.ones(len(u))]
  lr=LogisticRegression(C=c['domain_C'],max_iter=1000).fit(Z,yy);pr=lr.predict_proba(Z[:n])[:,1]
  gold=((pr/(1-pr).clip(1e-8))*n/len(u))**c.get('power',1)
 if kind=='kmm':
  KK=rbf_kernel(A[:n],gamma=c['weight_gamma']);kappa=n*rbf_kernel(A[:n],A[n:],gamma=c['weight_gamma']).mean(1)
  solution=minimize(lambda w:.5*w@KK@w-kappa@w,np.ones(n),jac=lambda w:KK@w-kappa,method='SLSQP',bounds=[(.2,5)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-n,'jac':lambda w:np.ones(n)}],options={'maxiter':300,'ftol':1e-8})
  if not solution.success:raise RuntimeError(solution.message)
  gold=1-c.get('blend',1)+c.get('blend',1)*solution.x
 gold=np.clip(gold,.2,5);gold/=gold.mean();outputs=[]
 z=KMeans(n_clusters=10,n_init=10,random_state=42).fit_predict(A);cluster=np.full((len(u),5),.5)
 for k in np.unique(z):
  known=z[:n]==k
  if known.sum()>=2:cluster[z[n:]==k]=Y[known].mean(0)
 for j in range(5):
  y=Y[:,j]
  if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
  m=SVC(C=1,kernel='precomputed',class_weight='balanced').fit(K[:n,:n],y,sample_weight=gold)
  for fraction in [.1,1/3,1]:
   for _ in range(3):
    d=m.decision_function(K[n:,:m.n_features_in_]);pseudo=(d>=0).astype(int);w=np.full(len(u),.03*fraction)
    w[((cluster[:,j]>=.5)!=pseudo)&(np.maximum(cluster[:,j],1-cluster[:,j])>=.8)]*=.1
    m=SVC(C=1,kernel='precomputed',class_weight='balanced').fit(K,np.r_[y,pseudo],sample_weight=np.r_[gold,w])
  outputs.append(expit(2*m.decision_function(Kq)))
 p=np.array(outputs).T
 if return_unlabeled:return p[:len(B)],p[len(B):],u
 return p

e.predict=predict18

def configs():
 cs=[dict(model='weighted_ssl',weighting='year',slope=s) for s in [-1,.5,1,2]]
 cs += [dict(model='weighted_ssl',weighting='late',late_weight=w) for w in [1.5,3,5]]
 cs += [dict(model='weighted_ssl',weighting='domain',domain_C=C,power=p) for C in [.01,.1,1] for p in [.5,1]]
 cs += [dict(model='weighted_ssl',weighting='kmm',weight_gamma=g,blend=b) for g in [.001,.003,.01] for b in [.5,1]]
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase18-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase18')
