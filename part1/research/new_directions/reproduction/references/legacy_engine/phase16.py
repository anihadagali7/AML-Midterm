"""Alternative supervised losses with the same low-density pseudo-label schedule."""
import phase14,experiment as e
import numpy as np,json
from scipy.special import expit
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression,RidgeClassifier
legacy_predict=e.predict

def predict16(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
 if c['model']!='loss_ssl':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
 pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
 A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
 if c.get('kernel','rbf')=='rbf':
  featuremap=Nystroem(kernel='rbf',gamma=c.get('gamma',.001),n_components=min(c.get('components',300),len(A)),random_state=42);A=featuremap.fit_transform(A);Q=featuremap.transform(Q)
 outputs=[]
 def model():
  if c['loss']=='squared_hinge':return LinearSVC(C=c.get('C',1),class_weight='balanced',max_iter=10000,random_state=42,dual='auto')
  if c['loss']=='logistic':return LogisticRegression(C=c.get('C',1),class_weight='balanced',max_iter=1000)
  return RidgeClassifier(alpha=1/c.get('C',1),class_weight='balanced')
 for j in range(5):
  y=Y[:,j]
  if len(np.unique(y))<2:outputs.append(np.full(len(Q),y[0]));continue
  m=model().fit(A[:n],y)
  for fraction in [.1,1/3,1]:
   for _ in range(3):
    labels=(m.decision_function(A[n:])>=0).astype(int)
    m=model().fit(A,np.r_[y,labels],sample_weight=np.r_[np.ones(n),np.full(len(u),c.get('pseudo_weight',.03)*fraction)])
  outputs.append(expit(m.decision_function(Q)*(1 if c['loss']=='logistic' else 2)))
 p=np.array(outputs).T
 if return_unlabeled:return p[:len(B)],p[len(B):],u
 return p

e.predict=predict16

def configs():
 cs=[]
 for loss in ['squared_hinge','logistic','squared_error']:
  for kernel,values in [('rbf',[.3,1,3,10]),('linear',[.01,.1,1])]:
   for C in values:cs.append(dict(model='loss_ssl',loss=loss,kernel=kernel,C=C))
 for loss in ['squared_hinge','logistic']:
  for C in [1,3,10]:cs.append(dict(model='loss_ssl',loss=loss,kernel='rbf',C=C,components=600))
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase16-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase16')
