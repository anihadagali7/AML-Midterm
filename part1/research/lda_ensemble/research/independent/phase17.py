"""Classical tree students trained on fold-local cluster/SVM pseudo-labels."""
import phase16,experiment as e
import numpy as np,json
from sklearn.ensemble import ExtraTreesRegressor
legacy_predict=e.predict
TEACHER=dict(model='lowdensity_v2',cluster_gate=True,clusters=10,C=1,gamma=.001,pseudo_weight=.03)
CACHE={}

def predict17(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
 if c['model']!='tree_student':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
 pos=np.asarray(train_pos);n=len(pos);u=e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded_groups)]
 key=(tuple(pos),tuple(u))
 if key not in CACHE:CACHE[key]=legacy_predict(pos,e.DATA.iloc[u].drop(columns=e.LABELS),TEACHER,excluded_groups=excluded_groups)
 pseudo=CACHE[key]
 if c.get('hard',False):pseudo=e.decisions(pseudo,'t0.55').astype(float)
 A,B=e.features(np.r_[e.LAB[pos],u],query_frame,c);Y=e.Y[pos];Q=np.r_[B,A[n:]] if return_unlabeled else B
 yy=np.r_[Y,pseudo];outputs=[]
 for j in range(5):
  gold_count=np.bincount(Y[:,j],minlength=2);goldweights=np.array([(n/(2*max(gold_count[v],1)))**c.get('balance_power',1) for v in Y[:,j]])
  w=np.r_[goldweights,np.full(len(u),c.get('pseudo_weight',.1))]
  if c['student']=='xgb':
   import xgboost as xgb
   m=xgb.train(dict(objective='binary:logistic',max_depth=c.get('depth',3),eta=.03,min_child_weight=2,subsample=.8,colsample_bytree=.8,reg_lambda=c.get('regularization',1),nthread=1,seed=42),xgb.DMatrix(A,label=yy[:,j],weight=w),num_boost_round=300)
   pred=m.predict(xgb.DMatrix(Q))
  elif c['student']=='catboost':
   from catboost import CatBoostClassifier
   m=CatBoostClassifier(loss_function='CrossEntropy',iterations=300,depth=c.get('depth',4),learning_rate=.03,l2_leaf_reg=10,thread_count=1,random_seed=42,verbose=False,allow_writing_files=False)
   m.fit(A,yy[:,j],sample_weight=w);pred=m.predict_proba(Q)[:,1]
  else:
   m=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=c.get('leaf',2),max_features=.7,n_jobs=1,random_state=42).fit(A,yy[:,j],sample_weight=w);pred=m.predict(Q)
  outputs.append(pred)
 p=np.array(outputs).T
 if return_unlabeled:return p[:len(B)],p[len(B):],u
 return p

e.predict=predict17

def configs():
 cs=[]
 for student in ['xgb','catboost','extra']:
  for pca in [0,25,75]:
   for w in [.03,.3]:cs.append(dict(model='tree_student',student=student,pca=pca,pseudo_weight=w))
 for student in ['xgb','catboost','extra']:
  for pca in [0,25]:cs.append(dict(model='tree_student',student=student,pca=pca,pseudo_weight=.1,hard=True))
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase17-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase17')
