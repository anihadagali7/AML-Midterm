"""Nested training-only calibration of SSL margins and multilabel dependencies."""
import phase11,experiment as e
import numpy as np,json
from pathlib import Path
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
legacy_predict=e.predict
CACHE={}
BASE=dict(model='lowdensity_v2',cluster_gate=True,clusters=10,C=1,gamma=.001,pseudo_weight=.03)

def shifted(p,t):return expit(logit(np.clip(p,1e-7,1-1e-7))-logit(np.clip(t,1e-7,1-1e-7)))

def fit_thresholds(p,y,c):
 grid=np.array([.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75]);t=np.full(5,.5)
 def objective(t):
  h=e.decisions(shifted(p,t),'t0.5');s=f1_score(y,h,average='samples',zero_division=0);m=f1_score(y,h,average='macro',zero_division=0)
  return (1-c.get('macro_weight',0))*s+c.get('macro_weight',0)*m-c.get('penalty',.02)*np.mean((t-.5)**2)
 for _ in range(4):
  changed=False
  for j in range(5):
   opts=[]
   for v in grid:
    tt=t.copy();tt[j]=v;opts.append((objective(tt),-abs(v-.5),v))
   v=max(opts)[2]
   if t[j]!=v:changed=True;t[j]=v
  if not changed:break
 return t

def calibrated(p,y,q,c):
 kind=c['calibration'];X=logit(np.clip(p,1e-6,1-1e-6));Q=logit(np.clip(q,1e-6,1-1e-6))
 if kind=='thresholds':return shifted(q,fit_thresholds(p,y,c))
 if kind=='patterns':
  patterns,yy=np.unique(y,axis=0,return_inverse=True)
  model=LogisticRegression(C=c.get('meta_C',1),max_iter=2000).fit(X,yy);pp=model.predict_proba(Q)
  if c.get('gfm'):
   patterns=patterns[model.classes_];choices=((np.arange(1,32)[:,None]>>np.arange(5))&1)
   utility=2*(choices@patterns.T)/(choices.sum(1)[:,None]+patterns.sum(1)[None,:])
   return choices[np.argmax(pp@utility.T,axis=1)].astype(float)
  return pp@patterns[model.classes_]
 out=[]
 for j in range(5):
  xx=X if kind=='joint' else X[:,j:j+1];qq=Q if kind=='joint' else Q[:,j:j+1]
  if len(np.unique(y[:,j]))<2:out.append(np.full(len(q),y[0,j]));continue
  m=LogisticRegression(C=c.get('meta_C',1),class_weight=c.get('meta_balance'),max_iter=1000).fit(xx,y[:,j]);out.append(m.predict_proba(qq)[:,1])
 return np.array(out).T

def predict14(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
 if c['model']!='calibrated_ssl':return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)
 pos=np.asarray(train_pos);y=e.Y[pos];base=c.get('base',BASE);excluded=[] if excluded_groups is None else list(excluded_groups)
 # Cache is indexed by all label-bearing positions, exclusion groups and base config.
 inner_splits=c.get('inner_splits',3)
 key=(tuple(pos),tuple(sorted(excluded)),e.config_id(base),inner_splits)
 if key not in CACHE:
  p=np.zeros_like(y,float)
  for a,b in e.make_splits(y,groups=e.GROUP[e.LAB[pos]],seed=130713,n_splits=inner_splits):
   p[b]=legacy_predict(pos[a],e.DATA.iloc[e.LAB[pos[b]]],base,excluded_groups=np.r_[excluded,e.GROUP[e.LAB[pos[b]]]])
  CACHE[key]=p
 p=CACHE[key]
 pred=legacy_predict(pos,query_frame,base,excluded_groups=excluded_groups,return_unlabeled=return_unlabeled)
 if return_unlabeled:
  q,u,ix=pred;qq=calibrated(p,y,np.r_[q,u],c);return qq[:len(q)],qq[len(q):],ix
 return calibrated(p,y,pred,c)

e.predict=predict14

def configs():
 cs=[]
 for kind in ['independent','joint','patterns']:
  for C in [.1,1,10]:cs.append(dict(model='calibrated_ssl',calibration=kind,meta_C=C))
 for weight in [0,.25,.5,1]:cs.append(dict(model='calibrated_ssl',calibration='thresholds',macro_weight=weight))
 for C in [.1,1,10]:cs.append(dict(model='calibrated_ssl',calibration='patterns',gfm=True,meta_C=C))
 return cs
if __name__=='__main__':
 cs=configs();e.save_json('experiments/phase14-configs.json',cs)
 for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase14')
