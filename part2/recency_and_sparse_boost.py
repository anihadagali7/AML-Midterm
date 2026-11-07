"""Historical weighting sensitivity and sparse LightGBM comparisons."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from experiments import OUT, load_data, prepare, matrix, metrics


def run():
    data=load_data()
    for split in ['temporal','validation']:
        tr,ev,f,bt,be,bx=prepare(data,split)
        scores=[];pred={};tests={};configs={}
        def record(name,p,pt,config):
            pred[name]=p;tests[name]=pt;configs[name]=config
            scores.append(dict(name=name,**metrics(ev,p)))
            pd.DataFrame(scores).sort_values('phase_mean_logloss').to_csv(OUT/f'extra_{split}_scores.csv',index=False)
            print(split,name,metrics(ev,p)['phase_mean_logloss'],flush=True)
        for rep in ['criteria','clinical']:
            xs=[matrix(b,rep) for b in [bt,be,bx]]
            for half_life in [3.,6.,12.]:
                w=2**((tr.start_year.fillna(tr.start_year.median()).to_numpy()-tr.start_year.max())/half_life)
                w=w/w.mean()
                for C in [.5,2.]:
                    model=LogisticRegression(C=C,solver='liblinear',max_iter=1000,tol=1e-5,random_state=42)
                    model.fit(xs[0],tr.label,sample_weight=w)
                    name=f'recency_{rep}_C{C:g}_half{half_life:g}'
                    record(name,model.predict_proba(xs[1])[:,1],model.predict_proba(xs[2])[:,1],
                           dict(family='recency',representation=rep,C=C,half_life=half_life))
        for year in ['none','year']:
            xs=[matrix(b,'knowledge',year) for b in [bt,be,bx]]
            for leaves in [7,15,31]:
                model=LGBMClassifier(n_estimators=1000,learning_rate=.025,num_leaves=leaves,
                    min_child_samples=40,reg_lambda=10,max_bin=31,colsample_bytree=.8,
                    subsample=.8,subsample_freq=1,n_jobs=4,verbosity=-1,random_state=42)
                model.fit(xs[0],tr.label)
                joblib.dump(model,OUT/f'lightgbm_{split}_leaves{leaves}_{year}.joblib',compress=3)
                for n in [400,1000]:
                    name=f'lightgbm_leaves{leaves}_n{n}_{year}'
                    p=model.predict_proba(xs[1],num_iteration=n)[:,1]
                    pt=model.predict_proba(xs[2],num_iteration=n)[:,1]
                    record(name,p,pt,dict(family='lightgbm',year=year,leaves=leaves,iterations=n))
        np.savez_compressed(OUT/f'extra_{split}_predictions.npz',**pred)
        np.savez_compressed(OUT/f'extra_{split}_test_predictions.npz',**tests)
        (OUT/'extra_configs.json').write_text(json.dumps(configs,indent=2))

if __name__=='__main__':run()
