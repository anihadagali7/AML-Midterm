"""No-year boosting: random-seed, recency and phase-weight sensitivity."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from experiments import OUT, PHASES, load_data, prepare, metrics
from boosted import dense_features


def run():
    data=load_data()
    for split in ['validation','temporal']:
        tr,ev,f,bt,be,bx=prepare(data,split)
        frames,svds,mask=dense_features(tr,ev,data['test'],bt,be,bx)
        xs=[d.drop(columns='year') for d in frames]
        cats=xs[0].select_dtypes('str').columns.tolist()
        scores=[];pred={};test={};configs={}
        for variant,seed in [('seed43',43),('seed44',44),('recency6',43),('phasebalanced',43)]:
            w=np.ones(len(tr))
            if variant=='recency6':
                w=2**((tr.start_year.fillna(tr.start_year.median()).to_numpy()-tr.start_year.max())/6)
            elif variant=='phasebalanced':
                w=tr.phase.map({p:len(tr)/(3*sum(tr.phase==p)) for p in PHASES}).to_numpy()
            w=w/w.mean()
            model=CatBoostClassifier(iterations=1500,depth=6,learning_rate=.035,l2_leaf_reg=6,
                random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,loss_function='Logloss')
            model.fit(xs[0],tr.label,cat_features=cats,sample_weight=w)
            joblib.dump(model,OUT/f'robust_catboost_{variant}_{split}.joblib',compress=3)
            for n in [800,1500]:
                name=f'robust_catboost_{variant}_n{n}'
                p=model.predict_proba(xs[1],ntree_end=n)[:,1]
                pt=model.predict_proba(xs[2],ntree_end=n)[:,1]
                pred[name]=p;test[name]=pt
                configs[name]=dict(family='robust_catboost',variant=variant,seed=seed,iterations=n)
                scores.append(dict(name=name,**metrics(ev,p)))
                print(split,name,metrics(ev,p)['phase_mean_logloss'],flush=True)
            pd.DataFrame(scores).sort_values('phase_mean_logloss').to_csv(OUT/f'robust_{split}_scores.csv',index=False)
            # Save each completed fit so intermediate results are safely inspectable.
            np.savez_compressed(OUT/f'robust_{split}_predictions.npz',**pred)
            np.savez_compressed(OUT/f'robust_{split}_test_predictions.npz',**test)
            (OUT/'robust_configs.json').write_text(json.dumps(configs,indent=2))

if __name__=='__main__':run()
