"""Current inductive LDA ensemble, using isolated frozen source definitions."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from references.lda_current import functions as lda
import models,models_v5
BASE=Path(__file__).resolve().parent
CACHE={}

def fitting(gold,masked):
    sources=pd.concat([pd.read_csv(BASE/'data/part1_train.csv'),pd.read_csv(BASE/'data/part1_val.csv')],ignore_index=True)
    order={v:i for i,v in enumerate(sources.ID)}
    frame=pd.concat([gold,masked],ignore_index=True)
    return frame.iloc[np.argsort([order[v] for v in frame.ID])].reset_index(drop=True)

def predict(gold,masked,query,labels,emb):
    assert not any(l in query for l in labels)
    assert not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    key=models_v5.cache_key(gold,masked,query,{'family':'current_lda_reference'})
    if key in CACHE:return CACHE[key]
    frame=fitting(gold,masked);sc=StandardScaler().fit(frame[emb].to_numpy())
    known=np.flatnonzero(frame[labels].ge(0).all(axis=1))
    packet=dict(z=sc.transform(frame[emb].to_numpy()),gold=known,y=frame.iloc[known][labels].to_numpy(int))
    revised=lda.fitted_variant(packet,dict(C=.1,teacher_lda=.2,final_lda=.2))
    source=models_v5.centered(lda.predict_base(revised['base'],sc.transform(query[emb].to_numpy())))
    engine=BASE/'references/legacy_engine'
    if str(engine) not in sys.path:sys.path.append(str(engine))
    import run_cross_repo_benchmark as legacy
    with legacy.data_context(frame):
        outputs={}
        for short,name in [('tri','Classical tri-training'),('svm','Local SVM specialist hybrid')]:
            config=next(r['config'] for r in legacy.registry() if r['name']==name)
            outputs[short]=legacy.e.predict(np.arange(len(gold)),query,config)
    p=.7*source+.15*outputs['tri']+.15*outputs['svm']
    CACHE[key]=(p,dict(reference='Current C.1/LDA 70/15/15 ensemble',pseudo_rounds=revised['history']))
    return CACHE[key]

def baseline(pool):
    from study import metric
    parts=[np.load(BASE/'references/lda_current'/f'{pool}{n}.npz') for n in [3,5]]
    np.testing.assert_array_equal(parts[0]['ids'],parts[1]['ids'])
    np.testing.assert_array_equal(parts[0]['truth'],parts[1]['truth'])
    score=np.concatenate([z['scores'] for z in parts]);pred=np.concatenate([z['predictions'] for z in parts]);y=parts[0]['truth']
    return dict(scores=score,predictions=pred,truth=y,ids=parts[0]['ids'],repeat_f1=np.array([metric(y,p)['sample_f1'] for p in pred]))
