"""Focused xRFM follow-up: one AGOP update, with semantic-vector normalization."""
import numpy as np
from sklearn.preprocessing import StandardScaler,normalize
import models,models_v5,models_v8
CACHE={}
def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query for l in labels) and not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    config={k:v for k,v in c.items() if k not in ['name','blend']};key=models_v5.cache_key(gold,masked,query,config)
    if key not in CACHE:
        raw=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()];sc=StandardScaler().fit(raw)
        z=sc.transform(raw);q=sc.transform(query[emb].to_numpy())
        if c['unit']:z,q=normalize(z),normalize(q)
        chosen=dict(p=2.,q=c['q'],ridge=.01,binary=c.get('binary',False));n=len(gold)
        states=models_v8.leaf_path(z[:n],gold[labels].to_numpy(int),chosen,steps=1)
        p=models_v8.inference(q,z[:n],states[-1],chosen);cutoff=.5 if chosen['binary'] else .3
        CACHE[key]=(p,cutoff,[s['band'] for s in states])
    p,cutoff,bands=CACHE[key]
    if c.get('blend'):
        aligned=p if cutoff==.5 else models_v5.centered(p)
        p=(1-c['blend'])*models_v5.cached_old(gold,masked,query,labels,emb)+c['blend']*aligned;cutoff=.5
    return p,models.threshold(p,cutoff),dict(decoder=cutoff,agop_updates=1,unit_normalization=c['unit'],bandwidths=bands,
                                          method='xRFM leaf, fixed one-step follow-up',generated_labels_used=bool(c.get('blend')))
