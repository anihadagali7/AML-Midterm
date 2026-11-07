"""Jacobian-preconditioned tree learners, inspired by JARF (December 2025).

Kernel regression supplies analytic derivatives instead of forest finite
differences. All transforms and pseudo labels use only permitted fitting rows.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
import models,models_v5,models_v8,current_reference
from references.lda_current import functions as lda
CACHE={};GEOMETRY={};PSEUDO={}

def representation(gold,masked,query,emb,labels,geometry):
    key=models_v5.cache_key(gold,masked,gold[emb],{'geometry':geometry})
    if key not in GEOMETRY:
        raw=np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()];sc=StandardScaler().fit(raw);z=sc.transform(raw)
        n,d=len(gold),len(emb);y=gold[labels].to_numpy(int)
        if geometry=='raw':H=np.eye(d);audit={}
        elif geometry=='pca':
            pc=PCA(svd_solver='full').fit(z);H=pc.components_.T;audit={}
        else:
            state=models_v8.leaf_path(z[:n],y,dict(p=2.,q=1.,ridge=.01,binary=True),steps=0)[0]
            rng=np.random.default_rng(13071);selected=rng.choice(len(z),min(256,len(z)),replace=False)
            M=np.zeros((d,d))
            for chunk in np.array_split(selected,8):
                G=models_v8.gradients(z[chunk],z[:n],state['S'],state['band'],2.,1.,state['alpha'])
                M+=np.einsum('ncd,nce->de',G,G,optimize=True)
            M/=len(selected);M/=max(float(np.trace(M))/d,1e-30)
            M+=.01*np.eye(d);M/=np.trace(M)/d
            ev,U=np.linalg.eigh(M)
            # H=M matches the paper's map; eigenvector coordinates are a
            # separately named adaptation to expose the principal directions.
            H=M if geometry=='jacobian' else U[:,::-1]
            audit=dict(gradient_probe_rows=len(selected),gradient_probe_ids=np.r_[gold.ID,masked.ID][selected].tolist(),
                       kernel_bandwidth=state['band'],top_eigenvalues=ev[-10:].tolist(),surrogate='binary Laplace kernel ridge, gold targets only')
        GEOMETRY[key]=(sc,H,audit)
    sc,H,audit=GEOMETRY[key]
    return sc.transform(gold[emb].to_numpy())@H,sc.transform(masked[emb].to_numpy())@H,sc.transform(query[emb].to_numpy())@H,audit

def generated(gold,masked,labels,emb):
    key=models_v5.cache_key(gold,masked,gold[emb],{'teacher':'current_lda'})
    if key not in PSEUDO:
        frame=current_reference.fitting(gold,masked);sc=StandardScaler().fit(frame[emb].to_numpy())
        known=np.flatnonzero(frame[labels].ge(0).all(1));packet=dict(z=sc.transform(frame[emb].to_numpy()),gold=known,y=frame.iloc[known][labels].to_numpy(int))
        revised=lda.fitted_variant(packet,dict(C=.1,teacher_lda=.2,final_lda=.2))
        ids=frame.iloc[revised['rows'][len(known):]].ID.tolist();lookup={v:i for i,v in enumerate(masked.ID)}
        PSEUDO[key]=(np.array([lookup[v] for v in ids],int),revised['targets'][len(known):],revised['history'])
    return PSEUDO[key]

def predict(gold,masked,query,c,labels,emb):
    assert not any(l in query for l in labels) and not (set(gold.ID)|set(masked.ID))&set(query.ID)
    assert masked[labels].eq(-100).all().all()
    cfg={k:v for k,v in c.items() if k not in ['name','blend']};key=models_v5.cache_key(gold,masked,query,cfg)
    if key not in CACHE:
        a,u,q,audit=representation(gold,masked,query,emb,labels,c['geometry']);y=gold[labels].to_numpy(int);weights=np.ones(len(gold));pseudo=0
        if c.get('pseudo'):
            ix,py,history=generated(gold,masked,labels,emb);pseudo=len(ix)
            if len(ix):
                a=np.r_[a,u[ix]];y=np.r_[y,py];weights=np.r_[weights,np.full(len(ix),.25*len(gold)/len(ix))]
            audit={**audit,'pseudo_history':history,'pseudo_ids':masked.iloc[ix].ID.tolist()}
        if c['learner']=='forest':
            model=RandomForestClassifier(n_estimators=200,criterion='entropy',max_features=.5,min_samples_leaf=1,n_jobs=1,random_state=13019)
            model.fit(a,y,sample_weight=weights);raw=model.predict_proba(q);p=np.column_stack([pr[:,list(classes).index(1)] if 1 in classes else np.zeros(len(q)) for pr,classes in zip(raw,model.classes_)])
        else:
            values=[]
            for j in range(5):
                if len(np.unique(y[:,j]))==1:values.append(np.full(len(q),float(y[0,j])));continue
                model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=7,max_depth=3,min_samples_leaf=5,l2_regularization=1.,early_stopping=False,random_state=13019)
                model.fit(a,y[:,j],sample_weight=weights);values.append(model.predict_proba(q)[:,1])
            p=np.column_stack(values)
        CACHE[key]=(p,dict(**audit,pseudo_rows_used=pseudo,geometry=c['geometry'],learner=c['learner']))
    p,audit=CACHE[key]
    if c.get('blend'):
        old,_=current_reference.predict(gold,masked,query,labels,emb);p=(1-c['blend'])*old+c['blend']*p
    return p,models.threshold(p,.5),dict(**audit,decoder=.5,method='Jacobian-preconditioned classical trees')
