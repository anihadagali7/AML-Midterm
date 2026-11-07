"""Average Ani fits on grouped gold subsets; retain fixed ensemble partners."""
import numpy as np
import ani_combinations as combinations

base=combinations.base
e=base.e
LEGACY=e.predict
CACHE={}
LAST_AUDIT={}


def clear():
    CACHE.clear()
    combinations.clear()


def predict(pos,query,config,excluded=None):
    global LAST_AUDIT
    if 'bagging' not in config:
        return combinations.predict(pos,query,config,excluded)
    pos=np.asarray(pos);settings=config['bagging']
    key=base.key(pos,query,config,excluded)
    if key in CACHE:return CACHE[key].copy()
    outer=[] if excluded is None else list(excluded)
    raw=[];members=[]
    for a,h in e.make_splits(e.Y[pos],groups=e.GROUP[e.LAB[pos]],seed=settings['seed'],n_splits=settings['folds']):
        ex=np.r_[outer,e.GROUP[e.LAB[pos[h]]]]
        raw.append(base.cached(pos[a],query,{},ex))
        members.append(dict(gold_rows=len(a),excluded_gold_rows=len(h)))
    ani=np.mean(raw,axis=0)
    full_weight=settings['full_weight']
    if full_weight:
        ani=(1-full_weight)*ani+full_weight*base.cached(pos,query,{},excluded)
    weights=config['weights'];assert abs(sum(weights.values())-1)<1e-12
    score=weights['ani']*base.centered(ani)
    if weights.get('tri',0):score+=weights['tri']*base.LEGACY(pos,query,base.TRI,excluded)
    if weights.get('svm',0):score+=weights['svm']*base.LEGACY(pos,query,base.SVM,excluded)
    LAST_AUDIT=dict(members=members,full_weight=full_weight,weights=weights)
    CACHE[key]=score.copy()
    return score


def dispatch(pos,query,config,excluded_groups=None,return_unlabeled=False):
    if config.get('model')!='ani_bagged':
        return LEGACY(pos,query,config,excluded_groups,return_unlabeled)
    assert not return_unlabeled
    return predict(pos,query,config,excluded_groups)


e.predict=dispatch
