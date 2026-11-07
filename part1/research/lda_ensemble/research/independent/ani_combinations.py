"""One-change-at-a-time combinations on top of the two frozen ensemble leaders."""
import numpy as np
from scipy.special import logit
import ani_improvements as base

e = base.e
LEGACY = e.predict
INNER_CACHE = {}
LAST_AUDIT = {}


def clear():
    INNER_CACHE.clear()
    base.clear()


def ensemble(pos, query, config, excluded=None):
    raw = []
    for seed in config.get('seeds', [0]):
        variant = dict(config.get('variant', {}), gmm_seed=seed)
        raw.append(base.cached(pos, query, variant, excluded))
    ani = base.centered(np.mean(raw, axis=0))
    weights = config['weights']
    assert abs(sum(weights.values())-1) < 1e-12
    score = weights['ani'] * ani
    if weights.get('tri', 0):
        score += weights['tri'] * base.LEGACY(pos, query, base.TRI, excluded)
    if weights.get('svm', 0):
        score += weights['svm'] * base.LEGACY(pos, query, base.SVM, excluded)
    return score


def inner(pos, config, excluded):
    plain = {k:v for k,v in config.items() if k not in ['other', 'source_sha256']}
    key = base.key(pos, e.DATA.iloc[e.LAB[pos]], plain, excluded)
    if key not in INNER_CACHE:
        scores, specialist = np.zeros((len(pos), 5)), np.zeros(len(pos))
        outer = [] if excluded is None else list(excluded)
        for a, h in e.make_splits(e.Y[pos], groups=e.GROUP[e.LAB[pos]], seed=270117, n_splits=3):
            ex = np.r_[outer, e.GROUP[e.LAB[pos[h]]]]
            query = e.DATA.iloc[e.LAB[pos[h]]].drop(columns=e.LABELS)
            scores[h] = ensemble(pos[a], query, plain, ex)
            specialist[h] = base.specialist(pos[a], query)
        INNER_CACHE[key] = scores, specialist
    return INNER_CACHE[key]


def predict(pos, query, config, excluded=None):
    global LAST_AUDIT
    pos = np.asarray(pos)
    score = ensemble(pos, query, config, excluded)
    LAST_AUDIT = dict(seeds=config.get('seeds',[0]), variant=config.get('variant',{}), weights=config['weights'])
    if 'other' not in config:
        return score
    weight = config['other']['specialist_weight']
    p, specialist = inner(pos, config, excluded)
    p = p.copy()
    p[:,4] = (1-weight)*p[:,4] + weight*specialist
    choices=[]
    for threshold in [.35,.40,.45,.50,.55,.60]:
        adjusted=p.copy();adjusted[:,4]=base.centered(adjusted[:,4],threshold)
        f1=e.metrics(e.Y[pos],adjusted,'t0.5')['sample_f1']
        choices.append((f1-.005*logit(threshold)**2,-abs(threshold-.5),threshold,f1))
    selected=max(choices)[2]
    if weight:
        score[:,4]=(1-weight)*score[:,4]+weight*base.specialist(pos,query)
    score[:,4]=base.centered(score[:,4],selected)
    LAST_AUDIT.update(selected_other_threshold=selected,specialist_weight=weight,
        inner_seed=270117,inner_folds=3,
        threshold_candidates=[dict(threshold=t,inner_sample_f1=f,penalized_score=c) for c,_,t,f in choices])
    return score


def dispatch(pos,query,config,excluded_groups=None,return_unlabeled=False):
    if config.get('model')!='ani_combination':
        return LEGACY(pos,query,config,excluded_groups,return_unlabeled)
    assert not return_unlabeled
    return predict(pos,query,config,excluded_groups)


e.predict=dispatch
