"""Inductive, frozen experimental extensions of Ani's self-training recipe.

Source recipes are unchanged. Every learned object uses the supplied fitting
gold and permitted masked rows. Inner selection removes held-out groups from
the masked pool as well as the verified-label pool.
"""
import json
import hashlib
import numpy as np
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
import run_cross_repo_benchmark as common

e = common.e
LEGACY = e.predict
PRED_CACHE = {}
INNER_CACHE = {}
LAST_AUDIT = {}
TRI = next(r['config'] for r in common.registry() if r['name'] == 'Classical tri-training')
SVM = next(r['config'] for r in common.registry() if r['name'] == 'Local SVM specialist hybrid')


def clear():
    PRED_CACHE.clear()
    INNER_CACHE.clear()
    common.pooled.clear_caches()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def key(pos, query, config, excluded):
    # Include labels and raw data to make cache reuse safe in pooled/temporal
    # contexts and under held-out-label/feature mutation tests.
    return (tuple(pos), digest(e.DATA[e.EMB].to_numpy().tobytes()), digest(e.Y.tobytes()),
            tuple(e.DATA.ID), digest(query[e.EMB].to_numpy().tobytes()), tuple(query.ID),
            json.dumps(config, sort_keys=True), tuple(sorted([] if excluded is None else excluded)))


def centered(probability, cutoff=.3):
    return expit(logit(np.clip(probability, 1e-9, 1-1e-9)) - logit(cutoff))


def base_fit(x, targets, trust):
    rows, classes, weights = [], [], []
    for i, target in enumerate(targets):
        positive = np.flatnonzero(target)
        rows.extend([i] * len(positive))
        classes.extend(positive)
        weights.extend([trust[i] / len(positive)] * len(positive))
    classes, weights = np.asarray(classes), np.asarray(weights)
    counts = np.bincount(classes, weights=weights, minlength=5) + 1e-9
    weights = weights * counts.mean() / counts[classes]
    # Preserve the native overall loss scale while altering relative trust.
    # For native trust=1, leave the original arithmetic exactly unchanged.
    if not np.all(trust == 1):
        weights *= len(targets) / weights.sum()
    model = LogisticRegression(C=.3, max_iter=3000).fit(x[rows], classes, sample_weight=weights)
    return model, normalize(x), targets / targets.sum(axis=1, keepdims=True), trust


def base_components(packet, query):
    model, x, targets, trust = packet
    lr = np.zeros((len(query), 5))
    lr[:, model.classes_] = model.predict_proba(query)
    similarity = normalize(query) @ x.T
    indices = np.argsort(-similarity, axis=1)[:, :10]
    weight = np.take_along_axis(similarity, indices, axis=1).clip(0) ** 2
    weight *= trust[indices]
    knn = (weight[:, :, None] * targets[indices]).sum(axis=1) / (weight.sum(axis=1, keepdims=True) + 1e-9)
    return lr, knn


def native_variant(pos, query, config, excluded=None):
    global LAST_AUDIT
    unlabeled = e.UNLAB if excluded is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB], excluded)]
    pool = np.sort(np.r_[e.LAB[pos], unlabeled])
    known = np.isin(pool, e.LAB[pos])
    assert not set(e.DATA.iloc[pool].ID) & set(query.ID)
    if excluded is not None:
        assert not set(e.GROUP[pool]) & set(excluded)
    x = e.DATA.iloc[pool][e.EMB].to_numpy()
    y = e.DATA.iloc[pool[known]][e.LABELS].to_numpy(int)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    pca = PCA(30, random_state=0).fit(z)
    gmm = GaussianMixture(30, covariance_type='diag', n_init=3,
                          random_state=config.get('gmm_seed', 0)).fit(pca.transform(z))
    assert gmm.converged_
    responsibility = gmm.predict_proba(pca.transform(z))
    gold, u = np.flatnonzero(known), np.flatnonzero(~known)
    rows, target, trust = gold.copy(), y.copy(), np.ones(len(gold))
    history = []
    for iteration in range(2):
        base = base_fit(z[rows], target, trust)
        t = target / target.sum(axis=1, keepdims=True)
        if np.all(trust == 1):
            rates = (responsibility[rows].T @ t + t.mean(axis=0)) / (responsibility[rows].sum(axis=0)[:, None] + 1)
        else:
            rates = (responsibility[rows].T @ (t * trust[:, None]) + np.average(t, axis=0, weights=trust)) / ((responsibility[rows].T @ trust)[:, None] + 1)
        lr, knn = base_components(base, z[u])
        prob = .8 * (.7 * lr + .3 * knn) + .2 * (responsibility[u] @ rates)
        keep = prob.max(axis=1) >= .8
        accepted_before_agreement = int(keep.sum())
        if config.get('agreement', False):
            keep &= lr.argmax(axis=1) == knn.argmax(axis=1)
        rows = np.r_[gold, u[keep]]
        target = np.vstack([y, common.adapters.decide(prob, .3)[keep]])
        trust = np.r_[np.ones(len(gold)), np.full(int(keep.sum()), config.get('pseudo_weight', 1.))]
        history.append(dict(round=iteration+1, accepted=int(keep.sum()), before_agreement=accepted_before_agreement))
    base = base_fit(z[rows], target, trust)
    lr, knn = base_components(base, scaler.transform(query[e.EMB].to_numpy()))
    LAST_AUDIT = dict(gold_ids=e.DATA.iloc[pool[known]].ID.tolist(),
        masked_ids=e.DATA.iloc[pool[~known]].ID.tolist(), query_ids=query.ID.tolist(),
        gold_rows=int(known.sum()), masked_rows=len(u), history=history,
        pseudo_weight=config.get('pseudo_weight', 1.), gmm_seed=config.get('gmm_seed', 0),
        agreement=config.get('agreement', False))
    return .7 * lr + .3 * knn


def cached(pos, query, config, excluded=None):
    k = key(pos, query, config, excluded)
    if k not in PRED_CACHE:
        PRED_CACHE[k] = native_variant(pos, query, config, excluded)
    return PRED_CACHE[k]


def specialist(pos, query):
    # A deliberately small, regularized binary head; no pseudo-labels.
    x = e.DATA.iloc[e.LAB[pos]][e.EMB].to_numpy()
    y = e.Y[pos, 4]
    if len(np.unique(y)) < 2:
        return np.full(len(query), y[0])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=.3, class_weight='balanced', max_iter=2000).fit(scaler.transform(x), y)
    return model.predict_proba(scaler.transform(query[e.EMB].to_numpy()))[:, 1]


def other_inner(pos, excluded):
    k = key(pos, e.DATA.iloc[e.LAB[pos]], {'purpose':'inner_other'}, excluded)
    if k not in INNER_CACHE:
        ani, expert = np.zeros((len(pos), 5)), np.zeros(len(pos))
        outer_ex = [] if excluded is None else list(excluded)
        for a, h in e.make_splits(e.Y[pos], groups=e.GROUP[e.LAB[pos]], seed=270117, n_splits=3):
            held_groups = e.GROUP[e.LAB[pos[h]]]
            q = e.DATA.iloc[e.LAB[pos[h]]].drop(columns=e.LABELS)
            ani[h] = centered(cached(pos[a], q, {}, np.r_[outer_ex, held_groups]))
            expert[h] = specialist(pos[a], q)
        INNER_CACHE[k] = ani, expert
    return INNER_CACHE[k]


def predict(pos, query, config, excluded=None):
    global LAST_AUDIT
    pos = np.asarray(pos)
    mode = config['mode']
    if mode == 'variant':
        return centered(cached(pos, query, config.get('variant', {}), excluded))
    if mode == 'seed_average':
        raw = [cached(pos, query, {'gmm_seed':seed}, excluded) for seed in config['seeds']]
        return centered(np.mean(raw, axis=0))
    ani = centered(cached(pos, query, config.get('variant', {}), excluded))
    if mode == 'blend':
        weights = config['weights']
        result = weights['ani'] * ani
        if weights.get('tri', 0):
            result += weights['tri'] * LEGACY(pos, query, TRI, excluded)
        if weights.get('svm', 0):
            # The SVM hybrid emits hard label votes, not calibrated confidence.
            result += weights['svm'] * LEGACY(pos, query, SVM, excluded)
        assert abs(sum(weights.values())-1) < 1e-12
        return result
    if mode == 'other':
        weight = config.get('specialist_weight', 0.)
        inner_ani, inner_expert = other_inner(pos, excluded)
        inner = inner_ani.copy()
        inner[:, 4] = (1-weight) * inner[:, 4] + weight * inner_expert
        options = []
        for threshold in [.35, .40, .45, .50, .55, .60]:
            scores = inner.copy()
            scores[:, 4] = centered(scores[:, 4], threshold)
            f1 = e.metrics(e.Y[pos], scores, 't0.5')['sample_f1']
            # Fixed shrinkage toward the native decision point.
            criterion = f1 - .005 * logit(threshold)**2
            options.append((criterion, -abs(threshold-.5), threshold, f1))
        selected = max(options)[2]
        result = ani.copy()
        if weight:
            result[:, 4] = (1-weight) * result[:, 4] + weight * specialist(pos, query)
        result[:, 4] = centered(result[:, 4], selected)
        LAST_AUDIT = dict(mode='other', selected_threshold=selected, specialist_weight=weight,
            inner_seed=270117, inner_folds=3, inner_options=[dict(threshold=t, sample_f1=f, penalized_score=c) for c,_,t,f in options],
            selection_labels='Outer-training gold only; inner held-out groups removed from every fitting pool')
        return result
    raise ValueError(mode)


def install():
    def dispatch(pos, query, config, excluded_groups=None, return_unlabeled=False):
        if config.get('model') != 'ani_improvement':
            return LEGACY(pos, query, config, excluded_groups, return_unlabeled)
        assert not return_unlabeled
        return predict(pos, query, config, excluded_groups)
    e.predict = dispatch


install()
