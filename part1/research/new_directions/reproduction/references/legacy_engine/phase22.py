"""Phase 22 refinement copied explicitly from the verified phase21 core.
Defaults reproduce phase21; phase21 itself remains unchanged.
Binary-relevance tri-training with classical learners and fold-local guards.
Adapted from Zhou & Li (2005). Weighted pseudo-labels, cluster checks, stratified
bootstraps, confidence gates, and cross-fitted error floors are explicit variants;
we do not assert the original theoretical guarantees for these modifications.
"""
import phase21, experiment as e
import json
import numpy as np
from scipy.special import expit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

legacy_predict = e.predict
GUARD_CACHE = {}
LAST_DIAGNOSTICS = []


def member_specs(c):
    C = c.get('C', 1)
    base = dict(kind='rbf', C=C, gamma=c.get('gamma', .001))
    group = c.get('committee', 'rbf')
    if group == 'mixed':
        return [base, dict(kind='logistic', C=c.get('logistic_C', .3)), dict(kind='ridge', alpha=c.get('ridge_alpha', 10))]
    if group == 'scales':
        return [dict(kind='rbf', C=C, gamma=g) for g in [.0006, .001, .002]]
    if group == 'kernels':
        return [base, dict(kind='poly', C=C, gamma=.003), dict(kind='linear', C=C)]
    return [base.copy() for _ in range(3)]


def bootstrap(y, rng, stratified=True):
    # Preserve minority presence; bootstrap rows independently within each class.
    if not stratified:
        return rng.choice(len(y), size=len(y), replace=True)
    return np.concatenate([rng.choice(np.flatnonzero(y == v), (y == v).sum(), replace=True)
                           for v in np.unique(y)])


def update_size(error, previous_error, previous_size, available):
    """Return allowable pseudo-label count under the original error/size guards."""
    if not error < previous_error or available == 0:
        return 0, previous_size
    if previous_size == 0:
        previous_size = int(np.floor(error / (previous_error - error) + 1))
    if not previous_size < available:
        return 0, previous_size
    if error * available < previous_error * previous_size:
        return available, previous_size
    if error > 0 and previous_size > error / (previous_error - error):
        size = int(np.ceil(previous_error * previous_size / error - 1))
        if previous_size < size <= available:
            return size, previous_size
    return 0, previous_size


def predict_core(train_pos, query_frame, c, excluded_groups=None, return_unlabeled=False):
    global LAST_DIAGNOSTICS
    if c['model'] != 'tri_refined':
        return legacy_predict(train_pos, query_frame, c, excluded_groups, return_unlabeled)
    pos = np.asarray(train_pos)
    n = len(pos)
    u = e.UNLAB if excluded_groups is None else e.UNLAB[~np.isin(e.GROUP[e.UNLAB], excluded_groups)]
    A, B = e.features(np.r_[e.LAB[pos], u], query_frame, c)
    Y = e.Y[pos]
    Q = np.r_[B, A[n:]] if return_unlabeled else B
    specs = member_specs(c)
    matrices = []
    for spec in specs:
        kind = spec['kind']
        if kind == 'rbf':
            matrices.append((rbf_kernel(A, gamma=spec['gamma']), rbf_kernel(Q, A, gamma=spec['gamma'])))
        elif kind == 'poly':
            matrices.append((polynomial_kernel(A, gamma=spec['gamma'], degree=2, coef0=1),
                             polynomial_kernel(Q, A, gamma=spec['gamma'], degree=2, coef0=1)))
        elif kind == 'linear':
            matrices.append((A @ A.T / A.shape[1], Q @ A.T / A.shape[1]))
        else:
            matrices.append(None)

    def fit(k, rows, yy, weights):
        spec = specs[k]
        if len(np.unique(yy)) < 2:
            return (float(yy[0]), rows)
        if matrices[k] is not None:
            model = SVC(C=spec['C'], kernel='precomputed', class_weight=c.get('class_weight', 'balanced'))
            model.fit(matrices[k][0][np.ix_(rows, rows)], yy, sample_weight=np.ascontiguousarray(weights))
        else:
            if spec['kind'] == 'logistic':
                model = LogisticRegression(C=spec['C'], class_weight=c.get('class_weight', 'balanced'), max_iter=1500)
            else:
                model = RidgeClassifier(alpha=spec['alpha'], class_weight=c.get('class_weight', 'balanced'))
            model.fit(A[rows], yy, sample_weight=np.ascontiguousarray(weights))
        return model, rows

    def probability(k, fitted, rows=None, query=False):
        model, used = fitted
        size = len(Q) if query else len(rows)
        if isinstance(model, float):
            return np.full(size, model)
        if matrices[k] is not None:
            X = matrices[k][1][:, used] if query else matrices[k][0][np.ix_(rows, used)]
        else:
            X = Q if query else A[rows]
        temperature = 1 if specs[k]['kind'] == 'logistic' else 2
        return expit(temperature * model.decision_function(X))

    cluster = None
    if c.get('cluster_gate', True):
        z = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(A)
        cluster = np.full((len(u), 5), .5)
        for group in np.unique(z):
            gold = z[:n] == group
            if gold.sum() >= 2:
                cluster[z[n:] == group] = Y[gold].mean(0)

    guard_predictions = None
    if c.get('guard', 'textbook') == 'crossfit':
        geometry = {key: c.get(key) for key in ['scale', 'pca', 'white', 'norm', 'features', 'feature_weight']}
        key = (tuple(pos), tuple(u), Y.tobytes(), json.dumps(specs, sort_keys=True), json.dumps(geometry, sort_keys=True), c.get('stratified_bootstrap', True))
        if key not in GUARD_CACHE:
            gp = np.zeros((3, n, 5))
            for fi, (a, b) in enumerate(e.make_splits(Y, groups=e.GROUP[e.LAB[pos]], seed=210017, n_splits=3)):
                for j in range(5):
                    for k in range(3):
                        rng = np.random.default_rng(210000 + 100 * j + 10 * k + fi)
                        rows = a[bootstrap(Y[a, j], rng, c.get('stratified_bootstrap', True))]
                        model = fit(k, rows, Y[rows, j], np.ones(len(rows)))
                        gp[k, b, j] = probability(k, model, rows=b)
            GUARD_CACHE[key] = gp
        guard_predictions = GUARD_CACHE[key]

    outputs = []
    diagnostics = []
    unlabeled_rows = np.arange(n, len(A))
    all_gold = np.arange(n)
    for j in range(5):
        y = Y[:, j]
        if len(np.unique(y)) < 2:
            outputs.append(np.full(len(Q), y[0]))
            continue
        rngs = [np.random.default_rng(c.get('bootstrap_seed', 210123) + 100*j + k) for k in range(3)]
        models = []
        for k in range(3):
            rows = bootstrap(y, rngs[k], c.get('stratified_bootstrap', True))
            models.append(fit(k, rows, y[rows], np.ones(n)))
        previous_errors = np.full(3, .5)
        previous_sizes = np.zeros(3, int)
        for iteration in range(c.get('rounds', 8)):
            if c.get('no_teaching', False):
                break
            goldp = np.array([probability(k, models[k], rows=all_gold) for k in range(3)])
            up = np.array([probability(k, models[k], rows=unlabeled_rows) for k in range(3)])
            planned = []
            for k in range(3):
                peers = [v for v in range(3) if v != k]
                g0, g1 = goldp[peers]
                agree = (g0 >= .5) == (g1 >= .5)
                error = float(np.mean((g0[agree] >= .5) != y[agree])) if agree.any() else .5
                if guard_predictions is not None:
                    p0, p1 = guard_predictions[peers, :, j]
                    ok = (p0 >= .5) == (p1 >= .5)
                    # A fixed OOF floor prevents overly confident in-sample error estimates.
                    floor = (np.sum((p0[ok] >= .5) != y[ok]) + 1) / (ok.sum() + 2)
                    error = max(error, floor)
                p0, p1 = up[peers]
                match = (p0 >= .5) == (p1 >= .5)
                confidence = np.minimum(np.maximum(p0, 1-p0), np.maximum(p1, 1-p1))
                pool = np.flatnonzero(match & (confidence >= c.get('confidence', .5)))
                count, previous_sizes[k] = update_size(error, previous_errors[k], previous_sizes[k], len(pool))
                if count == 0:
                    continue
                selected = rngs[k].choice(pool, size=count, replace=False) if count < len(pool) else pool
                pseudo = (p0[selected] >= .5).astype(int)
                weights = np.full(count, c.get('pseudo_weight', .1))
                if cluster is not None:
                    cp = cluster[selected, j]
                    weights[((cp >= .5) != pseudo) & (np.maximum(cp, 1-cp) >= .8)] *= .1
                if c.get('cap_mass'):
                    for value in [0, 1]:
                        mask = pseudo == value
                        cap = c.get('mass_fraction', .5) * np.sum(y == value)
                        if weights[mask].sum() > cap:
                            weights[mask] *= cap / weights[mask].sum()
                planned.append((k, selected, pseudo, weights, error))
            if not planned:
                break
            # Simultaneous updates: no learner sees another's update from this round.
            for k, selected, pseudo, weights, error in planned:
                models[k] = fit(k, np.r_[all_gold, n+selected], np.r_[y, pseudo], np.r_[np.ones(n), weights])
                previous_errors[k] = error
                previous_sizes[k] = len(selected)
                diagnostics.append(dict(label=e.LABELS[j],round=iteration,learner=k,error=error,
                                        pseudo_count=len(selected),positive=int(pseudo.sum()),
                                        effective_weight=float(weights.sum())))
        qp = np.array([probability(k, models[k], query=True) for k in range(3)])
        if c.get('aggregation', 'mean') == 'vote':
            # Preserve majority decisions at 0.5 and use scores to break empty-output ties.
            p = .8 * (qp >= .5).mean(0) + .2 * qp.mean(0)
        else:
            p = qp.mean(0)
        outputs.append(p)
    LAST_DIAGNOSTICS = diagnostics
    p = np.array(outputs).T
    if return_unlabeled:
        return p[:len(B)], p[len(B):], u
    return p


# Cache predictions, never fitted query-dependent transforms. Label bytes in the
# key prevent stale caches during explicit training-label perturbation checks.
PREDICTION_CACHE = {}

def cached(train_pos, query_frame, c, excluded_groups=None, return_unlabeled=False):
    from pandas import util
    pos=np.asarray(train_pos)
    columns=[x for x in query_frame.columns if x not in e.LABELS]
    query_hash=util.hash_pandas_object(query_frame[columns],index=False).to_numpy().tobytes()
    excluded=tuple(sorted([] if excluded_groups is None else excluded_groups))
    key=(tuple(pos),e.Y[pos].tobytes(),query_hash,e.config_id(c),excluded,return_unlabeled)
    if key not in PREDICTION_CACHE:
        PREDICTION_CACHE[key]=predict22(pos,query_frame,c,excluded_groups,return_unlabeled)
    return PREDICTION_CACHE[key]

def predict22(train_pos, query_frame, c, excluded_groups=None, return_unlabeled=False):
    kind=c['model']
    if kind=='tri_refined':
        return predict_core(train_pos,query_frame,c,excluded_groups,return_unlabeled)
    if kind=='tri_seed_ensemble':
        members=[cached(train_pos,query_frame,dict(c['member'],bootstrap_seed=seed),excluded_groups,return_unlabeled)
                 for seed in c['seeds']]
        if return_unlabeled:
            assert all(np.array_equal(m[2],members[0][2]) for m in members)
            return np.mean([m[0] for m in members],0),np.mean([m[1] for m in members],0),members[0][2]
        return np.mean(members,0)
    if kind=='hybrid22':
        base=cached(train_pos,query_frame,c['base'],excluded_groups,return_unlabeled)
        tri=cached(train_pos,query_frame,c['tri'],excluded_groups,return_unlabeled)
        def combine(bp,tp):
            if c.get('combination','hard')=='soft':
                # Align each component's frozen decision boundary before averaging.
                from scipy.special import logit
                b=expit(logit(np.clip(bp,1e-7,1-1e-7))-logit(float(c['base_rule'][1:])))
                t=expit(logit(np.clip(tp,1e-7,1-1e-7))-logit(float(c['tri_rule'][1:])))
                return (1-c['tri_weight'])*b+c['tri_weight']*t
            hb=e.decisions(bp,c['base_rule']);ht=e.decisions(tp,c['tri_rule'])
            out=np.where(np.asarray(c['use_tri'],bool)[None,:],ht,hb)
            empty=out.sum(1)==0;out[empty]=hb[empty]
            return out.astype(float)
        if return_unlabeled:
            assert np.array_equal(base[2],tri[2])
            return combine(base[0],tri[0]),combine(base[1],tri[1]),base[2]
        return combine(base,tri)
    return legacy_predict(train_pos,query_frame,c,excluded_groups,return_unlabeled)

e.predict=predict22

