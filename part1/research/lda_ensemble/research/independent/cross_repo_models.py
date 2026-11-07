"""Frozen, inductive adapters for the three shortlisted Michael/Ani recipes.

The caller supplies only fitting gold rows and permitted original masked rows.
Query rows are transformed only. Sources and parity checks live in
experiments/cross_repo_benchmark_v1. No original experiment code is modified.
"""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

LAST_AUDIT = {}


def decide(p, cutoff):
    out = (p >= cutoff).astype(int)
    out[np.arange(len(p)), p.argmax(axis=1)] = 1
    return out


def fit_ani_base(x, y):
    rows, labels, weights = [], [], []
    for i, target in enumerate(y):
        positive = np.flatnonzero(target)
        rows.extend([i] * len(positive))
        labels.extend(positive)
        weights.extend([1. / len(positive)] * len(positive))
    labels, weights = np.asarray(labels), np.asarray(weights)
    counts = np.bincount(labels, weights=weights, minlength=5) + 1e-9
    weights = weights * counts.mean() / counts[labels]
    model = LogisticRegression(C=.3, max_iter=3000).fit(x[rows], labels, sample_weight=weights)
    return model, normalize(x), y / y.sum(axis=1, keepdims=True)


def ani_base_predict(base, query):
    model, x, targets = base
    lr = np.zeros((len(query), 5))
    lr[:, model.classes_] = model.predict_proba(query)
    sim = normalize(query) @ x.T
    nearest = np.argsort(-sim, axis=1)[:, :10]
    weights = np.take_along_axis(sim, nearest, axis=1).clip(0) ** 2
    knn = (weights[:, :, None] * targets[nearest]).sum(axis=1)
    knn /= weights.sum(axis=1, keepdims=True) + 1e-9
    return .7 * lr + .3 * knn


def partial_targets(responsibility, known, truth):
    support = responsibility[known].sum(axis=0)
    rates = (responsibility[known].T @ truth + 2 * truth.mean(axis=0)) / (support[:, None] + 2)
    probability = responsibility @ rates
    eligible = (~known) & (responsibility @ support >= 3)
    targets = np.full((len(known), 5), -100, dtype=int)
    weights = np.zeros((len(known), 5))
    targets[known], weights[known] = truth, 1.
    for j in range(5):
        positive = eligible & (probability[:, j] >= .6)
        negative = eligible & (probability[:, j] <= .1)
        np_, nn = int(positive.sum()), int(negative.sum())
        kp, kn = int(truth[:, j].sum()), int(len(truth) - truth[:, j].sum())
        if not np_ or not nn or not kp or not kn:
            continue
        fraction = min(.5, .25 * np_ / kp, .25 * nn / kn)
        targets[positive, j], targets[negative, j] = 1, 0
        weights[positive, j], weights[negative, j] = fraction * kp / np_, fraction * kn / nn
    assert np.array_equal(weights > 0, targets != -100)
    np.testing.assert_array_equal(targets[known], truth)
    return targets, weights


def predict(frame, gold_rows, unlabeled_rows, query, family, features, labels):
    global LAST_AUDIT
    gold_rows, unlabeled_rows = np.asarray(gold_rows), np.asarray(unlabeled_rows)
    pool = np.sort(np.r_[gold_rows, unlabeled_rows])
    assert not set(frame.iloc[pool].ID) & set(query.ID)
    known = np.isin(pool, gold_rows)
    x = frame.iloc[pool][features].to_numpy()
    y = frame.iloc[pool[known]][labels].to_numpy(int)
    assert np.isin(y, [0, 1]).all()
    assert (frame.iloc[unlabeled_rows][labels].to_numpy() == -100).all()
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    raw_query = query[features].to_numpy()
    audit = dict(family=family, gold_ids=frame.iloc[pool[known]].ID.tolist(),
                 masked_ids=frame.iloc[pool[~known]].ID.tolist(), query_ids=query.ID.tolist(),
                 gold_rows=int(known.sum()), masked_rows=int((~known).sum()))
    if family == 'repo_ani':
        pca = PCA(30, random_state=0).fit(z)
        mixture = GaussianMixture(30, covariance_type='diag', n_init=3, random_state=0).fit(pca.transform(z))
        assert mixture.converged_
        responsibilities = mixture.predict_proba(pca.transform(z))
        fitting = np.flatnonzero(known)
        unlab = np.flatnonzero(~known)
        augmented, target = fitting.copy(), y.copy()
        history = []
        for _ in range(2):
            base = fit_ani_base(z[augmented], target)
            t = target / target.sum(axis=1, keepdims=True)
            rates = (responsibilities[augmented].T @ t + t.mean(axis=0)) / (responsibilities[augmented].sum(axis=0)[:, None] + 1)
            prob = .8 * ani_base_predict(base, z[unlab]) + .2 * (responsibilities[unlab] @ rates)
            keep = prob.max(axis=1) >= .8
            augmented = np.r_[fitting, unlab[keep]]
            target = np.vstack([y, decide(prob, .3)[keep]])
            history.append(int(keep.sum()))
        base = fit_ani_base(z[augmented], target)
        result = ani_base_predict(base, scaler.transform(raw_query))
        audit.update(pseudo_rows_by_round=history, gmm_iterations=int(mixture.n_iter_))
    else:
        if family == 'repo_dbscan':
            near = NearestNeighbors(n_neighbors=5, algorithm='brute', n_jobs=1).fit(z).kneighbors(z)[0][:, -1]
            eps = max(float(np.quantile(near, .25)), 1e-6)
            clustering = DBSCAN(eps=eps, min_samples=5, metric='euclidean', algorithm='brute', n_jobs=1).fit(z)
            components = np.unique(clustering.labels_[clustering.labels_ >= 0])
            responsibility = np.column_stack([clustering.labels_ == c for c in components]).astype(float) if len(components) else np.empty((len(z), 0))
            audit['eps'] = eps
        elif family == 'repo_kmeans':
            clustering = KMeans(n_clusters=8, n_init=10, max_iter=300, tol=1e-4, random_state=42).fit(z)
            responsibility = np.eye(8)[clustering.labels_]
        else:
            raise ValueError(family)
        targets, weights = partial_targets(responsibility, known, y)
        weights *= len(y) / weights.sum(axis=0)
        classifier_scaler = StandardScaler().fit(x[known])
        a, q = classifier_scaler.transform(x), classifier_scaler.transform(raw_query)
        probabilities, iterations = [], []
        for j in range(5):
            included = weights[:, j] > 0
            model = LogisticRegression(C=1000., solver='lbfgs', max_iter=5000, tol=1e-4, random_state=42)
            model.fit(a[included], targets[included, j], sample_weight=weights[included, j])
            probabilities.append(model.predict_proba(q)[:, list(model.classes_).index(1)])
            iterations.append(int(model.n_iter_.max()))
        result = np.column_stack(probabilities)
        audit.update(iterations=iterations, generated_positive=((targets[~known] == 1) & (weights[~known] > 0)).sum(axis=0).tolist(),
                     generated_negative=((targets[~known] == 0) & (weights[~known] > 0)).sum(axis=0).tolist())
    assert result.shape == (len(query), 5) and np.isfinite(result).all()
    LAST_AUDIT = audit
    return result


def install(engine):
    previous = engine.predict

    def dispatch(train_pos, query_frame, config, excluded_groups=None, return_unlabeled=False):
        if not config['model'].startswith('repo_'):
            return previous(train_pos, query_frame, config, excluded_groups, return_unlabeled)
        assert not return_unlabeled
        u = engine.UNLAB
        if excluded_groups is not None:
            u = u[~np.isin(engine.GROUP[u], excluded_groups)]
        return predict(engine.DATA, engine.LAB[np.asarray(train_pos)], u, query_frame,
                       config['model'], engine.EMB, engine.LABELS)

    engine.predict = dispatch
