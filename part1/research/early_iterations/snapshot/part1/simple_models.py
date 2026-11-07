"""Small, inspectable Part 1 models. No test data or hidden labels are used here."""
from dataclasses import dataclass

import numpy as np
import warnings
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading

REASONS = ['Safety', 'Funding', 'Efficacy', 'Recruitment', 'Other']
FEATURES = [f'emb_{i}' for i in range(150)]


class Geometry(BaseEstimator, TransformerMixin):
    """Fit only the permitted training pool; scalar scaling preserves rotation."""
    def __init__(self, kind='standard'):
        self.kind = kind

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        if self.kind in ('scalar', 'scalar_unit'):
            self.scaler_.scale_[:] = np.sqrt(np.mean(self.scaler_.var_))
        self.pca_ = None
        if self.kind.startswith('pca'):
            self.pca_ = PCA(n_components=int(self.kind[3:]), svd_solver='full').fit(self.scaler_.transform(X))
        return self

    def transform(self, X):
        Z = self.scaler_.transform(X)
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return normalize(Z) if self.kind.endswith('unit') else Z


def decide(probability, cutoff=.5):
    prediction = (probability >= cutoff).astype(int)
    empty = np.flatnonzero(prediction.sum(axis=1) == 0)
    prediction[empty, probability[empty].argmax(axis=1)] = 1
    return prediction


def expanded_targets(X, Y, balanced=False):
    rows, labels = np.nonzero(Y)
    weights = 1. / Y.sum(axis=1)[rows]
    if balanced:
        counts = np.bincount(labels, weights=weights, minlength=5)
        weights *= counts.mean() / counts[labels]
    return X[rows], labels, weights


class SimpleModel:
    def __init__(self, spec):
        self.spec = dict(spec)

    def fit(self, X, Y, U):
        s = self.spec
        self.geometry = Geometry(s.get('geometry', 'standard')).fit(np.vstack([X, U]))
        Z = self.geometry.transform(X)
        self.Y = np.array(Y, copy=True)
        self.Z = Z
        family = s['family']
        if family == 'simple_ani':
            # Ani's joint linear classifier and optional cosine neighbors, with
            # no PCA, GMM, cluster-rate blending, or per-label heuristics.
            Zu = self.geometry.transform(U)
            Za, Ya = Z, Y
            self.round_counts = []
            for round_index in range(s.get('rounds', 2) + 1):
                Zx, labels, weights = expanded_targets(Za, Ya, s.get('balanced', True))
                self.model = LogisticRegression(C=s['C'], max_iter=3000, tol=1e-4).fit(Zx, labels, sample_weight=weights)
                neighbor_Z, neighbor_Y = (Z, Y) if s.get('neighbors_verified_only', False) else (Za, Ya)
                self.neighbor_Z = normalize(neighbor_Z)
                self.neighbor_Y = neighbor_Y / neighbor_Y.sum(axis=1, keepdims=True)
                if round_index == s.get('rounds', 2):
                    break
                P = self._joint_predict(Zu)
                keep = P.max(axis=1) >= s.get('confidence', .8)
                self.round_counts.append(int(keep.sum()))
                Za = np.vstack([Z, Zu[keep]])
                Ya = np.vstack([Y, decide(P[keep], s['cutoff'])])
        elif family == 'powerset':
            self.labelsets, classes = np.unique(Y, axis=0, return_inverse=True)
            self.model = LogisticRegression(C=s['C'], max_iter=3000, tol=1e-6).fit(Z, classes)
        elif family == 'kernel_ridge':
            self.model = KernelRidge(alpha=s['alpha'], kernel='rbf', gamma=s['gamma']).fit(Z, Y)
        elif family == 'graph':
            pool = self.geometry.transform(np.vstack([X, U]))
            self.models = []
            for j in range(Y.shape[1]):
                labels = np.r_[Y[:, j], np.full(len(U), -1)]
                graph = LabelSpreading(kernel=s.get('kernel', 'knn'), n_neighbors=s.get('k', 10),
                                       gamma=s.get('gamma', .01), alpha=s.get('alpha', .8),
                                       max_iter=1000, tol=1e-5)
                self.models.append(graph.fit(pool, labels))
        elif family == 'cluster':
            pool = self.geometry.transform(np.vstack([X, U]))
            self.cluster = KMeans(n_clusters=s['k'], random_state=0, n_init=10).fit(pool)
            assignment = self.cluster.predict(Z)
            counts = np.bincount(assignment, minlength=s['k'])
            sums = np.zeros((s['k'], 5))
            np.add.at(sums, assignment, Y)
            self.rates = (sums + Y.mean(axis=0)) / (counts[:, None] + 1.)
        elif family == 'self_train':
            base_spec = dict(s, family='logistic')
            base = SimpleModel(base_spec).fit(X, Y, U)
            probability = base.predict_proba(U)
            self.round_counts = []
            for _ in range(s.get('rounds', 1)):
                confidence = np.minimum(np.maximum(probability, 1-probability).min(axis=1), probability.max(axis=1))
                keep = confidence >= s.get('confidence', .9)
                self.generated_count = int(keep.sum())
                self.round_counts.append(self.generated_count)
                targets = decide(probability[keep], s['cutoff'])
                Zu = self.geometry.transform(U[keep])
                self.models = []
                for j in range(Y.shape[1]):
                    model = LogisticRegression(C=s['C'], max_iter=3000, tol=1e-6)
                    self.models.append(model.fit(np.vstack([Z, Zu]), np.r_[Y[:, j], targets[:, j]],
                                                 sample_weight=np.r_[np.ones(len(X)), np.full(keep.sum(), s.get('weight', .25))]))
                probability = self.predict_proba(U)
        elif family in ('logistic', 'svm'):
            self.models = []
            weights = np.ones(len(X))
            if s.get('deduplicate_power', 0):
                # Repeated reason texts should not dominate the linear fit.
                groups = DBSCAN(eps=.01, min_samples=1).fit_predict(Z)
                weights = np.bincount(groups)[groups].astype(float) ** (-s['deduplicate_power'])
                weights *= len(weights) / weights.sum()
            for j in range(Y.shape[1]):
                if family == 'logistic':
                    model = LogisticRegression(C=s['C'], max_iter=3000, tol=1e-6, class_weight=s.get('class_weight'))
                else:
                    model = SVC(C=s['C'], gamma=s.get('gamma', 'scale'), kernel='rbf')
                self.models.append(model.fit(Z, Y[:, j], sample_weight=weights))
        elif family in ('multinomial', 'lda'):
            Zx, labels, weights = expanded_targets(Z, Y, s.get('balanced', False))
            if family == 'multinomial':
                self.model = LogisticRegression(C=s['C'], max_iter=3000, tol=1e-6).fit(Zx, labels, sample_weight=weights)
            else:
                self.model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Zx, labels)
        elif family != 'knn':
            raise ValueError(family)
        return self

    def _joint_predict(self, Z):
        P = np.zeros((len(Z), 5))
        P[:, self.model.classes_] = self.model.predict_proba(Z)
        weight = self.spec.get('knn_weight', 0.)
        if weight:
            similarities = normalize(Z) @ self.neighbor_Z.T
            ix = np.argsort(-similarities, axis=1)[:, :self.spec.get('knn_count', 10)]
            w = np.take_along_axis(similarities, ix, axis=1).clip(0) ** 2
            if self.spec.get('neighbor_weighting') == 'inverse_distance':
                w = 1. / np.maximum(np.sqrt(np.maximum(2 - 2 * np.take_along_axis(similarities, ix, axis=1), 0)), 1e-8)
            neighbor = (w[:, :, None] * self.neighbor_Y[ix]).sum(axis=1) / (w.sum(axis=1, keepdims=True) + 1e-9)
            P = (1 - weight) * P + weight * neighbor
        return P

    def predict_proba(self, X):
        Z = self.geometry.transform(X)
        family = self.spec['family']
        if family == 'simple_ani':
            return self._joint_predict(Z)
        if family == 'powerset':
            return self.model.predict_proba(Z) @ self.labelsets[self.model.classes_]
        if family == 'kernel_ridge':
            return np.clip(self.model.predict(Z), 0., 1.)
        if family == 'graph':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)
                P = np.column_stack([m.predict_proba(Z)[:, 1] for m in self.models])
            disconnected = ~np.isfinite(P).all(axis=1)
            if disconnected.any():
                # Components without any known label have no inferred distribution.
                # A nearest verified neighbor supplies a defined inductive fallback.
                ix = pairwise_distances(Z[disconnected], self.Z).argmin(axis=1)
                P[disconnected] = self.Y[ix]
            return P
        if family in ('logistic', 'self_train'):
            return np.column_stack([m.predict_proba(Z)[:, 1] for m in self.models])
        if family == 'cluster':
            return self.rates[self.cluster.predict(Z)]
        if family == 'svm':
            # A monotone mapping of margins, not calibrated probabilities.
            return expit(np.column_stack([m.decision_function(Z) for m in self.models]))
        if family in ('multinomial', 'lda'):
            P = np.zeros((len(X), 5))
            P[:, self.model.classes_] = self.model.predict_proba(Z)
            return P
        D = pairwise_distances(Z, self.Z)
        ix = np.argsort(D, axis=1, kind='stable')[:, :self.spec['k']]
        distance = np.take_along_axis(D, ix, axis=1)
        weight = 1. / np.maximum(distance, 1e-8)
        if self.spec.get('weights') == 'uniform':
            weight[:] = 1.
        return (weight[:, :, None] * self.Y[ix]).sum(axis=1) / weight.sum(axis=1, keepdims=True)


def first_stage():
    specs = []
    for geometry in ['standard', 'scalar']:
        for k in [1, 3, 10]:
            specs.append(dict(name=f'knn_{geometry}_k{k}', family='knn', geometry=geometry, k=k, cutoff=.5))
        for C in [.1, 1., 10.]:
            for cutoff in [.3, .5]:
                specs.append(dict(name=f'logistic_{geometry}_C{C:g}_t{cutoff:g}', family='logistic', geometry=geometry, C=C, cutoff=cutoff))
        for balanced in [False, True]:
            for C in [.3, 3.]:
                specs.append(dict(name=f'multinomial_{geometry}_C{C:g}_balanced{int(balanced)}', family='multinomial', geometry=geometry, C=C, balanced=balanced, cutoff=.3))
        specs.append(dict(name=f'lda_{geometry}', family='lda', geometry=geometry, cutoff=.3))
        for C in [1., 10.]:
            specs.append(dict(name=f'svm_{geometry}_C{C:g}', family='svm', geometry=geometry, C=C, cutoff=.5))
    for geometry, C in [('pca30', 1.), ('standard_unit', 10.), ('scalar_unit', 10.)]:
        specs.append(dict(name=f'logistic_{geometry}_C{C:g}', family='logistic', geometry=geometry, C=C, cutoff=.3))
    return specs


def second_stage():
    """Use unlabeled rows: graph smoothing, clusters, or one conservative refit."""
    specs = []
    for geometry in ['standard', 'scalar']:
        for k in [5, 10, 20]:
            for alpha in [.2, .8]:
                specs.append(dict(name=f'graph_{geometry}_k{k}_a{alpha:g}', family='graph', geometry=geometry,
                                  k=k, alpha=alpha, cutoff=.3, disconnected_fallback='nearest_verified'))
        for gamma in [.003, .01, .03]:
            specs.append(dict(name=f'graph_rbf_{geometry}_g{gamma:g}', family='graph', geometry=geometry,
                              kernel='rbf', gamma=gamma, alpha=.8, cutoff=.3, disconnected_fallback='nearest_verified'))
        for k in [20, 40, 80]:
            specs.append(dict(name=f'cluster_{geometry}_k{k}', family='cluster', geometry=geometry, k=k, cutoff=.3))
    for confidence in [.8, .9, .95]:
        specs.append(dict(name=f'self_train_scalar_c{confidence:g}', family='self_train', geometry='scalar', C=1.,
                          confidence=confidence, weight=.25, cutoff=.3))
    return specs
