"""Focused follow-up: verified-label trust and multilabel decision rules.

The default reproduces the previous three-pass simple model exactly.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

from simple_models import REASONS, FEATURES, decide


def sample_f1(Y, prediction):
    return np.mean(2 * (Y * prediction).sum(axis=1) / (Y.sum(axis=1) + prediction.sum(axis=1)))


class TrustedReasonClassifier:
    def __init__(self, spec=None):
        self.spec = dict(spec or {})

    def fit(self, X, Y, U):
        X, Y, U = np.asarray(X), np.asarray(Y, int), np.asarray(U)
        s = self.spec
        assert Y.shape == (len(X), 5) and np.isin(Y, [0, 1]).all() and (Y.sum(axis=1) > 0).all()
        self.scaler_ = StandardScaler().fit(np.vstack([X, U]))
        Z, V = self.scaler_.transform(X), self.scaler_.transform(U)
        if s.get('geometry') == 'scalar':
            self.scaler_.scale_[:] = np.sqrt(self.scaler_.var_.mean())
            Z, V = self.scaler_.transform(X), self.scaler_.transform(U)
        verified_counts = (Y / Y.sum(axis=1, keepdims=True)).sum(axis=0)
        Za, Ya = Z, Y
        self.round_audit_ = []
        self.accepted_mask_ = np.zeros(len(U), bool)
        self.accepted_labels_ = np.empty((0, 5), int)
        for step in range(s.get('rounds', 3) + 1):
            row, reason = np.nonzero(Ya)
            weight = Ya[row, reason] / Ya.sum(axis=1)[row]
            counts = np.bincount(reason, weights=weight, minlength=5)
            origin_weight = np.where(row < len(X), 1., s.get('pseudo_weight', 1.))
            source = s.get('balance_source', 'augmented')
            if source == 'verified':
                counts = verified_counts
            elif source == 'weighted':
                counts = np.bincount(reason, weights=weight * origin_weight, minlength=5)
            elif source != 'augmented':
                raise ValueError(source)
            weight *= (counts.mean() / counts[reason]) ** s.get('balance_power', 1.)
            weight *= origin_weight
            weight *= np.where((row < len(X)) & (Ya.sum(axis=1)[row] > 1), s.get('verified_pair_weight', 1.), 1.)
            if s.get('normalize_mass', False):
                weight *= len(Za) / weight.sum()
            self.logistic_ = LogisticRegression(C=s.get('C', .3), max_iter=3000, tol=1e-4)
            self.logistic_.fit(Za[row], reason, sample_weight=weight)
            self.neighbor_X_ = normalize(Za)
            self.neighbor_Y_ = Ya if s.get('binary_neighbor_targets', False) else Ya / Ya.sum(axis=1, keepdims=True)
            self.neighbor_trust_ = np.r_[np.ones(len(X)), np.full(len(Za)-len(X), s.get('neighbor_pseudo_weight', 1.))]
            self.round_audit_.append(dict(step=step, generated=len(Za)-len(X),
                                           effective_verified_weight=float(weight[row < len(X)].sum()),
                                           effective_generated_weight=float(weight[row >= len(X)].sum()),
                                           class_counts=counts.tolist()))
            if step == s.get('rounds', 3):
                break
            P = self._base_scaled(V)
            self.accepted_mask_ = P.max(axis=1) >= s.get('confidence', .8)
            self.accepted_labels_ = decide(P[self.accepted_mask_], .3)
            Za = np.vstack([Z, V[self.accepted_mask_]])
            Ya = np.vstack([Y, self.accepted_labels_])
        self.other_model_ = None
        self.other_constant_ = None
        self.other_kind_ = s.get('other_head')
        if self.other_kind_:
            fitting = Y[:, 3] == 1 if self.other_kind_.startswith('conditional') else np.ones(len(Y), bool)
            self.other_known_X_ = Z[fitting]
            self.other_known_Y_ = Y[fitting, 4]
            if self.other_kind_.endswith('logistic'):
                if len(np.unique(Y[fitting, 4])) == 1:
                    self.other_constant_ = float(Y[fitting, 4].mean())
                else:
                    self.other_model_ = LogisticRegression(C=s.get('other_C', 1.), max_iter=3000, tol=1e-6,
                                                           class_weight=s.get('other_class_weight')).fit(Z[fitting], Y[fitting, 4])
        return self

    def _base_scaled(self, Z):
        P = np.zeros((len(Z), 5))
        P[:, self.logistic_.classes_] = self.logistic_.predict_proba(Z)
        similarity = normalize(Z) @ self.neighbor_X_.T
        neighbors = np.argsort(-similarity, axis=1)[:, :self.spec.get('knn_count', 10)]
        weight = np.take_along_axis(similarity, neighbors, axis=1).clip(0) ** 2
        weight *= self.neighbor_trust_[neighbors]
        local = (weight[:, :, None] * self.neighbor_Y_[neighbors]).sum(axis=1) / (weight.sum(axis=1, keepdims=True) + 1e-9)
        mix = self.spec.get('knn_weight', .3)
        return (1-mix) * P + mix * local

    def predict_proba(self, X):
        Z = self.scaler_.transform(X)
        P = self._base_scaled(Z)
        if self.other_kind_:
            if self.other_constant_ is not None:
                other = np.full(len(Z), self.other_constant_)
            elif self.other_kind_.endswith('logistic'):
                other = self.other_model_.predict_proba(Z)[:, 1]
            else:
                similarity = normalize(Z) @ normalize(self.other_known_X_).T
                indices = np.argsort(-similarity, axis=1)[:, :self.spec.get('other_k', 5)]
                weights = np.take_along_axis(similarity, indices, axis=1).clip(0) ** 2
                other = (weights * self.other_known_Y_[indices]).sum(axis=1) / (weights.sum(axis=1) + 1e-9)
            if self.other_kind_.startswith('conditional'):
                eligible = decide(P, .3)[:, 3] == 1
                P[eligible, 4] = np.maximum(P[eligible, 4], other[eligible])
            else:
                P[:, 4] = other
        return P

    def predict(self, X):
        return decide(self.predict_proba(X), self.spec.get('cutoffs', .3))


def inner_other_threshold(X, Y, U, spec, seed=8128):
    """Pick one cutoff from inner OOF predictions; outer rows never enter fit."""
    model_spec = {k: v for k, v in spec.items() if k not in ('decision', 'cutoffs')}
    splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
    P = np.full(Y.shape, np.nan, dtype=float)
    folds = np.full(len(Y), -1)
    for fold, (fit, held) in enumerate(splitter.split(X, Y[:, 4])):
        model = TrustedReasonClassifier(model_spec).fit(X[fit], Y[fit], U)
        P[held] = model.predict_proba(X[held])
        folds[held] = fold
    assert np.isfinite(P).all()
    # Fixed candidate list and tie preference; optimize total sample F1.
    thresholds = [.3, .25, .35, .2, .4, .5, .1]
    scores = []
    for threshold in thresholds:
        cutoffs = np.array([.3, .3, .3, .3, threshold])
        scores.append(float(sample_f1(Y, decide(P, cutoffs))))
    best = next(i for i, score in enumerate(scores) if score >= max(scores)-1e-12)
    cutoff = thresholds[best]
    return np.array([.3, .3, .3, .3, cutoff]), dict(threshold=cutoff, candidates=thresholds, scores=scores,
                                                 inner_fold=folds.tolist(), inner_probabilities=P.tolist())


def fit_and_predict(X, Y, U, Q, spec):
    audit = {}
    cutoffs = spec.get('cutoffs', .3)
    if spec.get('decision') == 'inner_other':
        cutoffs, audit = inner_other_threshold(X, Y, U, spec)
    model = TrustedReasonClassifier(spec).fit(X, Y, U)
    P = model.predict_proba(Q)
    prediction = decide(P, cutoffs)
    return model, P, prediction, dict(cutoffs=np.asarray(cutoffs).tolist(), inner=audit, training=model.round_audit_)
