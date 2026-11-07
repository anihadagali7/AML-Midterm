"""Selected small model: logistic regression + cosine neighbors + three refits.

This module is independent of the experiment grid. Input columns are emb_0..149.
The known labels use Safety, Funding, Efficacy, Recruitment, Other in that order.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize


def predict_labels(probabilities):
    labels = (probabilities >= .3).astype(int)
    empty = np.flatnonzero(labels.sum(axis=1) == 0)
    labels[empty, probabilities[empty].argmax(axis=1)] = 1
    return labels


class ReasonClassifier:
    def __init__(self, rounds=3):
        self.rounds = rounds

    def fit(self, known_embeddings, known_labels, unlabeled_embeddings):
        X = np.asarray(known_embeddings, dtype=float)
        Y = np.asarray(known_labels, dtype=int)
        U = np.asarray(unlabeled_embeddings, dtype=float)
        if X.shape[1] != 150 or U.shape[1] != 150 or Y.shape != (len(X), 5):
            raise ValueError('Expected 150 embeddings and five reason labels.')
        if not np.isin(Y, [0, 1]).all() or (Y.sum(axis=1) == 0).any():
            raise ValueError('Each known trial must have at least one binary reason.')
        if (Y.sum(axis=0) == 0).any():
            raise ValueError('The fitting labels must represent all five reasons.')
        self.scaler_ = StandardScaler().fit(np.vstack([X, U]))
        Z, V = self.scaler_.transform(X), self.scaler_.transform(U)
        augmented_X, augmented_Y = Z, Y
        self.round_counts_ = []
        self.accepted_mask_ = np.zeros(len(U), dtype=bool)
        self.accepted_labels_ = np.empty((0, 5), dtype=int)
        for step in range(self.rounds + 1):
            # A two-reason trial contributes half a training example to each.
            row, reason = np.nonzero(augmented_Y)
            weight = 1. / augmented_Y.sum(axis=1)[row]
            counts = np.bincount(reason, weights=weight, minlength=5)
            weight *= counts.mean() / counts[reason]
            self.logistic_ = LogisticRegression(C=.3, max_iter=3000, tol=1e-4)
            self.logistic_.fit(augmented_X[row], reason, sample_weight=weight)
            self.neighbor_X_ = normalize(augmented_X)
            self.neighbor_Y_ = augmented_Y / augmented_Y.sum(axis=1, keepdims=True)
            if step == self.rounds:
                break
            probability = self._predict_scaled(V)
            self.accepted_mask_ = probability.max(axis=1) >= .8
            self.accepted_labels_ = predict_labels(probability[self.accepted_mask_])
            self.round_counts_.append(int(self.accepted_mask_.sum()))
            # Rebuild from verified labels each pass; never overwrite them.
            augmented_X = np.vstack([Z, V[self.accepted_mask_]])
            augmented_Y = np.vstack([Y, self.accepted_labels_])
        return self

    def _predict_scaled(self, Z):
        probability = np.zeros((len(Z), 5))
        probability[:, self.logistic_.classes_] = self.logistic_.predict_proba(Z)
        similarity = normalize(Z) @ self.neighbor_X_.T
        neighbors = np.argsort(-similarity, axis=1)[:, :10]
        weight = np.take_along_axis(similarity, neighbors, axis=1).clip(0) ** 2
        local = (weight[:, :, None] * self.neighbor_Y_[neighbors]).sum(axis=1)
        local /= weight.sum(axis=1, keepdims=True) + 1e-9
        return .7 * probability + .3 * local

    def predict_proba(self, embeddings):
        return self._predict_scaled(self.scaler_.transform(np.asarray(embeddings)))

    def predict(self, embeddings):
        return predict_labels(self.predict_proba(embeddings))
