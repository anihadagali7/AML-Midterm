"""Selected simple recipe: trust verified labels, retain both neighbor reasons.

Input embeddings: emb_0..149. Label order: Safety, Funding, Efficacy,
Recruitment, Other. All preprocessing excludes query/test inputs.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize


def select_reasons(scores):
    labels = (scores >= .3).astype(int)
    empty = np.flatnonzero(labels.sum(axis=1) == 0)
    labels[empty, scores[empty].argmax(axis=1)] = 1
    return labels


class TrustedTwoPassClassifier:
    def fit(self, known_embeddings, known_labels, unlabeled_embeddings):
        X = np.asarray(known_embeddings, float)
        Y = np.asarray(known_labels, int)
        U = np.asarray(unlabeled_embeddings, float)
        if X.shape[1] != 150 or U.shape[1] != 150 or Y.shape != (len(X), 5):
            raise ValueError('Expected 150 embedding columns and five reason labels.')
        if not np.isin(Y, [0, 1]).all() or (Y.sum(axis=1) == 0).any() or (Y.sum(axis=0) == 0).any():
            raise ValueError('Known rows need binary, nonempty labels representing all five reasons.')
        self.scaler_ = StandardScaler().fit(np.vstack([X, U]))
        Z, V = self.scaler_.transform(X), self.scaler_.transform(U)
        # Fix class weights using verified labels; generated labels never change them.
        counts = (Y / Y.sum(axis=1, keepdims=True)).sum(axis=0)
        self.class_weights_ = counts.mean() / counts
        Za, Ya = Z, Y
        self.round_counts_ = []
        self.fit_weight_audit_ = []
        for step in range(3):  # Initial fit, then two augmented refits.
            row, reason = np.nonzero(Ya)
            weight = self.class_weights_[reason] / Ya.sum(axis=1)[row]
            weight *= np.where(row < len(X), 1., .25)
            self.fit_weight_audit_.append(dict(step=step, known=float(weight[row < len(X)].sum()),
                                               generated=float(weight[row >= len(X)].sum())))
            self.logistic_ = LogisticRegression(C=.3, max_iter=3000, tol=1e-4)
            self.logistic_.fit(Za[row], reason, sample_weight=weight)
            self.neighbor_X_ = normalize(Za)
            self.neighbor_Y_ = Ya.copy()  # A two-reason neighbor votes fully for both.
            if step == 2:
                break
            scores = self._scores_scaled(V)
            self.accepted_mask_ = scores.max(axis=1) >= .8
            self.accepted_labels_ = select_reasons(scores[self.accepted_mask_])
            self.round_counts_.append(int(self.accepted_mask_.sum()))
            Za = np.vstack([Z, V[self.accepted_mask_]])
            Ya = np.vstack([Y, self.accepted_labels_])
        return self

    def _scores_scaled(self, Z):
        scores = np.zeros((len(Z), 5))
        scores[:, self.logistic_.classes_] = self.logistic_.predict_proba(Z)
        similarity = normalize(Z) @ self.neighbor_X_.T
        neighbors = np.argsort(-similarity, axis=1)[:, :10]
        weight = np.take_along_axis(similarity, neighbors, axis=1).clip(0) ** 2
        local = (weight[:, :, None] * self.neighbor_Y_[neighbors]).sum(axis=1)
        local /= weight.sum(axis=1, keepdims=True) + 1e-9
        return .7 * scores + .3 * local

    def predict_proba(self, embeddings):
        """Return selection scores; these are not calibrated marginal probabilities."""
        return self._scores_scaled(self.scaler_.transform(np.asarray(embeddings)))

    def predict(self, embeddings):
        return select_reasons(self.predict_proba(embeddings))
