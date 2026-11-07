"""Two fixed target-input ablations of the frozen anchored classifier.

Only embedding arrays, never target labels, enter this interface. The original
classifier remains unchanged. Target self-training uses its exact fit method;
geometry-only copies that method with one change to the preprocessing pool.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize

from anchored_solution import AnchoredReasonClassifier, select_reasons


class GeometryOnlyAnchored(AnchoredReasonClassifier):
    def fit(self, embeddings, labels, target_embeddings):
        X, Y = np.asarray(embeddings, float), np.asarray(labels, int)
        Q = np.asarray(target_embeddings, float)
        if X.ndim != 2 or X.shape[1] != 150 or Y.shape != (len(X), 5):
            raise ValueError("Expected 150 embeddings and five labels per fitting row.")
        if Q.ndim != 2 or Q.shape[1] != 150:
            raise ValueError("Target inputs contain only 150 embeddings per row.")
        known = np.isin(Y, [0, 1]).all(axis=1)
        unknown = (Y == -100).all(axis=1)
        if (not (known | unknown).all() or (Y[known].sum(axis=1) == 0).any()
                or (Y[known].sum(axis=0) == 0).any()):
            raise ValueError("Complete nonempty known labels or five -100 values required.")
        # Sole algorithmic difference: fit geometry on eligible rows + targets.
        geometry = np.vstack([X, Q])
        self.scaler_ = StandardScaler().fit(geometry)
        Z = self.scaler_.transform(X)
        geometry_z = self.scaler_.transform(geometry)
        self.pca_ = PCA(30, random_state=0).fit(geometry_z)
        self.gmm_ = GaussianMixture(30, covariance_type="diag", n_init=3,
                                   random_state=0).fit(self.pca_.transform(geometry_z))
        assert self.gmm_.converged_
        K, V, targets = Z[known], Z[unknown], Y[known]
        Rk = self.gmm_.predict_proba(self.pca_.transform(K))
        Ru = self.gmm_.predict_proba(self.pca_.transform(V))
        counts = (targets / targets.sum(axis=1, keepdims=True)).sum(axis=0)
        self.class_weights_ = counts.mean() / counts
        Za, Ya, Ra = K, targets, Rk
        self.round_audit_ = []
        for step in range(3):
            row, reason = np.nonzero(Ya)
            weights = self.class_weights_[reason] / Ya.sum(axis=1)[row]
            weights *= np.where(row < len(K), 1., .25)
            self.logistic_ = LogisticRegression(C=.1, max_iter=3000, tol=1e-4).fit(
                Za[row], reason, sample_weight=weights)
            self.lda_ = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=np.ones(5)/5).fit(Za[row], reason)
            self.neighbor_X_ = normalize(Za)
            self.neighbor_Y_ = Ya.copy()
            self.round_audit_.append(dict(
                step=step, known=len(K), generated=len(Za)-len(K),
                known_weight=float(weights[row < len(K)].sum()),
                generated_weight=float(weights[row >= len(K)].sum())))
            if step == 2:
                break
            T = Ya / Ya.sum(axis=1, keepdims=True)
            rates = (Ra.T @ T + T.mean(axis=0)) / (Ra.sum(axis=0)[:, None] + 1)
            scores = .95*self._scores(V) + .05*(Ru @ rates)
            self.generated_mask_ = scores.max(axis=1) >= .8
            self.generated_labels_ = select_reasons(scores[self.generated_mask_])
            Za = np.vstack([K, V[self.generated_mask_]])
            Ya = np.vstack([targets, self.generated_labels_])
            Ra = np.vstack([Rk, Ru[self.generated_mask_]])
        return self


def fit_predict(x, y, query_x, mode):
    """Use the same fixed model; change only which inputs fitting may see."""
    x, y, query_x = np.asarray(x, float), np.asarray(y, int), np.asarray(query_x, float)
    if mode == "inductive":
        model = AnchoredReasonClassifier().fit(x, y)
    elif mode == "geometry_only":
        model = GeometryOnlyAnchored().fit(x, y, query_x)
    elif mode == "target_self_training":
        target_y = np.full((len(query_x), 5), -100, dtype=int)
        model = AnchoredReasonClassifier().fit(np.vstack([x, query_x]), np.vstack([y, target_y]))
    else:
        raise ValueError(mode)
    known = np.isin(y, [0, 1]).all(axis=1)
    extra = len(query_x) if mode != "inductive" else 0
    assert model.scaler_.n_samples_seen_ == len(x) + extra
    accepted_targets = (int(model.generated_mask_[-len(query_x):].sum())
                        if mode == "target_self_training" else 0)
    audit = dict(mode=mode, gold_rows=int(known.sum()), original_masked_rows=int((~known).sum()),
                 target_rows=len(query_x), geometry_rows=len(x)+extra,
                 target_labels_supplied=False, target_generated_rows_in_final_fit=accepted_targets,
                 target_pseudo_labeling=mode == "target_self_training", rounds=model.round_audit_)
    return model.predict_proba(query_x), audit
