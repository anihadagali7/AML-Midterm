"""Classical nonlinear head with an explicit, supervised error-diversity loss.

Anchored is frozen. Its held-out residuals guide a Nyström kernel logistic head;
the supervised loss is never reversed. This is not joint neural-head training.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler

STRENGTHS = {'ordinary': 0., 'diverse10': .1, 'diverse30': .3}
RIDGE = .005


def centered_logit(scores):
    return logit(np.clip(scores, 1e-6, 1-1e-6))-logit(.3)


def decide(scores):
    result = (scores >= .5).astype(int)
    result[np.arange(len(result)), scores.argmax(1)] = 1
    return result


def loss_gradient(flat, features, labels, anchor_error, strength, ridge=RIDGE):
    """Mean BCE per class + L2 + strength/2 * summed squared error correlations."""
    n, d = features.shape
    W = flat.reshape(d, 5)
    logits = features @ W
    probabilities = expit(logits)
    error = probabilities-labels
    loss = np.logaddexp(0., logits).sum()/n-(labels*logits).sum()/n
    loss += ridge/2*np.square(W[1:]).sum()+1e-6/2*np.square(W[0]).sum()
    derivative = error/n
    residual = error-error.mean(0)
    va = np.square(anchor_error).mean(0)+1e-8
    vb = np.square(residual).mean(0)+1e-8
    covariance = (anchor_error*residual).mean(0)
    corr2 = np.square(covariance)/(va*vb)
    if strength:
        loss += strength/2*corr2.sum()
        dprob = strength/n*(covariance/(va*vb)*anchor_error
                           -np.square(covariance)/(va*vb**2)*residual)
        derivative += dprob*probabilities*(1-probabilities)
    gradient = features.T @ derivative
    gradient[1:] += ridge*W[1:]
    gradient[0] += 1e-6*W[0]
    return float(loss), gradient.ravel()


class ComplementaryKernel:
    def fit(self, X, Y, anchor_oof):
        gold = (Y >= 0).all(1)
        y = Y[gold]
        assert anchor_oof.shape == y.shape
        self.scaler = StandardScaler().fit(X)
        self.map = Nystroem(kernel='rbf', gamma=1/150, n_components=128,
                           random_state=270303, n_jobs=1)
        transformed = self.map.fit_transform(self.scaler.transform(X))
        features = np.c_[np.ones(len(y)), transformed[gold]]
        error = expit(centered_logit(anchor_oof))-y
        error -= error.mean(0)
        initial = np.zeros((features.shape[1], 5))
        initial[0] = logit(np.clip(y.mean(0), .02, .98))
        self.weights = {}
        self.audit = {}
        for name, strength in STRENGTHS.items():
            start = initial.ravel() if name == 'ordinary' else self.weights['ordinary'].ravel()
            fit = minimize(loss_gradient, start, args=(features, y, error, strength),
                           method='L-BFGS-B', jac=True,
                           options={'maxiter': 1000, 'ftol': 1e-11, 'gtol': 1e-7})
            assert fit.success, fit.message
            self.weights[name] = fit.x.reshape(features.shape[1], 5)
            residual = expit(features @ self.weights[name])-y
            residual -= residual.mean(0)
            correlation = (error*residual).mean(0)/np.sqrt(
                (np.square(error).mean(0)+1e-8)*(np.square(residual).mean(0)+1e-8))
            self.audit[name] = dict(iterations=int(fit.nit), objective=float(fit.fun),
                max_gradient=float(np.abs(fit.jac).max()),
                training_error_correlation=correlation.tolist(),
                training_mean_squared_error_correlation=float(np.square(correlation).mean()))
        return self

    def predict_logits(self, X):
        features = np.c_[np.ones(len(X)), self.map.transform(self.scaler.transform(X))]
        return {name: features @ W for name, W in self.weights.items()}


def fusion_features(anchor, second_logits=None):
    # Comparable label-indexed pre-decision scores across independently fitted heads.
    # No fold-specific kernel landmarks are incorrectly treated as aligned features.
    a = np.clip(centered_logit(anchor), -6, 6)/3
    if second_logits is None:
        return np.c_[np.ones(len(a)), a]
    return np.c_[np.ones(len(a)), a, np.clip(second_logits, -6, 6)/3]


def fit_fusion(anchor_oof, second_oof, labels, anchor_query, second_query):
    """Five ridge-logistic readouts of all ten head logits, offset at anchored."""
    F = fusion_features(anchor_oof, second_oof)
    Q = fusion_features(anchor_query, second_query)
    offset = centered_logit(anchor_oof)
    ridge = 20.

    def objective(flat):
        W = flat.reshape(F.shape[1], 5)
        logits = offset+F @ W
        value = np.logaddexp(0., logits).sum()-(labels*logits).sum()+ridge/2*np.square(W).sum()
        gradient = F.T @ (expit(logits)-labels)+ridge*W
        return float(value), gradient.ravel()

    fit = minimize(objective, np.zeros(F.shape[1]*5), method='L-BFGS-B', jac=True,
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
    assert fit.success, fit.message
    weights = fit.x.reshape(F.shape[1], 5)
    return expit(centered_logit(anchor_query)+Q @ weights), weights
