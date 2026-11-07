"""Gold-only logistic corrections of cross-fitted Other decision scores."""
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler, normalize
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

PRIOR = Path(__file__).resolve().parent.parent / "feature_gap_cv_v1"
sys.path.insert(0, str(PRIOR))
import features as f

VIEWS = ["calibration", "context", "neighbors", "drug_targets", "diseases", "criteria_text"]
BLOCKS = {view + "_ridge" + str(ridge): f"Other offset-logistic correction: {view}, penalty {ridge}"
          for view in VIEWS for ridge in [1, 10]}


def score_context(p):
    clipped = np.clip(p, 1e-6, 1-1e-6)
    normalized = clipped / clipped.sum(axis=1, keepdims=True)
    entropy = -(normalized*np.log(normalized)).sum(axis=1)
    sorted_scores = np.sort(p, axis=1)
    return np.column_stack([logit(clipped), entropy, p[:, 4]-p[:, :4].max(axis=1),
                            sorted_scores[:, -1]-sorted_scores[:, -2]])


def geometry(packet, query):
    z = packet["scaler"].transform(query[f.EMB].to_numpy())
    similarity = normalize(z) @ normalize(packet["z"][packet["gold"]]).T
    features = []
    for j in range(5):
        indices = packet["y"][:, j] == 1
        if not indices.any():
            features.extend([np.zeros(len(query)), np.zeros(len(query))])
            continue
        selected = np.sort(similarity[:, indices], axis=1)
        features.extend([selected[:, -1], selected[:, -min(3, selected.shape[1]):].mean(axis=1)])
    features += [np.linalg.norm(z, axis=1), similarity.max(axis=1)]
    return np.column_stack(features)


def inner_features(packet, fit):
    if "residual_inner" in packet:
        return packet["residual_inner"]
    gold = packet["gold"]
    masked = np.flatnonzero(fit[f.LABELS].eq(-100).all(axis=1))
    groups = pd.util.hash_pandas_object(fit[f.EMB].round(10), index=False).to_numpy()
    unique, inverse = np.unique(groups[gold], return_inverse=True)
    y = packet["y"]
    group_y = np.array([y[inverse == i].max(axis=0) for i in range(len(unique))])
    scores, neighbors = np.full(y.shape, np.nan), np.full((len(y), 12), np.nan)
    details = []
    splitter = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=270201)
    for a, h in splitter.split(group_y, group_y):
        train = np.flatnonzero(np.isin(inverse, a))
        held = np.flatnonzero(np.isin(inverse, h))
        forbidden = groups[gold[held]]
        u = masked[~np.isin(groups[masked], forbidden)]
        rows = np.sort(np.r_[gold[train], u])
        inner_fit = fit.iloc[rows].reset_index(drop=True)
        query = fit.iloc[gold[held]].drop(columns=f.LABELS).reset_index(drop=True)
        assert not set(groups[rows]) & set(forbidden)
        assert not set(inner_fit.ID) & set(query.ID)
        native = f.train_native(inner_fit)
        scores[held] = f.native_prediction(native, query)
        neighbors[held] = geometry(native, query)
        details.append(dict(fit_gold_ids=fit.iloc[gold[train]].ID.tolist(),
                            fit_masked_ids=fit.iloc[u].ID.tolist(), query_ids=query.ID.tolist()))
    assert np.isfinite(scores).all() and np.isfinite(neighbors).all()
    packet["residual_inner"] = dict(scores=scores, neighbors=neighbors, splits=details)
    return packet["residual_inner"]


def loss_gradient(weights, x, y, offset, ridge):
    z = offset + x @ weights
    loss = np.sum(np.logaddexp(0, z) - y*z) + .5*ridge*(weights@weights)
    gradient = x.T @ (expit(z)-y) + ridge*weights
    return float(loss), gradient


def fit_addition(packet, fit, query, name):
    view, penalty = name.rsplit("_ridge", 1)
    penalty = float(penalty)
    inner = inner_features(packet, fit)
    base = f.native_prediction(packet, query)
    if view == "calibration":
        train_context = logit(np.clip(inner["scores"][:, 4:5], 1e-6, 1-1e-6))
        query_context = logit(np.clip(base[:, 4:5], 1e-6, 1-1e-6))
    else:
        train_context, query_context = score_context(inner["scores"]), score_context(base)
        if view == "neighbors":
            train_context = np.column_stack([train_context, inner["neighbors"]])
            query_context = np.column_stack([query_context, geometry(packet, query)])
    scaler = StandardScaler().fit(train_context)
    x = scaler.transform(train_context) / np.sqrt(train_context.shape[1])
    q = scaler.transform(query_context) / np.sqrt(train_context.shape[1])
    vocabulary_size = 0
    if view in ["drug_targets", "diseases", "criteria_text"]:
        transform = f.Block(view).fit(fit)
        a, b = transform.transform(fit), transform.transform(query)
        rms = max(np.sqrt(np.mean(np.sum(a*a, axis=1))), 1e-12)
        x, q = np.column_stack([x, a[packet["gold"]]/rms]), np.column_stack([q, b/rms])
        vocabulary_size = len(transform.feature_names)
    x, q = np.column_stack([np.ones(len(x)), x]), np.column_stack([np.ones(len(q)), q])
    offset = logit(np.clip(inner["scores"][:, 4], 1e-6, 1-1e-6))
    y = packet["y"][:, 4]
    result = minimize(loss_gradient, np.zeros(x.shape[1]), args=(x, y, offset, penalty),
                      method="L-BFGS-B", jac=True, options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8))
    assert result.success, result.message
    corrected = base.copy()
    corrected[:, 4] = expit(logit(np.clip(base[:, 4], 1e-6, 1-1e-6)) + q @ result.x)
    np.testing.assert_array_equal(corrected[:, :4], base[:, :4])
    audit = dict(block=name, dimensions=x.shape[1], ridge=penalty, vocabulary_size=vocabulary_size,
                 corrective_gold_rows=len(y), pseudo_targets_used_by_corrector=False,
                 inner_splits=inner["splits"], inner_seed=270201, inner_folds=3,
                 coefficient_norm=float(np.linalg.norm(result.x)), optimization_iterations=int(result.nit),
                 coefficients_sha256=hashlib.sha256(result.x.tobytes()).hexdigest(),
                 inner_predictions_sha256=hashlib.sha256(inner["scores"].tobytes()).hexdigest())
    return corrected, corrected.copy(), audit
