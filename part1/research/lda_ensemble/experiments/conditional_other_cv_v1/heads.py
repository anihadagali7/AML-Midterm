"""Other heads with explicit global versus Recruitment-conditional training."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit

SETTINGS = {
    "global_lda_empirical": ("global", "lda", "empirical"),
    "global_lda_balanced": ("global", "lda", "balanced"),
    "conditional_lda_empirical": ("conditional", "lda", "empirical"),
    "conditional_lda_balanced": ("conditional", "lda", "balanced"),
    "conditional_lr_empirical": ("conditional", "lr", "empirical"),
    "conditional_lr_balanced": ("conditional", "lr", "balanced"),
}


def fit_predict(packet, fit, query, name, source, features):
    scope, family, prior = SETTINGS[name]
    original, _, _ = source.fit_addition(packet, fit, query, "revision_full")
    use = np.ones(len(packet["gold"]), dtype=bool) if scope == "global" else packet["y"][:, 3] == 1
    rows = packet["gold"][use]
    y = packet["y"][use, 4]
    cache = packet.setdefault("conditional_other_heads", {})
    if name not in cache:
        if len(np.unique(y)) < 2:
            model = float(y[0]) if len(y) else 0.
        elif family == "lda":
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                priors=np.array([.5, .5]) if prior == "balanced" else None).fit(packet["z"][rows], y)
        else:
            model = LogisticRegression(C=.1, max_iter=3000,
                class_weight="balanced" if prior == "balanced" else None).fit(packet["z"][rows], y)
        cache[name] = model
    model = cache[name]
    zq = packet["scaler"].transform(query[features.EMB].to_numpy())
    probability = np.full(len(query), model) if isinstance(model, float) else model.predict_proba(zq)[:, 1]
    result = original.copy()
    if scope == "global":
        result[:, 4] = .5*original[:, 4]+.5*probability
        gate = np.ones(len(query), dtype=bool)
        cutoff = .5
    else:
        gate = original[:, 3] >= .5
        cutoff = .8 if prior == "balanced" else .5
        head = expit(logit(np.clip(probability, 1e-9, 1-1e-9))-logit(cutoff))
        result[gate, 4] = np.maximum(original[gate, 4], head[gate])
    np.testing.assert_array_equal(result[:, :4], original[:, :4])
    return result, np.column_stack([original[:, :4], probability]), dict(block=name,
        fitting_ids=fit.iloc[rows].ID.tolist(), conditional_training=scope == "conditional",
        training_rows=len(rows), positive_rows=int(y.sum()), model=family, prior=prior,
        cutoff=cutoff, gated_query_rows=int(gate.sum()), pseudo_labels_used=False,
        first_four_scores_unchanged=True, conditional_operation="raise Other only on source-predicted Recruitment" if scope == "conditional" else None)
