"""Joint-label-set shrinkage LDA and exact conditional sample-F1 decoding."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SETTINGS = {population+"_"+prior:(population, prior)
            for population in ["gold", "augmented"] for prior in ["empirical", "uniform"]}
CHOICES = ((np.arange(1, 32)[:, None] >> np.arange(5)) & 1).astype(int)


def decode(probability, patterns):
    utility = 2*(CHOICES @ patterns.T)/(CHOICES.sum(1)[:, None]+patterns.sum(1)[None, :])
    return CHOICES[np.argmax(probability @ utility.T, axis=1)]


def fit_predict(packet, fit, query, name, source, features):
    population, prior = SETTINGS[name]
    original, _, _ = source.fit_addition(packet, fit, query, "revision_full")
    cache = packet.setdefault("joint_lda_heads", {})
    if name not in cache:
        if population == "gold":
            rows, targets = packet["gold"], packet["y"]
        else:
            base = source.fitted_variant(packet, source.SETTINGS["revision_full"])
            rows, targets = base["rows"], base["targets"]
        patterns, encoded = np.unique(targets, axis=0, return_inverse=True)
        counts = np.bincount(encoded, minlength=len(patterns))
        if len(patterns)==1:
            model = None
        else:
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                priors=np.ones(len(patterns))/len(patterns) if prior=="uniform" else None)
            model.fit(packet["z"][rows], encoded)
            np.testing.assert_array_equal(model.classes_, np.arange(len(patterns)))
        cache[name] = dict(model=model, rows=rows, patterns=patterns, counts=counts)
    fitted = cache[name]
    z = packet["scaler"].transform(query[features.EMB].to_numpy())
    probability = np.ones((len(query), 1)) if fitted["model"] is None else fitted["model"].predict_proba(z)
    predicted = decode(probability, fitted["patterns"])
    changed = original.copy()
    gate = (original[:, 3]>=.5)&(predicted[:, 4]==1)
    changed[gate, 4] = 1.
    return changed, predicted.astype(float), dict(block=name, population=population, prior=prior,
        fitting_ids=fit.iloc[fitted["rows"]].ID.tolist(), patterns=fitted["patterns"].tolist(),
        pattern_counts=fitted["counts"].tolist(), query_rows=len(query),
        other_addition_rows=int(gate.sum()), first_four_scores_preserved=True,
        pseudo_labels_used=population=="augmented", decoder="Exact expected sample-F1 over 31 nonempty sets; numeric-mask tie order; hard Other addition only on source-predicted Recruitment")


def check_decoder():
    # Degenerate label-set distributions must recover every possible true set.
    np.testing.assert_array_equal(decode(np.eye(31), CHOICES), CHOICES)
    p = np.array([[.25, .75]])
    patterns = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(decode(p, patterns), patterns[1:])
