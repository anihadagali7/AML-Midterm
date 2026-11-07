"""Frozen fast iteration around the scored 0.58609 file 10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from extension_neighbors import x_meta
from improvement_models import recency_weights
from ani_models import onehot

BASELINE = "file10"
RECIPES = {}
for weight in [.25, .35, .65, .75, 1.]:
    RECIPES[f"fast_weight{weight:g}"] = dict(blend={"ani_stack": 1-weight, "fast2_neighbors": weight})
for name, weights in [("sharp", [.25, .5, .75]), ("trend", [.1, .25, .5]),
                      ("recent_half", [.25, .5, .75]), ("unweighted_nn", [.1, .25]),
                      ("legacy01", [.1, .2, .3]), ("legacy03", [.1, .2])]:
    for weight in weights:
        RECIPES[f"mix_{name}_{weight:g}"] = dict(blend={BASELINE: 1-weight, name: weight})
RECIPES["mix_sharp_trend"] = dict(blend={BASELINE: .5, "sharp": .25, "trend": .25})
for C, tau in [(.1, 2.), (.6, 2.), (1., 2.), (.3, 1.5), (.3, 2.5), (.3, 3.)]:
    RECIPES[f"meta_C{C:g}_tau{tau:g}"] = dict(meta=True, C=C, tau=tau, drop_knn=False, interactions=False)
for C in [.1, .3]:
    RECIPES[f"meta_drop_knn_C{C:g}"] = dict(meta=True, C=C, tau=2., drop_knn=True, interactions=False)
for C in [.05, .15]:
    RECIPES[f"meta_phase_C{C:g}"] = dict(meta=True, C=C, tau=2., drop_knn=False, interactions=True)


def design(old, extra, df, side, recipe):
    x = x_meta(old, extra, df, side, dict(bank="fast2", tau=2., trend=False, C=recipe["C"], neighbors=True))
    if recipe["drop_knn"]:
        x = np.delete(x, 7, axis=1)
    if recipe["interactions"]:
        # Shared effects plus strongly regularized phase-specific deviations.
        ph = onehot(df)
        x = np.column_stack([x] + [x*ph[:, i, None] for i in range(3)])
    return x


def predict_recipes(old, extra, train, ev, components):
    result = {}; fitted = {}
    for name, recipe in RECIPES.items():
        if "blend" in recipe:
            result[name] = sum(weight*components[key] for key, weight in recipe["blend"].items())
        else:
            xt, xe = [design(old, extra, df, side, recipe) for df, side in [(train, "oof"), (ev, "eval")]]
            weights = recency_weights(train, recipe["tau"], old["year_median"])
            model = LogisticRegression(C=recipe["C"], max_iter=3000).fit(xt, train.label, sample_weight=weights)
            result[name] = .5*components["ani_stack"] + .5*model.predict_proba(xe)[:, 1]
            fitted[name] = model
    return result, fitted
