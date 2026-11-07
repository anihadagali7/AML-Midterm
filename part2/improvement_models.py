"""Fixed proposals around both published Ani winners; no evaluation labels fit models."""
from __future__ import annotations

import gc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from ani_models import AniFeatures, BASES, fit_base, predict_base, onehot

TAUS = {"unweighted": None, "recent": 3., "moderate": 6.}
# Freeze this registry before scoring. Do not silently add a post-hoc winner.
RECIPES = {
    "ani_stack": dict(banks=["recent"], tau=3., C=1.),
    "ani_first": dict(banks=["unweighted"], tau=None, C=1.),
    "tau6_stack": dict(banks=["moderate"], tau=6., C=1.),
    "recent_meta_tau6": dict(banks=["recent"], tau=6., C=1.),
    "recent_equal_phase": dict(banks=["recent"], tau=3., C=1., balance=True),
    "recent_phase_shrink": dict(banks=["recent"], tau=3., C=.3, interactions=True),
    "recent_phase_C1": dict(banks=["recent"], tau=3., C=1., interactions=True),
    "tau6_phase_shrink": dict(banks=["moderate"], tau=6., C=.3, interactions=True),
    "recent_without_nb": dict(banks=["recent"], tau=3., C=1., drop_nb=True),
    "multiscale_stack": dict(banks=["recent", "unweighted"], tau=3., C=.3),
    "multiscale_phase": dict(banks=["recent", "unweighted"], tau=3., C=.3, interactions=True),
    "ani_phase3_trend": dict(banks=["recent"], tau=3., C=1., trend=True),
    "blend_winners50": dict(blend={"ani_stack": .5, "ani_first": .5}),
}


def recency_weights(df, tau, median):
    year = df.start_year.fillna(median).to_numpy()
    w = np.ones(len(df)) if tau is None else np.exp((year - year.max()) / tau)
    return w / w.mean()


def fit_bank(train, ev, features=None, retain_models=False, progress=None):
    """Share only unsupervised transforms across fixed recency settings.

    Strict mode refits preprocessing inside each grouped inner fold. Passing a
    feature space explicitly selects the separately reported transductive track.
    No labels from ev are read. Inner weights match Ani's outer-normalized recipe.
    """
    train = train.reset_index(drop=True)
    strict = features is None
    features = AniFeatures().fit(train) if strict else features
    weights = {k: recency_weights(train, tau, features.year_median) for k, tau in TAUS.items()}
    bank = {k: {"oof": np.zeros((len(train), len(BASES))),
                "eval": np.zeros((len(ev), len(BASES)))} for k in TAUS}
    fold_ids = np.full(len(train), -1)
    audit = []
    for fold, (a, b) in enumerate(GroupKFold(5).split(train, groups=train.trial_id)):
        assert not set(train.trial_id.iloc[a]) & set(train.trial_id.iloc[b])
        space = AniFeatures().fit(train.iloc[a]) if strict else features
        fa, fb = space.transform(train.iloc[a]), space.transform(train.iloc[b])
        if strict:
            assert set(space.fit_ids) == set(train.ID.iloc[a])
            assert not set(space.fit_ids) & (set(train.ID.iloc[b]) | set(ev.ID))
        for j, (name, blocks, C) in enumerate(BASES):
            for key in TAUS:
                if name == "criteria kNN" and key != "unweighted":
                    bank[key]["oof"][b, j] = bank["unweighted"]["oof"][b, j]
                    continue
                m = fit_base(name, blocks, C, fa, train.iloc[a], weights[key][a])
                bank[key]["oof"][b, j] = predict_base(name, blocks, m, fb, train.iloc[b])
        fold_ids[b] = fold
        audit.append(dict(fold=fold, fit_ids=train.ID.iloc[a].tolist(),
                          heldout_ids=train.ID.iloc[b].tolist(),
                          preprocessing_fit_ids=space.fit_ids, trial_overlap=0))
        del space, fa, fb, m
        gc.collect()
        if progress:
            progress(f"inner {fold + 1}/5")
    assert (fold_ids >= 0).all()
    ft, fe = features.transform(train), features.transform(ev)
    models = {k: [] for k in TAUS}
    for j, (name, blocks, C) in enumerate(BASES):
        for key in TAUS:
            if name == "criteria kNN" and key != "unweighted":
                bank[key]["eval"][:, j] = bank["unweighted"]["eval"][:, j]
                if retain_models:
                    models[key].append(models["unweighted"][-1])
                continue
            m = fit_base(name, blocks, C, ft, train, weights[key])
            bank[key]["eval"][:, j] = predict_base(name, blocks, m, fe, ev)
            if retain_models:
                models[key].append(m)
    for values in bank.values():
        assert all(np.isfinite(v).all() for v in values.values())
    packet = dict(bank=bank, fold_ids=fold_ids, audit=audit,
                  year_median=features.year_median, year_cap=float(train.start_year.max() + 2),
                  feature_fit_ids=features.fit_ids, training_ids=train.ID.tolist(),
                  evaluation_ids=ev.ID.tolist())
    if retain_models:
        packet.update(features=features, models=models)
    return packet


def meta_inputs(bank, side, df, recipe, median, cap):
    columns = []
    for key in recipe["banks"]:
        z = bank[key][side]
        if recipe.get("drop_nb"):
            z = z[:, [i for i, (name, _, _) in enumerate(BASES) if name != "criteria MNB"]]
        z = np.clip(z, 1e-4, 1 - 1e-4)
        columns.append(np.log(z / (1 - z)))
    logits = np.hstack(columns)
    phase = onehot(df)
    parts = [logits, phase]
    if recipe.get("interactions"):
        # Shared coefficients plus shrunk phase-specific deviations.
        parts.extend([logits * phase[:, i, None] for i in range(3)])
    if recipe.get("trend"):
        year = df.start_year.fillna(median).to_numpy()
        parts.append((phase[:, 2] * (np.minimum(year, cap) - 2007) / 3)[:, None])
    return np.hstack(parts)


def fit_recipes(packet, train, ev):
    predictions, meta_models = {}, {}
    for name, recipe in RECIPES.items():
        if "blend" in recipe:
            predictions[name] = sum(w * predictions[k] for k, w in recipe["blend"].items())
            continue
        w = recency_weights(train, recipe["tau"], packet["year_median"])
        if recipe.get("balance"):
            for phase in ("I", "II", "III"):
                mask = train.phase.eq(phase).to_numpy()
                w[mask] *= len(train) / (3 * w[mask].sum())
        args = (packet["year_median"], packet["year_cap"])
        xt = meta_inputs(packet["bank"], "oof", train, recipe, *args)
        xe = meta_inputs(packet["bank"], "eval", ev, recipe, *args)
        model = LogisticRegression(C=recipe["C"], max_iter=3000).fit(xt, train.label, sample_weight=w)
        predictions[name] = model.predict_proba(xe)[:, 1]
        meta_models[name] = model
    return predictions, meta_models


def replay(packet, meta_models, df):
    f = packet["features"].transform(df)
    bank = {key: {"eval": np.column_stack([
        predict_base(name, blocks, model, f, df)
        for (name, blocks, _), model in zip(BASES, packet["models"][key])])} for key in TAUS}
    out = {}
    for name, recipe in RECIPES.items():
        if "blend" in recipe:
            out[name] = sum(w * out[k] for k, w in recipe["blend"].items())
        else:
            x = meta_inputs(bank, "eval", df, recipe, packet["year_median"], packet["year_cap"])
            out[name] = meta_models[name].predict_proba(x)[:, 1]
    return out
