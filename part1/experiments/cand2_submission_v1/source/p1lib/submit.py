"""Final refit on all 235 labeled rows, test predictions, Kaggle CSV + manifest, filled train labels."""
from __future__ import annotations

import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd

from .context import make_ctx
from .data import LABELS, write_submission
from .head import apply_head, decode, sld_em
from .pipeline import run_model, validate


def final_fit(cfg, D, n_seeds=1):
    """Refit on labeled-train + val (235 rows); return calibrated probs for test and the pool."""
    validate(cfg)
    test, unl = D.idx["test"], D.idx["unl"]
    rows = np.r_[test, unl]
    Ps, priors, Ts = [], [], []
    for s in range(n_seeds):
        ctx = make_ctx(D, D.idx["labval"], test, transductive=cfg.get("transductive", True), seed=s, name="final")
        o = run_model(cfg["model"], ctx, rows)
        Ps.append(o["P"]); priors.append(np.asarray(o["prior"])); Ts.append(o["T"])
    P = np.mean(Ps, axis=0)
    prior = np.mean(priors, axis=0)
    return {"P_test": P[: len(test)], "P_pool": P[len(test):], "prior": prior, "T": Ts}


def predict_test(cfg, D, fit):
    head = cfg.get("head", {})
    tr = cfg.get("transductive", True)
    target = None if tr else fit["P_pool"]          # inductive: SLD prior estimated on the unlabeled pool
    Y, Q, pi = apply_head(fit["P_test"], fit["prior"], D.CM, head, target)
    return Y, Q, pi


def fill_pool_labels(cfg, D, fit):
    """Hard labels for the 1,014 masked train rows from the refit model (SLD toward the pool's own
    prior when the head uses SLD)."""
    head = cfg.get("head", {})
    P = fit["P_pool"]
    if head.get("prior") == "sld":
        P, _ = sld_em(P, fit["prior"], a=head.get("sld_a", 0), floor=head.get("sld_floor", 0.25))
    return decode(P, D.CM, head.get("decode", "map")), P


def write_candidate(cfg, D, out_dir, name, scores=None, n_seeds=1, sample_path=None):
    os.makedirs(out_dir, exist_ok=True)
    fit = final_fit(cfg, D, n_seeds=n_seeds)
    Y, Q, pi = predict_test(cfg, D, fit)
    path = os.path.join(out_dir, f"{name}.csv")
    sub = write_submission(D, Y, path)
    if sample_path:                                  # exact format / order check vs the sample file
        sample = pd.read_csv(sample_path)
        assert list(sub.columns) == list(sample.columns) and (sub["ID"].values == sample["ID"].values).all()
    sha = hashlib.sha1(open(path, "rb").read()).hexdigest()
    manifest = {
        "name": name, "csv": os.path.abspath(path), "sha1": sha,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": cfg, "transductive_test_features_used": bool(cfg.get("transductive", True)),
        "trained_on": "157 labeled train rows + 78 val rows (235); 1,014 unlabeled train rows as pool",
        "n_seeds": n_seeds,
        "scores": scores or {},
        "test_pred_label_rate": dict(zip(LABELS, Y.mean(0).round(3).tolist())),
        "test_pred_cardinality": float(Y.sum(1).mean()),
        "test_pred_combo_counts": {"+".join(LABELS[j] for j in range(5) if r[j]): int(c)
                                   for r, c in zip(*np.unique(Y, axis=0, return_counts=True))},
        "sld_test_prior_labels": None if pi is None else dict(zip(LABELS, (pi @ D.CM).round(3).tolist())),
        "train_prior_labels": dict(zip(LABELS, (fit["prior"] @ D.CM).round(3).tolist())),
        "temperature": fit["T"],
    }
    with open(os.path.join(out_dir, f"{name}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=float)
    return manifest, fit, Y


def write_filled_train(cfg, D, fit, path):
    Yp, P = fill_pool_labels(cfg, D, fit)
    tr = D.df.iloc[D.idx["train"]].copy()
    tr = tr.drop(columns=["split"])
    pos = {g: i for i, g in enumerate(D.idx["unl"])}
    lab = tr[LABELS].to_numpy().copy()
    src = np.array(["given"] * len(tr), dtype=object)
    for i, g in enumerate(D.idx["train"]):
        if g in pos:
            lab[i] = Yp[pos[g]]
            src[i] = "filled"
    tr[LABELS] = lab
    tr["label_source"] = src
    tr["fill_confidence"] = np.where(src == "filled", [P[pos[g]].max() if g in pos else 1.0 for g in D.idx["train"]], 1.0)
    tr.to_csv(path, index=False)
    return tr
