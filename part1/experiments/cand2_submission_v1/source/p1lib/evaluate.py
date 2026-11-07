"""Leak-free evaluation harness.

Protocols (all over the 235 labeled rows = 157 labeled-train + 78 val):
  A : RepeatedStratifiedKFold(5 x R) stratified on the label-combo id         (no shift)
  B : chronological holdout: train start_year < cut, evaluate >= cut         (shift, cut=2005)
  C : official split: train on the 157 labeled-train rows, evaluate on val   (tie-breaker)
Every fold builds a fresh Ctx; everything label-dependent is fit inside `run_model` from
ctx.train labels only.  Label-free fits are reused across folds only when their feature-row
set is identical (transductive mode).

Fold outputs (calibrated combo probabilities BEFORE the head) are cached on disk keyed by the
model part of the config, so SLD strength / decoding are post-hoc and free to vary.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from .context import make_ctx
from .data import LABELS, bootstrap_se, label_f1, metrics, row_f1
from .head import apply_head
from .pipeline import model_part, run_model, validate

# fold-output cache lives OUTSIDE the midterm/ folder (regenerable, ~0.5 GB); override with P1_CACHE
CACHE_DIR = os.environ.get("P1_CACHE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".p1_cache", "folds")))
CHRONO_CUT = 2005
CHRONO_CUTS = (2004, 2005, 2006)   # objective uses the mean over these cuts


def cfg_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:20]


# ----------------------------------------------------------------------------- folds
def make_folds(D, protocols=("A", "B", "C"), R=4, seed=0, rows=None, chrono_cuts=CHRONO_CUTS):
    """rows: restrict to a subset of labeled rows (used by the nested search-overfitting check)."""
    labval = D.idx["labval"] if rows is None else np.asarray(rows)
    y = D.y[labval]
    folds = []
    if "A" in protocols:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=seed)
        for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
            folds.append({"name": f"A{i // 5}_{i % 5}", "protocol": "A", "repeat": i // 5,
                          "train": labval[tr], "eval": labval[te]})
    if "B" in protocols:
        for cut in chrono_cuts:
            yr = D.year[labval]
            folds.append({"name": f"B{cut}", "protocol": "B", "cut": cut,
                          "train": labval[yr < cut], "eval": labval[yr >= cut]})
    if "C" in protocols:
        tr = np.intersect1d(labval, D.idx["lab"])
        te = np.intersect1d(labval, D.idx["val"])
        if len(tr) and len(te):
            folds.append({"name": "C", "protocol": "C", "train": tr, "eval": te})
    return folds


def fold_tag(fold):
    h = hashlib.md5(np.r_[fold["train"], [-1], fold["eval"]].astype(np.int64).tobytes()).hexdigest()[:10]
    return f"{fold['name']}_{h}"


# ----------------------------------------------------------------------------- one fold
def run_fold(cfg, D, fold, use_cache=True):
    mp = model_part(cfg)
    key = cfg_hash(mp)
    path = os.path.join(CACHE_DIR, key, fold_tag(fold) + ".npz")
    if use_cache and os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            out = {k: z[k] for k in z.files}
            out["T"] = out["T"].tolist()
            return out
        except Exception:
            pass
    tr = cfg.get("transductive", True)
    ctx = make_ctx(D, fold["train"], fold["eval"], transductive=tr, seed=0, name=fold["name"])
    t0 = time.time()
    rows = fold["eval"] if tr else np.r_[fold["eval"], ctx.pool]
    o = run_model(cfg["model"], ctx, rows)
    ne = len(fold["eval"])
    out = {"P_eval": o["P"][:ne].astype(np.float32), "prior": np.asarray(o["prior"], float),
           "T": np.atleast_1d(np.asarray(o["T"], float)), "secs": np.float32(time.time() - t0)}
    if not tr:
        out["P_target"] = o["P"][ne:].astype(np.float32)
    if use_cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
        mpath = os.path.join(CACHE_DIR, key, "config.json")
        if not os.path.exists(mpath):
            with open(mpath, "w") as f:
                json.dump(mp, f, indent=1, default=str)
    out["T"] = out["T"].tolist()
    return out


def fold_outputs(cfg, D, folds, n_jobs=1, use_cache=True):
    if n_jobs == 1:
        return [run_fold(cfg, D, f, use_cache) for f in folds]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(run_fold)(cfg, D, f, use_cache) for f in folds)


# ----------------------------------------------------------------------------- scoring
def score(cfg, D, folds, outs, bootstrap=True):
    """Apply the head to cached fold outputs and compute A/B/C metrics with SEs."""
    head = cfg.get("head", {})
    res = {}
    per = {}
    for f, o in zip(folds, outs):
        P = o["P_eval"].astype(float)
        P = P / P.sum(1, keepdims=True)
        Pt = o.get("P_target")
        Pt = None if Pt is None else Pt.astype(float) / Pt.astype(float).sum(1, keepdims=True)
        Yp, Q, pi = apply_head(P, o["prior"], D.CM, head, Pt)
        per[f["name"]] = (f, Yp, pi)
    # ---- A ----
    A = [v for v in per.values() if v[0]["protocol"] == "A"]
    if A:
        rows = np.unique(np.concatenate([f["eval"] for f, _, _ in A]))
        pos = {r: i for i, r in enumerate(rows)}
        reps = sorted({f["repeat"] for f, _, _ in A})
        Ytrue = D.Y[rows]
        Ypreds = []
        for r in reps:
            Yp = np.zeros((len(rows), 5), int)
            for f, yp, _ in A:
                if f["repeat"] == r:
                    Yp[[pos[i] for i in f["eval"]]] = yp
            Ypreds.append(Yp)
        rep_metrics = [metrics(Ytrue, Yp) for Yp in Ypreds]
        rf = np.mean([row_f1(Ytrue, Yp) for Yp in Ypreds], axis=0)
        a = {k: float(np.mean([m[k] for m in rep_metrics])) for k in rep_metrics[0] if k not in ("n",)}
        a["sample_f1_se"] = float(rf.std(ddof=1) / np.sqrt(len(rf)))
        a["sample_f1_rep_sd"] = float(np.std([m["sample_f1"] for m in rep_metrics], ddof=1)) if len(reps) > 1 else 0.0
        a["n"] = int(len(rows))
        a["R"] = len(reps)
        if bootstrap:
            a.update(bootstrap_se(Ytrue, Ypreds, B=200))
        res["A"] = a
        res["rowf1_A"] = rf.tolist()
        res["rows_A"] = rows.tolist()
    # ---- B ----  (one or several chronological cuts; B = mean over cuts, B_se = mean per-cut SE
    #               because the cuts share most rows and are highly correlated)
    Bs = [v for v in per.values() if v[0]["protocol"] == "B"]
    if Bs:
        bm = []
        for f, yp, pi in Bs:
            Yt = D.Y[f["eval"]]
            m = metrics(Yt, yp)
            if bootstrap:
                m.update(bootstrap_se(Yt, [yp], B=200))
            m["cut"] = f["cut"]
            if pi is not None:
                m["sld_prior_labels"] = (pi @ D.CM).round(3).tolist()
            m["true_prior_labels"] = Yt.mean(0).round(3).tolist()
            bm.append(m)
            res.setdefault("rowf1_B", []).extend(row_f1(Yt, yp).tolist())
        res["B_ncuts"] = len(bm)
        if len(bm) == 1:
            res["B"] = bm[0]
        else:
            agg = {k: float(np.mean([m[k] for m in bm])) for k in bm[0]
                   if isinstance(bm[0][k], (int, float)) and k != "cut"}
            agg["cuts"] = bm
            res["B"] = agg
        for m in bm:
            if m["cut"] == CHRONO_CUT:
                res["B2005"] = {k: v for k, v in m.items()}
    # ---- C ----
    Cs = [v for v in per.values() if v[0]["protocol"] == "C"]
    if Cs:
        f, yp, pi = Cs[0]
        Yt = D.Y[f["eval"]]
        m = metrics(Yt, yp)
        if bootstrap:
            m.update(bootstrap_se(Yt, [yp], B=200))
        if pi is not None:
            m["sld_prior_labels"] = (pi @ D.CM).round(3).tolist()
        res["C"] = m
        res["rowf1_C"] = row_f1(Yt, yp).tolist()
    # ---- robust objective ----
    if "A" in res and "B" in res:
        res["J"] = 0.5 * res["A"]["sample_f1"] + 0.5 * res["B"]["sample_f1"]
        res["J_se"] = 0.5 * float(np.hypot(res["A"]["sample_f1_se"], res["B"]["sample_f1_se"]))
    res["secs"] = float(sum(float(o["secs"]) for o in outs))
    res["T_mean"] = float(np.mean([np.mean(o["T"]) for o in outs]))
    return res


def evaluate(cfg, D, protocols=("A", "B", "C"), R=4, n_jobs=1, seed=0, use_cache=True, rows=None,
             bootstrap=True, chrono_cuts=CHRONO_CUTS):
    validate(cfg)
    folds = make_folds(D, protocols, R, seed, rows=rows, chrono_cuts=chrono_cuts)
    outs = fold_outputs(cfg, D, folds, n_jobs, use_cache)
    return score(cfg, D, folds, outs, bootstrap=bootstrap)


def summary_row(name, res):
    r = {"name": name}
    for p in ("A", "B", "B2005", "C"):
        if p in res:
            r[f"{p}"] = res[p]["sample_f1"]
            r[f"{p}_se"] = res[p]["sample_f1_se"]
            r[f"{p}_macro"] = res[p]["macro_f1"]
    if "J" in res:
        r["J"] = res["J"]
        r["J_se"] = res["J_se"]
    r["secs"] = res.get("secs")
    return r


def paired_se(res1, res2, w=(0.5, 0.5)):
    """SE of the difference in J between two configs using per-row paired differences."""
    dA = np.asarray(res1["rowf1_A"]) - np.asarray(res2["rowf1_A"])
    dB = np.asarray(res1["rowf1_B"]) - np.asarray(res2["rowf1_B"])
    nB = len(dB) / res1.get("B_ncuts", 1)          # cuts overlap: effective n ~ one cut
    return float(np.sqrt((w[0] ** 2) * dA.var(ddof=1) / len(dA) + (w[1] ** 2) * dB.var(ddof=1) / nB))
