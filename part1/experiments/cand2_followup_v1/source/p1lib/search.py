"""Optuna TPE search over the composed config space, robust objective, 1-SE selection and the
nested search-overfitting check.

Objective  J = 0.5 * A + 0.5 * B   (A = repeated stratified CV sample-F1, B = mean chronological
holdout sample-F1 over cuts 2004/2005/2006).  C (official val) is logged, never optimized.
Selection  = one-standard-error rule on J, then the SIMPLEST config (fewest active components),
then a guard: B must not be worse than the plain-LR baseline's B by more than one SE.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
import time
import traceback

import numpy as np

from .evaluate import evaluate, make_folds, fold_outputs, score, CHRONO_CUTS
from .pipeline import validate

GRAPH_SPACE = {"scale": True, "pca": 64, "l2": True}


# ============================================================================ search space
def suggest_config(t, allow_blocks=True, allow_stack=False, require_unsup=False):
    """Define-by-run space.  Returns a full experiment config."""
    pca = t.suggest_categorical("pca", [0, 32, 64, 96])
    l2 = t.suggest_categorical("l2", [False, True])
    feats = {"scale": True, "pca": pca or None, "l2": l2, "blocks": {}}
    if allow_blocks:
        if t.suggest_categorical("blk_criteria", [False, True]):
            feats["blocks"]["criteria"] = {"d": 16}
        if t.suggest_categorical("blk_drug", [False, True]):
            feats["blocks"]["pathways"] = {"d": 16, "max_level": 2}
            feats["blocks"]["drugmeta"] = {}

    prior = t.suggest_categorical("prior", ["none", "sld"])
    head = {"prior": prior, "decode": t.suggest_categorical("decode", ["map", "expf1"])}
    if prior == "sld":
        head["sld_a"] = t.suggest_categorical("sld_a", [0, 25, 100, 400])
        head["sld_floor"] = 0.25

    clf_t = t.suggest_categorical("clf", ["lr", "ovr", "lda", "knn"] if require_unsup else ["lr", "ovr", "lda", "knn", "cs"])
    temperature = t.suggest_categorical("temperature", [False, True])
    if clf_t == "cs":       # Correct & Smooth as the predictor (T2); base = powerset LR
        gk = t.suggest_categorical("cs_graph_k", [5, 10, 20])
        lcfg = {"type": "cs", "graph": {"space": GRAPH_SPACE, "k": gk, "s": 1.0, "sym": "max"},
                "a1": t.suggest_categorical("cs_a1", [0.0, 0.5, 0.8]),
                "a2": t.suggest_float("cs_a2", 0.1, 0.9),
                "base": {"type": "lr", "C": t.suggest_float("C", 0.01, 3.0, log=True)},
                "base_features": {"scale": True, "pca": pca or None}}
        model = {"features": None, "synth": [], "labeler": None,
                 "clf": {"type": "direct", "labeler": lcfg}, "temperature": temperature}
        return {"model": model, "head": head, "transductive": True}

    clf = {"type": clf_t}
    if clf_t in ("lr", "ovr"):
        clf["C"] = t.suggest_float("C", 0.01, 3.0, log=True)
    if clf_t == "knn":
        clf.update(k=t.suggest_categorical("knn_k", [3, 5, 10, 20]), weights="distance",
                   tau=t.suggest_categorical("knn_tau", [0.05, 0.1, 0.3]), beta=1.0)
    if prior != "sld" and clf_t in ("lr", "ovr", "lda"):
        if t.suggest_categorical("class_weight", ["none", "balanced"]) == "balanced":
            clf["class_weight"] = "balanced"

    synth = []
    if t.suggest_categorical("syn_gmm", [False, True]):
        synth.append({"type": "gmm", "k": t.suggest_categorical("syn_gmm_k", [8, 16, 32]), "cov": "diag",
                      "space": {"scale": True, "pca": 32, "l2": True}})
    if t.suggest_categorical("syn_kmeans", [False, True]):
        synth.append({"type": "kmeans", "k": t.suggest_categorical("syn_kmeans_k", [8, 16, 32]),
                      "out": t.suggest_categorical("syn_kmeans_out", ["dist", "soft"])})
    if t.suggest_categorical("syn_knn", [False, True]):
        synth.append({"type": "knn_vote", "k": t.suggest_categorical("syn_knn_k", [5, 10, 20])})
    if t.suggest_categorical("syn_graph", [False, True]):
        synth.append({"type": "graph", "mode": t.suggest_categorical("syn_graph_mode", ["spread", "cs"]),
                      "graph": {"space": GRAPH_SPACE, "k": 10, "s": 1.0, "sym": "max"},
                      "alpha": t.suggest_float("syn_graph_alpha", 0.2, 0.9)})
    if t.suggest_categorical("syn_clmap", [False, True]):
        synth.append({"type": "cluster_map", "algo": "kmeans", "k": t.suggest_categorical("syn_clmap_k", [8, 16, 32]), "m": 4})
    if t.suggest_categorical("syn_oof", [False, True]):
        synth.append({"type": "oof_model", "model": {"features": {"scale": True}, "clf": {"type": "lda"}, "temperature": True}})

    lab_t = t.suggest_categorical("labeler", ["cluster", "spread", "cs", "selftrain"] if require_unsup else
                                  ["none", "cluster", "spread", "cs", "sld", "selftrain"])
    labeler = None
    if lab_t != "none":
        lam = t.suggest_float("lam", 0.03, 0.5, log=True)
        if lab_t == "cluster":
            labeler = {"type": "cluster", "algo": t.suggest_categorical("cl_algo", ["kmeans", "gmm", "seeded_kmeans"]),
                       "k": t.suggest_categorical("cl_k", [8, 16, 24, 32, 48]), "m": t.suggest_categorical("cl_m", [1, 4]),
                       "lam": lam}
        elif lab_t in ("spread", "cs"):
            g = {"space": GRAPH_SPACE, "k": t.suggest_categorical("lab_graph_k", [5, 10, 20]), "s": 1.0, "sym": "max"}
            labeler = {"type": lab_t, "graph": g, "lam": lam}
            if lab_t == "spread":
                labeler["alpha"] = t.suggest_float("lab_alpha", 0.2, 0.9)
            else:
                labeler.update(a1=t.suggest_categorical("lab_a1", [0.0, 0.5, 0.8]), a2=t.suggest_float("lab_a2", 0.2, 0.9))
        elif lab_t == "sld":
            labeler = {"type": "sld", "base": {"type": "lr", "C": 0.1}, "a": t.suggest_categorical("lab_sld_a", [25, 100, 400]),
                       "lam": lam}
        elif lab_t == "selftrain":
            labeler = {"type": "selftrain", "lam": lam,
                       "members": t.suggest_categorical("st_members", ["lr", "lr+lda+cs"]).split("+"),
                       "rounds": t.suggest_categorical("st_rounds", [1, 2, 3]),
                       "tau": t.suggest_categorical("st_tau", [0.6, 0.75, 0.9]),
                       "align": t.suggest_categorical("st_align", ["none", "sld"]) if clf.get("class_weight") != "balanced" else "none",
                       "weighting": t.suggest_categorical("st_weighting", ["hard", "softmatch"]),
                       "ramp": t.suggest_categorical("st_ramp", [False, True]),
                       "cluster_filter": t.suggest_categorical("st_cluster_filter", [False, True])}
            if require_unsup and "cs" not in labeler["members"]:
                labeler["cluster_filter"] = True
        if lab_t != "selftrain" and t.suggest_categorical("lab_hard", [False, True]):
            labeler["hard"] = True
            labeler["tau"] = t.suggest_categorical("lab_tau", [0.5, 0.7, 0.9])
    model = {"features": feats, "synth": synth, "labeler": labeler, "clf": clf, "temperature": temperature}
    return {"model": model, "head": head, "transductive": True}


# ============================================================================ complexity
CLF_COST = {"lr": 0.0, "lda": 0.0, "ovr": 0.25, "nb": 0.25, "knn": 0.5, "direct": 1.0, "hgb": 2.0}


def complexity(cfg) -> float:
    """Number of active components (+ small costs for extra knobs).  Lower = simpler."""
    m = cfg["model"]
    singles = m["ensemble"] if "ensemble" in m else [m]
    c = 0.0
    for s in singles:
        c += len(s.get("synth") or [])
        for sy in s.get("synth") or []:
            if sy["type"] == "oof_model":
                c += complexity({"model": sy["model"], "head": {}})
        lab = s.get("labeler")
        if lab:
            c += 1.0 + (0.5 * (len(lab.get("members", [])) > 1) + 0.25 * (lab.get("rounds", 1) - 1)
                        if lab["type"] == "selftrain" else 0.0)
        f = s.get("features") or {}
        c += len(f.get("blocks") or {})
        clf = s["clf"]
        c += CLF_COST.get(clf["type"], 1.0)
        if clf.get("class_weight") == "balanced":
            c += 0.5
        if s.get("temperature"):
            c += 0.25
    if "ensemble" in m:
        c += len(singles) - 1
    if cfg.get("head", {}).get("prior") == "sld":
        c += 0.5
    return c


def unsup_labeling(cfg) -> bool:
    """Assignment rule: an UNSUPERVISED method must take part in filling the missing labels.
    True if the label-filling path uses clustering or a similarity graph: a cluster / label-
    spreading / C&S labeler, or guarded self-training with the K-means consistency filter or a
    C&S member, or a direct cluster/graph predictor."""
    m = cfg["model"]
    for s in (m["ensemble"] if "ensemble" in m else [m]):
        lab = s.get("labeler") or {}
        if lab.get("type") in ("cluster", "spread", "cs"):
            return True
        if lab.get("type") == "selftrain" and (lab.get("cluster_filter") or "cs" in lab.get("members", [])):
            return True
        if s["clf"]["type"] == "direct" and s["clf"]["labeler"]["type"] in ("cluster", "spread", "cs"):
            return True
    return False


# ============================================================================ objective
def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(rec, default=float) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def make_objective(D, log_path, R=4, rows=None, chrono_cuts=CHRONO_CUTS, prune_margin=0.05,
                   allow_blocks=True, cv_seed=0, tag="main", require_unsup=False):
    import optuna

    def objective(trial):
        cfg = suggest_config(trial, allow_blocks=allow_blocks, require_unsup=require_unsup)
        validate(cfg)
        t0 = time.time()
        # stage 1: chrono cuts + val + first CV repeat; prune clearly bad configs
        folds1 = make_folds(D, ("A", "B", "C"), R=R, seed=cv_seed, rows=rows, chrono_cuts=chrono_cuts)
        first = [f for f in folds1 if f["protocol"] != "A" or f["repeat"] == 0]
        rest = [f for f in folds1 if f["protocol"] == "A" and f["repeat"] > 0]
        try:
            outs1 = fold_outputs(cfg, D, first, n_jobs=1)
            r1 = score(cfg, D, first, outs1, bootstrap=False)
            j1 = r1.get("J", np.nan)
            trial.set_user_attr("J_stage1", float(j1))
            try:
                best = trial.study.best_value
            except ValueError:
                best = -np.inf
            if np.isfinite(best) and j1 < best - prune_margin:
                append_jsonl(log_path, {"tag": tag, "trial": trial.number, "state": "pruned", "J_stage1": j1,
                                        "params": trial.params, "config": cfg, "secs": time.time() - t0})
                raise optuna.TrialPruned()
            outs2 = fold_outputs(cfg, D, rest, n_jobs=1)
            res = score(cfg, D, first + rest, outs1 + outs2, bootstrap=False)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            append_jsonl(log_path, {"tag": tag, "trial": trial.number, "state": "fail", "error": repr(e),
                                    "tb": traceback.format_exc()[-2000:], "params": trial.params, "config": cfg})
            raise
        cx = complexity(cfg)
        attrs = {"A": res["A"]["sample_f1"], "A_se": res["A"]["sample_f1_se"], "A_macro": res["A"]["macro_f1"],
                 "B": res["B"]["sample_f1"], "B_se": res["B"]["sample_f1_se"], "B_macro": res["B"]["macro_f1"],
                 "J": res["J"], "J_se": res["J_se"], "complexity": cx, "secs": time.time() - t0}
        if "B2005" in res:
            attrs["B2005"] = res["B2005"]["sample_f1"]
        if "C" in res:
            attrs.update(C=res["C"]["sample_f1"], C_se=res["C"]["sample_f1_se"], C_macro=res["C"]["macro_f1"])
        for k, v in attrs.items():
            trial.set_user_attr(k, float(v))
        trial.set_user_attr("config", json.dumps(cfg))
        append_jsonl(log_path, {"tag": tag, "trial": trial.number, "state": "complete", **attrs, "params": trial.params,
                                "config": cfg, "rowf1_A": res["rowf1_A"], "rowf1_B": res["rowf1_B"],
                                "rowf1_C": res.get("rowf1_C"), "B_ncuts": res.get("B_ncuts", 1)})
        return res["J"]

    return objective


def run_worker(storage_path, study_name, n_trials, log_path, seed, R=4, rows=None, chrono_cuts=CHRONO_CUTS,
               allow_blocks=True, tag="main", timeout=None, sampler_name="tpe", cv_seed=0, require_unsup=False):
    """One search worker process (shares the study through a journal file)."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from .data import load
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    D = load()
    storage = JournalStorage(JournalFileBackend(storage_path))
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True,
                                             n_startup_trials=40)
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                                sampler=sampler, load_if_exists=True)
    obj = make_objective(D, log_path, R=R, rows=rows, chrono_cuts=chrono_cuts, allow_blocks=allow_blocks, tag=tag,
                         cv_seed=cv_seed, prune_margin=(np.inf if sampler_name == "random" else 0.05),
                         require_unsup=require_unsup)
    study.optimize(obj, n_trials=n_trials, timeout=timeout, catch=(Exception,))


def launch(storage_path, study_name, n_workers, trials_per_worker, log_path, R=4, rows=None,
           chrono_cuts=CHRONO_CUTS, allow_blocks=True, tag="main", timeout=None, base_seed=0, enqueue=None,
           sampler_name="tpe", cv_seed=0, require_unsup=False):
    """Run n_workers processes in parallel (spawn) against one journal-backed study."""
    import multiprocessing as mp
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    storage = JournalStorage(JournalFileBackend(storage_path))
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
    for params in enqueue or []:
        study.enqueue_trial(params, skip_if_exists=True)
    ctx = mp.get_context("spawn")
    procs = []
    for w in range(n_workers):
        p = ctx.Process(target=run_worker, args=(storage_path, study_name, trials_per_worker, log_path,
                                                 base_seed + w, R, None if rows is None else list(map(int, rows)),
                                                 chrono_cuts, allow_blocks, tag, timeout, sampler_name, cv_seed,
                                                 require_unsup))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    return study


# ============================================================================ selection
def load_log(log_path, tag=None, state="complete"):
    recs = []
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            if (tag is None or r.get("tag") == tag) and (state is None or r.get("state") == state):
                recs.append(r)
    return recs


def select_one_se(recs, baseline=None, key="J", se_key="J_se"):
    """1-SE rule: candidates with J >= J_best - SE(J_best); choose minimal complexity, ties by J.
    Guard: if a baseline record is given, the choice must have B >= B_base - B_se_base."""
    if not recs:
        return None, []
    best = max(recs, key=lambda r: r[key])
    thr = best[key] - best[se_key]
    cands = [r for r in recs if r[key] >= thr]
    cands.sort(key=lambda r: (r["complexity"], -r[key]))
    for r in cands:
        if baseline is None or r["B"] >= baseline["B"] - baseline["B_se"]:
            return r, cands
    return cands[0], cands


def dedupe(recs):
    """Keep one record per distinct config (TPE re-samples identical categorical configs)."""
    seen = {}
    for r in recs:
        k = json.dumps(r["config"], sort_keys=True)
        if k not in seen or r[ "J"] > seen[k]["J"]:
            seen[k] = r
    return list(seen.values())
