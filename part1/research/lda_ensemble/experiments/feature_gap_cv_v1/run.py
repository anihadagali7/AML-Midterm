"""Frozen feature-gap comparison. No Part 2 or Kaggle test inputs are loaded.

Commands: prepare, parity, screen, shortlist, confirm, supplementary, report, verify.
All outputs stay beside this file; historical recipes and results are read-only.
"""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

import features as m

OUT = Path(__file__).resolve().parent
PART1, ROOT = m.PART1, m.RESEARCH
HISTORY = ROOT / "experiments/ani_improvement_v1"
CONTROL_NAMES = {"ani": "Ani frozen control", "ensemble": "Ani + tri/SVM 15% each"}
SPLITS = {
    "train3": "benchmark_v1/protocol.json",
    "pooled3": "pooled_benchmark_v1/protocol.json",
    "train5": "cross_repo_benchmark_v1/confirmation8/train/protocol.json",
    "pooled5": "cross_repo_benchmark_v1/confirmation8/pooled/protocol.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def decide(scores):
    result = (scores >= .5).astype(int)
    empty = result.sum(axis=1) == 0
    result[np.flatnonzero(empty), scores[empty].argmax(axis=1)] = 1
    return result


def sample_values(truth, prediction):
    return 2 * (truth * prediction).sum(axis=1) / (truth.sum(axis=1) + prediction.sum(axis=1))


def metrics(truth, scores):
    prediction = decide(scores)
    p, r, f, s = precision_recall_fscore_support(truth, prediction, average=None, zero_division=0)
    return dict(sample_f1=float(f1_score(truth, prediction, average="samples", zero_division=0)),
                macro_f1=float(f.mean()), reasons={label: dict(precision=float(p[j]),
                recall=float(r[j]), f1=float(f[j]), support=int(s[j])) for j, label in enumerate(m.LABELS)})


def source_paths():
    paths = [OUT / "run.py", OUT / "features.py", OUT / "checks.py", PART1 / "TESTING_POLICY.md"]
    paths += list(ROOT.glob("*.py"))
    paths += [PART1 / "data" / name for name in ["part1_train.csv", "part1_val.csv", "pathway_catalog.csv"]]
    paths += [ROOT / "experiments" / value for value in SPLITS.values()]
    paths += [ROOT / "experiments/cross_repo_benchmark_v1/protocol.json"]
    paths += [PART1 / "experiments/feature_reassessment_v1/findings.json"]
    for stage in SPLITS:
        folder = HISTORY / stage
        paths.append(folder / "index.json")
        for row in read(folder / "index.json"):
            if row["candidate"] in CONTROL_NAMES.values():
                paths += [folder / (row["result_key"] + suffix) for suffix in [".npz", ".json"]]
    paths += [HISTORY / "supplementary/results.json"]
    for row in read(HISTORY / "supplementary/results.json"):
        if row["candidate"] in CONTROL_NAMES.values():
            paths.append(ROOT / row["prediction_file"])
    paths += [p for p in (PART1 / "submissions").rglob("*") if p.is_file()]
    return sorted(set(paths))


def hashes():
    return {str(path.relative_to(PART1)): sha(path) for path in source_paths()}


def prepare():
    if (OUT / "protocol.json").exists():
        guard()
        return
    registry = [dict(name=f"{parent}__{block}", parent=parent, block=block,
                     description=description, decoder="threshold .5 then nonempty argmax")
                for block, description in m.BLOCKS.items() for parent in CONTROL_NAMES]
    save(OUT / "registry.json", registry)
    save(OUT / "protocol.json", dict(
        version="feature-gap-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=hashes(), registry_sha256=sha(OUT / "registry.json"),
        inputs="Part 1 only. No Part 2 fields, external data, ID predictors, test fitting, or test predictions.",
        hypotheses=m.BLOCKS, candidates=len(registry),
        parent="Ani inductive and 70/15/15 ensemble. Modify only Ani's final multinomial LR.",
        fixed_recipe="Original scaler/PCA/GMM, two embedding-only pseudo-label rounds, augmented targets, native class weighting, C=.3, kNN and decoder unchanged. Append one metadata block to final LR. No metadata enters pseudo-label generation or kNN.",
        feature_scaling="Metadata RMS row norm on permitted fitting pool scaled to .3 times embedding RMS row norm. Numeric structure columns standardized on that same pool. Fixed weight, no grid.",
        feature_details="Text: word unigrams/bigrams, min_df3, max_df.95, max2000, sublinear TF, L2 normalization, no stop list; append empty-view indicator. Categories: parsed exact sets, train min_count3, missing flag. ICD full block includes category and full-code namespaces. Structure: log counts and section flags; no clinical numeric-bound inference.",
        metadata_diagnostic="Each block also gets five gold-only balanced binary LR heads, C=.3, cutoff .5 plus fallback. Diagnostics never qualify for promotion or select weights.",
        folds=SPLITS, initial="Three frozen repeats in BOTH pools, all candidates, grouped purge of masked duplicates.",
        shortlist="Positive paired means and >=2/3 wins in both pools vs exact parent; at most four recipes ordered by minimum paired gain, then name. If none qualify, at most two diagnostics with minimum gain>=-.003 and positive gain in either pool; marked diagnostic.",
        confirmation="Five additional frozen repeats for shortlisted recipes. Combine all eight; require positive paired mean and >=5/8 wins in BOTH pools vs exact parent. Compare to both named references as well.",
        supplementary="After eight-repeat results, freeze at most three confirmed recipes ordered by minimum gain. If none confirm, evaluate at most two shortlisted diagnostic recipes, explicitly labeled. Original validation and three frozen chronological windows; no subsequent tuning.",
        control_reuse="Refit Ani and match saved fold scores at atol1e-10. Ensemble uses original ensemble score + .7*(new Ani - original Ani); tri/SVM contributions stay identical. Saved references, folds and code hashes protected.",
        error_analysis="Per-label metrics, per-trial gains/losses, actual Other-positive cohort and the 26 previously identified shared persistent errors. These retrospective groups never affect fitting or advancement.",
        limitations="Reused development labels; multiple hypotheses and selection optimism. Input representations and fixed weight are bounded tests, not exhaustive feature judgments.",
        submissions_created=False, full235_refit=False,
    ))
    print("Frozen nine feature blocks, 18 parent-specific additions and nine metadata diagnostics.", flush=True)


def guard():
    protocol = read(OUT / "protocol.json")
    assert protocol["source_sha256"] == hashes(), "Frozen source or protected input changed"
    assert protocol["registry_sha256"] == sha(OUT / "registry.json")
    return protocol


def frame_for(kind):
    train = pd.read_csv(PART1 / "data/part1_train.csv")
    if kind == "pooled":
        train = pd.concat([train, pd.read_csv(PART1 / "data/part1_val.csv")], ignore_index=True)
    return train


def plan(stage, frame):
    protocol = read(ROOT / "experiments" / SPLITS[stage])
    gold_positions = np.flatnonzero(frame[m.LABELS].ge(0).all(axis=1))
    gold = frame.iloc[gold_positions]
    groups = pd.util.hash_pandas_object(frame[m.EMB].round(10), index=False).to_numpy()
    masked = np.flatnonzero(frame[m.LABELS].eq(-100).all(axis=1))
    assert protocol["gold_ids"] == gold.ID.tolist()
    for seed in protocol["seeds"]:
        folds = [f for f in protocol["folds"] if f["seed"] == seed]
        assert len(folds) == 5
        assert sorted(h for fold in folds for h in fold["heldout_gold_positions"]) == list(range(len(gold)))
        for fold in folds:
            a, h = np.array(fold["train_gold_positions"]), np.array(fold["heldout_gold_positions"])
            assert not set(a) & set(h) and set(a) | set(h) == set(range(len(gold)))
            assert not set(groups[gold_positions[a]]) & set(groups[gold_positions[h]])
            assert fold["heldout_ids"] == gold.iloc[h].ID.tolist()
            excluded = np.isin(groups[masked], groups[gold_positions[h]])
            assert int(excluded.sum()) == fold["masked_duplicates_excluded"]
    return protocol, gold_positions, groups, masked


def controls(stage, gold):
    result = {}
    for key, name in CONTROL_NAMES.items():
        folder = HISTORY / stage
        row = next(r for r in read(folder / "index.json") if r["candidate"] == name)
        with np.load(folder / (row["result_key"] + ".npz"), allow_pickle=True) as saved:
            assert saved["ids"].tolist() == gold.ID.tolist()
            np.testing.assert_array_equal(saved["truth"], gold[m.LABELS].to_numpy())
            result[key] = saved["scores"].copy()
            for i, scores in enumerate(result[key]):
                np.testing.assert_array_equal(decide(scores), saved["predictions"][i])
                assert abs(metrics(saved["truth"], scores)["sample_f1"] - row["repeat_scores"][i]["sample_f1"]) < 1e-12
    return result


def fold_inputs(frame, positions, groups, masked, fold):
    a, h = np.array(fold["train_gold_positions"]), np.array(fold["heldout_gold_positions"])
    u = masked[~np.isin(groups[masked], groups[positions[h]])]
    fitting = np.sort(np.r_[positions[a], u])
    fit = frame.iloc[fitting].reset_index(drop=True)
    query = frame.iloc[positions[h]].drop(columns=m.LABELS).reset_index(drop=True)
    assert not set(fit.ID) & set(query.ID)
    assert not set(groups[fitting]) & set(groups[positions[h]])
    return fit, query, h


def parity():
    guard()
    from checks import check_parsers, check_boundary
    check_parsers()
    frame = frame_for("pooled")
    p, positions, groups, masked = plan("pooled3", frame)
    fit, query, h = fold_inputs(frame, positions, groups, masked, p["folds"][0])
    refs = controls("pooled3", frame.iloc[positions])
    packet = m.train_native(fit)
    pred = m.native_prediction(packet, query)
    np.testing.assert_allclose(pred, refs["ani"][0, h], atol=1e-10, rtol=0)
    boundary = check_boundary(packet, fit, query)
    save(OUT / "parity.json", dict(passed=True, source_sha256=sha(OUT / "protocol.json"),
         max_control_difference=float(np.abs(pred - refs["ani"][0, h]).max()), **boundary))
    print("Native-control parity, structured parsers and inductive feature-boundary checks passed.", flush=True)


def run_stage(stage, blocks):
    guard()
    assert read(OUT / "parity.json")["passed"]
    folder = OUT / stage
    folder.mkdir(exist_ok=True)
    frame = frame_for("pooled" if stage.startswith("pooled") else "train")
    p, positions, groups, masked = plan(stage, frame)
    gold = frame.iloc[positions]
    refs = controls(stage, gold)
    values = {key: score.copy() for key, score in refs.items()}
    for block in blocks:
        for parent in [*CONTROL_NAMES, "metadata"]:
            values[f"{parent}__{block}"] = np.full_like(refs["ani"], np.nan)
    audit = []
    for ri, seed in enumerate(p["seeds"]):
        for fold in [f for f in p["folds"] if f["seed"] == seed]:
            started = time.monotonic()
            fit, query, h = fold_inputs(frame, positions, groups, masked, fold)
            filename = f"seed{seed}_fold{fold['fold']}"
            cached = folder / (filename + ".npz")
            if cached.exists():
                details = read(folder / (filename + ".json"))
                assert details["protocol_sha256"] == sha(OUT / "protocol.json")
                assert details["blocks"] == blocks
                with np.load(cached) as saved:
                    assert saved["ids"].tolist() == query.ID.tolist()
                    for key in values:
                        if key in saved:
                            values[key][ri, h] = saved[key]
                audit.append(details)
                continue
            packet = m.train_native(fit)
            original = m.native_prediction(packet, query)
            np.testing.assert_allclose(original, refs["ani"][ri, h], atol=1e-10, rtol=0)
            predictions = {"ids": query.ID.to_numpy(dtype=str), "ani": original,
                           "ensemble": refs["ensemble"][ri, h]}
            features_audit = []
            for block in blocks:
                new, meta, detail = m.fit_addition(packet, fit, query, block)
                predictions[f"ani__{block}"] = new
                predictions[f"ensemble__{block}"] = np.clip(refs["ensemble"][ri, h] + .7 * (new - original), 0, 1)
                predictions[f"metadata__{block}"] = meta
                features_audit.append(detail)
            for key in values:
                assert np.isfinite(predictions[key]).all()
                values[key][ri, h] = predictions[key]
            details = dict(protocol_sha256=sha(OUT / "protocol.json"), stage=stage,
                seed=seed, fold=fold["fold"], blocks=blocks, fit_ids=fit.ID.tolist(),
                query_ids=query.ID.tolist(), gold_rows=len(packet["gold"]),
                pseudo_rows_by_round=packet["history"], features=features_audit,
                control_max_difference=float(np.abs(original - refs["ani"][ri, h]).max()),
                group_separation=True, heldout_labels_removed=True, seconds=time.monotonic() - started)
            np.savez_compressed(cached, **predictions)
            save(folder / (filename + ".json"), details)
            audit.append(details)
            print(json.dumps(dict(stage=stage, seed=seed, fold=fold["fold"], blocks=len(blocks),
                                  seconds=round(details["seconds"], 1))), flush=True)
    rows = []
    truth = gold[m.LABELS].to_numpy(int)
    for key, scores in values.items():
        assert np.isfinite(scores).all()
        repeats = [metrics(truth, score) for score in scores]
        array = np.array([r["sample_f1"] for r in repeats])
        rows.append(dict(candidate=key, mean=float(array.mean()), sd=float(array.std()), repeat_scores=repeats))
        np.savez_compressed(folder / (key + ".npz"), scores=scores,
                            predictions=np.array([decide(score) for score in scores]),
                            ids=gold.ID.to_numpy(dtype=str), truth=truth)
    save(folder / "index.json", rows)
    save(folder / "audit.json", dict(folds=len(audit), all_control_parity=True, all_group_separation=True))
    print(pd.DataFrame([dict(candidate=r["candidate"], mean=r["mean"]) for r in rows]).to_string(index=False), flush=True)
    guard()


def comparisons(extra=False):
    names = read(OUT / "shortlist.json")["names"] if extra else [r["name"] for r in read(OUT / "registry.json")]
    result = []
    for name in names:
        parent, block = name.split("__", 1)
        row = dict(candidate=name, parent=parent, block=block)
        for kind in ["train", "pooled"]:
            stages = [kind + "3", kind + "5"] if extra else [kind + "3"]
            collected = {}
            for key in [name, *CONTROL_NAMES]:
                repeats = []
                for stage in stages:
                    entry = next(r for r in read(OUT / stage / "index.json") if r["candidate"] == key)
                    repeats += [r["sample_f1"] for r in entry["repeat_scores"]]
                collected[key] = np.array(repeats)
            values = collected[name]
            row.update({kind + "_mean": float(values.mean()), kind + "_sd": float(values.std()),
                        kind + "_repeats": values.tolist()})
            for reference in CONTROL_NAMES:
                delta = values - collected[reference]
                row.update({f"{kind}_vs_{reference}_delta": float(delta.mean()),
                            f"{kind}_vs_{reference}_wins": int((delta > 1e-12).sum()),
                            f"{kind}_vs_{reference}_ties": int((np.abs(delta) <= 1e-12).sum()),
                            f"{kind}_vs_{reference}_deltas": delta.tolist()})
        row["minimum_parent_gain"] = min(row[f"{kind}_vs_{parent}_delta"] for kind in ["train", "pooled"])
        row["qualifies"] = bool(row["minimum_parent_gain"] > 1e-12 and all(
            row[f"{kind}_vs_{parent}_wins"] >= (5 if extra else 2) for kind in ["train", "pooled"]))
        result.append(row)
    return sorted(result, key=lambda r: (-r["minimum_parent_gain"], r["candidate"]))


def shortlist():
    guard()
    rows = comparisons()
    passed = [r for r in rows if r["qualifies"]][:4]
    diagnostic = not bool(passed)
    if diagnostic:
        passed = [r for r in rows if r["minimum_parent_gain"] >= -.003 and max(
            r[f"{kind}_vs_{r['parent']}_delta"] for kind in ["train", "pooled"]) > 1e-12][:2]
    record = dict(names=[r["candidate"] for r in passed], diagnostic_only=diagnostic,
                  selection_rule=read(OUT / "protocol.json")["shortlist"])
    if (OUT / "shortlist.json").exists():
        assert read(OUT / "shortlist.json") == record
    else:
        save(OUT / "shortlist.json", record)
    save(OUT / "screen_comparison.json", rows)
    pd.DataFrame(rows).to_csv(OUT / "screen_comparison.csv", index=False)
    print(json.dumps(record), flush=True)


def confirm():
    guard()
    blocks = sorted({name.split("__", 1)[1] for name in read(OUT / "shortlist.json")["names"]})
    if not blocks:
        save(OUT / "confirmation.json", dict(comparison=[], supplementary_names=[], diagnostic_only=True))
        return
    for stage in ["train5", "pooled5"]:
        run_stage(stage, blocks)
    rows = comparisons(extra=True)
    selected = [r for r in rows if r["qualifies"]][:3]
    diagnostic = not bool(selected)
    if diagnostic:
        selected = rows[:2]
    record = dict(comparison=rows, supplementary_names=[r["candidate"] for r in selected], diagnostic_only=diagnostic)
    if (OUT / "confirmation.json").exists():
        assert read(OUT / "confirmation.json") == record
    else:
        save(OUT / "confirmation.json", record)
    pd.DataFrame(rows).to_csv(OUT / "confirmation.csv", index=False)
    print(json.dumps(record), flush=True)


def supplementary():
    guard()
    names = read(OUT / "confirmation.json")["supplementary_names"]
    if not names:
        save(OUT / "supplementary/results.json", [])
        return
    blocks = sorted({name.split("__", 1)[1] for name in names})
    train = frame_for("train")
    full = frame_for("pooled").set_index("ID", drop=False)
    val = pd.read_csv(PART1 / "data/part1_val.csv")
    windows = [dict(name="original_validation", training_ids=train.ID.tolist(), heldout_ids=val.ID.tolist())]
    windows += read(ROOT / "experiments/cross_repo_benchmark_v1/protocol.json")["chronological_splits"]
    old = read(HISTORY / "supplementary/results.json")
    rows = []
    for wi, window in enumerate(windows):
        fit = full.loc[window["training_ids"]].reset_index(drop=True)
        held = full.loc[window["heldout_ids"]].reset_index(drop=True)
        query = held.drop(columns=m.LABELS)
        assert not set(fit.ID) & set(query.ID)
        if wi:
            fg = pd.util.hash_pandas_object(fit[m.EMB].round(10), index=False)
            hg = pd.util.hash_pandas_object(query[m.EMB].round(10), index=False)
            assert not set(fg) & set(hg)
        refs = {}
        for key, name in CONTROL_NAMES.items():
            source = next(r for r in old if r["candidate"] == name and r["window"] == wi)
            frame = pd.read_csv(ROOT / source["prediction_file"]).set_index("ID")
            assert set(frame.index) == set(query.ID)
            refs[key] = frame.loc[query.ID, [f"score_{label}" for label in m.LABELS]].to_numpy()
        packet = m.train_native(fit)
        original = m.native_prediction(packet, query)
        np.testing.assert_allclose(original, refs["ani"], rtol=0, atol=1e-10)
        scores = dict(refs)
        for block in blocks:
            new, meta, audit = m.fit_addition(packet, fit, query, block)
            scores[f"ani__{block}"] = new
            scores[f"ensemble__{block}"] = np.clip(refs["ensemble"] + .7 * (new - original), 0, 1)
            save(OUT / "supplementary" / f"window{wi}_{block}_audit.json", audit)
        for name in [*CONTROL_NAMES, *names]:
            score = scores[name]
            truth = held[m.LABELS].to_numpy(int)
            path = OUT / "supplementary" / f"window{wi}_{name}.npz"
            np.savez_compressed(path, ids=query.ID.to_numpy(dtype=str), scores=score, truth=truth, predictions=decide(score))
            row = dict(candidate=name, window=wi, window_name=window["name"], **metrics(truth, score))
            rows.append(row)
            print(json.dumps({key: row[key] for key in ["candidate", "window", "sample_f1"]}), flush=True)
    save(OUT / "supplementary/results.json", rows)
    guard()


def report():
    guard()
    confirmed = read(OUT / "confirmation.json")
    supp = read(OUT / "supplementary/results.json")
    final = []
    for row in confirmed["comparison"]:
        item = dict(row)
        name = row["candidate"]
        if name in confirmed["supplementary_names"]:
            item["validation"] = next(r["sample_f1"] for r in supp if r["candidate"] == name and r["window"] == 0)
            arrays = [np.load(OUT / "supplementary" / f"window{wi}_{name}.npz") for wi in [1, 2, 3]]
            truth = np.concatenate([a["truth"] for a in arrays])
            score = np.concatenate([a["scores"] for a in arrays])
            assert len(truth) == 98
            item["chronological"] = metrics(truth, score)["sample_f1"]
        final.append(item)
    error_rows, label_rows, metadata_rows = [], [], []
    hard = set(read(PART1 / "experiments/feature_reassessment_v1/findings.json")["persistent_errors"]["ids"])
    chosen = set(r["name"] for r in read(OUT / "registry.json"))
    for name in sorted(chosen):
        parent = name.split("__", 1)[0]
        stages = ["pooled3", "pooled5"] if name in read(OUT / "shortlist.json")["names"] else ["pooled3"]
        chunks, originals = [], []
        for stage in stages:
            with np.load(OUT / stage / (name + ".npz")) as a, np.load(OUT / stage / (parent + ".npz")) as b:
                chunks.extend(a["predictions"])
                originals.extend(b["predictions"])
                ids, truth = a["ids"].copy(), a["truth"].copy()
        pred, control = np.array(chunks), np.array(originals)
        changes = np.array([sample_values(truth, a) - sample_values(truth, b) for a, b in zip(pred, control)])
        for i, trial_id in enumerate(ids):
            error_rows.append(dict(candidate=name, repeats=len(pred), ID=trial_id, persistent_error=trial_id in hard,
                actual_labels=" ".join(label for label, flag in zip(m.LABELS, truth[i]) if flag),
                mean_sample_f1_delta=float(changes[:, i].mean()), improved_repeats=int((changes[:, i] > 1e-12).sum()),
                worsened_repeats=int((changes[:, i] < -1e-12).sum())))
        for j, label in enumerate(m.LABELS):
            values = [precision_recall_fscore_support(truth[:, j], a[:, j], average="binary", zero_division=0)[:3] for a in pred]
            avg = np.mean(values, axis=0)
            label_rows.append(dict(candidate=name, repeats=len(pred), label=label,
                                   precision=float(avg[0]), recall=float(avg[1]), f1=float(avg[2]), support=int(truth[:, j].sum())))
    for kind in ["train", "pooled"]:
        for row in read(OUT / (kind + "3") / "index.json"):
            if row["candidate"].startswith("metadata__"):
                metadata_rows.append(dict(pool=kind, block=row["candidate"].split("__", 1)[1],
                    sample_f1=row["mean"], sd=row["sd"],
                    Other_f1=float(np.mean([r["reasons"]["Other"]["f1"] for r in row["repeat_scores"]]))))
    pd.DataFrame(error_rows).to_csv(OUT / "per_trial_changes.csv", index=False)
    pd.DataFrame(label_rows).to_csv(OUT / "per_label_metrics.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(OUT / "metadata_only.csv", index=False)
    save(OUT / "summary.json", dict(final_comparison=final, screening=read(OUT / "screen_comparison.json"),
         confirmed_recipes=[r["candidate"] for r in final if r["qualifies"]],
         protocol_sha256=sha(OUT / "protocol.json"), part1_only=True, submissions_created=False,
         diagnostic_supplementary=confirmed["diagnostic_only"]))
    pd.DataFrame(final).to_csv(OUT / "final_comparison.csv", index=False)
    def compact(rows):
        result = []
        for r in rows:
            parent = r["parent"]
            result.append({"candidate": r["candidate"], "train F1": r["train_mean"],
                "train delta": r[f"train_vs_{parent}_delta"], "train wins": r[f"train_vs_{parent}_wins"],
                "pooled F1": r["pooled_mean"], "pooled delta": r[f"pooled_vs_{parent}_delta"],
                "pooled wins": r[f"pooled_vs_{parent}_wins"], "qualifies": r["qualifies"],
                "validation": r.get("validation"), "chronology": r.get("chronological")})
        return pd.DataFrame(result).to_html(index=False, float_format=lambda v: f"{v:.6f}", border=0)
    body = '<!doctype html><meta charset="utf-8"><title>Part 1 feature gap experiments</title><style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 20px;line-height:1.5}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eee}</style><h1>Part 1: new feature structure and earlier-feature retries</h1>'
    body += '<p>Nine blocks added separately to Ani’s final classifier, evaluated alone and inside the fixed three-way ensemble. Four new representations; five retries. Pseudo-labeling, neighbors, regularization, feature weight and decoders stayed fixed. Only supplied Part 1 data were used.</p>'
    body += '<h2>Eight-repeat confirmation</h2>' + compact(final)
    body += '<p>Advancement requires positive paired means and at least five of eight repeat wins in both pools. Supplementary results marked diagnostic in summary.json do not establish a confirmed improvement. Chronology pools 98 predictions; validation uses the original 157-label fit.</p>'
    body += '<h2>Complete three-repeat screen</h2>' + compact(read(OUT / "screen_comparison.json"))
    body += '<h2>Metadata-only diagnostics</h2>' + pd.DataFrame(metadata_rows).to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0)
    body += '<p>All labels and folds are reused development data. Negative results apply to these encodings and the fixed feature weight; they do not rule out every use of a raw field. No Kaggle improvement or final model promotion is claimed.</p>'
    body += '<p><a href="protocol.json">Protocol</a> · <a href="verification.json">Verification</a> · <a href="per_trial_changes.csv">Per-trial corrections and regressions</a> · <a href="per_label_metrics.csv">Per-label results</a> · <a href="summary.json">Full results</a></p>'
    (OUT / "report.html").write_text(body)
    print(pd.DataFrame(final).drop(columns=[c for c in pd.DataFrame(final) if c.endswith(("repeats", "deltas"))]).to_string(index=False), flush=True)


def verify():
    guard()
    count, repeats, folds = 0, 0, 0
    for stage in SPLITS:
        folder = OUT / stage
        if not (folder / "index.json").exists():
            continue
        frame = frame_for("pooled" if stage.startswith("pooled") else "train")
        p, pos, groups, masked = plan(stage, frame)
        gold = frame.iloc[pos]
        for row in read(folder / "index.json"):
            with np.load(folder / (row["candidate"] + ".npz")) as a:
                assert a["ids"].tolist() == gold.ID.tolist()
                np.testing.assert_array_equal(a["truth"], gold[m.LABELS].to_numpy())
                assert len(a["scores"]) == len(p["seeds"])
                for i, score in enumerate(a["scores"]):
                    np.testing.assert_array_equal(a["predictions"][i], decide(score))
                    assert metrics(a["truth"], score) == row["repeat_scores"][i]
                    repeats += 1
                count += 1
        for fold in p["folds"]:
            fit, query, held = fold_inputs(frame, pos, groups, masked, fold)
            record = read(folder / f"seed{fold['seed']}_fold{fold['fold']}.json")
            assert record["fit_ids"] == fit.ID.tolist() and record["query_ids"] == query.ID.tolist()
            assert record["control_max_difference"] < 1e-10
            folds += 1
    if (OUT / "summary.json").exists():
        assert read(OUT / "screen_comparison.json") == comparisons()
        assert read(OUT / "confirmation.json")["comparison"] == comparisons(extra=True)
    save(OUT / "verification.json", dict(passed=True, stages_rescored=count, repeats_rescored=repeats,
         folds_verified=folds, protected_sources_unchanged=True, native_control_parity_all_folds=True,
         part1_only=True, test_inputs_loaded=False, submissions_created=False,
         parser_and_boundary_checks=read(OUT / "parity.json")))
    print(json.dumps(read(OUT / "verification.json")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "parity", "screen", "shortlist", "confirm", "supplementary", "report", "verify"])
    args = parser.parse_args()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        if args.command == "screen":
            for stage in ["train3", "pooled3"]:
                run_stage(stage, list(m.BLOCKS))
        else:
            globals()[args.command]()


if __name__ == "__main__":
    main()
