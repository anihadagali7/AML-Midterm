"""Direct, fixed-recipe comparison with the source of Ani's .89423 artifact."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import argparse
import datetime
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import models as m

OUT = Path(__file__).resolve().parent
r, f = m.runner, m.f
BASE = r.read(m.REVISION/"protocol.json")
CONTRASTS = [("lda_ensemble__"+a, "ani__"+a, "Primary matched-access comparison") for a in m.ACCESS]
CONTRASTS += [("lda_source__"+a, "ani__"+a, "Source revision without ensemble") for a in m.ACCESS]
CONTRASTS += [("lda_ensemble__"+a, "lda_source__"+a, "Contribution of ensemble partners") for a in m.ACCESS]
CONTRASTS += [(recipe+"__target_inputs", recipe+"__inductive", "Input-access diagnostic") for recipe in m.RECIPES]
CONTRASTS += [("lda_ensemble__inductive", "ani__target_inputs", "Practical comparison of current and original access policies; not an isolated model effect")]


def guard():
    protocol = r.read(OUT/"protocol.json")
    for path, digest in protocol["source_sha256"].items():
        assert r.sha(r.PART1/path)==digest, path
    assert r.sha(OUT/"registry.json")==protocol["registry_sha256"]
    return protocol


def prepare():
    if (OUT/"protocol.json").exists():
        guard()
        return
    sources = dict(BASE["source_sha256"])
    paths = [Path(__file__), OUT/"models.py", OUT/"report.py", OUT/"verify.py", m.GAP/"verify_saved.py",
        m.REVISION/"protocol.json", *m.NOTEBOOKS.values(),
        r.PART1/"research/ani_snapshots/a548721/part1_best_submission.csv",
        r.ROOT/"experiments/cross_repo_benchmark_v1/registry.json"]
    for stage in r.SPLITS:
        for name in ["ani__revision_full", "ensemble__revision_full"]:
            paths.append(m.REVISION/stage/(name+".npz"))
    for wi in range(4):
        paths.append(m.REVISION/"supplementary"/f"window{wi}_ensemble__revision_full.npz")
    for path in paths:
        sources[str(path.relative_to(r.PART1))] = r.sha(path)
    submitted = r.PART1/"research/ani_snapshots/a548721/part1_best_submission.csv"
    assert r.sha(submitted)=="d28c381fe35bd7fe787328646cd1ed5d27a7ba8c6eba3353894ba761e1ad4f29"
    registry = [dict(name=recipe+"__"+access, recipe=recipe, access=access,
        decoder="Ani and latest source: raw .3 cutoff plus argmax, represented by monotone .5 centering; ensemble: .5 cutoff plus nonempty argmax")
        for access in m.ACCESS for recipe in m.RECIPES]
    r.save(OUT/"registry.json", registry)
    protocol = dict(version="ani-direct-comparison-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, registry_sha256=r.sha(OUT/"registry.json"), folds=r.SPLITS,
        requested_comparison="Current LDA ensemble versus the original Ani recipe associated with reported .89423 Kaggle score, using identical tests.",
        original_source_commit="a548721", original_submission_sha256=r.sha(submitted), reported_original_kaggle=.89423,
        original_recipe="StandardScaler; PCA30; GMM30 diagonal, n_init3, seed0; balanced multinomial LR C.3 plus cosine kNN10 weights .7/.3; two GMM-assisted pseudo-label rounds, .2 GMM contribution, confidence.8, cutoff.3; native final cutoff.3 plus argmax.",
        lda_source="Exact 8a8c3fc function recipe: C.1; .5 LR/.3 cosine kNN/.2 shrinkage LDA with uniform class priors, used in teaching and final prediction. Unchanged original GMM/self-training settings.",
        lda_ensemble=".7 centered latest Ani source + .15 frozen classical tri-training + .15 frozen SVM specialist hard votes. Partners refitted from scratch in every partition.",
        access_regimes={"inductive":"Every learned step uses fitting gold plus original masked training rows after held-group purge; query inputs only predict.",
            "target_inputs":"Append only the current evaluation batch as masked rows, every label -100. Original masked duplicates of held groups are still purged. All components in both recipes get identical fitting inputs and can pseudo-label target inputs. No actual Kaggle test data is loaded."},
        original_notebook_distinction="Original notebook CV used global transforms, ungrouped KFold and actual test inputs. We replace that evaluation procedure with saved grouped folds. Target-input arm models its FINAL submission access, where queried test rows are also in the unlabeled pool. This is not replay of its historical CV numbers or .89423 artifact.",
        chronology_distinction="Identical earlier labeled/masked fitting IDs and 98 later target IDs in both arms. Target-input diagnostic additionally sees the current future evaluation batch with labels erased; it is not strictly inductive forecasting.",
        primary_contrasts=CONTRASTS[:2], other_contrasts=CONTRASTS[2:],
        schedule="All six fixed recipe/access combinations receive all eight saved repeats in both pools. No pruning, tuning or candidate selection. Only after CV completion, all six receive original validation and three fixed chronological windows as predeclared direct comparisons.",
        confirmation="Positive mean paired delta and >=5/8 wins in BOTH pools against the stated comparator. Repeated-fold wins are descriptive, not independent significance tests.",
        parity="In every fit, execute only function definitions from both original notebooks using the same permitted transforms and rows, and compare probabilities to atol1e-10. Refit and reproduce saved inductive Ani, old ensemble and latest LDA scores in every grouped fold. No notebook input-loading, search, export or visualization cells executed.",
        label_safety="Query target columns erased before any fitting logic; parity includes adversarial query-label mutation. IDs only audit/alignment, never predictors; gold labels unchanged.",
        metric="Complete-OOF sample-wise F1 per repeat; mean and population SD. All predictions, per-label and per-trial results saved.",
        limitations="Reused development labels and folds. Local transductive batches have 31-47 CV rows rather than the 312-row Kaggle batch. Neither regime proves a Kaggle improvement or course permission for transduction.",
        part1_only=True, test_inputs_loaded=False, submissions_created=False, final235_refit=False)
    r.save(OUT/"protocol.json", protocol)
    print("Frozen direct Ani versus LDA comparison: three recipes, two matched access regimes, all saved folds.", flush=True)


def parity():
    guard()
    frame = r.frame_for("pooled")
    plan, pos, groups, masked = r.plan("pooled3", frame)
    fit, query, held = r.fold_inputs(frame, pos, groups, masked, plan["folds"][0])
    refs = r.controls("pooled3", frame.iloc[pos])
    differences = {}
    for access in m.ACCESS:
        scores, audit, old_ensemble = m.predict(fit, query, access, source_parity=True)
        poisoned = query.assign(**{label:np.arange(len(query))%2 for label in f.LABELS})
        fit_a, query_a = m.fitting_frame(fit, query, access)
        fit_b, query_b = m.fitting_frame(fit, poisoned, access)
        pd.testing.assert_frame_equal(fit_a, fit_b)
        pd.testing.assert_frame_equal(query_a, query_b)
        if access=="inductive":
            np.testing.assert_allclose(scores["ani__inductive"], refs["ani"][0, held], atol=1e-10, rtol=0)
            np.testing.assert_allclose(old_ensemble, refs["ensemble"][0, held], atol=1e-10, rtol=0)
            with np.load(m.REVISION/"pooled3/ensemble__revision_full.npz") as saved:
                np.testing.assert_allclose(scores["lda_ensemble__inductive"], saved["scores"][0, held], atol=1e-10, rtol=0)
        differences[access] = audit["notebook_function_max_differences"]
    r.save(OUT/"parity.json", dict(passed=True, both_original_notebooks_verified=True,
        notebook_max_differences=differences, query_label_mutation_cannot_change_fitting_or_query_inputs=True,
        fresh_partners_reproduce_original_ensemble=True, latest_ensemble_control_reproduced=True,
        original_notebook_loading_and_export_cells_executed=False))
    print("Both notebook recipes and the fresh ensemble partners reproduced; label-erasure checks passed.", flush=True)


def run_stage(stage):
    guard()
    assert r.read(OUT/"parity.json")["passed"]
    folder = OUT/stage
    folder.mkdir(exist_ok=True)
    frame = r.frame_for("pooled" if stage.startswith("pooled") else "train")
    plan, pos, groups, masked = r.plan(stage, frame)
    gold = frame.iloc[pos]
    refs = r.controls(stage, gold)
    with np.load(m.REVISION/stage/"ensemble__revision_full.npz") as a:
        assert a["ids"].tolist()==gold.ID.tolist()
        revised_reference = a["scores"].copy()
    shape = (len(plan["seeds"]), len(gold), 5)
    complete = {name:np.full(shape, np.nan) for name in m.NAMES}
    for ri, seed in enumerate(plan["seeds"]):
        for fold in [row for row in plan["folds"] if row["seed"]==seed]:
            fit, query, held = r.fold_inputs(frame, pos, groups, masked, fold)
            prefix = folder/f"seed{seed}_fold{fold['fold']}"
            audit_path = prefix.with_suffix(".json")
            if audit_path.exists():
                audit = r.read(audit_path)
                assert audit["protocol_sha256"]==r.sha(OUT/"protocol.json")
                assert audit["fit_ids"]==fit.ID.tolist() and audit["query_ids"]==query.ID.tolist()
                with np.load(prefix.with_suffix(".npz")) as saved:
                    assert saved["ids"].tolist()==query.ID.tolist()
                    scores = {name:saved[name].copy() for name in m.NAMES}
            else:
                scores, fits, differences = {}, {}, {}
                for access in m.ACCESS:
                    current, details, old_ensemble = m.predict(fit, query, access, source_parity=True)
                    scores.update(current)
                    fits[access] = details
                    if access=="inductive":
                        for key, generated, expected in [
                            ("original_ani", current["ani__inductive"], refs["ani"][ri, held]),
                            ("old_ensemble", old_ensemble, refs["ensemble"][ri, held]),
                            ("current_ensemble", current["lda_ensemble__inductive"], revised_reference[ri, held])]:
                            differences[key] = float(np.max(abs(generated-expected)))
                            np.testing.assert_allclose(generated, expected, atol=1e-10, rtol=0)
                np.savez_compressed(prefix.with_suffix(".npz"), ids=query.ID.to_numpy(dtype=str), **scores)
                audit = dict(fit_ids=fit.ID.tolist(), query_ids=query.ID.tolist(), fits=fits,
                    control_max_differences=differences, control_max_difference=max(differences.values()),
                    protocol_sha256=r.sha(OUT/"protocol.json"), all_models_freshly_refitted=True)
                r.save(audit_path, audit)
            for name in m.NAMES:
                complete[name][ri, held] = scores[name]
            print(json.dumps(dict(stage=stage, seed=seed, fold=fold["fold"], rows=len(query), complete=True)), flush=True)
    records = []
    for name, scores in complete.items():
        assert np.isfinite(scores).all()
        repeats = [r.metrics(gold[f.LABELS].to_numpy(), values) for values in scores]
        values = [entry["sample_f1"] for entry in repeats]
        records.append(dict(candidate=name, mean=float(np.mean(values)), sd=float(np.std(values)), repeat_scores=repeats))
        np.savez_compressed(folder/(name+".npz"), scores=scores, predictions=np.array([r.decide(p) for p in scores]),
            ids=gold.ID.to_numpy(dtype=str), truth=gold[f.LABELS].to_numpy(int))
    r.save(folder/"index.json", records)
    guard()


def supplementary():
    guard()
    assert all((OUT/stage/"index.json").exists() for stage in r.SPLITS)
    train, full = r.frame_for("train"), r.frame_for("pooled").set_index("ID", drop=False)
    val = pd.read_csv(r.PART1/"data/part1_val.csv")
    windows = [dict(name="original_validation", training_ids=train.ID.tolist(), heldout_ids=val.ID.tolist())]
    windows += r.read(r.ROOT/"experiments/cross_repo_benchmark_v1/protocol.json")["chronological_splits"]
    directory = OUT/"supplementary"
    directory.mkdir(exist_ok=True)
    results = []
    history = r.read(r.HISTORY/"supplementary/results.json")
    for wi, window in enumerate(windows):
        fit = full.loc[window["training_ids"]].reset_index(drop=True)
        gold = full.loc[window["heldout_ids"]].reset_index(drop=True)
        query = gold.drop(columns=f.LABELS)
        if wi:
            assert not set(pd.util.hash_pandas_object(fit[f.EMB].round(10), index=False))&set(pd.util.hash_pandas_object(query[f.EMB].round(10), index=False))
        audits = {}
        for access in m.ACCESS:
            scores, audit, old_ensemble = m.predict(fit, query, access, source_parity=True)
            audits[access] = audit
            if access=="inductive":
                ref = next(row for row in history if row["candidate"]=="Ani frozen control" and row["window"]==wi)
                frame = pd.read_csv(r.ROOT/ref["prediction_file"]).set_index("ID")
                np.testing.assert_allclose(scores["ani__inductive"], frame.loc[gold.ID, ["score_"+l for l in f.LABELS]].to_numpy(), atol=1e-10, rtol=0)
                with np.load(m.REVISION/"supplementary"/f"window{wi}_ensemble__revision_full.npz") as saved:
                    assert saved["ids"].tolist()==query.ID.tolist()
                    np.testing.assert_allclose(scores["lda_ensemble__inductive"], saved["scores"], atol=1e-10, rtol=0)
            for name, score in scores.items():
                truth = gold[f.LABELS].to_numpy(int)
                np.savez_compressed(directory/f"window{wi}_{name}.npz", scores=score, predictions=r.decide(score),
                    truth=truth, ids=query.ID.to_numpy(dtype=str))
                results.append(dict(candidate=name, window=wi, window_name=window["name"], **r.metrics(truth, score)))
        r.save(directory/f"window{wi}_audit.json", dict(fit_ids=fit.ID.tolist(), query_ids=query.ID.tolist(),
            fits=audits, protocol_sha256=r.sha(OUT/"protocol.json")))
        print(json.dumps(dict(supplementary_window=wi, rows=len(query), complete=True)), flush=True)
    r.save(directory/"results.json", results)
    guard()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "parity", *r.SPLITS, "supplementary"])
    args = parser.parse_args()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        if args.command in r.SPLITS:
            run_stage(args.command)
        else:
            globals()[args.command]()
