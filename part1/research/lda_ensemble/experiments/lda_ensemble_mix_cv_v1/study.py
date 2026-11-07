"""Five predeclared ensemble mixes using exact, verified grouped-fold scores."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import sys
import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
GAP = OUT.parent/"feature_gap_cv_v1"
REV = OUT.parent/"ani_lda_revision_cv_v1"
sys.path.insert(0, str(GAP))
import run as runner
import features as f
BASE = runner.read(REV/"protocol.json")
ORIGINAL_COMPARE, ORIGINAL_REPORT = runner.comparisons, runner.report
WEIGHTS = {"tri25": (.75, .25, 0.), "tri40": (.6, .4, 0.),
           "tri15": (.85, .15, 0.), "svm15": (.85, 0., .15),
           "partners25": (.5, .25, .25)}
BLOCKS = {key: f"LDA Ani / tri / SVM weights {weights}" for key, weights in WEIGHTS.items()}


def source_paths():
    paths = {runner.PART1/p for p in BASE["source_sha256"]}
    paths |= {Path(__file__), REV/"protocol.json", REV/"verification.json", REV/"independent_verification.json"}
    for stage in runner.SPLITS:
        paths.add(runner.HISTORY/stage/"index.json")
        for r in runner.read(runner.HISTORY/stage/"index.json"):
            if r["candidate"] in [*runner.CONTROL_NAMES.values(), "Ani + tri 25%"]:
                paths.add(runner.HISTORY/stage/(r["result_key"]+".npz"))
        for name in ["ani__revision_full", "ensemble__revision_full"]:
            paths.add(REV/stage/(name+".npz"))
    paths.add(runner.HISTORY/"supplementary/results.json")
    for r in runner.read(runner.HISTORY/"supplementary/results.json"):
        if r["candidate"] in [*runner.CONTROL_NAMES.values(), "Ani + tri 25%"]:
            paths.add(runner.ROOT/r["prediction_file"])
    for wi in range(4):
        for name in ["ani__revision_full", "ensemble__revision_full"]:
            paths.add(REV/"supplementary"/f"window{wi}_{name}.npz")
    return {str(p.relative_to(runner.PART1)): runner.sha(p) for p in sorted(paths)}


def guard():
    p = runner.read(OUT/"protocol.json")
    for path, digest in p["source_sha256"].items():
        assert runner.sha(runner.PART1/path) == digest, path
    assert runner.sha(OUT/"registry.json") == p["registry_sha256"]
    return p


def prepare():
    if (OUT/"protocol.json").exists():
        guard()
        return
    registry = [dict(name="ensemble__"+name, parent="ensemble", block=name, description=description)
                for name, description in BLOCKS.items()]
    runner.save(OUT/"registry.json", registry)
    protocol = dict(BASE)
    protocol.update(version="lda-ensemble-mix-cv-v1", frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=source_paths(), registry_sha256=runner.sha(OUT/"registry.json"), candidates=len(registry),
        hypotheses=BLOCKS, weights=WEIGHTS, parent="Latest LDA 70/15/15 ensemble",
        fixed_recipe="Five fixed convex mixes of latest inductive Ani LDA, frozen classical tri-training, and SVM scores. Default .5 cutoff and nonempty fallback. No fitting or threshold search. Historical 75/25 and 60/40 tri mixes are retried with new source, plus removing either partner and equal .25 partner votes.",
        control_reuse="Recover exact tri and SVM scores algebraically from historical mixtures on identical folds. Assert ranges, SVM binary votes, and reconstruction of both original and latest LDA ensemble. Verify source hashes, row order, truth and folds. No outer truth enters mixing.",
        initial="All five recipes receive all eight saved repeats in both pools; no pruning.",
        shortlist="All five recipes, predeclared.",
        confirmation="Positive paired means and at least 5/8 wins in BOTH pools against latest LDA ensemble.",
        supplementary="At most three confirmed recipes or two diagnostic exceptions, selected after all eight comparisons. Reuse identical source fits and inputs for validation and chronology.",
        metadata_diagnostic="No metadata-only model or new model fit.",
        limitations="Weights are predeclared but folds and labels are reused development data. No independent test or Kaggle improvement claim.")
    runner.save(OUT/"protocol.json", protocol)
    print("Frozen five ensemble mixes against latest LDA 70/15/15 control.", flush=True)


def decompose(old, old_ensemble, old_tri25, new, new_ensemble):
    tri = (old_tri25-.75*old)/.25
    svm = (old_ensemble-.7*old-.15*tri)/.15
    for value in [tri, svm]:
        assert value.min() >= -1e-10 and value.max() <= 1+1e-10
    np.testing.assert_allclose(svm, np.round(svm), atol=1e-10, rtol=0)
    np.testing.assert_allclose(.7*new+.15*tri+.15*svm, new_ensemble, atol=1e-10, rtol=0)
    return new, tri.clip(0, 1), svm.clip(0, 1)


def load_scores(path, gold):
    # Historical controlled NPZ files store object IDs; their bytes are hash-guarded.
    with np.load(path, allow_pickle=True) as a:
        assert a["ids"].tolist() == gold.ID.tolist()
        np.testing.assert_array_equal(a["truth"], gold[f.LABELS].to_numpy())
        return a["scores"].copy()


def components(stage, gold, refs):
    r = next(r for r in runner.read(runner.HISTORY/stage/"index.json") if r["candidate"]=="Ani + tri 25%")
    tri25 = load_scores(runner.HISTORY/stage/(r["result_key"]+".npz"), gold)
    new = load_scores(REV/stage/"ani__revision_full.npz", gold)
    ensemble = load_scores(REV/stage/"ensemble__revision_full.npz", gold)
    return decompose(refs["ani"], refs["ensemble"], tri25, new, ensemble), ensemble


def mix(parts, block):
    return sum(weight*value for weight, value in zip(WEIGHTS[block], parts))


def parity():
    guard()
    for stage in runner.SPLITS:
        frame = runner.frame_for("pooled" if stage.startswith("pooled") else "train")
        _, pos, _, _ = runner.plan(stage, frame)
        gold = frame.iloc[pos]
        parts, _ = components(stage, gold, runner.controls(stage, gold))
        for block in BLOCKS:
            result = mix(parts, block)
            assert np.isfinite(result).all() and result.min()>=0 and result.max()<=1+1e-10
            np.testing.assert_allclose(result[:, :1], mix(tuple(p[:, :1] for p in parts), block), atol=1e-10, rtol=0)
    runner.save(OUT/"parity.json", dict(passed=True, exact_source_alignment=True,
        both_ensemble_controls_reconstructed=True, svm_binary_votes_verified=True,
        label_free_elementwise_mix=True, query_batch_invariance=True, source_models_previously_verified=True))


def run_stage(stage, blocks):
    guard()
    directory = OUT/stage
    directory.mkdir(exist_ok=True)
    frame = runner.frame_for("pooled" if stage.startswith("pooled") else "train")
    plan, pos, groups, masked = runner.plan(stage, frame)
    gold = frame.iloc[pos]
    scores = runner.controls(stage, gold)
    parts, baseline = components(stage, gold, scores)
    scores["ensemble__revision_full"] = baseline
    for block in blocks:
        scores["ensemble__"+block] = mix(parts, block)
    records = []
    for name, value in scores.items():
        metrics = [runner.metrics(gold[f.LABELS].to_numpy(), v) for v in value]
        numbers = [v["sample_f1"] for v in metrics]
        records.append(dict(candidate=name, mean=float(np.mean(numbers)), sd=float(np.std(numbers)), repeat_scores=metrics))
        np.savez_compressed(directory/(name+".npz"), scores=value, predictions=np.array([runner.decide(v) for v in value]),
            ids=gold.ID.to_numpy(dtype=str), truth=gold[f.LABELS].to_numpy(int))
    for ri, seed in enumerate(plan["seeds"]):
        for fold in [v for v in plan["folds"] if v["seed"]==seed]:
            fit, query, held = runner.fold_inputs(frame, pos, groups, masked, fold)
            filename = f"seed{seed}_fold{fold['fold']}"
            np.savez_compressed(directory/(filename+".npz"), ids=query.ID.to_numpy(dtype=str),
                **{name:value[ri, held] for name, value in scores.items()})
            runner.save(directory/(filename+".json"), dict(fit_ids=fit.ID.tolist(), query_ids=query.ID.tolist(),
                blocks=blocks, source_predictions_verified=True, control_max_difference=0.,
                new_model_fits=0, protocol_sha256=runner.sha(OUT/"protocol.json")))
    runner.save(directory/"index.json", records)


def comparisons(extra=False):
    rows = ORIGINAL_COMPARE(extra)
    for row in rows:
        for pool in ["train", "pooled"]:
            reference = []
            for stage in [pool+"3", pool+"5"] if extra else [pool+"3"]:
                entry = next(r for r in runner.read(OUT/stage/"index.json") if r["candidate"]=="ensemble__revision_full")
                reference.extend(r["sample_f1"] for r in entry["repeat_scores"])
            delta = np.array(row[pool+"_repeats"])-reference
            row[pool+"_vs_revision_delta"] = float(delta.mean())
            row[pool+"_vs_revision_wins"] = int((delta>1e-12).sum())
            row[pool+"_vs_revision_ties"] = int((abs(delta)<=1e-12).sum())
            row[pool+"_vs_revision_deltas"] = delta.tolist()
        row["matched_revision_control"] = "ensemble__revision_full"
        row["minimum_original_parent_gain"] = row["minimum_parent_gain"]
        row["minimum_parent_gain"] = min(row[pool+"_vs_revision_delta"] for pool in ["train", "pooled"])
        row["qualifies"] = bool(row["minimum_parent_gain"]>1e-12 and all(row[pool+"_vs_revision_wins"] >= (5 if extra else 2) for pool in ["train", "pooled"]))
    return sorted(rows, key=lambda r:(-r["minimum_parent_gain"], r["candidate"]))


def shortlist():
    rows = comparisons()
    runner.save(OUT/"screen_comparison.json", rows)
    pd.DataFrame(rows).to_csv(OUT/"screen_comparison.csv", index=False)
    runner.save(OUT/"shortlist.json", dict(names=[r["name"] for r in runner.read(OUT/"registry.json")],
        diagnostic_only=False, selection_rule="All five candidates; predeclared before scores"))


def supplementary():
    guard()
    names = runner.read(OUT/"confirmation.json")["supplementary_names"]
    train = runner.frame_for("train")
    full = runner.frame_for("pooled").set_index("ID", drop=False)
    val = pd.read_csv(runner.PART1/"data/part1_val.csv")
    windows = [dict(name="original_validation", training_ids=train.ID.tolist(), heldout_ids=val.ID.tolist())]
    windows += runner.read(runner.ROOT/"experiments/cross_repo_benchmark_v1/protocol.json")["chronological_splits"]
    history = runner.read(runner.HISTORY/"supplementary/results.json")
    (OUT/"supplementary").mkdir(exist_ok=True)
    records = []
    for wi, window in enumerate(windows):
        gold = full.loc[window["heldout_ids"]].reset_index(drop=True)
        refs = {}
        for key, title in {**runner.CONTROL_NAMES, "tri25":"Ani + tri 25%"}.items():
            r = next(r for r in history if r["window"]==wi and r["candidate"]==title)
            frame = pd.read_csv(runner.ROOT/r["prediction_file"]).set_index("ID")
            assert set(frame.index)==set(gold.ID)
            refs[key] = frame.loc[gold.ID, ["score_"+label for label in f.LABELS]].to_numpy()
        new, ensemble = [load_scores(REV/"supplementary"/f"window{wi}_{name}__revision_full.npz", gold) for name in ["ani", "ensemble"]]
        parts = decompose(refs["ani"], refs["ensemble"], refs["tri25"], new, ensemble)
        values = {key:refs[key] for key in runner.CONTROL_NAMES}
        values["ensemble__revision_full"] = ensemble
        for name in names:
            values[name] = mix(parts, name.split("__", 1)[1])
        for name, score in values.items():
            truth = gold[f.LABELS].to_numpy(int)
            np.savez_compressed(OUT/"supplementary"/f"window{wi}_{name}.npz", scores=score,
                predictions=runner.decide(score), truth=truth, ids=gold.ID.to_numpy(dtype=str))
            records.append(dict(candidate=name, window=wi, window_name=window["name"], **runner.metrics(truth, score)))
    runner.save(OUT/"supplementary/results.json", records)


def report():
    ORIGINAL_REPORT()
    path = OUT/"report.html"
    body = path.read_text().replace("Part 1 feature gap experiments", "LDA ensemble mixes")
    body = body.replace("Part 1: new feature structure and earlier-feature retries", "Retesting ensemble partners after the LDA revision")
    start = body.index("<p>Nine blocks")
    stop = body.index("</p>", start)+4
    body = body[:start]+"<p>Five fixed ensemble mixes using exact source predictions from the same grouped folds. Confirmation is against the latest LDA 70/15/15 ensemble; the original-parent columns are historical references. Latest paired differences and wins are in final_comparison.csv. All source files are hash-guarded; the component decomposition reconstructs both controls. No new fitting or outer-label-dependent mixing.</p>"+body[stop:]
    body = body.replace("Metadata-only diagnostics", "No metadata-only fits")
    path.write_text(body)
    (OUT/"metadata_only.csv").unlink()


def independent_verify():
    import verify_saved
    verify_saved.HERE = OUT
    verify_saved.main()


runner.OUT = OUT
runner.m = SimpleNamespace(BLOCKS=BLOCKS, LABELS=f.LABELS, EMB=f.EMB)
runner.prepare, runner.guard, runner.parity = prepare, guard, parity
runner.run_stage, runner.comparisons, runner.shortlist = run_stage, comparisons, shortlist
runner.supplementary, runner.report = supplementary, report
if __name__=="__main__":
    if len(sys.argv)==2 and sys.argv[1]=="independent_verify":
        independent_verify()
    else:
        runner.main()
