"""One fixed benchmark for all ten numbered submissions and both Ani winners."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "4")
import argparse
import hashlib
import importlib.metadata
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from all_submission_models import LegacySubmissions, recipes
from benchmark_protocol import initialize, resolve, digest, DIRECTORY
from experiments import ROOT, load_data
from local_benchmark import predict_and_score
from improvement_report import audit_track

OUT = ROOT / "experiments/all_submissions"
EXTENSION = ROOT / "experiments/ani_extensions/decision"
ALIASES = {"ani_first": "ani_first", "ani_stack": "ani_stack", "ani_phase3_trend": "ani_phase3_trend",
           "09_ani_replace_knn5": "replace_knn5", "10_ani_fast2_neighbors_half": "fast2_neighbors_half"}
NAMES = list(recipes()) + ["07_ani_mild_phase3_trend", "08_ani_linear25"] + list(ALIASES)
SOURCES = ["all_submission_models.py", "all_submissions_benchmark.py", "benchmark_models.py",
           "benchmark_protocol.py", "boosted.py", "experiments.py", "ensemble_search.py",
           "local_benchmark.py", "improvement_report.py"]


def register():
    universe, protocol = initialize()
    gaps = json.loads((EXTENSION / "gaps/protocol.json").read_text())
    # Check inherited controls' original model implementations before reusing them.
    source_manifest = ROOT / "experiments/stack_improvements/runs/097ac8161372126d/manifest.json"
    for name, expected in json.loads(source_manifest.read_text())["source_sha256"].items():
        assert digest(ROOT / name) == expected, name
    selection = json.loads((EXTENSION / "selection.json").read_text())
    for path in selection["source_runs"].values():
        for name, expected in json.loads((ROOT / path / "manifest.json").read_text())["source_sha256"].items():
            assert digest(ROOT / name) == expected, name
    inputs = ["submissions/manifest.json", "submissions/followup_manifest.json",
              "submissions/ani_candidates_manifest.json", "submissions/ani_extensions_manifest.json",
              "submissions/leaderboard_results.json"]
    inputs += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "submissions").glob("[0-9][0-9]_*.csv"))]
    manifest = dict(models=NAMES, recipes=recipes(), aliases=ALIASES,
        protocol_sha256=digest(DIRECTORY / "protocol.json"),
        gap_protocol_sha256=digest(EXTENSION / "gaps/protocol.json"),
        policy_sha256=digest(ROOT / "ALL_SUBMISSIONS_PROTOCOL.md"),
        source_sha256={f: digest(ROOT / f) for f in SOURCES},
        input_sha256={f: digest(ROOT / f) for f in inputs},
        versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib"]},
        preprocessing=dict(strict="All models training-fold-only",
            published="Original recipe policies: Ani covariate-transductive; legacy components training-fold-only"))
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    out = OUT / "runs" / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "current_run.json").write_text(json.dumps(dict(run=key), indent=2) + "\n")
    return universe, protocol, gaps, out


def current():
    return OUT / "runs" / json.loads((OUT / "current_run.json").read_text())["run"]


def original_parity(model, ev, out):
    """Verify this fold adapter against the artifacts actually submitted."""
    errors = []
    components = model.components(ev)
    for family in ["refined", "extra", "boosted", "yearcap"]:
        path = ROOT / "experiments/model_search" / f"{family}_validation_predictions.npz"
        if not path.exists():
            continue
        with np.load(path) as old:
            for name in set(old.files) & set(components):
                errors.append(dict(kind="validation_component", model=name,
                    max_probability_difference=float(np.max(np.abs(old[name] - components[name])))))
    test = load_data()["test"]
    ps = model.predict(test)
    for name in recipes():
        path = ROOT / "submissions" / f"{name}.csv"
        saved = pd.read_csv(path)
        assert saved.ID.tolist() == test.ID.tolist()
        errors.append(dict(kind="submitted_test_file", model=name,
            max_probability_difference=float(np.max(np.abs(saved.label - ps[name]))),
            original_sha256=digest(path)))
    changed = ev.copy()
    changed["ID"] = "unused"; changed["trial_id"] = "unused"; changed["label"] = 1 - changed.label
    before, after = model.predict(ev), model.predict(changed)
    invariance = {name: float(np.max(np.abs(before[name] - after[name]))) for name in before}
    assert max(invariance.values()) == 0
    pd.DataFrame(errors).to_csv(out / "original_replay.csv", index=False)
    (out / "inference_invariance.json").write_text(json.dumps(invariance, indent=2) + "\n")
    assert max(r["max_probability_difference"] for r in errors) < 1e-6, errors
    print("Original replay max error", max(r["max_probability_difference"] for r in errors), flush=True)


def run_split(task):
    split, is_gap, out_string = task
    out = Path(out_string)
    base = out / "gaps" if is_gap else out
    done = all((base / track / f"scores_{split['name']}.csv").exists() for track in ["strict", "published"])
    if done:
        print("cached", split["name"], flush=True)
        return
    universe, _ = initialize(); train, ev = resolve(universe, split)
    cache = OUT / "cache" / out.name
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{split['name']}.joblib"
    started = time.time()
    def progress(message):
        print(split["name"], message, f"{time.time()-started:.1f}s", flush=True)
    if path.exists():
        model = joblib.load(path)
    else:
        model = LegacySubmissions().fit(train, progress)
        joblib.dump(model, path, compress=3)
    assert model.training_ids == train.ID.tolist()
    assert not set(model.training_trial_ids) & set(ev.trial_id)
    legacy = model.predict(ev)
    if split["name"] == "official":
        original_parity(model, ev, out)
    input_hashes = {}
    for track in ["strict", "published"]:
        source = EXTENSION / "gaps" / track if is_gap else EXTENSION / track
        predictions = {name: legacy[name] for name in recipes()}
        for name, alias in ALIASES.items():
            p = source / "predictions" / split["name"] / f"{alias}.csv"
            d = pd.read_csv(p)
            for col in ["ID", "trial_id", "phase", "label"]:
                assert d[col].tolist() == ev[col].tolist(), (p, col)
            predictions[name] = d.probability.to_numpy()
            input_hashes[str(p.relative_to(ROOT))] = digest(p)
        predictions["07_ani_mild_phase3_trend"] = .75 * predictions["ani_stack"] + .25 * predictions["ani_phase3_trend"]
        predictions["08_ani_linear25"] = .75 * predictions["ani_stack"] + .25 * legacy["our_linear"]
        predict_and_score(base / track, split, ev, predictions)
    folder = base / "audits"; folder.mkdir(exist_ok=True)
    audit = dict(training_ids=train.ID.tolist(), evaluation_ids=ev.ID.tolist(),
        preprocessing_fit_ids=model.training_ids, training_trial_ids=model.training_trial_ids,
        reused_predictions_sha256=input_hashes, model_sha256=digest(path),
        legacy_preprocessing="Fitting rows only: vocabulary, IDF, SVD, metadata mask, year statistics",
        seconds=time.time()-started)
    (folder / f"{split['name']}.json").write_text(json.dumps(audit, indent=2) + "\n")
    progress("all 13 recipes scored in both tracks")


def summarize(out, universe, protocol, gaps):
    for is_gap, spec in [(False, protocol), (True, gaps)]:
        for track in ["strict", "published"]:
            folder = out / "gaps" / track if is_gap else out / track
            table, scores, _ = audit_track(folder, universe, spec, names=NAMES)
            table.to_csv(folder / "summary.csv")
            scores.to_csv(folder / "fold_scores.csv", index=False)
            print("gaps" if is_gap else "main", track, table.sort_values("temporal").to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--official-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    universe, protocol, gaps, out = register()
    tasks = [(s, False, str(out)) for s in protocol["splits"]]
    tasks += [(s, True, str(out)) for s in gaps["splits"]]
    if args.official_only:
        run_split(tasks[0]); return
    if not args.summarize:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_split, tasks))
    summarize(out, universe, protocol, gaps)
    print("Results", out, flush=True)


if __name__ == "__main__":
    main()
