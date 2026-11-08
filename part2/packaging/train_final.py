"""Refit file 13 from the supplied CSVs without reading existing fitted caches.

The same implementation is embedded in the finalized model notebook. Run with
Python 3.12 and part2/requirements/requirements-all-submissions.txt installed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")


def refit_final(part2_root, output_dir):
    """Train the frozen recipe, replay it on feature-only rows, and save a new CSV."""
    root, output = Path(part2_root).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    workspace = runpy.run_path(str(root / "run.py"))["prepare_workspace"](root)
    sys.path.insert(0, str(workspace))
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from ani_models import AniFeatures
    from experiments import load_data
    from improvement_models import fit_bank, fit_recipes as fit_base_metas
    from extension_neighbors import fit_extended, fit_recipes as fit_neighbor_metas
    from all_submission_models import LegacySubmissions
    from sprint_export import native_packets, original_components

    started = time.monotonic()
    def progress(message):
        print(f"{time.monotonic() - started:.1f}s: {message}", flush=True)

    data = load_data()
    train = pd.concat([data["train"], data["val"]], ignore_index=True)
    test = data["test"]
    assert len(train) == 9050 and len(test) == 1500
    assert not set(train.trial_id) & set(test.trial_id)
    assert "label" not in test
    # Frozen published-recipe preprocessing: vocabulary/IDF see all supplied
    # texts; supervised fits and grouped stacking folds use labeled rows only.
    with threadpool_limits(limits=2):
        features = AniFeatures().fit(pd.concat(list(data.values()), ignore_index=True))
        progress("fitted vocabulary/IDF on supplied train, validation, and test text")
        old = fit_bank(train, test, features=features, retain_models=True,
                       progress=lambda message: progress("base bank " + message))
        baseline, base_metas = fit_base_metas(old, train, test)
        extra = fit_extended(train, test, old, space=features, retain=True,
                             progress=lambda fold: progress(f"neighbor bank inner {fold}/5"))
        neighbors, neighbor_metas = fit_neighbor_metas(old, extra, train, test, baseline)
        legacy = LegacySubmissions().fit(train, progress=progress)
        legacy_probability = legacy.predict(test)["01_balanced_ensemble"]
        probability = np.clip(.7 * neighbors["fast2_neighbors_half"]
                              + .3 * legacy_probability, .001, .999)

        # Replay actual fitted models after removing outcomes and replacing IDs.
        query = test.copy()
        query["ID"] = "unused"
        query["trial_id"] = "unused"
        query["label"] = 0
        bundle = dict(old=old, extra=extra, base_metas=base_metas,
                      neighbor_metas=neighbor_metas)
        replay_old, replay_extra = native_packets(bundle, query)
        components = original_components(bundle, replay_old, replay_extra, query)
        replay = np.clip(.7 * components["file10"]
                         + .3 * legacy.predict(query)["01_balanced_ensemble"], .001, .999)
    replay_error = float(np.max(np.abs(probability - replay)))
    assert replay_error < 1e-12, replay_error
    assert old["training_ids"] == extra["train_ids"] == legacy.training_ids == train.ID.tolist()
    for packet in (old, extra):
        for fold in packet["audit"]:
            heldout = fold.get("heldout_ids", fold.get("eval_ids"))
            assert not set(fold["fit_ids"]) & set(heldout)
            indexed = train.set_index("ID")
            assert not (set(indexed.loc[fold["fit_ids"], "trial_id"])
                        & set(indexed.loc[heldout, "trial_id"]))

    sample = pd.read_csv(root / "data/part2_sample_submission.csv")
    frame = pd.DataFrame(dict(ID=test.ID, label=probability)).set_index("ID").loc[sample.ID].reset_index()
    assert frame.ID.is_unique and frame.ID.equals(sample.ID)
    assert np.isfinite(frame.label).all() and frame.label.between(0, 1, inclusive="neither").all()
    destination = output / "13_file10_legacy30_refit.csv"
    frame.to_csv(destination, index=False)
    recorded = pd.read_csv(root / "submissions/13_file10_legacy30.csv")
    assert frame.ID.equals(recorded.ID)
    error = float(np.max(np.abs(frame.label - recorded.label)))
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt = dict(
        recipe={"file10": .7, "legacy01": .3}, training_rows=len(train),
        test_rows=len(test), existing_model_caches_read=False,
        native_replay_max_error=replay_error, recorded_csv_max_error=error,
        numerically_matches_recorded_at_1e_7=error < 1e-7,
        recorded_csv_sha256=sha(root / "submissions/13_file10_legacy30.csv"),
        refit_csv_sha256=sha(destination), elapsed_seconds=time.monotonic()-started,
        versions={p: importlib.metadata.version(p) for p in
                  ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib"]},
        data_sha256={p.name: sha(p) for p in sorted((root / "data").glob("*.csv"))},
        source_sha256={p.name: sha(p) for p in sorted((root / "src").glob("*.py"))},
        note="The Kaggle score belongs to the original CSV. A refit does not acquire that score.")
    (output / "refit_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    progress(f"saved refit; maximum difference from scored CSV: {error:.3g}")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part2-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refit_final(args.part2_root, args.output), indent=2))
