"""Capture original native-model component probabilities for a portable replay.

This requires the historical local caches; the final training notebook does not.
It never derives a component by algebraically subtracting the winning CSV.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def capture(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    workspace = runpy.run_path(str(root / "run.py"))["prepare_workspace"](root)
    sys.path.insert(0, str(workspace))
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from experiments import load_data
    from sprint_export import native_packets, original_components
    paths = dict(ani="experiments/ani_extensions/cache/decision/final/submission_bundle.joblib",
                 legacy="experiments/leaderboard_sprint/cache/legacy_full.joblib")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = json.loads((root / "experiments/leaderboard_sprint/runs/4dc0f274d45bce9e/exports/13_file10_legacy30.json").read_text())
    assert sha(root / paths["ani"]) == receipt["original_bundle_sha256"]
    assert sha(root / paths["legacy"]) == receipt["legacy_bundle_sha256"]
    test = load_data()["test"]
    bundle, legacy = (joblib.load(root / paths[k]) for k in ["ani", "legacy"])
    with threadpool_limits(limits=2):
        old, extra = native_packets(bundle, test)
        components = original_components(bundle, old, extra, test)
        legacy01 = legacy["model"].predict(test)["01_balanced_ensemble"]
    p = np.clip(.7*components["file10"] + .3*legacy01, .001, .999)
    recorded = pd.read_csv(root / "submissions/13_file10_legacy30.csv")
    assert recorded.ID.tolist() == test.ID.tolist()
    difference = float(np.max(np.abs(p - recorded.label)))
    assert difference < 1e-12
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "recorded_components.npz", ID=test.ID.to_numpy(dtype=str),
                        ani_stack=components["ani_stack"],
                        fast2_neighbors=components["fast2_neighbors"], legacy01=legacy01)
    audit = dict(kind="Native inference from original fitted caches; not a fresh fit",
                 original_caches={p: sha(root/p) for p in paths.values()},
                 component_archive_sha256=sha(output / "recorded_components.npz"),
                 recorded_submission_sha256=sha(root / "submissions/13_file10_legacy30.csv"),
                 maximum_probability_difference=difference)
    (output / "recorded_components_provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.part2_root, args.output)
