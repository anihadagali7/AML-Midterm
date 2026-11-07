"""Independently rescore the complete, common-population comparison."""
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from all_submissions_benchmark import current, NAMES, EXTENSION, ALIASES
from all_submission_models import recipes
from benchmark_protocol import initialize, resolve, digest, DIRECTORY
from experiments import ROOT, load_data
from improvement_report import audit_track


def verify():
    out = current()
    manifest = json.loads((out / "manifest.json").read_text())
    universe, protocol = initialize()
    gaps = json.loads((EXTENSION / "gaps/protocol.json").read_text())
    assert digest(DIRECTORY / "protocol.json") == manifest["protocol_sha256"]
    assert digest(EXTENSION / "gaps/protocol.json") == manifest["gap_protocol_sha256"]
    assert digest(ROOT / "ALL_SUBMISSIONS_PROTOCOL.md") == manifest["policy_sha256"]
    for category in ["source_sha256", "input_sha256"]:
        for name, expected in manifest[category].items():
            path = ROOT / name
            if name == "submissions/leaderboard_results.json" and digest(path) != expected:
                # Later leaderboard results do not change the frozen benchmark inputs.
                snapshot = out / "input_snapshots" / name
                assert digest(snapshot) == expected, name
                before, after = json.loads(snapshot.read_text()), json.loads(path.read_text())
                for key, value in before.items():
                    if key == "submissions":
                        assert all(after[key][file] == record for file, record in value.items())
                    else:
                        assert after[key] == value
            else:
                assert digest(path) == expected, name
    ledger = json.loads((ROOT / "submissions/leaderboard_results.json").read_text())
    for entry in ledger["ani_winners"].values():
        assert digest(ROOT / "submissions" / entry["submission"]) == entry["submission_sha256"]
    for filename, entry in ledger["submissions"].items():
        assert digest(ROOT / "submissions" / filename) == entry["sha256"]
    count = 0; reused = 0; native = 0
    for is_gap, spec in [(False, protocol), (True, gaps)]:
        parent = out / "gaps" if is_gap else out
        all_frames = {}
        for track in ["strict", "published"]:
            folder = parent / track
            summary, scores, frames = audit_track(folder, universe, spec, names=NAMES)
            saved = pd.read_csv(folder / "summary.csv", index_col="model")
            assert np.allclose(summary.astype(float).sort_index(), saved.loc[summary.index].astype(float).sort_index(), equal_nan=True, atol=1e-12, rtol=0)
            all_frames[track] = frames
            count += len(scores)
            for split in spec["splits"]:
                f = frames[split["name"]]
                p = {name: f[name].probability.to_numpy() for name in NAMES}
                assert np.allclose(p["07_ani_mild_phase3_trend"], .75*p["ani_stack"]+.25*p["ani_phase3_trend"], atol=3e-15, rtol=0)
                # Recover the old linear component independently from two fixed blends.
                linear = 2*p["02_text_heavy"]-p["01_balanced_ensemble"]
                assert np.allclose(p["08_ani_linear25"], .75*p["ani_stack"]+.25*linear, atol=3e-15, rtol=0)
                phase1 = f["02_text_heavy"].phase.eq("I")
                for name in ["03_phase_specialized", "05_text_later_phases", "06_lighter_tree_blend"]:
                    assert np.allclose(p[name][phase1], p["02_text_heavy"][phase1], atol=3e-15, rtol=0)
                source = EXTENSION / "gaps" / track if is_gap else EXTENSION / track
                for name, alias in ALIASES.items():
                    old = pd.read_csv(source / "predictions" / split["name"] / f"{alias}.csv")
                    assert old.ID.tolist() == f[name].ID.tolist()
                    assert np.allclose(old.probability, p[name], atol=1e-15, rtol=0)
                    reused += 1
        for split in spec["splits"]:
            train, ev = resolve(universe, split)
            audit = json.loads((parent / "audits" / f"{split['name']}.json").read_text())
            assert audit["training_ids"] == audit["preprocessing_fit_ids"] == train.ID.tolist()
            assert audit["evaluation_ids"] == ev.ID.tolist()
            assert not set(audit["training_trial_ids"]) & set(ev.trial_id)
            for name, expected in audit["reused_predictions_sha256"].items():
                assert digest(ROOT / name) == expected
            path = ROOT / "experiments/all_submissions/cache" / out.name / f"{split['name']}.joblib"
            if path.exists():
                assert digest(path) == audit["model_sha256"]
                native += 1
            for name in recipes():
                a = all_frames["strict"][split["name"]][name]
                b = all_frames["published"][split["name"]][name]
                assert a.equals(b)
    replay = pd.read_csv(out / "original_replay.csv")
    assert len(replay[replay.kind.eq("validation_component")]) == 6
    assert len(replay[replay.kind.eq("submitted_test_file")]) == 6
    assert replay.max_probability_difference.max() < 1e-6
    invariance = json.loads((out / "inference_invariance.json").read_text())
    assert max(invariance.values()) == 0
    official_cache = ROOT / "experiments/all_submissions/cache" / out.name / "official.joblib"
    byte_replay = {}
    if official_cache.exists():
        # Replay after deserialization, including exact historical CSV rounding.
        model = joblib.load(official_cache)
        test = load_data()["test"]
        predictions = model.predict(test)
        for name in recipes():
            frame = pd.DataFrame(dict(ID=test.ID, label=predictions[name]))
            sha = hashlib.sha256(frame.to_csv(index=False, float_format="%.10f").encode()).hexdigest()
            assert sha == digest(ROOT / "submissions" / f"{name}.csv"), name
            byte_replay[name] = sha
    result = dict(models=len(NAMES), original_splits=len(protocol["splits"]),
        larger_gap_splits=len(gaps["splits"]), preprocessing_tracks=2,
        independently_rescored_model_split_results=count, source_and_input_hashes_match=True,
        same_trial_ids_and_labels_for_every_model=True, legacy_training_boundaries_verified=True,
        unchanged_control_prediction_files=reused, native_cache_hashes_verified=native,
        original_component_and_submission_replay_max_error=float(replay.max_probability_difference.max()),
        label_and_identifier_inference_max_error=max(invariance.values()),
        deserialized_original_csv_byte_replay=byte_replay,
        reporting_source_sha256={name: digest(ROOT / name) for name in
            ["all_submissions_verify.py", "all_submissions_report.py"]},
        current_leaderboard_sha256=digest(ROOT / "submissions/leaderboard_results.json"),
        new_kaggle_submissions=0)
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
