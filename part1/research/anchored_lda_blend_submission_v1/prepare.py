"""Package the tested fixed blend of the two existing full-235 fits; no upload."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PART1, ROOT = HERE.parents[1], HERE.parents[2]
STUDY = PART1/"research/anchored_neighbor_blend_v1"
ANCHORED = PART1/"experiments/paired_improvement/candidate"
LDA = PART1/"research/lda_ensemble/experiments/final_lda_ensemble_trainval235_v1"
SUBMISSION = PART1/"submissions/ready/part1_anchored75_lda25_trainval235.csv"
LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
sys.path[:0] = [str(PART1), str(STUDY)]
from neighbor_models import centered, decide


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def main():
    assert not (HERE/"manifest.json").exists(), "Use verify.py for the completed export."
    protocol = read(STUDY/"protocol.json")
    for name, digest in protocol["sources"].items():
        assert sha(ROOT/name) == digest, name
    am = read(ANCHORED/"manifest.json")
    lm, lr, la = [read(LDA/name) for name in ["manifest.json", "frozen_recipe.json", "fit_audit.json"]]
    protected = {ROOT/name: digest for name, digest in protocol["sources"].items()}
    for name in ["anchored_model.joblib", "test_scores.csv", "part1_anchored_submission.csv", "anchored_solution.py"]:
        path = ANCHORED/name
        assert sha(path) == am["artifact_sha256"][name], name
        protected[path] = sha(path)
    retained = read(PART1/"research/consolidation_v1/retention.json")["retained_files"]
    for row in retained:
        if row["path"].startswith("experiments/final_lda_ensemble_trainval235_v1/"):
            path = PART1/"research/lda_ensemble"/row["path"]
            assert sha(path) == row["sha256"], row["path"]
            protected[path] = row["sha256"]
    for record_name in ["anchored-trainval235.json", "lda-ensemble-trainval235.json"]:
        result = read(PART1/"submissions/results"/record_name)
        for path_field, sha_field in [("submission_file", "submission_sha256"), ("frozen_recipe_file", "frozen_recipe_sha256")]:
            path = ROOT/result[path_field]
            assert sha(path) == result[sha_field]
            protected[path] = result[sha_field]
    assert am["known_labels"] == lm["supervised_gold_rows"] == lr["final_gold_rows"] == 235
    assert am["originally_unlabeled"] == lm["masked_training_rows"] == 1014
    assert not lm["test_inputs_used_for_fit"] and not lr["test_inputs_used_for_fit"]
    assert la["main"]["target_input_rows"] == la["main"]["latest_final_pseudo_query_rows"] == 0
    assert all(row["test_inputs_excluded"] for row in la["partner_feature_boundaries"])
    assert read(STUDY/"results/verification.json")["passed"]
    extra = [Path(__file__), HERE/"verify.py", STUDY/"protocol.json", STUDY/"results/verification.json",
             STUDY/"results/cv_summary.csv", STUDY/"results/paired_summary.csv", STUDY/"results/supplementary_scores.csv",
             ANCHORED/"manifest.json", PART1/"research/consolidation_v1/retention.json",
             PART1/"data/part1_sample_submission.csv"]
    protected.update({path: sha(path) for path in extra})
    recipe = dict(version="anchored75-lda25-trainval235-v1",
        authorization="User requested preparing the tested 75/25 blend for a manual Kaggle trial, after discussion of its pooled-CV improvement and failure to meet the strict both-pool promotion rule.",
        study_commit="8ec2f84", registered_recipe=protocol["candidates"]["anchored75_lda25"],
        weights=dict(anchored=.75, existing_lda_ensemble=.25),
        decision="Center anchored native cutoff .3 to .5 using the registered logit transform; blend; cutoff .5 plus nonempty argmax.",
        component_population="Both existing final fits used all 235 verified labels plus the 1,014 original masked training rows, inductively.",
        preparation="Reload existing anchored fitted model for prediction and use frozen full-precision LDA ensemble scores. No new fitting, tuning or weight selection.",
        promotion_status="Manual trial authorized; not a both-pool CV-qualified or Kaggle-confirmed replacement for anchored.",
        source_sha256={str(path.relative_to(ROOT)): digest for path, digest in sorted(protected.items())})
    save(HERE/"frozen_recipe.json", recipe)

    train = pd.read_csv(PART1/"data/part1_train.csv")
    val = pd.read_csv(PART1/"data/part1_val.csv")
    fit = pd.concat([train, val], ignore_index=True)
    test = pd.read_csv(PART1/"data/part1_test.csv")
    sample = pd.read_csv(PART1/"data/part1_sample_submission.csv")
    gold = fit[LABELS].isin([0, 1]).all(axis=1)
    masked = fit[LABELS].eq(-100).all(axis=1)
    assert len(fit) == 1249 and gold.sum() == 235 and masked.sum() == 1014 and (gold | masked).all()
    assert len(test) == test.ID.nunique() == 312 and list(sample.columns) == ["ID", "Predicted"]
    assert test.ID.tolist() == sample.ID.tolist() and not set(LABELS) & set(test.columns)
    assert not set(fit.ID) & set(test.ID)
    assert la["main"]["fitting_ids"] == fit.ID.tolist()
    assert set(la["main"]["gold_ids"]) == set(fit.loc[gold, "ID"])
    assert set(la["main"]["masked_ids"]) == set(fit.loc[masked, "ID"])

    with threadpool_limits(limits=1):
        model = joblib.load(ANCHORED/"anchored_model.joblib")
        assert model.round_audit_ == am["final_fit_audit"]
        raw = model.predict_proba(test[[f"emb_{i}" for i in range(150)]])
    saved = pd.read_csv(ANCHORED/"test_scores.csv", float_precision="round_trip").set_index("ID").loc[test.ID, LABELS].to_numpy()
    np.testing.assert_allclose(raw, saved, atol=1e-14, rtol=0)
    anchored = centered(raw)
    with np.load(LDA/"prediction_scores.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"], test.ID)
        lda = z["scores"].copy()
        np.testing.assert_array_equal(lda, z["lda_ensemble__inductive"])
        np.testing.assert_array_equal(decide(lda), z["predictions"])
    saved = pd.read_csv(LDA/"test_scores.csv", float_precision="round_trip").set_index("ID").loc[test.ID, ["score_"+s for s in LABELS]].to_numpy()
    np.testing.assert_allclose(lda, saved, atol=1e-14, rtol=0)
    old = pd.read_csv(ANCHORED/"part1_anchored_submission.csv").set_index("ID").loc[test.ID].reset_index()
    old_lda = pd.read_csv(LDA/"part1_lda_ensemble_trainval235.csv").set_index("ID").loc[test.ID].reset_index()
    render = lambda p: [" ".join(label for label, flag in zip(LABELS, row) if flag) for row in p]
    assert render(decide(anchored)) == old.Predicted.tolist()
    assert render(decide(lda)) == old_lda.Predicted.tolist()
    scores = .75*anchored+.25*lda
    hard = decide(scores)
    assert scores.shape == (312, 5) and np.isfinite(scores).all()
    submission = pd.DataFrame(dict(ID=test.ID, Predicted=render(hard)))
    payload = submission.to_csv(index=False).encode()
    SUBMISSION.parent.mkdir(exist_ok=True)
    if SUBMISSION.exists():
        assert SUBMISSION.read_bytes() == payload, "Refusing to overwrite a different submission."
    else:
        SUBMISSION.write_bytes(payload)
    np.savez_compressed(HERE/"prediction_scores.npz", ids=test.ID.to_numpy(str), labels=np.array(LABELS),
                        anchored_raw=raw, anchored_centered=anchored, lda_ensemble=lda, scores=scores, predictions=hard)
    changes = pd.DataFrame(dict(ID=test.ID, anchored=old.Predicted, blend=submission.Predicted, lda=old_lda.Predicted))
    changed = changes.anchored != changes.blend
    changes[changed].to_csv(HERE/"changes_vs_anchored.csv", index=False)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), status="Ready for manual upload; score unknown",
        recipe="anchored75-lda25-trainval235", submission_file=str(SUBMISSION.relative_to(ROOT)),
        submission_sha256=sha(SUBMISSION), frozen_recipe_sha256=sha(HERE/"frozen_recipe.json"),
        both_components_verified_labels=235, original_masked_training_rows=1014, rows=312,
        new_model_fits=0, anchored_saved_model_predictions_recomputed=True, lda_saved_full235_scores_reused=True,
        test_inputs_used_for_fit=False, test_labels_used=False, uploaded=False, kaggle_score=None,
        changed_labelsets_vs_anchored=int(changed.sum()), changed_labelsets_vs_lda=int((changes.lda != changes.blend).sum()),
        minimum_score_distance_from_cutoff=float(abs(scores-.5).min()),
        predicted_label_counts={label: int(n) for label, n in zip(LABELS, hard.sum(axis=0))},
        artifacts={name: sha(HERE/name) for name in ["prediction_scores.npz", "changes_vs_anchored.csv"]},
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))
    for path, digest in protected.items():
        assert sha(path) == digest, str(path)
    save(HERE/"manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
