"""Authorized final fit of the frozen inductive LDA ensemble; no upload."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(variable, "1")
from pathlib import Path
import datetime
import importlib.metadata
import json
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
DIRECT = OUT.parent/"ani_direct_comparison_v1"
sys.path.insert(0, str(DIRECT))
import study as direct
r, f, m = direct.r, direct.f, direct.m
SUBMISSION = OUT/"part1_lda_ensemble_trainval235.csv"
PREVIOUS = r.ROOT/"predictions/three_way_trainval235"


def freeze():
    direct.guard()
    sources = dict(r.read(DIRECT/"protocol.json")["source_sha256"])
    paths = [Path(__file__), OUT/"verify.py", DIRECT/"protocol.json", DIRECT/"summary.json", DIRECT/"verification.json",
        PREVIOUS/"manifest.json", PREVIOUS/"frozen_recipe.json", PREVIOUS/"test_scores.csv",
        r.ROOT/"predictions/part1_three_way_trainval235.csv",
        *[r.PART1/"data"/name for name in ["part1_train.csv", "part1_val.csv", "part1_test.csv", "part1_sample_submission.csv"]]]
    for path in paths:
        sources[str(path.relative_to(r.PART1))] = r.sha(path)
    recipe = dict(version="final-lda-ensemble-trainval235-v1", candidate="lda_ensemble__inductive",
        authorization="User requested final training and an scp command for laptop transfer; upload remains manual.",
        final_gold_rows=235, original_train_gold=157, former_validation_gold=78, original_masked_rows=1014,
        source_recipe="Frozen ani_direct_comparison_v1/models.py predict(..., access='inductive')",
        outer_weights=dict(latest_ani_lda=.7, classical_tri_training=.15, svm_specialist_hard_vote=.15),
        latest_ani_weights=dict(logistic=.5, cosine_knn=.3, shrinkage_lda=.2), logistic_C=.1,
        decision_rule="Center Ani source cutoff.3 to.5; fixed ensemble cutoff.5 plus nonempty argmax",
        test_inputs_used_for_fit=False, test_labels_used=False, tuning=False,
        source_sha256=sources, comparison_protocol_sha256=r.sha(DIRECT/"protocol.json"),
        supplementary_validation_belongs_to_157_label_fit=True, independent_validation_of_final_refit=None,
        python_version=sys.version, package_versions={name:importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]})
    path = OUT/"frozen_recipe.json"
    if path.exists():
        assert r.read(path)==recipe, "Refusing to change the frozen export recipe"
    else:
        r.save(path, recipe)
    return recipe


def guard(recipe):
    for name, digest in recipe["source_sha256"].items():
        assert r.sha(r.PART1/name)==digest, name
    direct.guard()


def main():
    recipe = freeze()
    assert r.read(DIRECT/"verification.json")["passed"]
    evidence = next(row for row in r.read(DIRECT/"summary.json")["comparison"] if row["candidate"]==recipe["candidate"])
    fit = r.frame_for("pooled")
    original_fit = fit.copy(deep=True)
    test = pd.read_csv(r.PART1/"data/part1_test.csv")
    sample = pd.read_csv(r.PART1/"data/part1_sample_submission.csv")
    assert len(fit)==1249 and fit.ID.is_unique
    known = fit[f.LABELS].ge(0).all(axis=1)
    masked = fit[f.LABELS].eq(-100).all(axis=1)
    assert int(known.sum())==235 and int(masked.sum())==1014 and (known|masked).all()
    assert len(test)==312 and test.ID.is_unique
    assert test.ID.tolist()==sample.ID.tolist()
    assert not set(test.ID)&set(fit.ID)
    assert not set(f.LABELS)&set(test.columns)
    assert list(sample.columns)==["ID", "Predicted"]
    gold_ids, masked_ids = set(fit.loc[known, "ID"]), set(fit.loc[masked, "ID"])
    engine = m.partners.e
    original_features = engine.features
    boundary_checks = []

    def guarded_features(indices, query, config):
        permitted = engine.DATA.iloc[indices]
        assert set(permitted.ID)<=set(fit.ID)
        assert not set(permitted.ID)&set(test.ID)
        boundary_checks.append(dict(fitting_rows=len(permitted),
            fitting_gold=len(set(permitted.ID)&gold_ids), fitting_masked=len(set(permitted.ID)&masked_ids),
            query_rows=len(query), test_inputs_excluded=True))
        return original_features(indices, query, config)

    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        engine.features = guarded_features
        try:
            scores, audit, original_ensemble = m.predict(fit, test, "inductive", source_parity=True)
            print("Final 235-label fit completed; notebook functions reproduced.", flush=True)
            small, small_audit, _ = m.predict(fit, test.iloc[:3], "inductive", source_parity=False)
            for name in scores:
                np.testing.assert_allclose(small[name], scores[name][:3], atol=1e-10, rtol=0)
        finally:
            engine.features = original_features
    pd.testing.assert_frame_equal(fit, original_fit)
    assert audit["fitting_ids"]==fit.ID.tolist()
    assert set(audit["gold_ids"])==gold_ids and set(audit["masked_ids"])==masked_ids
    assert audit["target_input_rows"]==audit["latest_final_pseudo_query_rows"]==0
    assert not set(audit["fitting_ids"])&set(test.ID)
    assert boundary_checks and all(row["test_inputs_excluded"] for row in boundary_checks)
    previous = pd.read_csv(PREVIOUS/"test_scores.csv").set_index("ID")
    previous_scores = previous.loc[test.ID, ["score_"+label for label in f.LABELS]].to_numpy()
    np.testing.assert_allclose(original_ensemble, previous_scores, atol=1e-10, rtol=0)
    original_submission = pd.read_csv(r.ROOT/"predictions/part1_three_way_trainval235.csv")
    assert original_submission.ID.tolist()==test.ID.tolist()
    previous_hard = np.array([[int(label in value.split()) for label in f.LABELS] for value in original_submission.Predicted])
    np.testing.assert_array_equal(r.decide(original_ensemble), previous_hard)
    chosen = scores[recipe["candidate"]]
    hard = r.decide(chosen)
    assert chosen.shape==(312, 5) and np.isfinite(chosen).all()
    assert (hard.sum(1)>0).all()
    submission = pd.DataFrame(dict(ID=test.ID,
        Predicted=[" ".join(label for label, value in zip(f.LABELS, row) if value) for row in hard]))
    payload = submission.to_csv(index=False).encode()
    if SUBMISSION.exists():
        assert SUBMISSION.read_bytes()==payload, "Refusing to overwrite different prediction bytes"
    else:
        SUBMISSION.write_bytes(payload)
    pd.testing.assert_frame_equal(pd.read_csv(SUBMISSION), submission)
    np.savez_compressed(OUT/"prediction_scores.npz", ids=test.ID.to_numpy(dtype=str), scores=chosen,
        predictions=hard, original_ensemble=original_ensemble, **scores)
    frame = pd.DataFrame(chosen, columns=["score_"+label for label in f.LABELS])
    frame.insert(0, "ID", test.ID)
    frame.to_csv(OUT/"test_scores.csv", index=False)
    r.save(OUT/"fit_audit.json", dict(main=audit, small_batch=small_audit, partner_feature_boundaries=boundary_checks))
    guard(recipe)
    manifest = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="Ready for manual Kaggle upload; not submitted", candidate=recipe["candidate"],
        submission_file=str(SUBMISSION.relative_to(r.PART1)), submission_sha256=r.sha(SUBMISSION),
        frozen_recipe_sha256=r.sha(OUT/"frozen_recipe.json"), supervised_gold_rows=235, masked_training_rows=1014,
        submission_rows=312, final_train_plus_validation_refit=True, test_inputs_used_for_fit=False,
        independent_validation_of_refit=None, historical_development_evidence=evidence,
        checks=dict(notebook_function_parity=True, previous_submitted_ensemble_reproduced=True,
            small_query_batch_predictions_identical=True, gold_labels_and_prior_submissions_unchanged=True,
            sample_id_order_correct=True, csv_columns_correct=True, labels_valid_nonempty=True),
        changed_test_label_sets_vs_previous_ensemble=int((hard!=previous_hard).any(1).sum()),
        test_predicted_label_counts={label:int(count) for label, count in zip(f.LABELS, hard.sum(0))},
        original_gold_label_counts={label:int(count) for label, count in zip(f.LABELS, fit.loc[known,f.LABELS].sum())},
        known_remaining_submissions_before_manual_upload=2, uploaded=False, kaggle_score=None)
    r.save(OUT/"manifest.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!="historical_development_evidence"}, indent=2))
    print("READY:", SUBMISSION, flush=True)


if __name__=="__main__":
    main()
