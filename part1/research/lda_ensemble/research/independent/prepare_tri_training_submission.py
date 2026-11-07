"""Prepare the user-selected 235-label tri-training submission, without upload.

Uses the exact frozen recipe from the common comparison. Test rows are queries
only. Original validation F1 remains a historical measurement of the 157-label
fit; it is never presented as a validation score for this final refit.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_cross_repo_benchmark as r
import phase21

ROOT = r.ROOT
OUT = ROOT / 'predictions/tri_training_trainval235'
SUBMISSION = ROOT / 'predictions/part1_tri_training_trainval235.csv'


def main():
    r.check_contract()
    before = r.protected()
    candidate = next(c for c in r.registry() if c['name'] == 'Classical tri-training')
    assert candidate['config'] == dict(model='tri_training', committee='mixed', cluster_gate=False,
        pseudo_weight=1., aggregation='vote', stratified_bootstrap=False)
    assert candidate['rule'] == 't0.5'
    OUT.mkdir(exist_ok=True)
    recipe = dict(name=candidate['name'], config=candidate['config'], rule=candidate['rule'],
        supervised_labels=235, original_training_gold=157, former_validation_gold=78,
        original_masked_training_rows=1014, final_refit=True, test_inputs_for_fitting=False,
        source_sha256=r.sources(), export_script_sha256=r.b.file_hash(Path(__file__)),
        registry_sha256=r.b.file_hash(r.OUT / 'registry.json'), protected_sha256=before)
    lock = OUT / 'frozen_recipe.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == recipe
    else:
        r.save(lock, recipe)

    train, val, full = r.frames()
    test = pd.read_csv(ROOT / 'data/part1_test.csv')
    sample = pd.read_csv(ROOT / 'data/part1_sample_submission.csv')
    assert len(test) == 312 and test.ID.is_unique
    assert test.ID.tolist() == sample.ID.tolist()
    assert not set(test.ID) & set(full.ID)
    assert not set(r.e.LABELS) & set(test.columns)
    fit_audit = []
    with threadpool_limits(limits=1), warnings.catch_warnings(), r.data_context(full):
        warnings.simplefilter('error', ConvergenceWarning)
        assert len(r.e.Y) == 235 and len(r.e.UNLAB) == 1014
        original_features = r.e.features
        def guarded_features(indices, query, config):
            fitting = r.e.DATA.iloc[indices]
            assert len(fitting) == 1249 and fitting.ID.is_unique
            assert set(fitting.ID) == set(full.ID)
            assert not set(fitting.ID) & set(test.ID)
            fit_audit.append(dict(fitting_rows=len(fitting), fitting_gold=235, fitting_masked=1014,
                query_rows=len(query), test_inputs_excluded=True))
            return original_features(indices, query, config)
        r.e.features = guarded_features
        try:
            p, pu, u = r.e.predict(np.arange(235), test, candidate['config'], return_unlabeled=True)
            diagnostics = list(phase21.LAST_DIAGNOSTICS)
            masked_ids = r.e.DATA.iloc[u].ID.to_numpy()
            # A small-batch deterministic refit verifies prediction-time inputs
            # cannot change fitting or the first rows of the submitted artifact.
            r.pooled.clear_caches()
            check = r.e.predict(np.arange(235), test.iloc[:3], candidate['config'])
            np.testing.assert_allclose(check, p[:3], rtol=0, atol=1e-10)
        finally:
            r.e.features = original_features
    assert p.shape == (312, 5) and np.isfinite(p).all()
    assert pu.shape == (1014, 5) and np.isfinite(pu).all()
    hard = r.e.decisions(p, candidate['rule'])
    submission = pd.DataFrame(dict(ID=test.ID,
        Predicted=[' '.join(label for label, flag in zip(r.e.LABELS, row) if flag) for row in hard]))
    assert list(submission.columns) == list(sample.columns) == ['ID', 'Predicted']
    assert (hard.sum(axis=1) > 0).all()
    payload = submission.to_csv(index=False).encode()
    if SUBMISSION.exists():
        assert SUBMISSION.read_bytes() == payload, 'Refusing to replace different submission bytes.'
    else:
        SUBMISSION.write_bytes(payload)
    reread = pd.read_csv(SUBMISSION)
    pd.testing.assert_frame_equal(reread, submission)
    decoded = np.array([[int(label in value.split()) for label in r.e.LABELS] for value in reread.Predicted])
    np.testing.assert_array_equal(decoded, hard)
    assert reread.Predicted.map(lambda text: set(text.split()) <= set(r.e.LABELS)).all()
    scores = pd.DataFrame(p, columns=['score_'+label for label in r.e.LABELS])
    scores.insert(0, 'ID', test.ID)
    scores.to_csv(OUT / 'test_scores.csv', index=False)
    generated = pd.DataFrame(r.e.decisions(pu, candidate['rule']), columns=r.e.LABELS)
    generated.insert(0, 'ID', masked_ids)
    generated.to_csv(OUT / 'generated_training_labels.csv', index=False)
    assert set(masked_ids) == set(train.loc[train[r.e.LABELS].eq(-100).all(axis=1), 'ID'])
    assert before == r.protected()
    r.check_contract()
    results = json.loads((r.OUT / 'confirmation8/summary.json').read_text())
    evidence = next(row for row in results['comparison'] if row['candidate'] == candidate['name'])
    manifest = dict(created_at=r.b.now(), status='Ready for manual Kaggle upload; not submitted',
        candidate=candidate['name'], config=candidate['config'], decision_rule=candidate['rule'],
        supervised_gold_rows=235, masked_training_rows=1014, submission_rows=312,
        final_train_plus_validation_refit=True, test_inputs_used_for_fit=False,
        independent_validation_of_refit=None,
        historical_157_label_validation_f1=candidate['recorded_validation']['sample_f1'],
        development_evidence=evidence, recipe_sha256=r.b.file_hash(lock),
        submission_file=str(SUBMISSION.relative_to(ROOT)), submission_sha256=r.b.file_hash(SUBMISSION),
        csv_columns_correct=True, sample_id_order_matches=True, all_ids_unique=True,
        all_labels_valid_and_nonempty=True, small_batch_refit_predictions_match=True,
        original_data_and_current_submission_unchanged=True, fit_audit=fit_audit,
        training_ids=full.ID.tolist(), gold_ids=full.loc[full[r.e.LABELS].ge(0).all(axis=1), 'ID'].tolist(),
        test_predicted_label_counts=dict(zip(r.e.LABELS, hard.sum(axis=0).astype(int).tolist())))
    r.save(OUT / 'manifest.json', manifest)
    r.save(OUT / 'training_diagnostics.json', diagnostics)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['training_ids','gold_ids','development_evidence','fit_audit']}, indent=2))
    print('READY:', SUBMISSION)


if __name__ == '__main__':
    main()
