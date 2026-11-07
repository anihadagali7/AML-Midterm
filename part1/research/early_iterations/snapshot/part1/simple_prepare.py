"""Save a separate reviewable candidate; never upload or replace past artifacts."""
import json
import platform
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from simple_benchmark import ROOT, BASE, OUT, digest, load_data
from simple_models import FEATURES, REASONS, SimpleModel, decide
from simple_solution import ReasonClassifier


def main():
    selected = json.loads((OUT / 'selected.json').read_text())
    spec = next(c for c in selected['candidates'] if c['recipe']['name'] == selected['primary'])
    assert spec['recipe']['name'] == 'final_control_r3'
    dest = OUT / 'candidate'
    dest.mkdir(exist_ok=True)
    known, unknown = load_data()
    X, Y, U = known[FEATURES].to_numpy(), known[REASONS].to_numpy(int), unknown[FEATURES].to_numpy()
    checks = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        # Check the independent small implementation on EVERY existing fold.
        expected = np.load(OUT / spec['stage'] / 'probabilities.npz')[selected['primary']]
        for repeat in range(8):
            folds = pd.read_csv(BASE / 'splits' / f'repeat{repeat}.csv').set_index('ID').loc[known.ID, 'fold'].to_numpy()
            for fold in range(5):
                held = folds == fold
                model = ReasonClassifier().fit(X[~held], Y[~held], U)
                actual = model.predict_proba(X[held])
                np.testing.assert_allclose(actual, expected[repeat, held], atol=1e-10, rtol=1e-8)
                checks.append(dict(repeat=repeat, fold=fold, max_abs_error=float(np.abs(actual - expected[repeat, held]).max())))
        model = ReasonClassifier().fit(X, Y, U)
        joblib.dump(model, dest / 'reason_classifier.joblib', compress=3)
        # Fit has finished before the test file is opened.
        test = pd.read_csv(ROOT / 'data/part1_test.csv')
        template = pd.read_csv(ROOT / 'data/part1_sample_submission.csv')
        assert set(test.ID).isdisjoint(known.ID) and set(test.ID).isdisjoint(unknown.ID)
        probability = model.predict_proba(test[FEATURES].to_numpy())
        reloaded = joblib.load(dest / 'reason_classifier.joblib')
        np.testing.assert_allclose(reloaded.predict_proba(test[FEATURES].to_numpy()), probability, atol=0, rtol=0)
        prediction = model.predict(test[FEATURES].to_numpy())
        predictions = [' '.join(r for r, present in zip(REASONS, row) if present) for row in prediction]
        submission = pd.DataFrame(dict(ID=test.ID, Predicted=predictions)).set_index('ID').loc[template.ID].reset_index()
        assert submission.ID.tolist() == template.ID.tolist()
        assert len(submission) == 312 and submission.ID.is_unique and submission.Predicted.str.len().gt(0).all()
        submission.to_csv(dest / 'part1_simple_submission.csv', index=False)
        pd.DataFrame(probability, index=test.ID, columns=REASONS).to_csv(dest / 'test_probabilities.csv')
        pseudo = unknown[['ID', 'start_year']].copy()
        pseudo['accepted_for_classifier_fit'] = model.accepted_mask_
        U_probability = model.predict_proba(U)
        U_prediction = model.predict(U)
        for j, reason in enumerate(REASONS):
            pseudo[reason] = U_prediction[:, j]
            pseudo[f'probability_{reason}'] = U_probability[:, j]
            fit_labels = np.full(len(U), -100)
            fit_labels[model.accepted_mask_] = model.accepted_labels_[:, j]
            pseudo[f'fit_target_{reason}'] = fit_labels
        pseudo.to_csv(dest / 'generated_labels.csv', index=False)
        train = pd.read_csv(ROOT / 'data/part1_train.csv')
        masked = train[REASONS].lt(0).all(axis=1)
        assert train.loc[masked, 'ID'].tolist() == unknown.ID.tolist()
        augmented = train.copy()
        augmented.loc[masked, REASONS] = U_prediction
        augmented['label_origin'] = np.where(masked, 'model_prediction', 'verified_training')
        np.testing.assert_array_equal(augmented.loc[~masked, REASONS], train.loc[~masked, REASONS])
        augmented.to_csv(dest / 'part1_train_augmented.csv', index=False)
        # The PDF requires an unsupervised label-generation experiment. Persist
        # the evaluated K-means baseline's augmented dataset, separately labeled.
        cluster_spec = dict(family='cluster', geometry='scalar', k=40, cutoff=.3)
        cluster = SimpleModel(cluster_spec).fit(X, Y, U)
        cluster_prediction = decide(cluster.predict_proba(U), .3)
        cluster_augmented = train.copy()
        cluster_augmented.loc[masked, REASONS] = cluster_prediction
        cluster_augmented['label_origin'] = np.where(masked, 'kmeans_generated', 'verified_training')
        cluster_augmented.to_csv(dest / 'kmeans_comparison_train_augmented.csv', index=False)
    pd.DataFrame(checks).to_csv(dest / 'independent_40_fold_replay.csv', index=False)
    manifest = dict(status='unsubmitted development candidate; historical reference unchanged',
                    selection=selected, known_labels=235, originally_unlabeled_rows=1014,
                    final_self_training_counts=model.round_counts_, test_rows=312,
                    held_features_used_in_cv_fit=False, test_features_used_for_fit=False,
                    prediction_policy='Probability >=0.3; select the highest probability reason when none qualify.',
                    unsupervised_comparison='K-means k40 augmented labels are a separate evaluated baseline, not the selected classifier targets.',
                    cv_mean_sample_f1=.9131914893617021, cv_reference_sample_f1=.9088652482269504,
                    kaggle_score=None, independent_fold_replay_count=len(checks), reload_exact=True,
                    versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__),
                    source_sha256={p.name:digest(p) for p in [ROOT/'simple_solution.py', ROOT/'simple_models.py', ROOT/'simple_benchmark.py', ROOT/'simple_verify.py', ROOT/'simple_prepare.py']},
                    input_sha256={p.name:digest(p) for p in (ROOT/'data').glob('part1_*.csv')},
                    artifact_sha256={p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'})
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:manifest[k] for k in ['status','known_labels','final_self_training_counts','cv_mean_sample_f1','independent_fold_replay_count','reload_exact']}, indent=2))


if __name__ == '__main__':
    main()
