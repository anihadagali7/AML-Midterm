"""Reference replay and fixed later-year diagnostics, after candidate selection."""
import ast
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from simple_models import SimpleModel, FEATURES, REASONS, decide
from simple_benchmark import BASE, ROOT, OUT, digest, load_data, load_references, metrics


def reference_functions():
    """Execute only the original notebook's pure recipe functions, never its driver."""
    notebook = json.loads((ROOT / 'paired_recipe_cv.ipynb').read_text())
    source = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    names = {'fit_ani_base', 'predict_ani_base', 'decide', 'fit_ani', 'predict_packet'}
    tree = ast.parse(source)
    definitions = '\n\n'.join(ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
    namespace = dict(np=np, warnings=warnings, ConvergenceWarning=ConvergenceWarning,
                     LogisticRegression=LogisticRegression, normalize=normalize, FEATURES=FEATURES, REASONS=REASONS)
    exec(compile(definitions, str(ROOT / 'paired_recipe_cv.ipynb'), 'exec'), namespace)
    return namespace


def fit_reference(frame, fit_rows, unknown_rows, held_rows):
    ns = reference_functions()
    X, Y = frame[FEATURES].to_numpy(), frame[REASONS].to_numpy(int)
    pool = np.sort(np.r_[fit_rows, unknown_rows])
    assert not np.intersect1d(pool, held_rows).size
    scaler = StandardScaler().fit(X[pool])
    Z = scaler.transform(X)
    pca = PCA(30, random_state=0).fit(Z[pool])
    gmm = GaussianMixture(30, covariance_type='diag', n_init=3, random_state=0).fit(pca.transform(Z[pool]))
    assert gmm.converged_
    ns.update(X=X, Y=Y, original_unlabeled=unknown_rows, test_rows=np.array([], int), all_frame=frame)
    prep = dict(name='independent_replay', scaler=scaler, ani_responsibility=gmm.predict_proba(pca.transform(Z)))
    packet, _ = ns['fit_ani']('independent_replay', 'train_only', fit_rows, held_rows, prep, persist=False)
    return ns['predict_packet'](packet, X[held_rows])


def main():
    audit = OUT / 'verification'
    audit.mkdir(exist_ok=True)
    selected = json.loads((OUT / 'selected.json').read_text())
    specs = selected['candidates']
    known, unknown = load_data()
    train = pd.read_csv(ROOT / 'data/part1_train.csv')
    valid = pd.read_csv(ROOT / 'data/part1_val.csv')
    frame = pd.concat([train, valid], ignore_index=True)
    is_known = frame[REASONS].ge(0).all(axis=1).to_numpy()
    known_rows, unknown_rows = np.flatnonzero(is_known), np.flatnonzero(~is_known)
    row_by_id = pd.Series(np.arange(len(frame)), index=frame.ID)
    assignments = pd.read_csv(BASE / 'splits/repeat0.csv').set_index('ID').loc[known.ID, 'fold'].to_numpy()
    held_rows = row_by_id.loc[known.loc[assignments == 0, 'ID']].to_numpy()
    fitting = np.setdiff1d(known_rows, held_rows)
    protocol = dict(selected_sha256=digest(OUT / 'selected.json'),
                    source_reference_sha256=digest(ROOT / 'paired_recipe_cv.ipynb'),
                    diagnostics=['original 157/78 split (already used in development)', 'known-label chronological cutoffs 2003, 2005, 2007'],
                    temporal_unlabeled_policy='Only originally unlabeled rows starting at or before the cutoff; all later labeled/unknown features excluded from fitting.',
                    caution='Diagnostics, not untouched test estimates. No candidate retuning after these scores.')
    (audit / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows, predictions = [], []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        P = fit_reference(frame, fitting, unknown_rows, held_rows)
        saved = pd.read_csv(BASE / 'predictions/repeat0_fold0__ani_train_only.csv').set_index('ID').loc[frame.iloc[held_rows].ID]
        expected = saved[[f'probability_{r}' for r in REASONS]].to_numpy()
        np.testing.assert_allclose(P, expected, atol=1e-10, rtol=1e-8)
        print('Original Ani fold replay matches saved probabilities.', flush=True)
        # Refit selected models independently and check the persisted OOF file.
        for spec in specs:
            s = spec['recipe']
            saved = np.load(OUT / spec['stage'] / 'probabilities.npz')[s['name']][0, assignments == 0]
            model = SimpleModel(s).fit(frame.iloc[fitting][FEATURES].to_numpy(), frame.iloc[fitting][REASONS].to_numpy(int), frame.iloc[unknown_rows][FEATURES].to_numpy())
            replay = model.predict_proba(frame.iloc[held_rows][FEATURES].to_numpy())
            np.testing.assert_allclose(replay, saved, atol=1e-10, rtol=1e-8)
        print('Selected independent fold refits match saved probabilities.', flush=True)
        splits = [('original_validation', np.flatnonzero(is_known & (np.arange(len(frame)) < len(train))), unknown_rows,
                   np.arange(len(train), len(frame)))]
        for year in [2003, 2005, 2007]:
            early = frame.start_year.to_numpy() <= year
            splits.append((f'after_{year}', np.flatnonzero(is_known & early), np.flatnonzero(~is_known & early), np.flatnonzero(is_known & ~early)))
        for label, fit, unlab, held in splits:
            X = frame.iloc[fit][FEATURES].to_numpy()
            Y = frame.iloc[fit][REASONS].to_numpy(int)
            U = frame.iloc[unlab][FEATURES].to_numpy()
            Q = frame.iloc[held][FEATURES].to_numpy()
            actual = frame.iloc[held][REASONS].to_numpy(int)
            results = {'ani_train_only': (fit_reference(frame, fit, unlab, held), .3)}
            for spec in specs:
                s = spec['recipe']
                model = SimpleModel(s).fit(X, Y, U)
                results[s['name']] = (model.predict_proba(Q), s['cutoff'])
            for name, (P, cutoff) in results.items():
                pred = decide(P, cutoff)
                rows.append(dict(split=label, name=name, verified_fit=len(fit), unlabeled_fit=len(unlab), held=len(held), **metrics(actual, pred)))
                result = pd.DataFrame(dict(ID=frame.iloc[held].ID, split=label, name=name))
                for j, reason in enumerate(REASONS):
                    result[f'actual_{reason}'] = actual[:, j]
                    result[f'predicted_{reason}'] = pred[:, j]
                    result[f'probability_{reason}'] = P[:, j]
                predictions.append(result)
            print(label, {r['name']: round(r['sample_f1'], 6) for r in rows if r['split'] == label}, flush=True)
    pd.DataFrame(rows).to_csv(audit / 'diagnostic_scores.csv', index=False)
    pd.concat(predictions).to_csv(audit / 'diagnostic_predictions.csv', index=False)
    # Trial-level paired uncertainty: keep all 8 predictions for a trial together.
    refs = load_references(known, None)
    Y = known[REASONS].to_numpy(int)
    def trial_f1(pred):
        return 2 * (pred * Y).sum(axis=2) / (pred.sum(axis=2) + Y.sum(axis=1))
    rng = np.random.default_rng(20261105)
    draws = rng.integers(0, 235, size=(10000, 235))
    uncertainty = []
    for spec in specs:
        s = spec['recipe']
        pred = np.load(OUT / spec['stage'] / 'predictions.npz')[s['name']]
        for ref, ref_pred in refs.items():
            difference = (trial_f1(pred) - trial_f1(ref_pred)).mean(axis=0)
            interval = np.quantile(difference[draws].mean(axis=1), [.025, .975])
            uncertainty.append(dict(name=s['name'], reference=ref, mean_delta=difference.mean(),
                                    lower_95=interval[0], upper_95=interval[1],
                                    caveat='Trial bootstrap of development predictions; not adjusted for model selection or correlated training fits.'))
    pd.DataFrame(uncertainty).to_csv(audit / 'paired_trial_bootstrap.csv', index=False)
    (audit / 'checks.json').write_text(json.dumps(dict(reference_fold_replay=True, selected_fold_refits=True,
                                                      held_and_test_features_excluded=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
