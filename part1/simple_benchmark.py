"""Use the existing 40 saved folds; retain every result and out-of-fold prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from simple_models import FEATURES, REASONS, SimpleModel, decide, first_stage

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/paired_recipe_cv'
OUT = ROOT / 'experiments/simple_methodical'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    train = pd.read_csv(ROOT / 'data/part1_train.csv')
    valid = pd.read_csv(ROOT / 'data/part1_val.csv')
    visible = train[REASONS].ge(0).all(axis=1)
    known = pd.concat([train.loc[visible], valid], ignore_index=True)
    unknown = train.loc[~visible].reset_index(drop=True)
    assert len(known) == 235 and len(unknown) == 1014
    assert set(known.ID).isdisjoint(unknown.ID)
    return known, unknown


def metrics(Y, prediction):
    return dict(sample_f1=float(f1_score(Y, prediction, average='samples', zero_division=0)),
                macro_f1=float(f1_score(Y, prediction, average='macro', zero_division=0)),
                **dict(zip([f'f1_{r}' for r in REASONS], f1_score(Y, prediction, average=None, zero_division=0))))


def load_references(known, folds):
    result = {}
    for name in ['ani_train_only', 'dbscan_train_only']:
        all_pred = []
        for repeat in range(8):
            frame = pd.concat([pd.read_csv(BASE / 'predictions' / f'repeat{repeat}_fold{f}__{name}.csv') for f in range(5)]).set_index('ID').loc[known.ID]
            np.testing.assert_array_equal(frame[[f'actual_{r}' for r in REASONS]], known[REASONS])
            all_pred.append(frame[[f'predicted_{r}' for r in REASONS]].to_numpy())
        result[name] = np.stack(all_pred)
    return result


def report(stage, known, folds, predictions):
    Y = known[REASONS].to_numpy(int)
    repeats, fold_rows = [], []
    all_predictions = {**load_references(known, folds), **predictions}
    for name, P in all_predictions.items():
        for repeat in range(8):
            repeats.append(dict(name=name, repeat=repeat, **metrics(Y, P[repeat])))
            for fold in range(5):
                held = folds[repeat] == fold
                fold_rows.append(dict(name=name, repeat=repeat, fold=fold, **metrics(Y[held], P[repeat, held])))
    rs, fs = pd.DataFrame(repeats), pd.DataFrame(fold_rows)
    summary = rs.groupby('name').agg(mean_f1=('sample_f1', 'mean'), repeat_sd=('sample_f1', 'std'), macro_f1=('macro_f1', 'mean'), **{f'f1_{r}': (f'f1_{r}', 'mean') for r in REASONS})
    for ref, short in [('ani_train_only', 'ani'), ('dbscan_train_only', 'dbscan')]:
        rr = rs.loc[rs.name == ref, 'sample_f1'].to_numpy()
        fr = fs.loc[fs.name == ref, 'sample_f1'].to_numpy()
        for name in summary.index:
            dr = rs.loc[rs.name == name, 'sample_f1'].to_numpy() - rr
            df = fs.loc[fs.name == name, 'sample_f1'].to_numpy() - fr
            summary.loc[name, f'delta_{short}'] = dr.mean()
            summary.loc[name, f'repeat_wins_{short}'] = (dr > 1e-12).sum()
            summary.loc[name, f'fold_wins_{short}'] = (df > 1e-12).sum()
            summary.loc[name, f'fold_ties_{short}'] = (np.abs(df) < 1e-12).sum()
    summary = summary.sort_values('mean_f1', ascending=False)
    rs.to_csv(stage / 'repeat_scores.csv', index=False)
    fs.to_csv(stage / 'fold_scores.csv', index=False)
    summary.to_csv(stage / 'summary.csv')
    print(summary[['mean_f1', 'repeat_sd', 'macro_f1', 'delta_ani', 'repeat_wins_ani', 'fold_wins_ani']].to_string(), flush=True)
    return summary


def run(stage_name, specs):
    stage = OUT / stage_name
    stage.mkdir(parents=True, exist_ok=True)
    known, unknown = load_data()
    X, Y, U = known[FEATURES].to_numpy(), known[REASONS].to_numpy(int), unknown[FEATURES].to_numpy()
    fold_paths = [BASE / 'splits' / f'repeat{r}.csv' for r in range(8)]
    folds = np.array([pd.read_csv(p).set_index('ID').loc[known.ID, 'fold'].to_numpy() for p in fold_paths])
    assert all(np.array_equal(np.bincount(f), np.repeat(47, 5)) for f in folds)
    protocol = dict(stage=stage_name, hypotheses='Direct reason embeddings; simple geometry, local neighbors, and linear boundaries. No auxiliary features or IDs as predictors.',
                    specs=specs, fixed_folds={p.name: digest(p) for p in fold_paths},
                    inputs={n: digest(ROOT / 'data' / n) for n in ['part1_train.csv', 'part1_val.csv']},
                    policy='Fit only 188 known fitting rows plus 1014 unlabeled training rows. Exclude held/test inputs from every fit.',
                    metric='Mean sample F1 of all 235 OOF rows per repeat, then average 8 repeats.',
                    interpretation='Development benchmark; repeated folds overlap and are not independent test sets.')
    protocol_path = stage / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Use a new stage for changed hypotheses.'
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    predictions = {s['name']: np.full((8, 235, 5), -1, dtype=np.int8) for s in specs}
    probabilities = {s['name']: np.full((8, 235, 5), np.nan) for s in specs}
    start = time.time()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for repeat in range(8):
            for fold in range(5):
                held = folds[repeat] == fold
                assert held.sum() == 47 and (~held).sum() == 188
                # Cache only fits that differ in the decision cutoff.
                cache = {}
                for s in specs:
                    ignored = ('name',) if s['family'] in ('simple_ani', 'self_train') else ('name', 'cutoff')
                    key = json.dumps({k: v for k, v in s.items() if k not in ignored}, sort_keys=True)
                    if key not in cache:
                        model = SimpleModel(s).fit(X[~held], Y[~held], U)
                        cache[key] = model.predict_proba(X[held])
                    P = cache[key]
                    assert np.isfinite(P).all()
                    probabilities[s['name']][repeat, held] = P
                    predictions[s['name']][repeat, held] = decide(P, s['cutoff'])
            print(f'{stage_name}: repeat {repeat + 1}/8 complete, {time.time() - start:.1f}s', flush=True)
    assert all((p >= 0).all() for p in predictions.values())
    np.savez_compressed(stage / 'predictions.npz', ids=known.ID.to_numpy(str), truth=Y, folds=folds, **predictions)
    np.savez_compressed(stage / 'probabilities.npz', **probabilities)
    # Independent disk rescore, so summary is tied to persisted predictions.
    with np.load(stage / 'predictions.npz') as saved:
        reloaded = {s['name']: saved[s['name']] for s in specs}
    report(stage, known, folds, reloaded)
    return stage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specs', type=Path)
    parser.add_argument('--stage', default='01_simple_baselines')
    args = parser.parse_args()
    run(args.stage, json.loads(args.specs.read_text()) if args.specs else first_stage())
