"""Independently rescore temporary predictions, or verify retained compact evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run import HERE, LABELS, VARIANTS, check_frozen, cases, fitting_context, strict, write_json


def row_f1(truth, pred):
    return 2 * (truth * pred).sum(1) / (truth.sum(1) + pred.sum(1))


def label_metrics(truth, pred):
    tp = (truth * pred).sum(0)
    fp = ((1 - truth) * pred).sum(0)
    fn = (truth * (1 - pred)).sum(0)
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
    return divide(tp, tp + fp), divide(tp, tp + fn), divide(2 * tp, 2 * tp + fp + fn), truth.sum(0)


def verify_compact(output):
    check_frozen()
    repeats = pd.read_csv(output / 'repeat_scores.csv')
    summary = pd.read_csv(output / 'summary.csv')
    labels = pd.read_csv(output / 'per_label_repeat_scores.csv')
    averaged = labels.groupby(['pool', 'recipe', 'label'])[['precision', 'recall', 'f1', 'support']].mean()
    saved = pd.read_csv(output / 'per_label_summary.csv').set_index(['pool', 'recipe', 'label'])
    np.testing.assert_allclose(averaged.to_numpy(), saved.loc[averaged.index, averaged.columns].to_numpy(), atol=1e-14, rtol=0)
    paired = []
    for _, row in summary.iterrows():
        sub = repeats[(repeats.pool == row.pool) & (repeats.recipe == row.recipe)].set_index('seed')
        parent = repeats[(repeats.pool == row.pool) & (repeats.recipe == 'anchored')].set_index('seed')
        delta = sub.sample_f1 - parent.loc[sub.index, 'sample_f1']
        np.testing.assert_allclose([sub.sample_f1.mean(), sub.sample_f1.std(ddof=0),
            sub.macro_f1.mean(), delta.mean()], [row.sample_f1_mean, row.sample_f1_population_sd,
            row.macro_f1_mean, row.paired_delta_mean], atol=1e-14, rtol=0)
        assert len(sub) == row.repeats
        assert ((delta > 1e-12).sum(), (abs(delta) <= 1e-12).sum(), (delta < -1e-12).sum()) == (row.wins, row.ties, row.losses)
        for seed, d in delta.items():
            paired.append(dict(pool=row.pool, recipe=row.recipe, seed=seed, delta_vs_anchored=float(d)))
    pd.DataFrame(paired).to_csv(output / 'paired_repeat_scores.csv', index=False)
    screen = repeats[repeats.seed.isin([20261126, 20261127, 20261128])]
    passers = []
    for variant in VARIANTS:
        passed = []
        for pool in ['train', 'pooled']:
            sub = screen[(screen.pool == pool) & (screen.recipe == variant)].set_index('seed')
            parent = screen[(screen.pool == pool) & (screen.recipe == 'anchored')].set_index('seed')
            delta = sub.sample_f1 - parent.sample_f1
            passed.append(delta.mean() > 0 and (delta > 1e-12).sum() >= 2)
        if all(passed): passers.append(variant)
    assert passers == json.loads((output / 'screen_gate.json').read_text())['passers']
    return dict(compact_metrics_and_gate_verified=True, source_and_protected_hashes_verified=True,
                screen_passers=passers)


def verify_predictions(output):
    counts = dict(outer_cases=0, inner_boundaries=0, prediction_rows=0)
    repeat_scores = pd.read_csv(output / 'repeat_scores.csv').set_index(['pool', 'seed', 'recipe'])
    label_scores = pd.read_csv(output / 'per_label_repeat_scores.csv').set_index(['pool', 'seed', 'recipe', 'label'])
    complementarity = []
    for stage in ['screen', 'confirm', 'supplementary']:
        folder = output / stage
        if not folder.exists(): continue
        grouped = {}
        for case in cases(stage):
            key = f'{case["pool"]}_{case["seed"]}_{case["fold"]}'
            p = folder / f'{key}.npz'
            assert p.exists(), p
            a = np.load(p)
            np.testing.assert_array_equal(a['ids'], case['query'].ID.to_numpy(str))
            np.testing.assert_array_equal(a['truth'], case['query'][LABELS].to_numpy(int))
            for name in a.files:
                if not name.startswith('scores_'): continue
                scores = a[name]
                assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1 + 1e-12).all()
                pred = (scores >= .3).astype(int)
                pred[np.arange(len(pred)), scores.argmax(1)] = 1
                np.testing.assert_array_equal(pred, a[name.replace('scores_', 'predictions_')])
                counts['prediction_rows'] += len(pred)
            ctx = fitting_context(case['fit'])
            g = strict.groups(ctx.D)
            coverage = np.zeros(len(ctx.train), int)
            for tr, te in strict.inner_folds(ctx):
                query = ctx.train[te]
                train = ctx.train[tr]
                pool = ctx.pool[~np.isin(g[ctx.pool], g[query])]
                assert not np.intersect1d(g[np.r_[train, pool]], g[query]).size
                assert not np.intersect1d(np.r_[train, pool], query).size
                assert (ctx.D.Y[train].sum(0) > 0).all()
                coverage[te] += 1
                counts['inner_boundaries'] += 1
            assert (coverage == 1).all()
            counts['outer_cases'] += 1
            grouped.setdefault((case['pool'], case['seed']), []).append(a)
        for (pool, seed), arrays in grouped.items():
            truth = np.concatenate([a['truth'] for a in arrays])
            anchor = np.concatenate([a['predictions_anchored'] for a in arrays])
            ids = np.concatenate([a['ids'] for a in arrays])
            assert len(ids) == len(np.unique(ids))
            anchor_f1 = row_f1(truth, anchor)
            for name in arrays[0].files:
                if not name.startswith('predictions_'): continue
                recipe = name.removeprefix('predictions_')
                pred = np.concatenate([a[name] for a in arrays])
                rf = row_f1(truth, pred)
                precision, recall, lf, support = label_metrics(truth, pred)
                record = repeat_scores.loc[(pool, seed, recipe)]
                np.testing.assert_allclose([rf.mean(), lf.mean()], [record.sample_f1, record.macro_f1], atol=1e-14, rtol=0)
                for j, label in enumerate(LABELS):
                    row = label_scores.loc[(pool, seed, recipe, label)]
                    np.testing.assert_allclose([precision[j], recall[j], lf[j], support[j]],
                        [row.precision, row.recall, row.f1, row.support], atol=1e-14, rtol=0)
                if recipe == 'anchored': continue
                exact = (truth == pred).all(1)
                anchor_exact = (truth == anchor).all(1)
                complementarity.append(dict(pool=pool, seed=seed, recipe=recipe,
                    changed_rows=int((pred != anchor).any(1).sum()),
                    higher_row_f1=int((rf > anchor_f1).sum()), lower_row_f1=int((rf < anchor_f1).sum()),
                    exact_fixes=int((exact & ~anchor_exact).sum()),
                    exact_breaks=int((~exact & anchor_exact).sum()),
                    total_row_f1_gain=float((rf - anchor_f1).clip(0).sum()),
                    total_row_f1_loss=float((anchor_f1 - rf).clip(0).sum())))
    pd.DataFrame(complementarity).to_csv(output / 'complementarity.csv', index=False)
    audits = json.loads((output / 'fit_audits.json').read_text())
    assert all(a['gold_unchanged'] for a in audits)
    checks = [a for a in audits if a.get('no_teacher_clone_exact')]
    assert len(checks) == 1 and checks[0]['zero_residual_identity'] and checks[0]['query_batch_independence']
    return dict(**counts, independent_row_and_label_rescoring=True, decoder_verified=True,
                ids_truth_and_fold_coverage_verified=True, nested_group_boundaries_verified=True,
                baseline_reference_max_abs=max(a['reference_score_max_abs'] or 0 for a in audits),
                preflight_no_teacher_clone_exact=True, preflight_zero_residual_identity=True,
                preflight_query_batch_independence=True, gold_unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--predictions', action='store_true')
    args = parser.parse_args()
    result = verify_compact(args.output)
    if args.predictions:
        result.update(verify_predictions(args.output))
        write_json(args.output / 'verification.json', result)
    else:
        assert json.loads((args.output / 'verification.json').read_text())['independent_row_and_label_rescoring']
        result['historical_full_rescoring_record_present'] = True
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
