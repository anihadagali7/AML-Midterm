"""Independent rescoring and paired summaries; no fitting or parameter selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
PART = HERE.parents[1]
REPO = PART.parent
OLD = PART / 'experiments/derived_benchmark_audit_v1'
LABELS = ['Safety', 'Funding', 'Efficacy', 'Recruitment', 'Other']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def metrics(y, p):
    pr, re, f, support = precision_recall_fscore_support(y, p, zero_division=0)
    return dict(sample_f1=f1_score(y, p, average='samples', zero_division=0), macro_f1=f.mean(),
                **{f'{key}_{label}': float(v[j]) for key, v in [('precision', pr), ('recall', re), ('f1', f), ('support', support)] for j, label in enumerate(LABELS)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--five-seed', action='store_true')
    args = ap.parse_args()
    OUT = HERE / 'five_seed' if args.five_seed else HERE
    protocol = read(HERE / 'protocol.json')
    protected = dict(protocol['source_hashes'])
    if args.five_seed:
        protected.update(read(OUT / 'protocol.json')['source_hashes'])
    for p, h in protected.items(): assert sha(REPO / p) == h, p
    plan = read(HERE / 'case_plan.json')
    previous_plan = read(OLD / 'splits.json')
    expected_plan = []
    for access in ['native_dylan', 'strict_inductive']:
        for c in previous_plan:
            take = (c['track'] == 'dylan' or c['kind'] == 'validation') if access == 'native_dylan' else c['track'] == 'canonical'
            if take:
                expected_plan.append(dict(c, access=access, source_key=c['key'], key=access+'__'+c['key']))
    assert plan == expected_plan
    models = protocol['models']
    full = pd.concat([pd.read_csv(PART / 'data/part1_train.csv'), pd.read_csv(PART / 'data/part1_val.csv')], ignore_index=True).set_index('ID')
    groups = pd.util.hash_pandas_object(full[[f'emb_{i}' for i in range(150)]].round(10), index=False)
    all_gold = set(full.index[full[LABELS].ge(0).all(axis=1)])
    train_ids = set(pd.read_csv(PART / 'data/part1_train.csv').ID)
    true_train = all_gold & train_ids
    test_ids = set(pd.read_csv(PART / 'data/part1_test.csv').ID)
    case_rows, row_scores, collected, coverage = [], [], {}, {}
    anchor_checks, nested_contexts = 0, 0
    for c in plan:
        pth = OUT / 'results' / (c['key'] + '.npz')
        meta = read(pth.with_suffix('.json')); assert sha(pth) == meta['sha256']
        nested_contexts += meta['nested_context_count']
        assert meta['nested_gold_group_overlaps'] == meta['nested_masked_group_overlaps'] == 0
        fit, query = set(c['fit_ids']), set(c['query_ids'])
        assert not fit & query and not (fit | query) & test_ids
        if c['access'] == 'strict_inductive' and c['kind'] != 'validation':
            assert not set(groups.loc[c['fit_ids']]) & set(groups.loc[c['query_ids']])
        if c['kind'] == 'train': assert (fit | query) <= train_ids
        if c['kind'] == 'chronology': assert full.loc[c['fit_ids']].start_year.max() < full.loc[c['query_ids']].start_year.min()
        if c['kind'] in ['A', 'train', 'pooled']:
            coverage.setdefault((c['access'], c['kind'], c['repeat']), []).extend(c['query_ids'])
        with np.load(pth) as z, np.load(OLD / 'results' / (c['source_key'] + '.npz')) as old:
            assert list(z['ids']) == c['query_ids']
            np.testing.assert_array_equal(z['truth'], full.loc[c['query_ids'], LABELS].to_numpy())
            np.testing.assert_array_equal(z['ids'], old['ids'])
            np.testing.assert_array_equal(z['anchored_predictions'], old['anchored_predictions'])
            np.testing.assert_allclose(z['anchored_scores'], old['anchored_scores'], atol=1e-12, rtol=0)
            anchor_checks += 1
            for model in models:
                y, p, s = z['truth'], z[model + '_predictions'], z[model + '_scores']
                assert p.shape == s.shape == y.shape and np.isfinite(s).all()
                assert (p.sum(1) > 0).all() and set(np.unique(p)) <= {0, 1}
                if model.startswith('cand'):
                    Q, CM = z[model + '_Q'], z['combo_matrix']
                    np.testing.assert_allclose(Q.sum(1), 1, atol=1e-10)
                    np.testing.assert_allclose(s, Q @ CM, atol=1e-12)
                    # Independently enumerate all nonempty decisions for expected F1.
                    choices = np.array([[(value >> j) & 1 for j in range(5)] for value in range(1, 32)])
                    F = 2 * (choices @ CM.T) / (choices.sum(1)[:, None] + CM.sum(1))
                    np.testing.assert_array_equal(p, choices[(Q @ F.T).argmax(1)])
                else:
                    expected = (s >= .3).astype(int)
                    expected[np.arange(len(s)), s.argmax(1)] = 1
                    np.testing.assert_array_equal(p, expected)
                    if model != 'anchored':
                        members = protocol['blends']['members'][model]
                        np.testing.assert_allclose(s, sum(z[m + '_scores'] for m in members) / len(members), atol=1e-12)
                case_rows.append(dict(model=model, access=c['access'], kind=c['kind'], repeat=c.get('repeat'),
                                      cut=c.get('cut'), case=c['key'], **metrics(y, p)))
                values = 2 * (y * p).sum(1) / (y.sum(1) + p.sum(1))
                for ident, val in zip(z['ids'], values):
                    row_scores.append(dict(model=model, access=c['access'], kind=c['kind'], repeat=c.get('repeat'),
                                           cut=c.get('cut'), case=c['key'], ID=ident, f1=val))
                block = c['repeat'] if c['kind'] in ['A', 'train', 'pooled'] else c['cut'] if c['kind'] == 'B' else 'all'
                collected.setdefault((model, c['access'], c['kind'], block), []).append((y.copy(), p.copy()))
    for (access, kind, repeat), ids in coverage.items():
        assert len(ids) == len(set(ids)) and set(ids) == (true_train if kind == 'train' else all_gold)
    assert len(plan) == anchor_checks == 138 and len(coverage) == 26
    pd.DataFrame(case_rows).to_csv(OUT / 'case_metrics.csv', index=False)
    rf = pd.DataFrame(row_scores); rf.to_csv(OUT / 'row_f1.csv.gz', index=False)
    blocks = []
    for (model, access, kind, block), vals in collected.items():
        y, p = [np.concatenate([v[j] for v in vals]) for j in [0, 1]]
        blocks.append(dict(model=model, access=access, kind=kind, block=block, **metrics(y, p)))
    bdf = pd.DataFrame(blocks); bdf.to_csv(OUT / 'block_metrics.csv', index=False)
    aggregated, comparisons, repeats = [], [], []
    for (model, access, kind), block in bdf.groupby(['model', 'access', 'kind']):
        value = dict(model=model, access=access, kind=kind, blocks=len(block),
                     sample_f1_sd_population=block.sample_f1.std(ddof=0) if kind in ['A','train','pooled'] else None,
                     **block.drop(columns=['model','access','kind','block']).mean().to_dict())
        aggregated.append(value)
    adf = pd.DataFrame(aggregated); adf.to_csv(OUT / 'aggregate_metrics.csv', index=False)
    for access in ['native_dylan', 'strict_inductive']:
        table = adf[adf.access == access].pivot(index='model', columns='kind', values='sample_f1').reindex(models)
        if access == 'native_dylan': table['J'] = (table.A + table.B) / 2
        table.to_csv(OUT / (access + '_comparison.csv'))
        comparisons.append(table)
        for kind in (['A'] if access == 'native_dylan' else ['train', 'pooled']):
            r = bdf[(bdf.access == access) & (bdf.kind == kind)].pivot(index='block', columns='model', values='sample_f1')
            for model in models:
                d = r[model] - r.anchored
                repeats.append(dict(access=access, kind=kind, model=model, repeats=len(d),
                                    mean_delta=d.mean(), delta_sd_population=d.std(ddof=0),
                                    wins=int((d > 1e-12).sum()), ties=int((abs(d) <= 1e-12).sum()), losses=int((d < -1e-12).sum())))
    pd.DataFrame(repeats).to_csv(OUT / 'paired_repeat_summary.csv', index=False)

    native = comparisons[0]
    reproduction = []
    stems = dict(cand1='cand1_selected_1se', cand2='cand2_compliant_argmax', cand4='cand4_ens3_cs_sld', cand5='cand5_lr_cs_sld_robust')
    for model, stem in stems.items():
        expected = read(HERE / 'manifests' / (stem + '.manifest.json'))['scores']
        for metric in ['A', 'B', 'validation', 'J']:
            key = 'C' if metric == 'validation' else metric
            ref = expected[key] if key == 'J' else expected[key]['sample_f1']
            reproduction.append(dict(model=model, metric=metric, supplied=ref, measured=native.loc[model, metric],
                                     delta=native.loc[model, metric] - ref))
    supplied_blends = read(HERE / 'supplied_ensemble_results.json')
    for model, key in [('anchored_cand1', '== M+cand1 equal'), ('anchored_cand5', '== M+cand5 equal'), ('anchored_cand1_cand5', '== M+cand1+cand5 equal')]:
        ref = supplied_blends[key]
        for metric in ['A','B','validation','J']:
            k = 'C' if metric == 'validation' else metric
            expected = ref[k] if k == 'J' else ref[k]['sample_f1']
            reproduction.append(dict(model=model, metric=metric, supplied=expected, measured=native.loc[model, metric], delta=native.loc[model, metric]-expected))
    repro = pd.DataFrame(reproduction)
    repro.to_csv(OUT / ('native_vs_supplied_single_seed.csv' if args.five_seed else 'native_reproduction.csv'), index=False)

    # Shared group resamples preserve pairing and overlaps between A/B and repeats.
    gold_order = full.index[full[LABELS].ge(0).all(axis=1)]
    group_pos = pd.Series(pd.factorize(groups.loc[gold_order])[0], index=gold_order)
    n = group_pos.nunique(); NBOOT = 5000
    weights = np.random.default_rng(20261107).multinomial(n, np.full(n, 1/n), NBOOT)
    def bootstrap(frame):
        scores = frame.groupby(['ID','model']).f1.mean().unstack('model')[models]
        g = group_pos.loc[scores.index].to_numpy()
        count = np.bincount(g, minlength=n)
        sums = np.zeros((n, len(models))); np.add.at(sums, g, scores.to_numpy())
        return (weights @ sums) / (weights @ count)[:,None]
    intervals = []
    for access, table in zip(['native_dylan','strict_inductive'], comparisons):
        bs = {}
        for kind in table.columns:
            frame = rf[(rf.access == access) & (rf.kind == kind)]
            if kind == 'J': continue
            bs[kind] = sum(bootstrap(frame[frame.cut == cut]) for cut in [2004,2005,2006])/3 if kind == 'B' else bootstrap(frame)
        if access == 'native_dylan': bs['J'] = (bs['A'] + bs['B'])/2
        for kind, samples in bs.items():
            for j, model in enumerate(models):
                d = samples[:,j] - samples[:,0]
                lo, hi = np.quantile(d, [.025,.975])
                intervals.append(dict(access=access, metric=kind, model=model,
                                      delta=table.loc[model,kind]-table.loc['anchored',kind],
                                      conditional_ci_low=lo, conditional_ci_high=hi, bootstrap_se=d.std(ddof=1)))
    pd.DataFrame(intervals).to_csv(OUT / 'paired_intervals.csv', index=False)
    save(OUT / 'verification.json', dict(status='passed', cases=len(plan), recipe_evaluations=len(case_rows),
         seed_convention='five-inner-seed raw posterior average' if args.five_seed else 'published benchmark seed0',
         deterministic_anchored_reused=args.five_seed,
         anchored_historical_prediction_checks=anchor_checks, complete_OOF_repeats=len(coverage),
         protected_source_hashes=len(protected), nested_context_audits=nested_contexts,
         **({'native_difference_vs_supplied_single_seed_max_abs':float(abs(repro.delta).max())} if args.five_seed else
            {'native_reproduction_max_abs_error':float(abs(repro.delta).max()),'native_exact_to_1e_12':bool((abs(repro.delta)<1e-12).all())}),
         bootstrap={'resamples':NBOOT,'seed':20261107,'unit':'222 embedding groups, common weights across models and occurrences',
                    'limitation':'Conditional on fitted predictions; not independent validation or uncertainty of a new Kaggle result.'}))
    for access, table in zip(['native_dylan','strict_inductive'], comparisons):
        print(access); print(table.to_string())
    print('Native difference from supplied single-seed scores:', abs(repro.delta).max())


if __name__ == '__main__':
    main()
