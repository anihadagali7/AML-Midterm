"""Common evaluation of six frozen Part 1 recipes; no submission/refit.

Commands: prepare, parity, train, pooled, validation, chronology, report, verify.
The existing benchmark runners and saved folds remain the scoring authority.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import html
import importlib.metadata
import json
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

import benchmark_cv as b
import benchmark_pooled_cv as pooled
import cross_repo_models as adapters

e = b.e
ROOT = b.ROOT
OUT = ROOT / 'experiments/cross_repo_benchmark_v1'
adapters.install(e)
LOCAL = ['Local SVM specialist hybrid', 'Classical tri-training', 'Current best hybrid']
FAMILIES = [('Ani training-only adaptation', 'repo_ani', 't0.3'),
            ('DBSCAN + logistic C1000', 'repo_dbscan', 't0.5'),
            ('K-means k8 + logistic C1000', 'repo_kmeans', 't0.5')]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    b.save(path, value)


def registry():
    return json.loads((OUT / 'registry.json').read_text())


def sources():
    paths = [ROOT / 'experiment.py', ROOT / 'benchmark_cv.py', ROOT / 'benchmark_pooled_cv.py',
             ROOT / 'cross_repo_models.py', ROOT / 'run_cross_repo_benchmark.py']
    paths += sorted(ROOT.glob('phase[0-9]*.py'))
    paths += sorted((ROOT / 'experiments').glob('phase*-baseline.json'))
    return {str(p.relative_to(ROOT)): b.file_hash(p) for p in paths}


def protected():
    paths = list((ROOT / 'data').glob('*.csv')) + [ROOT / 'experiments/current-best.json']
    paths += list((ROOT / 'predictions/current_best').glob('*'))
    for d in ['benchmark_v1', 'pooled_benchmark_v1']:
        paths += [ROOT / 'experiments' / d / n for n in ['protocol.json', 'registry.json', 'comparison.csv']]
    return {str(p.relative_to(ROOT)): b.file_hash(p) for p in paths if p.is_file()}


def frames():
    train = pd.read_csv(ROOT / 'data/part1_train.csv')
    val = pd.read_csv(ROOT / 'data/part1_val.csv')
    return train, val, pd.concat([train, val], ignore_index=True)


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        check_contract()
        print('Existing frozen protocol verified.', flush=True)
        return
    OUT.mkdir(exist_ok=True)
    rs = json.loads((ROOT / 'experiments/benchmark_v1/registry.json').read_text())
    selected = [next(r for r in rs if r['name'] == name) for name in LOCAL]
    for name, family, rule in FAMILIES:
        selected.append(dict(name=name, config=dict(model=family, adapter_sha256=b.file_hash(ROOT / 'cross_repo_models.py')),
                             rule=rule, incumbent=False, recorded_validation=None,
                             validation_source='New inductive evaluation of frozen imported recipe'))
    train, val, full = frames()
    gold = full[e.LABELS].ge(0).all(axis=1).to_numpy()
    group = pd.util.hash_pandas_object(full[e.EMB].round(10), index=False).to_numpy()
    chronological = []
    for cutoff, end in [(2004, 2006), (2006, 2007), (2007, 2010)]:
        held = np.flatnonzero(gold & full.start_year.gt(cutoff) & full.start_year.le(end))
        eligible = full.start_year.le(cutoff).to_numpy()
        removed = eligible & np.isin(group, group[held])
        allowed = np.flatnonzero(eligible & ~removed)
        labeled_fit = allowed[gold[allowed]]
        assert len(labeled_fit) >= 100 and len(held) >= 20
        assert (full.iloc[labeled_fit][e.LABELS].sum(axis=0) >= 2).all()
        assert not set(group[allowed]) & set(group[held])
        chronological.append(dict(name=f'through_{cutoff}_test_{cutoff + 1}_{end}', cutoff=cutoff, test_end=end,
            training_ids=full.iloc[allowed].ID.tolist(), heldout_ids=full.iloc[held].ID.tolist(),
            gold_rows=len(labeled_fit), masked_rows=int((~gold[allowed]).sum()), heldout_rows=len(held),
            purged_earlier_gold=int((removed & gold).sum()), purged_earlier_masked=int((removed & ~gold).sum()),
            gold_label_counts=full.iloc[labeled_fit][e.LABELS].sum().astype(int).to_dict(),
            heldout_label_counts=full.iloc[held][e.LABELS].sum().astype(int).to_dict()))
    protocol = dict(version='part1-cross-repo-v1', created_at=b.now(), candidates=[r['name'] for r in selected],
        primary='Existing grouped training-label CV, 5 folds x 3 frozen repeats',
        pooled='Existing grouped 235-label CV, 5 folds x 3 frozen repeats; separate from primary',
        frozen_split_sources={n: b.file_hash(ROOT / 'experiments' / n / 'protocol.json') for n in ['benchmark_v1', 'pooled_benchmark_v1']},
        metric='Sample-wise F1 pooled across all OOF rows within each repeat; mean of repeats; descriptive repeat SD',
        validation='All six frozen recipes evaluated on original 78 rows, with all original training inputs available. No validation/test input fitting. Historical local scores checked for reproduction.',
        chronological_splits=chronological,
        chronological_boundary_selection='Pre-score year/count inspection: first window has >=125 original gold, >=100 masked and >=3 positive examples per label before group purging; successive whole-year test windows have >=20 gold rows, merging the final undersized tail. No score-based cutoff selection.',
        chronological_metric='Pool the 98 non-overlapping held-out trial predictions for sample F1; report each window and per-label precision/recall/F1 separately. Windows are not independent repeats.',
        chronological_feature_policy='Only original masked and gold rows from years <= cutoff, with matching held-out embedding groups purged from all fitting. Future years excluded even if unlabeled.',
        chronological_limitation='Start year is the supplied proxy for time, not the date a stopping label became observable. Gold-label enrichment and sparse late labels prevent a direct estimate of Kaggle class proportions.',
        no_tuning=True, final_all_label_refit=False, kaggle_inputs_used=False, submissions_created=False,
        feature_access='Inductive; fold-local preprocessing, clustering, pseudo-labeling and fitting. Imported adapters preserve native recipe settings.',
        selection='Compare paired CV means and repeat differences in both pools; validation and chronology supplementary. No automatic promotion.',
        limitations='All labels were used in historical model development. Grouped folds emphasize unseen embedding patterns. No score is an untouched estimate of future performance.',
        protected_sha256=protected(), source_sha256=sources(),
        versions={n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'iterative-stratification']},
        source_manifest_sha256=b.file_hash(OUT / 'source_manifest.json'))
    save(OUT / 'registry.json', selected)
    protocol['registry_sha256'] = b.file_hash(OUT / 'registry.json')
    save(path, protocol)
    for p in chronological:
        print(json.dumps({k:v for k,v in p.items() if not k.endswith('_ids')}), flush=True)


def check_contract():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['protected_sha256'] == protected(), 'Protected input or benchmark changed.'
    assert p['source_sha256'] == sources(), 'Code changed after protocol freeze; document a new version.'
    assert p['registry_sha256'] == b.file_hash(OUT / 'registry.json')
    assert p['source_manifest_sha256'] == b.file_hash(OUT / 'source_manifest.json')
    return p


@contextmanager
def data_context(frame):
    keys = ['DATA', 'LAB', 'UNLAB', 'Y', 'GROUP', 'make_splits']
    old = {k:getattr(e, k) for k in keys}
    try:
        pooled.clear_caches()
        e.DATA = frame.reset_index(drop=True).copy()
        e.LAB = np.flatnonzero(e.DATA[e.LABELS].ge(0).all(axis=1))
        e.UNLAB = np.flatnonzero(e.DATA[e.LABELS].eq(-100).all(axis=1))
        e.Y = e.DATA.iloc[e.LAB][e.LABELS].to_numpy(int)
        e.GROUP = pd.util.hash_pandas_object(e.DATA[e.EMB].round(10), index=False).to_numpy()
        def make_splits(y=None, groups=None, seed=20261104, n_splits=5):
            return old['make_splits'](e.Y if y is None else y, e.GROUP[e.LAB] if groups is None else groups, seed, n_splits)
        e.make_splits = make_splits
        yield
    finally:
        pooled.clear_caches()
        for k,v in old.items():
            setattr(e, k, v)


def detailed(y, probability, rule):
    prediction = e.decisions(probability, rule)
    p, r, f, s = precision_recall_fscore_support(y, prediction, average=None, zero_division=0)
    return dict(**e.metrics(y, probability, rule), rows=len(y),
                reasons={label:dict(precision=float(p[j]), recall=float(r[j]), f1=float(f[j]),
                    positives=int(s[j]), predicted_positive=int(prediction[:,j].sum())) for j,label in enumerate(e.LABELS)})


def persist_predictions(path, frame, y, probability, rule):
    out = frame[['ID', 'start_year']].copy()
    prediction = e.decisions(probability, rule)
    for j, label in enumerate(e.LABELS):
        out[f'actual_{label}'] = y[:,j]
        out[f'score_{label}'] = probability[:,j]
        out[f'predicted_{label}'] = prediction[:,j]
    out['trial_f1'] = 2 * (prediction * y).sum(axis=1) / (prediction.sum(axis=1) + y.sum(axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parity():
    check_contract()
    train, val, frame = frames()
    manifest = json.loads((OUT / 'source_manifest.json').read_text())['files']
    def read(key):
        entry = manifest[key]
        assert b.file_hash(ROOT / entry['local']) == entry['sha256']
        return pd.read_csv(ROOT / entry['local'])
    assignment = read('part1/experiments/paired_recipe_cv/splits/repeat0.csv')
    fitting_ids = set(assignment.loc[assignment.fold != 0, 'ID'])
    gold = np.flatnonzero(frame.ID.isin(fitting_ids))
    u = np.flatnonzero(frame[e.LABELS].eq(-100).all(axis=1))
    results = []
    for name, family, rule in FAMILIES:
        ending = {'repo_ani':'paired_recipe_cv/predictions/repeat0_fold0__ani_train_only.csv',
                  'repo_dbscan':'paired_recipe_cv/predictions/repeat0_fold0__dbscan_train_only.csv',
                  'repo_kmeans':'leading_recipes_cv/predictions/repeat0_fold0__kmeans_k8.csv'}[family]
        expected = read('part1/experiments/' + ending)
        query = frame.set_index('ID').loc[expected.ID].reset_index().drop(columns=e.LABELS)
        probability = adapters.predict(frame, gold, u, query, family, e.EMB, e.LABELS)
        reference = expected[[f'probability_{l}' for l in e.LABELS]].to_numpy()
        error = float(np.max(np.abs(probability - reference)))
        labels_equal = bool(np.array_equal(e.decisions(probability, rule), expected[[f'predicted_{l}' for l in e.LABELS]].to_numpy()))
        row = dict(name=name, max_probability_difference=error, identical_predictions=labels_equal,
                   reference='Original ungrouped first fold, training-only feature access, for adapter parity only')
        results.append(row)
        print(json.dumps(row), flush=True)
        assert labels_equal and error < 1e-6, 'Adapter must reproduce saved source predictions before common benchmarking.'
    save(OUT / 'parity.json', dict(passed=True, checks=results, finished_at=b.now()))


def randomized(kind):
    check_contract()
    assert json.loads((OUT / 'parity.json').read_text())['passed']
    results = []
    def run():
        protocol = pooled.protocol() if kind == 'pooled' else b.protocol()
        if kind == 'train': b.validate_protocol(protocol)
        else: pooled.validate(protocol)
        for i, candidate in enumerate(registry()):
            native_out = b.OUT
            dest = OUT / (kind + '_cv')
            dest.mkdir(exist_ok=True)
            try:
                # Reuse exact-key results from previous identical local runs.
                # Imported recipes are stored solely in the new comparison.
                b.OUT = native_out if candidate['name'] in LOCAL else dest
                result = b.evaluate(candidate, protocol)
                for suffix in ['.json', '.npz']:
                    src = b.OUT / (result['key'] + suffix)
                    target = dest / src.name
                    if src != target: shutil.copy2(src, target)
                entry = dict(candidate=candidate['name'], result_file=str((dest / (result['key'] + '.json')).relative_to(ROOT)),
                    result_key=result['key'], cv_mean=result['cv_mean'], repeat_std=result['repeat_std'],
                    cv_macro=result['cv_macro'], reused_previous_identical_run=candidate['name'] in LOCAL,
                    repeat_scores=result['repeat_scores'])
                results.append(entry)
                save(dest / 'index.json', results)
                print(json.dumps(dict(stage=kind, **{k:v for k,v in entry.items() if k != 'repeat_scores'})), flush=True)
            finally:
                b.OUT = native_out
                pooled.clear_caches()
    if kind == 'pooled':
        with pooled.pooled_context(): run()
    else:
        run()
    check_contract()


def validation():
    check_contract()
    # Evaluation is deliberately after both randomized comparisons are complete.
    assert all(len(json.loads((OUT / (k + '_cv') / 'index.json').read_text())) == 6 for k in ['train', 'pooled'])
    train, val, _ = frames()
    results = []
    with data_context(train):
        for i, candidate in enumerate(registry()):
            pooled.clear_caches()
            query = val.drop(columns=e.LABELS)
            probability = e.predict(np.arange(len(e.Y)), query, candidate['config'])
            score = detailed(val[e.LABELS].to_numpy(int), probability, candidate['rule'])
            historical = candidate.get('recorded_validation')
            if historical is not None:
                assert abs(score['sample_f1'] - historical['sample_f1']) < 1e-12
            persist_predictions(OUT / 'validation' / f'model{i}.csv', val, val[e.LABELS].to_numpy(int), probability, candidate['rule'])
            row = dict(candidate=candidate['name'], model_index=i, **score,
                prior_local_validation_reproduced=historical is not None,
                adapter_audit=adapters.LAST_AUDIT if candidate['name'] not in LOCAL else None)
            results.append(row)
            save(OUT / 'validation/results.json', results)
            print(json.dumps(dict(stage='validation', candidate=candidate['name'], sample_f1=score['sample_f1'])), flush=True)
    check_contract()


def chronology():
    protocol = check_contract()
    assert (OUT / 'validation/results.json').exists()
    _, _, full = frames()
    results = []
    for j, split in enumerate(protocol['chronological_splits']):
        fit = full.set_index('ID').loc[split['training_ids']].reset_index()
        query = full.set_index('ID').loc[split['heldout_ids']].reset_index()
        y = query[e.LABELS].to_numpy(int)
        assert fit.start_year.max() <= split['cutoff'] < query.start_year.min()
        with data_context(fit):
            qgroups = pd.util.hash_pandas_object(query[e.EMB].round(10), index=False).to_numpy()
            assert not set(e.GROUP) & set(qgroups)
            for i, candidate in enumerate(registry()):
                pooled.clear_caches()
                probability = e.predict(np.arange(len(e.Y)), query.drop(columns=e.LABELS), candidate['config'], excluded_groups=qgroups)
                score = detailed(y, probability, candidate['rule'])
                persist_predictions(OUT / 'chronology' / f'window{j}_model{i}.csv', query, y, probability, candidate['rule'])
                row = dict(candidate=candidate['name'], model_index=i, window=j, split=split['name'], **score,
                    adapter_audit=adapters.LAST_AUDIT if candidate['name'] not in LOCAL else None)
                results.append(row)
                save(OUT / 'chronology/results.json', results)
                print(json.dumps(dict(stage='chronology', candidate=candidate['name'], window=j, sample_f1=score['sample_f1'])), flush=True)
    check_contract()


def report():
    protocol = check_contract()
    train = {r['candidate']:r for r in json.loads((OUT / 'train_cv/index.json').read_text())}
    pool = {r['candidate']:r for r in json.loads((OUT / 'pooled_cv/index.json').read_text())}
    valid = {r['candidate']:r for r in json.loads((OUT / 'validation/results.json').read_text())}
    chrono = json.loads((OUT / 'chronology/results.json').read_text())
    rows, chronology_scores, paired = [], [], []
    for i, candidate in enumerate(registry()):
        name = candidate['name']
        data = pd.concat([pd.read_csv(OUT / 'chronology' / f'window{j}_model{i}.csv') for j in range(3)], ignore_index=True)
        assert data.ID.is_unique and len(data) == 98
        y = data[[f'actual_{l}' for l in e.LABELS]].to_numpy(int)
        probability = data[[f'score_{l}' for l in e.LABELS]].to_numpy()
        cs = detailed(y, probability, candidate['rule'])
        chronology_scores.append(dict(candidate=name, **cs))
        rows.append(dict(candidate=name, train_cv=train[name]['cv_mean'], train_sd=train[name]['repeat_std'],
            pooled_cv=pool[name]['cv_mean'], pooled_sd=pool[name]['repeat_std'], validation=valid[name]['sample_f1'],
            chronological=cs['sample_f1'], chronological_macro=cs['macro_f1'],
            **{f'window{j}':next(r['sample_f1'] for r in chrono if r['candidate']==name and r['window']==j) for j in range(3)}))
        for other in registry():
            if other['name'] == name: continue
            for key, records in [('train', train), ('pooled', pool)]:
                delta = np.array([s['sample_f1'] for s in records[name]['repeat_scores']]) - np.array([s['sample_f1'] for s in records[other['name']]['repeat_scores']])
                paired.append(dict(candidate=name, reference=other['name'], benchmark=key, mean_difference=float(delta.mean()),
                    repeat_differences=delta.tolist(), repeats_won=int((delta > 1e-12).sum()), repeats_tied=int((np.abs(delta) <= 1e-12).sum())))
    table = pd.DataFrame(rows).sort_values('train_cv', ascending=False)
    table.to_csv(OUT / 'comparison.csv', index=False)
    save(OUT / 'paired_comparisons.json', paired)
    save(OUT / 'chronology/pooled_scores.json', chronology_scores)
    save(OUT / 'summary.json', dict(created_at=b.now(), protocol_sha256=b.file_hash(OUT/'protocol.json'),
        models=rows, original_local_randomized_fits_reused=90, new_imported_randomized_fits=90,
        validation_fits=6, chronological_fits=18, source_parity_fits=3,
        current_model_and_submission_unchanged=True, kaggle_inputs_used=False))
    intro = '''<p>Six frozen recipes, identical saved grouped five-fold splits and three repeats in each randomized benchmark.
    All fitting is inductive. Original local CV artifacts were reused only after their exact code/configuration/fold cache keys matched.
    All six validation fits and all 18 chronological fits were run for this comparison. No hyperparameter search or full-data submission refit.</p>
    <p>Training CV contains 157 gold rows; pooled CV contains 235. Original validation contains 78 rows and is separate from pooled CV.
    Chronological F1 pools 98 distinct later trials across three expanding training windows. Its windows differ in difficulty and are not independent repeats.</p>
    <p>Ani here is its <strong>training-only adaptation</strong>, not the transductive submitted pipeline. The team's reported Kaggle leader remains Ani's original 0.89423 submission.
    No new Kaggle results were obtained. Historical tuning used these labels: these are development results, not untouched test estimates.</p>'''
    temporal = pd.DataFrame([{k:v for k,v in r.items() if k in ['name','cutoff','test_end','gold_rows','masked_rows','heldout_rows','purged_earlier_gold','purged_earlier_masked']} for r in protocol['chronological_splits']])
    reasons = pd.DataFrame([dict(candidate=r['candidate'], reason=l, **values) for r in chronology_scores for l,values in r['reasons'].items()])
    body = '<!doctype html><meta charset="utf-8"><title>Common Part 1 recipe comparison</title><style>body{font:16px system-ui;max-width:1450px;margin:40px auto;padding:0 24px;color:#172331}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{text-align:left;border-bottom:1px solid #d9e0e7;padding:9px}th{background:#eef3f8}p{max-width:1050px;line-height:1.55}a{color:#1467b3}</style><h1>Common Part 1 recipe comparison</h1>'
    body += intro + table.to_html(index=False, float_format=lambda x:f'{x:.5f}', border=0)
    body += '<h2>Chronological windows fixed before scoring</h2>' + temporal.to_html(index=False, border=0)
    body += '<p>Rows with matching held-out embedding groups were purged from both fitting gold and masked pools. Only years at or before each cutoff entered fitting. Boundaries were chosen from sample/label counts, never model scores. Start year is a stress-test proxy, not label-availability time.</p>'
    body += '<h2>Chronological per-reason results</h2>' + reasons.to_html(index=False, float_format=lambda x:f'{x:.5f}', border=0)
    body += '<p><a href="protocol.json">Frozen protocol</a> · <a href="registry.json">Recipes</a> · <a href="paired_comparisons.json">Paired CV differences</a> · <a href="parity.json">Source parity checks</a> · <a href="verification.json">Verification</a></p>'
    (OUT / 'report.html').write_text(body)
    print(table.to_string(index=False), flush=True)


def verify():
    check_contract()
    checks = []
    for kind in ['train', 'pooled']:
        results = json.loads((OUT / (kind + '_cv') / 'index.json').read_text())
        assert len(results) == 6
        for record, candidate in zip(results, registry()):
            assert record['candidate'] == candidate['name']
            data = np.load(OUT / (kind + '_cv') / (record['result_key'] + '.npz'), allow_pickle=True)
            assert len(data['scores']) == 3 and len(data['truth']) == (157 if kind == 'train' else 235)
            for i, prob in enumerate(data['scores']):
                np.testing.assert_array_equal(e.decisions(prob, candidate['rule']), data['predictions'][i])
                score = e.metrics(data['truth'], prob, candidate['rule'])
                assert abs(score['sample_f1'] - record['repeat_scores'][i]['sample_f1']) < 1e-12
            checks.append(dict(stage=kind, candidate=candidate['name'], oof_repeats_rescored=3))
    for stage in ['validation', 'chronology']:
        results = json.loads((OUT / stage / 'results.json').read_text())
        assert len(results) == (6 if stage == 'validation' else 18)
        for row in results:
            file = f'model{row["model_index"]}.csv' if stage == 'validation' else f'window{row["window"]}_model{row["model_index"]}.csv'
            data = pd.read_csv(OUT / stage / file)
            pred = data[[f'predicted_{l}' for l in e.LABELS]].to_numpy(int)
            truth = data[[f'actual_{l}' for l in e.LABELS]].to_numpy(int)
            score = e.metrics(truth, pred, 't0.5')
            assert abs(score['sample_f1'] - row['sample_f1']) < 1e-12
            checks.append(dict(stage=stage, candidate=row['candidate'], predictions_rescored=len(data)))
    save(OUT / 'verification.json', dict(passed=True, verified_at=b.now(), checks=checks,
         protected_sha256=protected(), sources_sha256=sources(), original_benchmarks_unchanged=True))
    print('Verified all predictions, metrics, frozen sources, and protected artifacts.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','parity','train','pooled','validation','chronology','report','verify'])
    args = parser.parse_args()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        if args.stage in ['train', 'pooled']: randomized(args.stage)
        else: globals()[args.stage]()


if __name__ == '__main__':
    main()
