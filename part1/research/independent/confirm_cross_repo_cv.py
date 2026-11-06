"""Predeclared five-repeat extension for all six recipes, retaining original CV.

Run prepare once before any extension scores; then train, pooled, and report.
The first three saved repeats are reused exactly, giving eight in total.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_cross_repo_benchmark as r

b, e, pooled = r.b, r.e, r.pooled
OUT = r.OUT / 'confirmation8'
EXTRA_SEEDS = [20261129, 20261130, 20261201, 20261202, 20261203]


def protocol(kind, make=False):
    path = OUT / kind / 'protocol.json'
    if make:
        original = json.loads((r.ROOT / 'experiments' / ('benchmark_v1' if kind == 'train' else 'pooled_benchmark_v1') / 'protocol.json').read_text())
        assert set(EXTRA_SEEDS).isdisjoint(original['seeds'])
        folds = []
        for seed in EXTRA_SEEDS:
            for i, (fit, held) in enumerate(e.make_splits(y=e.Y, groups=e.GROUP[e.LAB], seed=seed, n_splits=5)):
                folds.append(dict(seed=seed, fold=i, train_gold_positions=fit.tolist(), heldout_gold_positions=held.tolist(),
                    heldout_ids=e.DATA.iloc[e.LAB[held]].ID.tolist(),
                    masked_duplicates_excluded=int(np.isin(e.GROUP[e.UNLAB], e.GROUP[e.LAB[held]]).sum())))
        p = dict(version='cross-repo-additional-five-repeats-v1', benchmark=kind, created_at=b.now(),
            parent_protocol_sha256=b.digest(original), parent_seeds=original['seeds'], seeds=EXTRA_SEEDS,
            parent_comparison_protocol_sha256=b.file_hash(r.OUT/'protocol.json'),
            gold_ids=e.DATA.iloc[e.LAB].ID.tolist(), label_order=e.LABELS,
            folds=folds, folds_sha256=b.digest(folds), runner_sha256=b.file_hash(Path(__file__)),
            reason='Initial paired comparison is close; add five prespecified repeats for every one of the six recipes, retaining all previous scores and frozen recipes.',
            changes='Split assignments only. All preprocessing, grouping, fitting boundaries, features, hyperparameters and decision rules unchanged.',
            metric='Pool all OOF trials within each repeat; final mean and descriptive population SD over all eight repeats.',
            interpretation='Additional split-sensitivity evidence on reused development data, not an independent test. No seed or recipe search.')
        r.save(path, p)
    else:
        p = json.loads(path.read_text())
    assert p['runner_sha256'] == b.file_hash(Path(__file__))
    assert p['parent_comparison_protocol_sha256'] == b.file_hash(r.OUT/'protocol.json')
    assert p['gold_ids'] == e.DATA.iloc[e.LAB].ID.tolist()
    assert p['folds_sha256'] == b.digest(p['folds'])
    for seed in p['seeds']:
        fs = [f for f in p['folds'] if f['seed'] == seed]
        assert len(fs) == 5
        assert sorted(x for f in fs for x in f['heldout_gold_positions']) == list(range(len(e.Y)))
        for f in fs:
            a, h = np.array(f['train_gold_positions']), np.array(f['heldout_gold_positions'])
            assert not set(a) & set(h) and len(a) + len(h) == len(e.Y)
            assert not set(e.GROUP[e.LAB[a]]) & set(e.GROUP[e.LAB[h]])
            assert f['heldout_ids'] == e.DATA.iloc[e.LAB[h]].ID.tolist()
    return p


def prepare():
    r.check_contract()
    for kind in ['train', 'pooled']:
        assert len(json.loads((r.OUT / (kind + '_cv') / 'index.json').read_text())) == 6
    assert not (OUT / 'train/protocol.json').exists() and not (OUT / 'pooled/protocol.json').exists()
    protocol('train', make=True)
    with pooled.pooled_context(): protocol('pooled', make=True)
    r.save(OUT / 'manifest.json', dict(frozen_at=b.now(), candidates=[x['name'] for x in r.registry()],
        extra_seeds=EXTRA_SEEDS, original_repeats=3, added_repeats=5, total_repeats=8,
        added_fits=300, source_sha256=b.file_hash(Path(__file__))))
    print('Both sets of 25 additional grouped folds frozen for all six recipes.', flush=True)


def run(kind):
    r.check_contract()
    def work():
        p = protocol(kind)
        old_out = b.OUT
        rows = []
        try:
            b.OUT = OUT / kind
            for candidate in r.registry():
                result = b.evaluate(candidate, p)
                rows.append(dict(candidate=candidate['name'], result_key=result['key'], cv_mean=result['cv_mean'],
                    repeat_std=result['repeat_std'], repeat_scores=result['repeat_scores']))
                r.save(OUT / kind / 'index.json', rows)
                pooled.clear_caches()
                print(json.dumps(dict(stage='extra_'+kind, candidate=candidate['name'], mean=result['cv_mean'])), flush=True)
        finally:
            b.OUT = old_out
    if kind == 'pooled':
        with pooled.pooled_context(): work()
    else: work()
    r.check_contract()


def report():
    r.check_contract()
    original = pd.read_csv(r.OUT / 'comparison.csv').set_index('candidate')
    scores, checks = {}, []
    for kind in ['train', 'pooled']:
        first = {x['candidate']:x for x in json.loads((r.OUT / (kind+'_cv') / 'index.json').read_text())}
        extra = {x['candidate']:x for x in json.loads((OUT / kind / 'index.json').read_text())}
        assert len(first) == len(extra) == 6
        scores[kind] = {}
        for candidate in r.registry():
            name = candidate['name']
            old = np.load(r.OUT / (kind+'_cv') / (first[name]['result_key']+'.npz'), allow_pickle=True)
            new = np.load(OUT / kind / (extra[name]['result_key']+'.npz'), allow_pickle=True)
            np.testing.assert_array_equal(old['ids'], new['ids'])
            np.testing.assert_array_equal(old['truth'], new['truth'])
            combined = np.concatenate([old['scores'], new['scores']])
            assert combined.shape == (8, len(e.Y) if kind == 'train' else 235, 5)
            metrics = [r.detailed(old['truth'], p, candidate['rule']) for p in combined]
            for i, metric in enumerate(metrics[3:]):
                assert abs(metric['sample_f1']-extra[name]['repeat_scores'][i]['sample_f1']) < 1e-12
            values = np.array([s['sample_f1'] for s in metrics])
            scores[kind][name] = dict(mean=float(values.mean()), repeat_sd=float(values.std()),
                                      repeat_f1=values.tolist(), repeat_metrics=metrics)
            checks.append(dict(benchmark=kind, candidate=name, repeats_verified=8,
                original_predictions_preserved=True, complete_oof_coverage=True))
    paired = []
    for kind, models in scores.items():
        for name, result in models.items():
            for other, reference in models.items():
                if name == other: continue
                difference = np.array(result['repeat_f1']) - np.array(reference['repeat_f1'])
                paired.append(dict(benchmark=kind, candidate=name, reference=other, mean_difference=float(difference.mean()),
                    differences=difference.tolist(), wins=int((difference>1e-12).sum()),
                    ties=int((np.abs(difference)<=1e-12).sum()), losses=int((difference < -1e-12).sum())))
    rows = []
    for candidate in r.registry():
        name = candidate['name']
        rows.append(dict(candidate=name, train_cv_8=scores['train'][name]['mean'], train_sd_8=scores['train'][name]['repeat_sd'],
            pooled_cv_8=scores['pooled'][name]['mean'], pooled_sd_8=scores['pooled'][name]['repeat_sd'],
            validation=float(original.loc[name,'validation']), chronological=float(original.loc[name,'chronological']),
            train_cv_3=float(original.loc[name,'train_cv']), pooled_cv_3=float(original.loc[name,'pooled_cv'])))
    table = pd.DataFrame(rows).sort_values('train_cv_8', ascending=False)
    table.to_csv(OUT/'comparison.csv', index=False)
    r.save(OUT/'summary.json', dict(scores=scores, comparison=rows, paired_comparisons=paired, created_at=b.now()))
    r.save(OUT/'verification.json', dict(passed=True, checks=checks, protected_sha256=r.protected()))
    original_html = (r.OUT/'report.html').read_text()
    location = original_html.index('<h1>')
    head = original_html[:location]
    body = '<h1>Eight-repeat confirmation: six frozen Part 1 recipes</h1><p>The original three repeats are preserved, with five additional seeds fixed before running this extension. Every recipe received the same 40 grouped folds in each label pool. No recipe or decision rule changed.</p>'
    body += table.to_html(index=False, float_format=lambda x:f'{x:.5f}', border=0)
    body += '<p>SD describes sensitivity to the splits. Eight repeats reuse the same trials and are not eight independent datasets. Original validation and chronology remain unchanged. Ani is the training-only adaptation; this comparison does not supply a new Kaggle score.</p>'
    body += '<p><a href="summary.json">All repeat scores and paired differences</a> · <a href="train/protocol.json">Additional training folds</a> · <a href="pooled/protocol.json">Additional pooled folds</a> · <a href="verification.json">Verification</a> · <a href="../report.html">Initial comparison and chronological details</a></p>'
    body += '<h2>Ani versus local SVM: paired repeat differences</h2>'
    pairtable = pd.DataFrame([x for x in paired if x['candidate']=='Ani training-only adaptation' and x['reference']=='Local SVM specialist hybrid'])
    body += pairtable[['benchmark','mean_difference','wins','ties','losses']].to_html(index=False, float_format=lambda x:f'{x:.5f}', border=0)
    (OUT/'report.html').write_text(head+body)
    print(table.to_string(index=False), flush=True)
    print(pairtable[['benchmark','mean_difference','wins','ties','losses']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','train','pooled','report'])
    args = parser.parse_args()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        if args.stage in ['train','pooled']: run(args.stage)
        else: globals()[args.stage]()


if __name__ == '__main__': main()
