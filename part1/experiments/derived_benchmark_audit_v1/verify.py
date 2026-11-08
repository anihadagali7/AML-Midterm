"""Independent rescoring, coverage, provenance and historical control checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
PART = HERE.parents[1]
REPO = PART.parent
LABELS = ['Safety', 'Funding', 'Efficacy', 'Recruitment', 'Other']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = read(HERE / 'protocol.json')
    for path, digest in protocol['source_hashes'].items():
        assert sha(REPO / path) == digest, path
    full = pd.concat([pd.read_csv(PART / 'data/part1_train.csv'), pd.read_csv(PART / 'data/part1_val.csv')]).set_index('ID')
    test_ids = set(pd.read_csv(PART / 'data/part1_test.csv').ID)
    hashes = pd.util.hash_pandas_object(full[[f'emb_{i}' for i in range(150)]].round(10), index=False)
    gold_ids = set(full.index[full[LABELS].ge(0).all(axis=1)])
    train_ids = set(pd.read_csv(PART / 'data/part1_train.csv').ID)
    train_gold = gold_ids & train_ids
    masked_ids = set(full.index[full[LABELS].eq(-100).all(axis=1)])
    cases = read(HERE / 'splits.json')
    comp = pd.read_csv(HERE / 'comparison.csv').set_index('model')
    models = protocol['models']
    checks = 0
    occurrences = {}
    per_model = {name: [] for name in models}
    for c in cases:
        fit, query, gold, masked = (set(c[k]) for k in ['fit_ids', 'query_ids', 'train_gold', 'masked_ids'])
        assert len(fit) == len(c['fit_ids']) and len(query) == len(c['query_ids'])
        assert fit == gold | masked and not gold & masked
        assert gold <= gold_ids and query <= gold_ids and masked <= masked_ids
        assert not fit & query and not (fit | query) & test_ids
        if c['kind'] == 'train':
            assert (fit | query) <= train_ids
        require_groups = c['track'] == 'strict_B' or (c['track'] == 'canonical' and c['kind'] != 'validation')
        if require_groups:
            assert not set(hashes.loc[c['fit_ids']]) & set(hashes.loc[c['query_ids']])
        if c['track'] == 'strict_B':
            assert full.loc[c['fit_ids']].start_year.max() < c['cut']
            assert full.loc[c['query_ids']].start_year.min() >= c['cut']
        if c['kind'] == 'chronology':
            assert full.loc[c['fit_ids']].start_year.max() < full.loc[c['query_ids']].start_year.min()
        if c['kind'] in ['A', 'train', 'pooled']:
            occurrences.setdefault((c['kind'], c['repeat']), []).extend(c['query_ids'])
        path = HERE / 'results' / f'{c["key"]}.npz'
        sidecar = read(path.with_suffix('.json'))
        assert sha(path) == sidecar['sha256']
        with np.load(path) as z:
            assert list(z['ids']) == c['query_ids']
            np.testing.assert_array_equal(z['truth'], full.loc[c['query_ids'], LABELS].to_numpy())
            for name in models:
                y, p, s = z['truth'], z[name + '_predictions'], z[name + '_scores']
                assert p.shape == y.shape == s.shape and np.isfinite(s).all()
                assert set(np.unique(p)) <= {0, 1} and (p.sum(1) > 0).all()
                cutoff = .3 if name == 'anchored' else .5
                expected = (s >= cutoff).astype(int)
                empty = np.flatnonzero(expected.sum(1) == 0)
                expected[empty, s[empty].argmax(1)] = 1
                np.testing.assert_array_equal(expected, p)
                actual = f1_score(y, p, average='samples', zero_division=0)
                assert abs(actual - sidecar['metrics'][name]['sample_f1']) < 1e-12
                pr, re, f1, support = precision_recall_fscore_support(y, p, zero_division=0)
                assert abs(f1.mean() - sidecar['metrics'][name]['macro_f1']) < 1e-12
                for key, values in [('precision', pr), ('recall', re), ('f1', f1), ('support', support)]:
                    for j, label in enumerate(LABELS):
                        assert abs(values[j] - sidecar['metrics'][name][key + '_' + label]) < 1e-12
                per_model[name].append(dict(case=c, ids=z['ids'].copy(), y=y.copy(), p=p.copy(), score=actual))
                checks += 1
    for (kind, repeat), ids in occurrences.items():
        assert len(ids) == len(set(ids))
        assert set(ids) == (train_gold if kind == 'train' else gold_ids)
    assert len(occurrences) == 26
    assert len(cases) == 140 and checks == 840

    def pooled(records):
        return f1_score(np.concatenate([r['y'] for r in records]), np.concatenate([r['p'] for r in records]), average='samples', zero_division=0)

    for name in models:
        rec = per_model[name]
        def select(track, kind):
            return [r for r in rec if r['case']['track'] == track and r['case']['kind'] == kind]
        # Independently rebuild all main metric expressions from label predictions.
        expected = dict(dylan_A=pooled(select('dylan', 'A')),
                        dylan_B=np.mean([r['score'] for r in select('dylan', 'B')]),
                        strict_B=np.mean([r['score'] for r in select('strict_B', 'B')]),
                        validation=pooled(select('canonical', 'validation')),
                        grouped_train=pooled(select('canonical', 'train')),
                        grouped_pooled=pooled(select('canonical', 'pooled')),
                        chronology98=pooled(select('canonical', 'chronology')),
                        dylan_B2005=next(r['score'] for r in select('dylan', 'B') if r['case']['cut'] == 2005))
        expected['dylan_J'] = (expected['dylan_A'] + expected['dylan_B']) / 2
        expected['aligned_J'] = (expected['grouped_pooled'] + expected['strict_B']) / 2
        expected['current_J_diagnostic'] = (expected['grouped_pooled'] + expected['chronology98']) / 2
        for key, value in expected.items():
            assert abs(value - comp.loc[name, key]) < 1e-12, (name, key)

    archive = PART / 'research/lda_ensemble/research/independent/experiments/cross_repo_benchmark_v1/confirmation8/comparison.csv'
    historical = pd.read_csv(archive).set_index('candidate')
    old_checks = []
    for name, ref in [('dbscan', 'DBSCAN + logistic C1000'), ('tri_training', 'Classical tri-training')]:
        for metric, oldmetric in [('grouped_train', 'train_cv_8'), ('grouped_pooled', 'pooled_cv_8'),
                                  ('validation', 'validation'), ('chronology98', 'chronological')]:
            delta = float(comp.loc[name, metric] - historical.loc[ref, oldmetric])
            assert abs(delta) < 1e-12
            old_checks.append(dict(model=name, metric=metric, delta=delta))
    current = pd.read_csv(PART / 'experiments/current_leaders_grouped_v1/comparison.csv')
    current = current.set_index('name')
    current_checks = 0
    for name in ['original_ani', 'previous_ensemble', 'lda_ensemble', 'anchored']:
        ref = 'anchored_candidate' if name == 'anchored' else name
        for metric, oldmetric in [('grouped_train', 'train'), ('grouped_pooled', 'pooled'),
                                  ('validation', 'validation'), ('chronology98', 'chronology98')]:
            assert abs(comp.loc[name, metric] - current.loc[ref, oldmetric]) < 1e-12
            current_checks += 1
    # The runner additionally verifies exact predictions and aligned probability
    # scores for all four retained leaders on 80 canonical folds (320 checks).
    replay = read(HERE / 'control_replay_checks.json')
    assert len(replay) == 320 and all(r['identical_predictions'] and r['max_score_error'] <= 1e-7 for r in replay)
    supplied = read(HERE / 'source_snapshot/benchmark_results.json')['michael_anchored']
    for key in ['A', 'B', 'C', 'J']:
        expected = supplied[key] if key == 'J' else supplied[key]['sample_f1']
        assert abs(comp.loc['anchored', 'validation' if key == 'C' else 'dylan_' + key] - expected) < 1e-12
    for artifact in protocol['scored_artifacts']:
        assert sha(REPO / artifact['csv']) == artifact['sha256']
    result = dict(status='passed', case_prediction_rescores=checks, protected_source_hashes=len(protocol['source_hashes']),
                  exact_historical_fold_controls=len(replay), DBSCAN_tri_aggregate_controls=old_checks,
                  current_leader_aggregate_controls=current_checks,
                  complete_OOF_repeats=len(occurrences), supplied_anchored_A_B_C_J_reproduced=True,
                  zero_test_input_access=True, strict_and_canonical_group_boundaries=True,
                  study_outputs='Predictions are matched benchmark controls, not new submission artifacts.')
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
