"""Read-only checks of the relocated research; no training or artifact rewrites.

Prints JSON to stdout. Redirect it to integration_verification.json when desired.
Run from any cwd with the recorded research Python environment.
"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PART1 = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(ROOT / 'IMPORT_MANIFEST.json')
    for item in manifest['files']:
        path = ROOT / item['destination']
        assert path.stat().st_size == item['bytes'], str(path)
        assert sha(path) == item['sha256'], str(path)
    for item in read(PART1 / 'research/ani_snapshots/provenance.json'):
        assert sha(PART1 / item['file']) == item['sha256'], item['file']
    snapshots = PART1 / 'research/ani_snapshots'
    assert sha(snapshots/'a548721/part1_best_submission.csv') == sha(snapshots/'8a8c3fc/part1_first_submission.csv')
    assert sha(snapshots/'a548721/part1_best_submission.csv') != sha(snapshots/'8a8c3fc/part1_best_submission.csv')

    # Isolate runners whose import configures shared module state.
    guard_commands = {
        'cross_repo': 'import run_cross_repo_benchmark as m; m.check_contract()',
        'improvements': 'import run_ani_improvements as m; m.guard()',
        'combinations': 'import run_ani_combinations as m; m.guard()',
        'bagging': 'import run_ani_bagging as m; m.guard()',
        'refit': 'import diagnose_refit as m; m.guard()',
        'transductive': 'import run_three_way_transductive as m; m.guard()',
        'train_folds': 'import run_ani_improvements as m; m.stage_protocol("train"); m.stage_protocol("train", True)',
        'pooled_folds': 'import run_ani_improvements as m\nwith m.pooled.pooled_context():\n m.stage_protocol("pooled"); m.stage_protocol("pooled", True)',
        'original_benchmark_scores': 'import verify_benchmarks as v\nimport json\nb=v.b\nv.verify_predictions(b.OUT,json.loads((b.OUT/"summary.json").read_text()))\nwith v.pooled.pooled_context():\n v.verify_predictions(v.pooled.OUT,json.loads((v.pooled.OUT/"summary.json").read_text()),True)',
    }
    for name, code in guard_commands.items():
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(f'{name}: {result.stderr}\n{result.stdout}')

    # Independently rescore all stages in the three recent inductive searches.
    import benchmark_cv as b
    label_order = b.e.LABELS
    train = pd.read_csv(ROOT/'data/part1_train.csv')
    val = pd.read_csv(ROOT/'data/part1_val.csv')
    gold = train.loc[train[label_order].ge(0).all(axis=1)]
    records = repeats = 0
    for study in ['ani_improvement_v1', 'ani_combination_v1', 'ani_bagging_v1']:
        for stage in ['train3', 'train5', 'pooled3', 'pooled5']:
            folder = ROOT/'experiments'/study/stage
            expected = pd.concat([gold, val], ignore_index=True) if stage.startswith('pooled') else gold
            for row in read(folder/'index.json'):
                meta = read(folder/(row['result_key']+'.json'))
                with np.load(folder/(row['result_key']+'.npz'), allow_pickle=True) as saved:
                    assert saved['ids'].tolist() == expected.ID.tolist()
                    np.testing.assert_array_equal(saved['truth'], expected[label_order].to_numpy())
                    assert len(saved['scores']) == int(stage[-1])
                    values = []
                    for i, scores in enumerate(saved['scores']):
                        pred = b.e.decisions(scores, meta['rule'])
                        np.testing.assert_array_equal(pred, saved['predictions'][i])
                        metric = b.e.metrics(saved['truth'], scores, meta['rule'])
                        for key, value in row['repeat_scores'][i].items():
                            assert abs(metric[key]-value) < 1e-12, (study, stage, key)
                        values.append(metric['sample_f1'])
                    assert abs(np.mean(values)-row['mean']) < 1e-12
                    assert abs(np.std(values)-row['sd']) < 1e-12
                    records += 1
                    repeats += len(values)

    with (PART1/'submissions/index.csv').open(newline='') as f:
        submissions = list(csv.DictReader(f))
    sample = pd.read_csv(PART1/'data/part1_sample_submission.csv')
    archive = read(PART1/'submissions/archive_manifest.json')
    assert {x['file']: x['sha256'] for x in archive['records']} == {x['file']: x['sha256'] for x in submissions}
    for record in submissions:
        path = PART1/record['file']
        assert sha(path) == record['sha256'], record['file']
        frame = pd.read_csv(path)
        assert list(frame.columns) == list(sample.columns)
        assert frame.ID.tolist() == sample.ID.tolist() and len(frame) == 312
        for labels in frame.Predicted:
            assert isinstance(labels, str) and labels.strip()
            tokens = labels.replace(';', ' ').replace(',', ' ').split()
            assert len(tokens) == len(set(tokens)) and set(tokens) <= set(label_order), labels

    # Recheck after importing/running guards: frozen files must remain unchanged.
    for item in manifest['files']:
        assert sha(ROOT/item['destination']) == item['sha256'], item['destination']
    result = dict(passed=True, imported_files_verified=len(manifest['files']),
                  source_code_and_frozen_artifacts_unchanged=True,
                  isolated_contract_checks=list(guard_commands),
                  grouped_folds_verified=80, recent_oof_records_rescored=records,
                  recent_oof_repeats_rescored=repeats,
                  submission_files_verified=len(submissions), ani_revision_distinction_verified=True,
                  models_fitted=0, uploads_performed=0,
                  python=sys.version.split()[0], clean_environment_install_tested=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
