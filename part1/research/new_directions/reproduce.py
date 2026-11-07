"""Replay one saved configuration in an isolated output folder using frozen source.

No downloads, test prediction export, or modifications to the compact record.
Only downloaded reference documents are absent from old source hash lists;
all executable files, recipes, data and split hashes are still checked.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
import types
from pathlib import Path

from verify import ROOT, read_csv, registry_path, verify


def prepare(output, data):
    # Exclusive creation prevents mixing a rerun with unrelated or stale results.
    output.mkdir(parents=True, exist_ok=False)
    engine = output / 'engine'
    shutil.copytree(ROOT / 'reproduction', engine)
    (engine / 'data').symlink_to(data, target_is_directory=True)
    for n in range(2, 15):
        folder = engine / 'studies' / f'v{n}'
        for name in ('data', 'protocols', 'references'):
            (folder / name).symlink_to(Path('../..') / name, target_is_directory=True)
    (engine / 'references/legacy_engine/data').symlink_to('../../data', target_is_directory=True)
    return engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=[f'v{n}' for n in range(1, 15)])
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--pool', choices=['train', 'pooled'], default='train')
    parser.add_argument('--repeats', type=int, choices=range(1, 9), default=3)
    parser.add_argument('--supplementary', action='store_true', help='Replay the recorded validation and three chronological windows.')
    parser.add_argument('--output', type=Path, help='New directory for regenerated results; defaults to ignored .runs/.')
    parser.add_argument('--data', type=Path, default=ROOT.parents[1] / 'data')
    parser.add_argument('--check', action='store_true', help='Check inputs and imports without fitting or creating output.')
    args = parser.parse_args()
    data = args.data.resolve()
    verify(data)
    reg = json.loads((ROOT / 'reproduction' / registry_path(args.study)).read_text())
    config = next((c for c in reg['recipes'] if c['name'] == args.recipe), None)
    is_reference = args.study == 'v5' and args.recipe == 'submitted_reference' and args.supplementary
    if config is None and not is_reference:
        parser.error('Recipe is not in this study registry.')
    if args.supplementary:
        if not any(r['study'] == args.study and r['recipe'] == args.recipe for r in read_csv('supplementary.csv')):
            parser.error('No historical supplementary evaluation for this recipe.')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    env = json.loads((ROOT / 'reproduction/environment.json').read_text())
    packages = {name: importlib.metadata.version(name) for name in
                ('numpy', 'pandas', 'scipy', 'scikit-learn', 'iterative-stratification', 'threadpoolctl')}
    differences = {name: dict(recorded=env['packages'][name], installed=ver)
                   for name, ver in packages.items() if ver != env['packages'][name]}
    if differences:
        print('Library version differences may affect numerical replay: ' + json.dumps(differences), flush=True)
    if args.check:
        # Model imports do not need data links; legacy engine imports happen only at prediction time.
        sys.path.insert(0, str(ROOT / 'reproduction'))
        name = 'models' if args.study == 'v1' else 'study_v4' if args.study == 'v4' else 'models_' + args.study
        importlib.import_module(name)
        print(json.dumps(dict(passed=True, study=args.study, recipe=args.recipe, fitting=False)))
        return
    output = (args.output or ROOT / '.runs' / f'{args.study}-{args.recipe}-{args.pool}').resolve()
    if output == ROOT or ROOT in output.parents and '.runs' not in output.relative_to(ROOT).parts:
        parser.error('Use .runs/ or an output directory outside the retained research record.')
    engine = prepare(output, data)
    sys.path.insert(0, str(engine))
    e = importlib.import_module('study')
    original_baseline = e.baseline
    models = importlib.import_module('models')
    modules = {f'v{n}': importlib.import_module(f'study_v{n}') for n in range(2, 15)}
    # Adapt only storage verification: frozen registries include hashes of three downloaded papers.
    # verify() above checks every retained source and the explicit document-only omissions.
    for version, module in modules.items():
        frozen = json.loads((engine / registry_path(version)).read_text())
        module.registry = lambda record=frozen: record
    e.registry = lambda: reg
    n = int(args.study[1:])
    e.ROOT = engine if n == 1 else engine / 'studies' / args.study
    predict = models.predict if n == 1 else modules['v4'].predict if n == 4 else importlib.import_module('models_' + args.study).predict
    e.models = types.SimpleNamespace(predict=predict, threshold=models.threshold)
    e.baseline = original_baseline if n <= 4 else modules['v5'].baseline if n <= 10 else importlib.import_module('current_reference').baseline
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.supplementary:
            if n <= 3:
                helper = importlib.import_module('extended_checks')
                helper.supplementary(args.study, config)
            elif n in (5, 6):
                importlib.import_module('continuation_checks').supplementary(args.study, args.recipe)
            elif n in (8, 10):
                importlib.import_module('recent_checks').supplementary(args.study, args.recipe)
            else:
                raise ValueError('No recorded supplementary case for this study.')
        else:
            e.evaluate(config, args.pool, args.repeats)
    info = dict(study=args.study, recipe=args.recipe, pool=args.pool, repeats=args.repeats,
                supplementary=args.supplementary, python=sys.version, packages=packages,
                version_differences=differences, frozen_sources_changed=False,
                omitted_documents=json.loads((ROOT/'manifest.json').read_text())['omitted_reference_documents'],
                note='Fresh replay, not additional evidence or model promotion. Full per-sample records exist only in this disposable output folder.')
    (output / 'replay.json').write_text(json.dumps(info, indent=2) + '\n')
    print('Regenerated results: ' + str(e.ROOT), flush=True)


if __name__ == '__main__':
    main()
