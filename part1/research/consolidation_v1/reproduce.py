"""Rerun frozen blocks on a saved CV stage in a new disposable directory.

Example: reproduce.py feature_gap_cv_v1 --block drug_targets --stage train3 --output /tmp/feature-replay
Requires the original ML environment. Discarded historical outputs are explicitly
excluded from storage hash checks; recipes and fitting boundaries stay unchanged.
"""
import argparse
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from evidence import ROOT, HERE, sha


def verify_inputs():
    record = json.loads((HERE / 'retention.json').read_text())
    for root, entries in [(ROOT, record['retained_files']), (HERE.parents[1], record['shared_data'])]:
        for row in entries:
            assert sha(root / row['path']) == row['sha256'], row['path']
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study')
    parser.add_argument('--block', action='append', default=[], help='Frozen block name; may be repeated.')
    parser.add_argument('--stage', choices=['train3', 'pooled3', 'train5', 'pooled5'], default='train3')
    parser.add_argument('--output', type=Path, help='Required unless --check; must be a new directory.')
    parser.add_argument('--check', action='store_true', help='Verify inputs and import without fitting.')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'experiments/lda_research_checkpoint_v1/catalog.json').read_text())
    info = next((s for s in catalog['studies'] if s['study'] == args.study), None)
    if info is None:
        parser.error('Study must be listed in the retained catalog.')
    if not args.check and (not args.output or not args.block and args.study != 'ani_direct_comparison_v1'):
        parser.error('Replay requires --output and --block (direct comparison runs all fixed recipes).')
    if not args.check and info['status'] != 'completed':
        parser.error('This historical study failed or was partial; use its corrected version for replay.')
    record = verify_inputs()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = '1'
    temporary = tempfile.TemporaryDirectory(prefix='aml-import-check-') if args.check else None
    output = Path(temporary.name).resolve() if temporary else args.output.resolve()
    if not temporary:
        if output == ROOT or ROOT in output.parents:
            parser.error('Choose output outside the retained research.')
        output.mkdir(parents=True, exist_ok=False)
    engine = output / 'part1'
    engine.mkdir()
    for row in record['retained_files']:
        src = ROOT / row['path']; dst = engine / row['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    (engine / 'data').symlink_to(HERE.parents[1] / 'data', target_is_directory=True)
    (engine / 'research/independent/data').symlink_to('../../data', target_is_directory=True)
    target = engine / 'experiments' / args.study
    source = target / ('run.py' if args.study == 'feature_gap_cv_v1' else 'study.py')
    sys.path.insert(0, str(target))
    spec = importlib.util.spec_from_file_location('replay_study', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = module if args.study == 'feature_gap_cv_v1' else getattr(module, 'runner', getattr(module, 'r', None))
    protocol = json.loads((target / 'protocol.json').read_text())
    omitted = {}

    def guard():
        for path, expected in protocol['source_sha256'].items():
            if path in record['omitted_protocol_output_dependencies']:
                assert record['omitted_protocol_output_dependencies'][path] == expected
                omitted[path] = expected
            else:
                assert sha(engine / path) == expected, path
        assert sha(target / 'registry.json') == protocol['registry_sha256']
        return protocol

    module.guard = runner.guard = guard
    guard()
    registry = json.loads((target / 'registry.json').read_text())
    allowed = {r.get('block') for r in registry}
    if any(b not in allowed for b in args.block):
        parser.error('Block must match registry: ' + ', '.join(sorted(b for b in allowed if b)))
    if args.check:
        print(json.dumps(dict(passed=True, study=args.study, models_fitted=0, omitted_historical_outputs=len(omitted))))
        temporary.cleanup()
        return
    # Regenerate earlier negative-candidate inputs only when a later study needs
    # them. They are not permanently retained as caches in this package.
    required = {}
    for path in omitted:
        p = Path(path)
        if len(p.parts) == 4 and p.parts[2] == args.stage:
            required.setdefault(p.parts[1], set())
            if '__' in p.stem:
                required[p.parts[1]].add(p.stem.split('__', 1)[1])
    for study, blocks in required.items():
        child = output / ('dependency-' + study)
        command = [sys.executable, str(Path(__file__).resolve()), study, '--stage', args.stage, '--output', str(child)]
        for block in sorted(blocks):
            command.extend(['--block', block])
        subprocess.run(command, check=True)
        shutil.copytree(child / 'part1/experiments' / study / args.stage,
                        engine / 'experiments' / study / args.stage, dirs_exist_ok=True)
    shutil.rmtree(target / args.stage, ignore_errors=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.study == 'ani_direct_comparison_v1':
            module.run_stage(args.stage)
        else:
            runner.run_stage(args.stage, args.block)
    result = dict(study=args.study, stage=args.stage, blocks=args.block,
                  frozen_source_modified=False, omitted_historical_output_hashes=omitted,
                  regenerated_dependencies=list(required),
                  packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy']},
                  note='Fresh disposable replay; discarded historical outputs were not claimed verified.')
    (output / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Regenerated results: ' + str(target / args.stage))


if __name__ == '__main__':
    main()
