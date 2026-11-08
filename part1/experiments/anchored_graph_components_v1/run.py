"""Replay the registered screen, then only qualifying confirmations. No test CSV read."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from models import (HERE, LABELS, EMB, CS, VARIANTS, AnchoredReasonClassifier,
                    select_reasons, fit_pseudo, ResidualCorrector, fitting_context, strict)
from pseudo_anchored import PseudoTeacherAnchored

P1 = HERE.parents[1]
REF = P1 / 'experiments/current_leaders_grouped_v1'
SNAP = REF / 'snapshots'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def protected_paths():
    return ([P1 / 'anchored_solution.py', P1 / 'TESTING_POLICY.md',
             P1 / 'data/part1_train.csv', P1 / 'data/part1_val.csv']
            + list(SNAP.glob('*.json')) + list(REF.glob('anchored_*.npz')))


def freeze():
    target = HERE / 'protocol.json'
    assert not target.exists(), 'Do not overwrite a frozen study.'
    source = [HERE / 'run.py', HERE / 'models.py', HERE / 'pseudo_anchored.py']
    source += sorted((HERE / 'vendor').rglob('*.py'))
    p = dict(version='anchored-graph-components-v1', created_at=datetime.now(timezone.utc).isoformat(),
        base_main='256fa8dabb2cd243f1162a0356e2c55b7a369a59', parent='anchored 0.89636',
        source_sha256={str(f.relative_to(HERE)): digest(f) for f in source},
        protected_sha256={str(f.relative_to(P1)): digest(f) for f in protected_paths()},
        python=platform.python_version(), requirements_sha256=digest(HERE / 'requirements-lock.txt'),
        hypotheses=dict(cs_pseudo='Graph teacher labels the masked fitting pool more usefully than anchored self-training.',
            graph_residual_raw='Nearby fitting rows have transferable anchored OOF errors.',
            graph_residual_autoscale='C&S automatic residual magnitude scaling makes that correction useful.'),
        variants=VARIANTS, cs_teacher=CS,
        cs_parent_settings_preserved=dict(pseudo_confidence=.8, pseudo_decode='.3 + argmax',
            pseudo_weight=.25, rounds=3, blend=[.5, .3, .2]),
        cs_teacher_schedule='One fit from original fitting gold only; identical teacher on both refreshes.',
        residual=dict(inner_folds=5, inner_seed=0, group_purges=True, alpha=.8,
            graph=CS['graph'], modes=['raw', 'autoscale'],
            autoscale='mean gold OOF residual L1 / node propagated residual L1, floor 1e-12',
            extension='RBF weighted nearest fitting graph nodes',
            output='clip(base + field,0,1) independently by label; no row normalization or smoothing'),
        fixed='Parent blend, classifiers, weights and .3+argmax decoder unchanged; no parameter search.',
        access='Inductive; only eligible fitting inputs; Train157 excludes validation entirely; no test inputs.',
        inner='All five inner anchored fits refit preprocessing, clusters, pseudo labels and classifiers; same group purge.',
        screen='First three saved repeats, BOTH pools: positive mean paired delta and >=2/3 wins each.',
        confirm='Only screen passers: five extra saved repeats; positive mean and >=5/8 wins BOTH pools over all eight.',
        supplementary='Only shortlisted recipes: original validation plus three saved chronology windows.',
        diagnostic_exceptions=[], seeds='Parent0, graph PCA0, teacher/inner0; saved outer assignments unchanged.',
        metrics='Complete OOF repeat sample F1; mean, population SD; paired wins/ties/losses; macro and label PRF/support.',
        limitation='Repeated development labels; no untouched holdout or Kaggle outcome. Overlapping repeats are descriptive.')
    write_json(target, p)
    print('Frozen', target, digest(target), flush=True)


def check_frozen():
    p = json.loads((HERE / 'protocol.json').read_text())
    for name, sha in p['source_sha256'].items():
        assert digest(HERE / name) == sha, name
    for name, sha in p['protected_sha256'].items():
        assert digest(P1 / name) == sha, name
    return p


def group_hash(frame):
    return pd.util.hash_pandas_object(frame[EMB].round(10), index=False).to_numpy()


def cases(stage):
    train = pd.read_csv(P1 / 'data/part1_train.csv')
    val = pd.read_csv(P1 / 'data/part1_val.csv')
    both = pd.concat([train, val], ignore_index=True)
    if stage == 'supplementary':
        yield dict(pool='validation', seed=0, fold=0, fit=train, query=val, positions=np.arange(len(val)))
        for c in json.loads((SNAP / 'chronology.json').read_text())['chronological_splits']:
            fit = both[both.ID.isin(c['training_ids'])].copy()
            query = both.set_index('ID').loc[c['heldout_ids']].reset_index()
            assert not np.intersect1d(group_hash(fit), group_hash(query)).size
            yield dict(pool=c['name'], seed=0, fold=0, fit=fit, query=query, positions=np.arange(len(query)))
        return
    for pool, frame in [('train', train), ('pooled', both)]:
        plan = json.loads((SNAP / f'{pool}{3 if stage == "screen" else 5}.json').read_text())
        known = (frame[LABELS].to_numpy() >= 0).all(1)
        gold = np.flatnonzero(known)
        assert frame.iloc[gold].ID.tolist() == plan['gold_ids']
        groups = group_hash(frame)
        for fold in plan['folds']:
            query_pos = gold[fold['heldout_gold_positions']]
            allowed_masked = (~known) & (~np.isin(groups, groups[query_pos]))
            assert int((~known).sum() - allowed_masked.sum()) == fold['masked_duplicates_excluded']
            fit_pos = np.sort(np.r_[gold[fold['train_gold_positions']], np.flatnonzero(allowed_masked)])
            assert not np.intersect1d(groups[fit_pos], groups[query_pos]).size
            yield dict(pool=pool, seed=fold['seed'], fold=fold['fold'], fit=frame.iloc[fit_pos].copy(),
                query=frame.iloc[query_pos].copy(), positions=np.array(fold['heldout_gold_positions']))


def parent_reference(case, scores):
    if case['pool'] not in ['train', 'pooled']:
        return None
    r = np.load(REF / f'anchored_{case["pool"]}.npz')
    seeds = [20261126, 20261127, 20261128, 20261129, 20261130, 20261201, 20261202, 20261203]
    ix = seeds.index(case['seed'])
    positions = case['positions']
    assert r['ids'][positions].tolist() == case['query'].ID.tolist()
    assert np.array_equal(r['truth'][positions], case['query'][LABELS].to_numpy(int))
    delta = float(np.abs(scores - r['scores'][ix, positions]).max())
    assert delta < 1e-10, delta
    assert np.array_equal(select_reasons(scores), r['predictions'][ix, positions])
    return delta


def fit_case(case, variants, output, preflight=False):
    with threadpool_limits(limits=1):
        start = time.monotonic()
        frame, query = case['fit'], case['query']
        X, Y, Xq = frame[EMB].to_numpy(float), frame[LABELS].to_numpy(int), query[EMB].to_numpy(float)
        frame_before = frame.copy(deep=True)
        parent = AnchoredReasonClassifier().fit(X, Y)
        scores = dict(anchored=parent.predict_proba(Xq))
        audit = dict(pool=case['pool'], seed=case['seed'], fold=case['fold'],
            gold=int((Y >= 0).all(1).sum()), masked=int((Y == -100).all(1).sum()),
            fit_ids_sha256=hashlib.sha256('\n'.join(frame.ID).encode()).hexdigest(),
            parent_rounds=parent.round_audit_,
            reference_score_max_abs=parent_reference(case, scores['anchored']))
        if preflight:
            clone = PseudoTeacherAnchored().fit(X, Y)
            np.testing.assert_array_equal(clone.predict_proba(Xq), scores['anchored'])
            audit['no_teacher_clone_exact'] = True
        if 'cs_pseudo' in variants:
            cs = fit_pseudo(frame)
            scores['cs_pseudo'] = cs.predict_proba(Xq)
            audit['cs_pseudo'] = cs.component_audit_
            if preflight:
                np.testing.assert_allclose(cs.predict_proba(Xq[:2]), scores['cs_pseudo'][:2], atol=1e-12, rtol=0)
        if any(v.startswith('graph_residual_') for v in variants):
            residual = ResidualCorrector().fit(frame, parent)
            audit['residual'] = residual.audit
            for name in variants:
                if not name.startswith('graph_residual_'):
                    continue
                mode = name.removeprefix('graph_residual_')
                correction = residual.correction(Xq, mode)
                scores[name] = residual.predict_proba(Xq, mode)
                audit[name] = dict(correction_abs_mean=float(np.abs(correction).mean()),
                    clipped_fraction=float(((scores['anchored'] + correction < 0) |
                                            (scores['anchored'] + correction > 1)).mean()),
                    changed_rows=int(np.any(select_reasons(scores[name]) != select_reasons(scores['anchored']), axis=1).sum()))
                if preflight:
                    np.testing.assert_allclose(residual.predict_proba(Xq[:2], mode), scores[name][:2], atol=1e-12, rtol=0)
            if preflight:
                residual.fields['zero'] = np.zeros_like(residual.fields['raw'])
                np.testing.assert_array_equal(residual.predict_proba(Xq, 'zero'), scores['anchored'])
                audit['zero_residual_identity'] = True
                audit['query_batch_independence'] = True
        pd.testing.assert_frame_equal(frame, frame_before)
        audit['gold_unchanged'] = True
        audit['elapsed_seconds'] = time.monotonic() - start
        key = f'{case["pool"]}_{case["seed"]}_{case["fold"]}'
        arrays = dict(ids=query.ID.to_numpy(str), truth=query[LABELS].to_numpy(int), positions=case['positions'])
        for name, s in scores.items():
            assert np.isfinite(s).all()
            arrays[f'scores_{name}'] = s
            arrays[f'predictions_{name}'] = select_reasons(s)
        output = Path(output)
        np.savez_compressed(output / f'{key}.npz', **arrays)
        write_json(output / f'{key}.json', audit)
        return key, audit['elapsed_seconds']


def summarize(output, stages):
    """Independent sklearn metric computation after prediction files are written."""
    output = Path(output)
    rows, label_rows, audits = [], [], []
    for stage in stages:
        by_group = {}
        for p in sorted((output / stage).glob('*.npz')):
            a = json.loads(p.with_suffix('.json').read_text()); audits.append(a)
            by_group.setdefault((a['pool'], a['seed']), []).append(p)
        for (pool, seed), paths in by_group.items():
            arrays = [np.load(p) for p in paths]
            truth = np.concatenate([a['truth'] for a in arrays])
            ids = np.concatenate([a['ids'] for a in arrays])
            assert len(np.unique(ids)) == len(ids)
            if pool in ['train', 'pooled']:
                expected = json.loads((SNAP / f'{pool}3.json').read_text())['gold_ids']
                assert len(paths) == 5 and set(ids) == set(expected)
            for key in arrays[0].files:
                if not key.startswith('predictions_'): continue
                name = key.removeprefix('predictions_')
                pred = np.concatenate([a[key] for a in arrays])
                assert np.isin(pred, [0, 1]).all() and (pred.sum(1) > 0).all()
                row = dict(pool=pool, seed=seed, recipe=name, n=len(ids),
                    sample_f1=f1_score(truth, pred, average='samples'),
                    macro_f1=f1_score(truth, pred, average='macro', zero_division=0))
                rows.append(row)
                pr, re, f, support = precision_recall_fscore_support(truth, pred, zero_division=0)
                for j, label in enumerate(LABELS):
                    label_rows.append(dict(pool=pool, seed=seed, recipe=name, label=label,
                        precision=pr[j], recall=re[j], f1=f[j], support=int(support[j])))
    repeats = pd.DataFrame(rows)
    labels = pd.DataFrame(label_rows)
    summary = []
    for (pool, name), group in repeats.groupby(['pool', 'recipe']):
        parent = repeats[(repeats.pool == pool) & (repeats.recipe == 'anchored')].set_index('seed')
        delta = group.sample_f1.to_numpy() - parent.loc[group.seed, 'sample_f1'].to_numpy()
        summary.append(dict(pool=pool, recipe=name, repeats=len(group),
            sample_f1_mean=group.sample_f1.mean(), sample_f1_population_sd=group.sample_f1.std(ddof=0),
            macro_f1_mean=group.macro_f1.mean(), paired_delta_mean=float(delta.mean()),
            wins=int((delta > 1e-12).sum()), ties=int((np.abs(delta) <= 1e-12).sum()), losses=int((delta < -1e-12).sum())))
    summary = pd.DataFrame(summary)
    repeats.to_csv(output / 'repeat_scores.csv', index=False)
    labels.to_csv(output / 'per_label_repeat_scores.csv', index=False)
    labels.groupby(['pool', 'recipe', 'label'])[['precision', 'recall', 'f1', 'support']].mean().to_csv(output / 'per_label_summary.csv')
    summary.to_csv(output / 'summary.csv', index=False)
    write_json(output / 'fit_audits.json', audits)
    if stages == ['screen']:
        passes = []
        for v in VARIANTS:
            sub = summary[(summary.recipe == v) & summary.pool.isin(['train', 'pooled'])]
            if len(sub) == 2 and ((sub.paired_delta_mean > 0) & (sub.wins >= 2)).all(): passes.append(v)
        write_json(output / 'screen_gate.json', dict(passers=passes, rule='positive mean + 2/3 wins in BOTH pools'))
    print(summary.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['freeze', 'preflight', 'screen', 'confirm', 'supplementary', 'summarize'])
    p.add_argument('--output', type=Path)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.action == 'freeze': freeze(); return
    check_frozen()
    assert args.output is not None
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'summarize':
        summarize(args.output, [s for s in ['screen', 'confirm', 'supplementary'] if (args.output / s).exists()]); return
    stage = 'screen' if args.action == 'preflight' else args.action
    folder = args.output / stage
    folder.mkdir(exist_ok=True)
    variants = VARIANTS
    if stage != 'screen':
        variants = json.loads((args.output / 'screen_gate.json').read_text())['passers']
        assert variants, 'No screen passers; stop under the registered rule.'
    jobs = list(cases(stage))
    if args.action == 'preflight':
        print(fit_case(jobs[0], variants, folder, preflight=True), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        for c in jobs:
            key = f'{c["pool"]}_{c["seed"]}_{c["fold"]}'
            if (folder / f'{key}.npz').exists() and (folder / f'{key}.json').exists(): continue
            pending[executor.submit(fit_case, c, variants, folder)] = key
        for future in as_completed(pending):
            print('Completed', *future.result(), flush=True)
    summarize(args.output, [s for s in ['screen', 'confirm', 'supplementary'] if (args.output / s).exists()])


if __name__ == '__main__':
    main()
