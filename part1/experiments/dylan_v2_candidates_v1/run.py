"""Fixed Dylan v2 candidates on their native benchmark and our grouped benchmark."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PART = HERE.parents[1]
REPO = PART.parent
PREVIOUS = PART / 'experiments/derived_benchmark_audit_v1'
sys.path.insert(0, str(HERE / 'source'))
sys.path.insert(0, str(PART))
from p1lib.data import load, LABELS, EMB
from p1lib import cache
from p1lib.context import make_ctx
from p1lib.pipeline import run_model, validate
from p1lib.head import apply_head
from anchored_solution import AnchoredReasonClassifier
import strict_adapter

CANDIDATES = {'cand1': 'cand1_selected_1se', 'cand2': 'cand2_compliant_argmax',
              'cand4': 'cand4_ens3_cs_sld', 'cand5': 'cand5_lr_cs_sld_robust'}
BLENDS = {'anchored_cand1': ['anchored', 'cand1'], 'anchored_cand5': ['anchored', 'cand5'],
          'anchored_cand1_cand5': ['anchored', 'cand1', 'cand5']}
MODELS = ['anchored', *CANDIDATES, *BLENDS]
DATA = None


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data():
    global DATA
    if DATA is None:
        DATA = load(str(PART / 'data'))
    return DATA


def cases():
    old = read(PREVIOUS / 'splits.json')
    native = [dict(c, access='native_dylan') for c in old if c['track'] == 'dylan' or c['kind'] == 'validation']
    strict = [dict(c, access='strict_inductive') for c in old if c['track'] == 'canonical']
    for c in native + strict:
        c['source_key'] = c['key']
        c['key'] = c['access'] + '__' + c['key']
    assert len(native) == 54 and len(strict) == 84
    return native + strict


def freeze():
    files = [HERE / 'run.py', HERE / 'strict_adapter.py', PART / 'anchored_solution.py', PART / 'TESTING_POLICY.md',
             PREVIOUS / 'splits.json', PREVIOUS / 'protocol.json', PREVIOUS / 'comparison.csv', HERE / 'requirements-lock.txt']
    files += list((PART / 'data').glob('*.csv'))
    files += list((HERE / 'source/p1lib').glob('*.py')) + list((HERE / 'manifests').glob('*.json'))
    files += [HERE / f for f in ['candidate_definitions.md', 'supplied_candidates.csv', 'supplied_ensemble_results.json',
                                'supplied_ensemble.py', 'supplied_REPORT.md', 'supplied_requirements.txt']]
    hashes = {str(p.relative_to(REPO)): sha(p) for p in files}
    if (HERE / 'protocol.json').exists():
        assert read(HERE / 'protocol.json')['source_hashes'] == hashes
        return
    configs = {name: read(HERE / 'manifests' / f'{stem}.manifest.json')['config'] for name, stem in CANDIDATES.items()}
    for cfg in configs.values(): validate(cfg)
    save(HERE / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
         base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
         branch='codex/part1-derived-benchmark-audit', parent='Anchored 0.89636', models=MODELS, configs=configs,
         hypothesis='Do the four fixed Dylan v2 candidates and three supplied fixed blends improve on anchored under our benchmark, and how do they score under the supplied native benchmark?',
         plans='All saved 80 grouped CV folds + validation + three chronology windows; native Dylan A50+B3+C1. All seven candidates/blends are diagnostic full comparisons, including unfavorable results; no shortlist selection.',
         native_access='Unmodified p1lib model code, full1561 label-free preprocessing including query and actual test features. Source inner folds/calibration retained. Gold labels restricted by native ctx.train. SLD on query batch. Anchor remains inductive, as in supplied comparison.',
         strict_access='Only eligible fitting rows and query rows exist in per-case Data; query labels masked, no test rows. Explicit fitting gold/masked sets from saved plans. All preprocessing and pseudo labels refit within grouped calibration folds. Inner group-matching masked rows purged. C&S residual base OOF uses grouped folds and refit scaling. KNN synthetic training features use leave-embedding-group-out fitting and scaling. Kmeans soft-feature temperature estimated on fitting rows only. SLD uses eligible masked training pool, never query batch.',
         schema='Eight source label combinations are frozen recipe support, not selected during this experiment. All prediction labels are five fixed task labels; no query labels enter fit.',
         seeds='One inner seed0 per benchmark fit, matching Dylan evaluate.run_fold. The preexisting final submission CSVs average five inner seeds; these runs evaluate the supplied benchmark convention, not a five-seed final artifact.',
         blends={'weights': 'equal, supplied definitions unchanged', 'decode': 'argmax label plus all label scores >=0.3', 'members': BLENDS},
         advancement='No automatic promotion. Primary check: positive paired mean and >=5/8 wins in BOTH grouped pools; report validation and chronology regressions. No tuning hyperparameters, cutoffs, heads, or blend weights.',
         retention='Keep compact matched comparisons and source/recipes/summaries; remove transformed features, graph/fitted caches, temporary logs. Prediction retention decided by usefulness of matched access comparison, not by winning within this study.',
         references_excluded='The two reference CSVs are explicitly not proposed assignment-compliant candidates; their supplied scores remain in candidate_definitions.md. cand3 duplicates cand1.',
         source_hashes=hashes, versions={n: importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','iterative-stratification','joblib']},
         package_version_note='Use the current repository environment for matched fits. Supplied native score replay will report any difference from Dylan package versions; never silently substitute reported scores.',
         no_new_submission=True))
    save(HERE / 'case_plan.json', cases())


def prepare(case):
    D = data()
    pos = {v: i for i, v in enumerate(D.df.ID)}
    train, query, pool, fit = [np.array([pos[v] for v in case[key]]) for key in ['train_gold', 'query_ids', 'masked_ids', 'fit_ids']]
    if case['access'] == 'native_dylan':
        ctx = make_ctx(D, train, query, seed=0, name=case['key'])
        return D, ctx, fit, query
    keep = np.union1d(fit, query)
    remap = {v: i for i, v in enumerate(keep)}
    tr, qu, po = [np.array([remap[v] for v in rows]) for rows in [train, query, pool]]
    idx = {'train': np.array([i for i, v in enumerate(keep) if v in set(D.idx['train'])]),
           'lab': np.array([i for i, v in enumerate(keep) if v in set(D.idx['lab'])]),
           'val': np.array([i for i, v in enumerate(keep) if v in set(D.idx['val'])]),
           'unl': po, 'test': np.array([], int), 'labval': np.r_[tr, qu]}
    sub = replace(D, df=D.df.iloc[keep].reset_index(drop=True), emb=D.emb[keep], Y=D.Y[keep], y=D.y[keep], idx=idx, year=D.year[keep])
    ctx = strict_adapter.context(sub, tr, qu, po, seed=0, name=case['key'])
    return ctx.D, ctx, np.union1d(tr, po), qu


def evaluate_case(case, write=True):
    path = HERE / 'results' / case['key']
    if write and path.with_suffix('.json').exists():
        meta = read(path.with_suffix('.json'))
        assert sha(path.with_suffix('.npz')) == meta['sha256']
        return dict(key=case['key'], cached=True)
    start = time.time()
    cache.clear(); strict_adapter.AUDIT.clear()
    D, ctx, fitting, query = prepare(case)
    source = data().df.set_index('ID')
    y = source.loc[case['query_ids'], LABELS].to_numpy(int)
    fitframe = source.loc[case['fit_ids']]
    qframe = source.loc[case['query_ids']]
    with threadpool_limits(limits=1):
        anchor = AnchoredReasonClassifier().fit(fitframe[EMB], fitframe[LABELS])
        scores = {'anchored': anchor.predict_proba(qframe[EMB])}
        predictions = {'anchored': anchor.predict(qframe[EMB])}
        arrays = dict(ids=qframe.index.to_numpy(str), truth=y, combo_matrix=D.CM)
        metadata = {}
        for name, stem in CANDIDATES.items():
            cfg = read(HERE / 'manifests' / f'{stem}.manifest.json')['config']
            rows = query if case['access'] == 'native_dylan' else np.r_[query, ctx.pool]
            runner = run_model if case['access'] == 'native_dylan' else strict_adapter.run_model
            out = runner(cfg['model'], ctx, rows)
            # Match native evaluate.run_fold's float32 fold cache before the head.
            P = out['P'][:len(query)].astype(np.float32).astype(float); P /= P.sum(1, keepdims=True)
            target = None
            if case['access'] != 'native_dylan':
                target = out['P'][len(query):].astype(np.float32).astype(float)
                target /= target.sum(1, keepdims=True)
            yp, Q, prior = apply_head(P, out['prior'], D.CM, cfg['head'], target)
            scores[name], predictions[name] = Q @ D.CM, yp
            arrays[name + '_Q'] = Q
            arrays[name + '_P'] = P
            metadata[name] = dict(temperature=out['T'], source_prior=np.asarray(out['prior']).tolist(),
                                  target_prior=None if prior is None else prior.tolist(), info=out.get('info', {}))
        for name, members in BLENDS.items():
            scores[name] = sum(scores[m] for m in members) / len(members)
            predictions[name] = (scores[name] >= .3).astype(int)
            predictions[name][np.arange(len(query)), scores[name].argmax(1)] = 1
    for name in MODELS:
        p, s = predictions[name], scores[name]
        assert p.shape == s.shape == y.shape and np.isfinite(s).all() and (p.sum(1) > 0).all()
        arrays[name + '_scores'], arrays[name + '_predictions'] = s, p
    meta = dict(key=case['key'], seconds=time.time()-start, models=metadata,
                nested_context_count=len(strict_adapter.AUDIT),
                nested_gold_group_overlaps=sum(x['gold_group_overlap'] for x in strict_adapter.AUDIT),
                nested_masked_group_overlaps=sum(x['masked_group_overlap'] for x in strict_adapter.AUDIT),
                nested_audit_sha256=hashlib.sha256(json.dumps(strict_adapter.AUDIT, sort_keys=True).encode()).hexdigest())
    if not write:
        return arrays, meta
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    meta['sha256'] = sha(path.with_suffix('.npz'))
    save(path.with_suffix('.json'), meta)
    return dict(key=case['key'], seconds=round(meta['seconds'], 2), nested_contexts=meta['nested_context_count'])


def worker(case):
    try:
        return evaluate_case(case)
    except Exception:
        save(HERE / 'results' / (case['key'] + '.failure.json'), dict(traceback=traceback.format_exc()))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--access', choices=['native_dylan', 'strict_inductive'])
    args = ap.parse_args()
    (HERE / 'results').mkdir(exist_ok=True)
    freeze()
    plan = cases()
    if args.access:
        plan = [c for c in plan if c['access'] == args.access]
    if args.smoke:
        plan = [c for c in plan if c['kind'] == 'validation']
        for c in plan:
            print(worker(c), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(worker, c) for c in plan]
        for i, f in enumerate(as_completed(pending), 1):
            print(i, len(plan), json.dumps(f.result()), flush=True)


if __name__ == '__main__':
    main()
