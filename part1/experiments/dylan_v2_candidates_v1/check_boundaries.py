"""Meaningful leakage probes for the strict adapters, independent of scoring."""
import copy
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as study
from p1lib import cache
from p1lib.head import apply_head
import strict_adapter as strict

HERE = Path(__file__).resolve().parent


def predictions(ctx):
    cache.clear()
    result = {}
    for name, stem in study.CANDIDATES.items():
        cfg = study.read(HERE / 'manifests' / (stem + '.manifest.json'))['config']
        rows = np.r_[ctx.eval, ctx.pool]
        out = strict.run_model(cfg['model'], ctx, rows)
        P = out['P'].astype(np.float32).astype(float); P /= P.sum(1, keepdims=True)
        yp, Q, prior = apply_head(P[:len(ctx.eval)], out['prior'], ctx.D.CM,
                                 cfg['head'], P[len(ctx.eval):])
        result[name] = dict(predictions=yp, probabilities=Q, temperature=out['T'])
    return result


def assert_same(a, b, select=slice(None)):
    errors = {}
    for name in a:
        np.testing.assert_array_equal(a[name]['predictions'][select], b[name]['predictions'][select])
        np.testing.assert_allclose(a[name]['probabilities'][select], b[name]['probabilities'][select], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a[name]['temperature'], b[name]['temperature'], atol=1e-12, rtol=0)
        errors[name] = float(abs(a[name]['probabilities'][select]-b[name]['probabilities'][select]).max())
    return errors


def main():
    case = next(c for c in study.cases() if c['access'] == 'strict_inductive' and c['kind'] == 'validation')
    D, ctx, _, _ = study.prepare(case)
    assert len(D.idx['test']) == 0
    assert set(D.df.ID) == set(case['fit_ids']) | set(case['query_ids'])
    assert (D.y[ctx.eval] == -1).all() and (D.Y[ctx.eval] == -1).all()
    with threadpool_limits(limits=1):
        original = predictions(ctx)
        changed_labels = copy.deepcopy(ctx)
        changed_labels.D.y[ctx.eval] = np.arange(len(ctx.eval)) % D.K
        changed_labels.D.Y[ctx.eval] = D.CM[changed_labels.D.y[ctx.eval]]
        changed_labels.D.df.loc[ctx.eval, study.LABELS] = changed_labels.D.Y[ctx.eval]
        label_errors = assert_same(original, predictions(changed_labels))
        changed_features = copy.deepcopy(ctx)
        row = ctx.eval[0]
        changed_features.D.emb[row] += 7 * np.where(np.arange(150) % 2, 1, -1)
        changed_features.D.df.loc[row, study.EMB] = changed_features.D.emb[row]
        feature_errors = assert_same(original, predictions(changed_features), slice(1, None))
        # Check a full candidate2 inner pipeline (features and pseudo labels)
        # against changes to the labels its calibration fold is meant to hide.
        tr, te = next(strict.inner_folds(ctx))
        inner = strict.subcontext(ctx, ctx.train[tr], ctx.train[te])
        cfg = study.read(HERE / 'manifests/cand2_compliant_argmax.manifest.json')['config']['model']
        cache.clear(); a = strict.uncalibrated(cfg, inner, inner.eval)
        changed = copy.deepcopy(inner)
        changed.D.y[inner.eval] = np.arange(len(inner.eval)) % D.K
        changed.D.Y[inner.eval] = D.CM[changed.D.y[inner.eval]]
        changed.D.df.loc[inner.eval, study.LABELS] = changed.D.Y[inner.eval]
        cache.clear(); b = strict.uncalibrated(cfg, changed, changed.eval)
        np.testing.assert_allclose(a['P'], b['P'], atol=1e-12, rtol=0)
    out = dict(status='passed', test_case=case['key'],
               outer_query_label_change_max_errors=label_errors,
               changing_one_query_embedding_does_not_change_other_queries=feature_errors,
               calibration_inner_query_label_change_max_error=float(abs(a['P']-b['P']).max()),
               literal_test_rows_present=False, query_labels_masked_before_fit=True,
               caveat='These are targeted invariance probes, in addition to all saved-boundary audits; not a proof about every future pipeline.')
    study.save(HERE / 'boundary_checks.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
