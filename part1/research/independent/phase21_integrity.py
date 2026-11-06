"""Audit the selected hybrid without tuning its configuration or decoder."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import phase21,experiment as e

def run():
    candidate=json.loads(Path('experiments/current-best.json').read_text())['candidate']
    c=candidate['config']; rule=candidate['rule']
    train,held=e.make_splits(seed=20261107,n_splits=5)[0]
    excluded=e.GROUP[e.LAB[held]]
    query=e.DATA.iloc[e.LAB[held]].copy()
    original=e.Y.copy()
    with threadpool_limits(limits=1):
        before=e.predict(train,query,c,excluded_groups=excluded)
        try:
            e.Y[held]=1-e.Y[held]
            query.loc[:,e.LABELS]=1-query[e.LABELS].to_numpy()
            after=e.predict(train,query,c,excluded_groups=excluded)
        finally:
            e.Y[:]=original
        single=e.predict(train,query.iloc[:1],c,excluded_groups=excluded)
        with_masked,pu,ix=e.predict(train,query,c,excluded_groups=excluded,return_unlabeled=True)
    assert np.array_equal(before,after)
    assert np.array_equal(before[:1],single)
    assert np.array_equal(before,with_masked)
    assert np.array_equal(ix,e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded)])
    assert not np.isin(e.GROUP[ix],excluded).any()
    assert np.isin(before,[0,1]).all() and (before.sum(1)>0).all()
    assert pu.shape==(len(ix),5) and np.isin(pu,[0,1]).all()
    assert (pu.sum(1)>0).all()
    weights=dict(np.load('experiments/phase21-weights.npz'))
    for w in weights.values():
        assert len(w)==len(e.Y) and np.isfinite(w).all() and (w>0).all()
        assert abs(w.mean()-1)<1e-12
    h=e.decisions(before,rule);y=e.Y[held]
    row=2*(h*y).sum(1)/(h.sum(1)+y.sum(1))
    assert abs(row.mean()-f1_score(y,h,average='samples'))<1e-12
    source=['phase21.py','phase21_weights.py','phase21_select.py','phase21_hybrid.py',
            'phase21_plain.py','phase21_classical_hybrid.py','phase21_integrity.py']
    checks=dict(selected_config=c,heldout_rows=len(held),
        selected_hybrid_holdout_label_invariance=True,
        selected_hybrid_query_batch_invariance=True,
        selected_hybrid_return_unlabeled_query_invariance=True,
        matching_masked_embedding_duplicates_excluded=True,
        all_predictions_binary_nonempty=True,
        positive_normalized_risk_weights=True,uniform_row_f1_matches_sklearn=True,
        max_label_perturbation=float(abs(before-after).max()),
        max_batch_difference=float(abs(before[:1]-single).max()),
        validation_labels_used_for_training=0,refit_with_validation_performed=False,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source})
    e.save_json('experiments/phase21-integrity-checks.json',checks)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':run()
