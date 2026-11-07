"""Uncached perturbation check of nested specialist fitting and query isolation."""
from pathlib import Path
import json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
import phase23 as m,phase22 as p22,experiment as e

def run():
    c=json.loads(Path('experiments/phase22-baseline.json').read_text())['candidate']['config']
    c={**c,'model':'joint_specialist','meta':'other','meta_C':.3,'inner_splits':15}
    a,b=e.make_splits(seed=20261110)[0];q=e.DATA.iloc[e.LAB[b]].copy();excluded=e.GROUP[e.LAB[b]]
    original=e.Y.copy()
    def clear():m.INNER_CACHE.clear();p22.PREDICTION_CACHE.clear()
    with threadpool_limits(limits=1):
        clear();p=e.predict(a,q,c,excluded)
        try:
            e.Y[b]=1-e.Y[b];q.loc[:,e.LABELS]=1-q[e.LABELS].to_numpy()
            clear();altered=e.predict(a,q,c,excluded)
        finally:e.Y[:]=original
        one=e.predict(a,q.iloc[:1],c,excluded)
    assert np.array_equal(p,altered)
    assert np.array_equal(p[:1],one)
    assert np.isin(p,[0,1]).all() and (p.sum(1)>0).all()
    checks=dict(nested_heldout_label_invariance=True,caches_cleared_before_perturbation=True,
                query_batch_invariance=True,binary_nonempty=True,max_difference=float(abs(p-altered).max()),
                heldout_rows=len(b),validation_used_for_training=False,
                source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['phase22.py','phase23.py']})
    e.save_json('experiments/phase23-integrity-checks.json',checks)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':run()
