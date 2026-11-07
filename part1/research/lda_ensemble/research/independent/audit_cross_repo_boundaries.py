"""Reproduce the first-pooled-fold fitting-boundary audit for all six recipes."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
import run_cross_repo_benchmark as r


def main():
    r.check_contract()
    checks = []
    with threadpool_limits(limits=1), r.pooled.pooled_context():
        p = r.pooled.protocol()
        f = p['folds'][0]
        a, h = np.array(f['train_gold_positions']), np.array(f['heldout_gold_positions'])
        held_rows = r.e.LAB[h]
        excluded = r.e.GROUP[held_rows]
        masked = r.e.UNLAB[np.isin(r.e.GROUP[r.e.UNLAB], excluded)]
        forbidden = np.r_[held_rows, masked]
        q = r.e.DATA.iloc[held_rows].drop(columns=r.e.LABELS).copy()
        for candidate in r.registry():
            r.pooled.clear_caches()
            base = r.e.predict(a, q, candidate['config'], excluded_groups=excluded)
            raw, old_y = r.e.DATA.iloc[forbidden].copy(), r.e.Y.copy()
            try:
                r.e.Y[h] = 1 - r.e.Y[h]
                r.e.DATA.loc[held_rows, r.e.LABELS] = 1 - r.e.DATA.loc[held_rows, r.e.LABELS].to_numpy()
                r.e.DATA.loc[forbidden, r.e.EMB] = 1e6
                r.pooled.clear_caches()
                changed = r.e.predict(a, q, candidate['config'], excluded_groups=excluded)
            finally:
                r.e.DATA.iloc[forbidden] = raw.to_numpy()
                r.e.Y[:] = old_y
                r.pooled.clear_caches()
            one = r.e.predict(a, q.iloc[:1], candidate['config'], excluded_groups=excluded)
            np.testing.assert_allclose(base, changed, rtol=0, atol=1e-9)
            np.testing.assert_allclose(base[:1], one, rtol=0, atol=1e-9)
            check = dict(candidate=candidate['name'], heldout_labels_and_features_cannot_change_fit=True,
                matching_masked_duplicates_cannot_change_fit=True, query_batch_invariance=True,
                max_difference=float(np.max(np.abs(base-changed))), masked_duplicates_tested=len(masked))
            checks.append(check)
            print(json.dumps(check), flush=True)
    r.save(r.OUT/'fitting_boundary_audit.json', dict(passed=True, checks=checks))
    r.check_contract()


if __name__ == '__main__':
    main()
