"""Mutation and fitting-boundary checks for all new combination code paths."""
from pathlib import Path
import json
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_ani_combinations as task


def main():
    task.guard()
    checks=[]
    names=['70/15/15 + '+s for s in ['pseudo weight 0.5','three-seed average',
           'inner Other threshold','Other specialist 50%']]
    with threadpool_limits(limits=1),warnings.catch_warnings(),task.pooled.pooled_context():
        warnings.simplefilter('error',ConvergenceWarning)
        f=task.pooled.protocol()['folds'][0]
        a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
        e=task.e;held=e.LAB[h];ex=e.GROUP[held]
        masked=e.UNLAB[np.isin(e.GROUP[e.UNLAB],ex)]
        forbidden=np.r_[held,masked]
        q=e.DATA.iloc[held].drop(columns=e.LABELS).copy()
        native=task.models.base.native_variant
        seen=[]

        def inspected(pos,query,config,excluded=None):
            fit_ids=set(e.DATA.iloc[e.LAB[pos]].ID)
            assert fit_ids<=set(e.DATA.iloc[e.LAB[a]].ID)
            query_groups=e.pd.util.hash_pandas_object(query[e.EMB].round(10),index=False).to_numpy()
            assert not set(e.GROUP[e.LAB[pos]]) & set(query_groups)
            assert set(ex)<=set(excluded)
            assert set(query_groups)<=set(excluded)
            result=native(pos,query,config,excluded)
            audit=task.models.base.LAST_AUDIT
            allowed_masked=set(e.DATA.iloc[e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded)]].ID)
            assert set(audit['masked_ids'])==allowed_masked
            assert set(audit['gold_ids'])==fit_ids
            seen.append(dict(fitting_gold=len(pos),query_rows=len(query),excluded_groups=len(set(excluded))))
            return result

        task.models.base.native_variant=inspected
        try:
            for name in names:
                c=next(c for c in task.registry() if c['name']==name)
                seen.clear();task.models.clear();original=e.predict(a,q,c['config'],ex)
                fitting_checks=seen.copy()
                raw=e.DATA.iloc[forbidden].copy();truth=e.Y.copy()
                try:
                    e.Y[h]=1-e.Y[h]
                    e.DATA.loc[held,e.LABELS]=1-e.DATA.loc[held,e.LABELS].to_numpy()
                    e.DATA.loc[forbidden,e.EMB]=1e6
                    task.models.clear();changed=e.predict(a,q,c['config'],ex)
                finally:
                    e.DATA.iloc[forbidden]=raw.to_numpy();e.Y[:]=truth;task.models.clear()
                one=e.predict(a,q.iloc[:1],c['config'],ex)
                np.testing.assert_allclose(original,changed,rtol=0,atol=1e-9)
                np.testing.assert_allclose(original[:1],one,rtol=0,atol=1e-9)
                row=dict(candidate=name,heldout_label_and_feature_invariance=True,
                    masked_duplicate_rows_tested=len(masked),query_batch_invariance=True,
                    nested_fitting_checks=fitting_checks,max_difference=float(np.max(np.abs(original-changed))))
                checks.append(row);print(json.dumps(row),flush=True)
        finally:task.models.base.native_variant=native
    task.r.save(task.OUT/'boundary_audit.json',dict(passed=True,checks=checks,script_sha256=task.b.file_hash(Path(__file__))))
    task.guard()


if __name__=='__main__':main()
