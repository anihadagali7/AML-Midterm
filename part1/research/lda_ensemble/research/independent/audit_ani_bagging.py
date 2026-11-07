"""Check every subset-averaging fit respects outer and omitted-gold groups."""
from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_limits
import run_ani_bagging as study


def main():
    task=study.task;task.guard();checks=[]
    names=['70/15/15 + half subset average','70/15/15 + full subset average']
    with threadpool_limits(limits=1),task.pooled.pooled_context():
        e=task.e;f=task.pooled.protocol()['folds'][0]
        a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
        held=e.LAB[h];ex=e.GROUP[held]
        masked=e.UNLAB[np.isin(e.GROUP[e.UNLAB],ex)];forbidden=np.r_[held,masked]
        q=e.DATA.iloc[held].drop(columns=e.LABELS).copy()
        native=task.models.base.native_variant;seen=[]

        def inspected(pos,query,config,excluded=None):
            fit_ids=set(e.DATA.iloc[e.LAB[pos]].ID)
            assert fit_ids<=set(e.DATA.iloc[e.LAB[a]].ID)
            omitted_groups=set(e.GROUP[e.LAB[a]])-set(e.GROUP[e.LAB[pos]])
            assert set(ex)|omitted_groups<=set(excluded)
            result=native(pos,query,config,excluded)
            audit=task.models.base.LAST_AUDIT
            expected=set(e.DATA.iloc[e.UNLAB[~np.isin(e.GROUP[e.UNLAB],excluded)]].ID)
            assert set(audit['masked_ids'])==expected
            assert set(audit['gold_ids'])==fit_ids
            seen.append(dict(gold_rows=len(pos),omitted_groups=len(omitted_groups),masked_rows=len(expected)))
            return result

        task.models.base.native_variant=inspected
        try:
            for name in names:
                c=next(c for c in task.registry() if c['name']==name)
                seen.clear();task.models.clear();original=e.predict(a,q,c['config'],ex)
                fits=seen.copy();raw=e.DATA.iloc[forbidden].copy();truth=e.Y.copy()
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
                    matching_masked_rows_tested=len(masked),query_batch_invariance=True,
                    subset_fitting_checks=fits,max_difference=float(np.max(np.abs(original-changed))))
                checks.append(row);print(json.dumps(row),flush=True)
        finally:task.models.base.native_variant=native
    task.r.save(task.OUT/'boundary_audit.json',dict(passed=True,checks=checks,script_sha256=task.b.file_hash(Path(__file__))))
    task.guard()


if __name__=='__main__':main()
