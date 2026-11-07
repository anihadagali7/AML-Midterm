"""Boundary audit for each new fitting/selection path on one pooled fold."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_ani_improvements as task


def main():
    task.guard()
    names=['Ani + tri 25%','Ani + SVM 10%','Ani pseudo weight 0.5',
           'Ani agreement, pseudo weight 0.5','Ani three-GMM-seed average',
           'Ani Other specialist 25%','Ani inner-CV Other threshold']
    checks=[]
    with threadpool_limits(limits=1),task.pooled.pooled_context():
        p=task.pooled.protocol();f=p['folds'][0]
        a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
        e=task.e
        held=e.LAB[h];ex=e.GROUP[held]
        masked=e.UNLAB[np.isin(e.GROUP[e.UNLAB],ex)]
        forbidden=np.r_[held,masked]
        q=e.DATA.iloc[held].drop(columns=e.LABELS).copy()
        for name in names:
            c=next(c for c in task.registry() if c['name']==name)
            task.models.clear();original=e.predict(a,q,c['config'],ex)
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
                masked_duplicate_exclusion=True,masked_duplicate_rows_tested=len(masked),
                query_batch_invariance=True,max_difference=float(np.max(np.abs(original-changed))))
            checks.append(row);print(json.dumps(row),flush=True)
    task.r.save(task.OUT/'boundary_audit.json',dict(passed=True,checks=checks,script_sha256=task.b.file_hash(Path(__file__))))
    task.guard()


if __name__=='__main__':main()
