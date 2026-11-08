"""Explanatory discrimination/error checks; never selects thresholds or fits models."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as study


def main():
    output=study.HERE/'results';rows=[];changes=[];groups={}
    for case in study.cases():
        if case['pool']=='supplementary':continue
        path=output/'predictions'/(case['key']+'.npz')
        if not path.exists():continue
        z=study.loadnp(path);a,c=study.refs(case)
        z['native_parent']=.8*a+.2*c
        groups.setdefault((case['pool'],case['seed']),[]).append(z)
    for (pool,seed),zs in groups.items():
        if len(zs)!=5:continue
        y=np.concatenate([z['truth'] for z in zs]);parent=np.concatenate([z['predictions'][0] for z in zs])
        values=dict(parent=np.concatenate([z['native_parent'][:,4] for z in zs]),
                    global_svc=np.concatenate([z['outer_specialists'][:,0] for z in zs]),
                    conditional_svc=np.concatenate([z['outer_specialists'][:,2] for z in zs]))
        for subgroup,keep in [('all',np.ones(len(y),bool)),('gold_recruitment',y[:,3]==1)]:
            for name,s in values.items():
                rows.append(dict(pool=pool,seed=seed,subgroup=subgroup,score=name,n=int(keep.sum()),
                    positives=int(y[keep,4].sum()),auc=float(roc_auc_score(y[keep,4],s[keep])),
                    average_precision=float(average_precision_score(y[keep,4],s[keep]))))
        base_f1=2*(y*parent).sum(1)/(y.sum(1)+parent.sum(1))
        for j,name in enumerate(study.NAMES):
            p=np.concatenate([z['predictions'][j] for z in zs]);new_f1=2*(y*p).sum(1)/(y.sum(1)+p.sum(1))
            changes.append(dict(pool=pool,seed=seed,recipe=name,
                improved_rows=int((new_f1>base_f1+1e-12).sum()),worsened_rows=int((new_f1<base_f1-1e-12).sum()),
                changed_label_sets=int((p!=parent).any(1).sum()),
                other_misses_fixed=int(((y[:,4]==1)&(parent[:,4]==0)&(p[:,4]==1)).sum()),
                other_false_positives_added=int(((y[:,4]==0)&(parent[:,4]==0)&(p[:,4]==1)).sum()),
                recruitment_other_recovered=int(((y[:,4]==1)&(y[:,3]==1)&(parent[:,4]==0)&(p[:,4]==1)).sum())))
    pd.DataFrame(rows).to_csv(output/'summary/discrimination_diagnostic.csv',index=False)
    pd.DataFrame(changes).to_csv(output/'summary/change_diagnostic.csv',index=False)
    print(pd.DataFrame(rows).groupby(['pool','subgroup','score'])[['auc','average_precision']].mean().round(5).to_string())


if __name__=='__main__':main()
