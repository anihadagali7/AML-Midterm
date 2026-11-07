"""Reproduce Ani's validation model and exact train+validation submission fit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import json,time
import joblib
import numpy as np
import pandas as pd
from experiments import ROOT,load_data,metrics
from ani_models import AniFeatures,AniStack,BASES

OUT=ROOT/'experiments/ani_comparison'


def run():
    data=load_data();all_df=pd.concat(list(data.values()),ignore_index=True)
    path=OUT/'transductive_features.joblib'
    if path.exists():space=joblib.load(path)
    else:
        space=AniFeatures().fit(all_df);joblib.dump(space,path,compress=3)
    rows=[]
    for population in ['train','trainval']:
        path=OUT/f'ani_exact_{population}.joblib'
        fit=data['train'] if population=='train' else pd.concat([data['train'],data['val']],ignore_index=True)
        start=time.time()
        if path.exists():model=joblib.load(path)
        else:
            model=AniStack(tau=3).fit(fit,features=space)
            joblib.dump(model,path,compress=3)
        target=data['val'] if population=='train' else data['test']
        p=model.predict_proba(target)
        pd.DataFrame({'ID':target.ID,'probability':p}).to_csv(OUT/f'ani_exact_{population}_predictions.csv',index=False)
        np.savez_compressed(OUT/f'ani_exact_{population}_bases.npz',oof=model.oof_base_predictions,pred=model.predict_base_matrix(target))
        row={'model':f'ani_exact_{population}','training_rows':len(fit),'seconds':time.time()-start}
        if population=='train':row.update(metrics(target,p))
        else:
            reference=pd.read_csv(OUT/'source/part2_best_submission.csv')
            expected=reference.set_index('ID').loc[target.ID,'label'].to_numpy()
            difference=np.abs(np.clip(p,.001,.999)-expected)
            row.update(max_submission_probability_difference=float(difference.max()),mean_submission_probability_difference=float(difference.mean()))
        rows.append(row)
        print(row,flush=True)
        print('Meta coefficients',dict(zip([x[0] for x in BASES]+['I','II','III'],model.meta.coef_[0])),flush=True)
        pd.DataFrame(rows).to_csv(OUT/'exact_reproduction.csv',index=False)
    (OUT/'exact_reproduction.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':run()
