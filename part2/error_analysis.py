"""Part 1 reason labels are joined exclusively after outcome prediction."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from experiments import ROOT, OUT, REASONS, PHASES, metrics, load_data


def reason_labels():
    path=ROOT.parent/'part1/experiments/additional_methods/classifier_models/dbscan_m5_q25__logistic_C1000.joblib'
    model=joblib.load(path)
    frames=[]
    for split in ['train','val','test']:
        df=pd.read_csv(ROOT.parent/f'part1/data/part1_{split}.csv')
        x=model['scaler'].transform(df[model['features']].to_numpy())
        p=np.column_stack([m.predict_proba(x)[:,1] for m in model['estimators']])
        y=p>=.5
        empty=~y.any(axis=1)
        y[empty,p[empty].argmax(axis=1)]=True
        known=df[REASONS].ge(0).all(axis=1).to_numpy() if all(c in df for c in REASONS) else np.zeros(len(df),bool)
        y=y.astype(int)
        if known.any():y[known]=df.loc[known,REASONS].to_numpy()
        result=pd.DataFrame(y,columns=REASONS)
        result['trial_id']=df.ID.to_numpy()
        result['reason_source']=np.where(known,'true_'+split,'predicted_'+split)
        frames.append(result)
    labels=pd.concat(frames,ignore_index=True)
    assert not labels.trial_id.duplicated().any()
    labels.to_csv(OUT/'reason_labels_for_analysis.csv',index=False)
    (OUT/'reason_model_provenance.json').write_text(json.dumps({
        'path':str(path.relative_to(ROOT.parent)),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'policy':'Visible Part 1 train/val labels when available; otherwise frozen DBSCAN/logistic Part 1 model, cutoff 0.5, argmax fallback.',
        'usage':'Error analysis only; none of these columns or embeddings is an outcome predictor.'},indent=2))
    return labels


def errors_by_reason(df, probabilities, labels):
    d=df.merge(labels,on='trial_id',how='left',validate='many_to_one',sort=False)
    assert np.array_equal(d.ID,df.ID)
    d['probability']=probabilities
    rows=[]
    for phase in PHASES:
        for reason in REASONS+['No recorded stopping reason']:
            group=d.reason_source.isna() if reason.startswith('No recorded') else d[reason].eq(1)
            v=d[d.phase.eq(phase)&d.label.eq(0)&group]
            assert not v.trial_id.duplicated().any()
            n=len(v);k=int((v.probability>=.5).sum())
            # Wilson intervals indicate small groups; predictions also have label uncertainty.
            if n:
                phat=k/n;z=1.96;denom=1+z*z/n
                center=(phat+z*z/(2*n))/denom
                radius=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denom
                lo,hi=center-radius,center+radius
            else:phat=lo=hi=np.nan
            rows.append(dict(phase=phase,reason=reason,n=n,false_success=k,false_success_rate=phat,
                wilson_lower=lo,wilson_upper=hi,
                true_reason_rows=int(v.reason_source.fillna('').str.startswith('true').sum()),
                predicted_reason_rows=int(v.reason_source.fillna('').str.startswith('predicted').sum())))
    return pd.DataFrame(rows)


def run():
    labels=reason_labels();data=load_data()
    base=np.load(OUT/'linear_validation_predictions.npz')['lr_criteria_none_C2_phase0']
    baseline=errors_by_reason(data['val'],base,labels)
    baseline.to_csv(OUT/'baseline_reason_errors.csv',index=False)
    selected=np.load(OUT/'selected_validation_predictions.npz')
    performance=[]
    for name,p in [('baseline',base)]+[(k,selected[k]) for k in selected.files]:
        for phase in PHASES:
            m=metrics(data['val'],p)
            performance.append(dict(model=name,phase=phase,n=int(data['val'].phase.eq(phase).sum()),
                log_loss=m[phase+'_logloss'],f1=m[phase+'_f1'],roc_auc=m[phase+'_roc_auc'],pr_auc=m[phase+'_pr_auc']))
        table=errors_by_reason(data['val'],p,labels)
        table.to_csv(OUT/f'{name}_reason_errors.csv',index=False)
        if name=='03_phase_specialized':
            compare=baseline.merge(table,on=['phase','reason','n'],suffixes=('_baseline','_improved'))
            compare['false_success_rate_change']=compare.false_success_rate_improved-compare.false_success_rate_baseline
            compare.to_csv(OUT/'reason_error_comparison.csv',index=False)
    pd.DataFrame(performance).to_csv(OUT/'phase_metrics.csv',index=False)

if __name__=='__main__':run()
