"""Run every new candidate on both saved label pools, against pinned current Ani."""
import argparse
import json
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from simple_benchmark import ROOT,BASE,digest,metrics
from simple_models import FEATURES,REASONS,decide
from paired_models import PairedModel,prepare,first_specs

OUT=ROOT/'experiments/paired_improvement'
REFERENCE=ROOT/'experiments/current_ani_pool_audit'


def pools():
    train=pd.read_csv(ROOT/'data/part1_train.csv');valid=pd.read_csv(ROOT/'data/part1_val.csv')
    unknown=np.flatnonzero(train[REASONS].lt(0).all(axis=1).to_numpy())
    return [('train157',train,ROOT/'experiments/classifier_cross_validation/splits',5,3,unknown),
            ('trainval235',pd.concat([train,valid],ignore_index=True),BASE/'splits',8,5,unknown)]


def summarize(stage,df):
    reasons=REASONS
    fold_rows=[];repeat_rows=[]
    for (pool,name,repeat,fold),g in df.groupby(['pool','name','repeat','fold'],sort=True):
        fold_rows.append(dict(pool=pool,name=name,repeat=repeat,fold=fold,**metrics(g[[f'actual_{r}' for r in reasons]].to_numpy(),g[[f'predicted_{r}' for r in reasons]].to_numpy())))
    for (pool,name,repeat),g in df.groupby(['pool','name','repeat'],sort=True):
        assert len(g)==g.ID.nunique()==(157 if pool=='train157' else 235)
        repeat_rows.append(dict(pool=pool,name=name,repeat=repeat,**metrics(g[[f'actual_{r}' for r in reasons]].to_numpy(),g[[f'predicted_{r}' for r in reasons]].to_numpy())))
    folds=pd.DataFrame(fold_rows);repeats=pd.DataFrame(repeat_rows)
    summary=repeats.groupby(['pool','name']).agg(mean_f1=('sample_f1','mean'),repeat_sd=('sample_f1','std'),macro_f1=('macro_f1','mean'),**{f'f1_{r}':(f'f1_{r}','mean') for r in reasons})
    for pool in ['train157','trainval235']:
        ref=repeats[(repeats.pool==pool)&(repeats.name=='ani_current')].sample_f1.to_numpy()
        ref_fold=folds[(folds.pool==pool)&(folds.name=='ani_current')].sample_f1.to_numpy()
        for name in summary.loc[pool].index:
            diff=repeats[(repeats.pool==pool)&(repeats.name==name)].sample_f1.to_numpy()-ref
            fd=folds[(folds.pool==pool)&(folds.name==name)].sample_f1.to_numpy()-ref_fold
            summary.loc[(pool,name),'delta_ani']=diff.mean()
            summary.loc[(pool,name),'repeat_wins']=sum(diff>1e-12);summary.loc[(pool,name),'repeat_ties']=sum(abs(diff)<1e-12)
            summary.loc[(pool,name),'fold_wins']=sum(fd>1e-12);summary.loc[(pool,name),'fold_ties']=sum(abs(fd)<1e-12)
    summary.to_csv(stage/'summary.csv');repeats.to_csv(stage/'repeat_scores.csv',index=False);folds.to_csv(stage/'fold_scores.csv',index=False)
    combined=summary.reset_index().pivot(index='name',columns='pool',values=['mean_f1','delta_ani','macro_f1'])
    combined.columns=['_'.join(c) for c in combined.columns]
    combined['minimum_delta']=combined[['delta_ani_train157','delta_ani_trainval235']].min(axis=1)
    combined.sort_values('minimum_delta',ascending=False).to_csv(stage/'both_pools.csv')
    print(combined.sort_values('minimum_delta',ascending=False).to_string(),flush=True)
    return summary


def run(stage_name,specs):
    stage=OUT/stage_name;stage.mkdir(parents=True,exist_ok=True)
    baseline_protocol=json.loads((REFERENCE/'protocol.json').read_text())
    baseline=pd.read_csv(REFERENCE/'predictions.csv')
    baseline=baseline[baseline.name.isin(['ani_current','trusted_two_pass'])].copy()
    protocol=dict(stage=stage_name,specs=specs,ani_commit=baseline_protocol['ani_commit'],pools=baseline_protocol['pools'],
                  features='150 supplied reason embeddings only',feature_policy=baseline_protocol['policy'],
                  selection='Prefer improvements on both pools; neither may be silently dropped. No test labels or inputs in fitting.',
                  reference_sha256=digest(REFERENCE/'predictions.csv'),input_sha256={p.name:digest(p) for p in [ROOT/'data/part1_train.csv',ROOT/'data/part1_val.csv']})
    if (stage/'protocol.json').exists():assert json.loads((stage/'protocol.json').read_text())==protocol
    else:(stage/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for filename in ['paired_models.py','paired_search.py']:
        source=Path(__file__).resolve().parent/filename
        if (stage/filename).exists():
            assert digest(source)==digest(stage/filename), f'Source changed: use a new stage or execute its frozen sources: {filename}'
        else:shutil.copyfile(source,stage/filename)
    outputs=[];audits=[];controls=[];start=time.time()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for pool,frame,splitdir,nrep,nfold,unknown in pools():
            verified=np.flatnonzero(frame[REASONS].ge(0).all(axis=1).to_numpy())
            X=frame[FEATURES].to_numpy();Y=frame[REASONS].to_numpy(int)
            for repeat in range(nrep):
                assignment=pd.read_csv(splitdir/f'repeat{repeat}.csv').set_index('ID').loc[frame.iloc[verified].ID,'fold'].to_numpy()
                assert digest(splitdir/f'repeat{repeat}.csv')==protocol['pools'][pool]['split_sha256'][f'repeat{repeat}.csv']
                for fold in range(nfold):
                    fit=verified[assignment!=fold];held=verified[assignment==fold];feature_rows=np.sort(np.r_[fit,unknown])
                    assert not np.intersect1d(feature_rows,held).size
                    prep=prepare(X[feature_rows],any(s.get('gmm_weight',0)>0 for s in specs))
                    for s in specs:
                        model=PairedModel(s).fit(X[fit],Y[fit],X[unknown],prep)
                        P=model.predict_proba(X[held]);pred=decide(P,s.get('cutoff',.3))
                        assert np.isfinite(P).all() and pred.sum(axis=1).min()>0
                        if s['name'] in ['ani_exact_control','trusted_exact_control']:
                            name='ani_current' if s['name']=='ani_exact_control' else 'trusted_two_pass'
                            old=baseline[(baseline.pool==pool)&(baseline.repeat==repeat)&(baseline.fold==fold)&(baseline.name==name)].set_index('ID').loc[frame.iloc[held].ID]
                            old_p=old[[f'probability_{r}' for r in REASONS]].to_numpy()
                            # Equivalent floating-point operation/scaler row order can
                            # perturb optimizer convergence. Require identical labels,
                            # and record the maximum score error, bounded to 1e-6.
                            np.testing.assert_allclose(P,old_p,atol=1e-6,rtol=0,err_msg=f'{pool}/{repeat}/{fold}/{name}')
                            np.testing.assert_array_equal(pred,old[[f'predicted_{r}' for r in REASONS]])
                            controls.append(dict(pool=pool,repeat=repeat,fold=fold,name=name,max_abs_score_error=float(abs(P-old_p).max()),identical_predictions=True))
                        out=pd.DataFrame(dict(ID=frame.iloc[held].ID,pool=pool,name=s['name'],repeat=repeat,fold=fold))
                        for j,r in enumerate(REASONS):out[f'actual_{r}']=Y[held,j];out[f'predicted_{r}']=pred[:,j];out[f'probability_{r}']=P[:,j]
                        outputs.append(out)
                        audits.append(dict(pool=pool,repeat=repeat,fold=fold,name=s['name'],fit_ids=frame.iloc[fit].ID.tolist(),held_ids=frame.iloc[held].ID.tolist(),unlabeled_count=len(unknown),feature_count=len(feature_rows),fit_audit=model.audit_))
                print(stage_name,pool,f'repeat {repeat+1}/{nrep}',round(time.time()-start,1),'s',flush=True)
    df=pd.concat([baseline,*outputs],ignore_index=True)
    df.to_csv(stage/'predictions.csv',index=False)
    (stage/'fit_audits.json').write_text(json.dumps(audits,separators=(',',':'))+'\n')
    (stage/'control_checks.json').write_text(json.dumps(controls,indent=2)+'\n')
    summarize(stage,pd.read_csv(stage/'predictions.csv'))
    return stage


def diagnostics(stage_name):
    stage=OUT/stage_name;specs=json.loads((stage/'protocol.json').read_text())['specs']
    _,frame,_,_,_,unknown=pools()[1]
    verified=np.flatnonzero(frame[REASONS].ge(0).all(axis=1).to_numpy());X=frame[FEATURES].to_numpy();Y=frame[REASONS].to_numpy(int)
    rows=[];outputs=[]
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for label in ['original_validation','after_2003','after_2005','after_2007']:
            if label=='original_validation':fit=verified[:157];held=verified[157:];u=unknown
            else:
                year=int(label.split('_')[-1]);early=frame.start_year.to_numpy()<=year
                fit=verified[early[verified]];held=verified[~early[verified]];u=unknown[early[unknown]]
            prep=prepare(X[np.sort(np.r_[fit,u])],any(s.get('gmm_weight',0)>0 for s in specs))
            for s in specs:
                model=PairedModel(s).fit(X[fit],Y[fit],X[u],prep);P=model.predict_proba(X[held]);pred=decide(P,s.get('cutoff',.3))
                rows.append(dict(split=label,name=s['name'],fit_rows=len(fit),held_rows=len(held),unlabeled_rows=len(u),**metrics(Y[held],pred)))
                out=pd.DataFrame(dict(ID=frame.iloc[held].ID,split=label,name=s['name']))
                for j,r in enumerate(REASONS):out[f'actual_{r}']=Y[held,j];out[f'predicted_{r}']=pred[:,j];out[f'probability_{r}']=P[:,j]
                outputs.append(out)
            print(label,{r['name']:round(r['sample_f1'],6) for r in rows if r['split']==label},flush=True)
    ref=pd.read_csv(REFERENCE/'current_diagnostic_scores.csv');ref=ref[ref.name.isin(['ani_current','trusted_two_pass'])]
    pd.concat([ref,pd.DataFrame(rows)]).to_csv(stage/'diagnostic_scores.csv',index=False)
    pd.concat(outputs).to_csv(stage/'diagnostic_predictions.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--specs',type=Path);p.add_argument('--diagnostics',action='store_true')
    a=p.parse_args();run(a.stage,json.loads(a.specs.read_text()) if a.specs else first_specs())
    if a.diagnostics:diagnostics(a.stage)
