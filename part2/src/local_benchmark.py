"""Run the frozen, grouped and chronological model-comparison protocol.

python local_benchmark.py --suite full
python local_benchmark.py --suite chronological
python local_benchmark.py --summarize

Results are cached under a code/version fingerprint. No Kaggle files are
submitted. This runner never fits on an evaluation trial or its text.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,copy,hashlib,importlib.metadata,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from ani_models import AniStack
from benchmark_models import OurLinear
from benchmark_protocol import DIRECTORY,initialize,resolve,digest
from experiments import ROOT,PHASES,metrics

MODELS=['ani_stack','ani_meta_C0.3','ani_phase3_trend','our_linear','blend_ani75_linear25','blend_ani50_linear50']


def run_directory(protocol):
    code={f:digest(ROOT/f) for f in ['ani_models.py','benchmark_models.py','benchmark_protocol.py','experiments.py','local_benchmark.py']}
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib']}
    manifest={'protocol_sha256':digest(DIRECTORY/'protocol.json'),'source_sha256':code,'versions':versions,
        'models':MODELS,'baseline':'ani_stack','ani_stack':{'tau':3,'meta_C':1,'inner_preprocessing':'inner-fold training only'},
        'model_selection':'Equal-phase log loss; prioritize mean and worst chronological windows, corroborate with repeated group CV; inspect paired uncertainty, not just validation F1.'}
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    out=DIRECTORY/'runs'/key;out.mkdir(parents=True,exist_ok=True)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    (DIRECTORY/'current_run.json').write_text(json.dumps({'run':key,'directory':str(out.relative_to(ROOT))},indent=2))
    return out


def predict_and_score(out,split,df,probabilities):
    folder=out/'predictions'/split['name'];folder.mkdir(parents=True,exist_ok=True)
    scores=[]
    for name,p in probabilities.items():
        assert len(p)==len(df) and np.isfinite(p).all() and ((p>0)&(p<1)).all()
        frame=df[['ID','trial_id','phase','start_year','label']].copy();frame['probability']=p
        frame.to_csv(folder/f'{name}.csv',index=False,float_format='%.15g')
        row={'model':name,'split':split['name'],'kind':split['kind'],'repeat':split['repeat'],**metrics(df,p)}
        row['brier']=brier_score_loss(df.label,p)
        for phase in PHASES:
            m=df.phase.eq(phase);row[phase+'_brier']=brier_score_loss(df.label[m],p[m])
            row[phase+'_n']=int(m.sum());row[phase+'_predicted_mean']=float(p[m].mean());row[phase+'_observed_mean']=float(df.label[m].mean())
        scores.append(row)
    pd.DataFrame(scores).to_csv(out/f"scores_{split['name']}.csv",index=False)


def fit_split(out,split,train,ev):
    folder=out/'models'/split['name'];folder.mkdir(parents=True,exist_ok=True)
    start=time.time();fitted={}
    for name,factory in [('ani_stack',lambda:AniStack(tau=3,inner_refit=True)),('our_linear',OurLinear)]:
        path=folder/f'{name}.joblib'
        if path.exists():model=joblib.load(path)
        else:
            model=factory().fit(train)
            assert model.training_ids==train.ID.tolist()
            joblib.dump(model,path,compress=3)
        fitted[name]=model
    baseline=fitted['ani_stack']
    assert set(baseline.features.fit_ids)==set(train.ID)
    assert not set(baseline.features.fit_ids)&set(ev.ID)
    assert all(x['trial_overlap']==0 for x in baseline.fold_audit)
    z=baseline.predict_base_matrix(ev)
    probabilities={'ani_stack':baseline.meta.predict_proba(baseline.meta_matrix(z,ev))[:,1],
                   'our_linear':fitted['our_linear'].predict_proba(ev)}
    # These two variants change only the meta-model; all base fits remain identical.
    for name,C,trend in [('ani_meta_C0.3',.3,False),('ani_phase3_trend',1.,True)]:
        candidate=copy.copy(baseline);candidate.meta_C=C;candidate.phase3_year=trend
        year=train.start_year.fillna(candidate.features.year_median).to_numpy()
        w=np.exp((year-year.max())/3);w=w/w.mean()
        candidate.meta=LogisticRegression(C=C,max_iter=3000).fit(candidate.meta_matrix(candidate.oof_base_predictions,train),train.label,sample_weight=w)
        probabilities[name]=candidate.meta.predict_proba(candidate.meta_matrix(z,ev))[:,1]
        joblib.dump(candidate,folder/f'{name}.joblib',compress=3)
    probabilities['blend_ani75_linear25']=.75*probabilities['ani_stack']+.25*probabilities['our_linear']
    probabilities['blend_ani50_linear50']=.5*probabilities['ani_stack']+.5*probabilities['our_linear']
    if split['kind']=='official':
        changed=ev.copy();changed['ID']='unused';changed['trial_id']='unused';changed['label']=1-ev.label
        for name in ['ani_stack','our_linear']:
            assert np.allclose(fitted[name].predict_proba(changed),probabilities[name],atol=1e-12,rtol=0)
    predict_and_score(out,split,ev,probabilities)
    audit={'split':split['name'],'fit_rows':len(train),'eval_rows':len(ev),'trial_overlap':0,
        'fit_ID_sha256':hashlib.sha256('\n'.join(train.ID).encode()).hexdigest(),
        'eval_ID_sha256':hashlib.sha256('\n'.join(ev.ID).encode()).hexdigest(),
        'preprocessing_population':'training rows only, both outer and stacking-inner folds',
        'inner_folds':baseline.fold_audit,'seconds':time.time()-start}
    (folder/'audit.json').write_text(json.dumps(audit,indent=2))
    print(split['name'],'completed',round(time.time()-start,1),'seconds',
          {k:round(metrics(ev,p)['phase_mean_logloss'],5) for k,p in probabilities.items()},flush=True)


def loss_rows(df):
    p=np.clip(df.probability.to_numpy(),1e-6,1-1e-6);y=df.label.to_numpy()
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def paired_bootstrap(out,protocol,model,kind,repetitions=1200):
    frames=[]
    for split in protocol['splits']:
        if split['kind']!=kind:continue
        folder=out/'predictions'/split['name']
        if not (folder/f'{model}.csv').exists() or not (folder/'ani_stack.csv').exists():continue
        a,b=pd.read_csv(folder/f'{model}.csv'),pd.read_csv(folder/'ani_stack.csv')
        assert a.ID.equals(b.ID)
        a['delta']=loss_rows(a)-loss_rows(b)
        a['period']=split['name'] if kind=='temporal' else 'group_cv'
        frames.append(a[['ID','trial_id','phase','period','delta']])
    if not frames:return {}
    d=pd.concat(frames,ignore_index=True)
    # Each trial's repeated-CV losses are averaged before resampling.
    d=d.groupby(['ID','trial_id','phase','period'],as_index=False).delta.mean()
    trials,index=np.unique(d.trial_id,return_inverse=True)
    rng=np.random.default_rng(42);weights=rng.poisson(1,size=(repetitions,len(trials))).astype(np.float32)[:,index]
    boot=[];points=[]
    for (_,phase),group in d.groupby(['period','phase']):
        ix=group.index.to_numpy();points.append(group.delta.mean())
        boot.append((weights[:,ix]@group.delta.to_numpy())/weights[:,ix].sum(axis=1))
    values=np.mean(boot,axis=0)
    return {kind+'_delta':float(np.mean(points)),kind+'_delta_lower':float(np.quantile(values,.025)),
            kind+'_delta_upper':float(np.quantile(values,.975))}


def summarize(out,protocol,bootstrap=True):
    files=list(out.glob('scores_*.csv'))
    if not files:return
    scores=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    scores.to_csv(out/'fold_scores.csv',index=False)
    rows=[]
    expected_cv=sum(s['kind']=='group_cv' for s in protocol['splits'])
    for model,g in scores.groupby('model'):
        row={'model':model,'completed_splits':len(g),'expected_splits':len(protocol['splits'])}
        official=g[g.kind.eq('official')]
        row['official_logloss']=float(official.phase_mean_logloss.iloc[0]) if len(official) else np.nan
        times=g[g.kind.eq('temporal')]
        row['temporal_windows']=len(times);row['temporal_mean_logloss']=times.phase_mean_logloss.mean();row['worst_temporal_logloss']=times.phase_mean_logloss.max()
        repeat_scores=[]
        for repeat in [0,1]:
            splits=[s for s in protocol['splits'] if s['kind']=='group_cv' and s['repeat']==repeat]
            paths=[out/'predictions'/s['name']/f'{model}.csv' for s in splits]
            if all(p.exists() for p in paths):
                d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
                assert d.ID.is_unique
                repeat_scores.append(metrics(d,d.probability.to_numpy())['phase_mean_logloss'])
        row['cv_completed_repeats']=len(repeat_scores)
        row['cv_mean_logloss']=float(np.mean(repeat_scores)) if repeat_scores else np.nan
        row['cv_repeat_std']=float(np.std(repeat_scores)) if len(repeat_scores)>1 else np.nan
        if bootstrap and model!='ani_stack':
            if len(times)==3:row.update(paired_bootstrap(out,protocol,model,'temporal'))
            if len(g[g.kind.eq('group_cv')])==expected_cv:row.update(paired_bootstrap(out,protocol,model,'group_cv'))
        rows.append(row)
    summary=pd.DataFrame(rows).sort_values('temporal_mean_logloss')
    baseline=summary[summary.model.eq('ani_stack')].iloc[0]
    summary['temporal_mean_delta_vs_ani']=summary.temporal_mean_logloss-baseline.temporal_mean_logloss
    summary['cv_mean_delta_vs_ani']=summary.cv_mean_logloss-baseline.cv_mean_logloss
    # A model cannot gain a complete-comparison status from a favorable subset.
    summary['complete_comparison']=summary.completed_splits.eq(summary.expected_splits)
    summary.to_csv(out/'summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['full','chronological','official'],default='full')
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--force',action='store_true',help='Recompute predictions using cached fitted models')
    args=parser.parse_args()
    universe,protocol=initialize();out=run_directory(protocol)
    if not args.summarize:
        for split in protocol['splits']:
            if args.suite=='chronological' and split['kind']=='group_cv':continue
            if args.suite=='official' and split['kind']!='official':continue
            score=out/f"scores_{split['name']}.csv"
            if score.exists() and not args.force:
                saved=pd.read_csv(score)
                if set(saved.model)==set(MODELS):continue
            train,ev=resolve(universe,split);fit_split(out,split,train,ev)
            summarize(out,protocol,bootstrap=False)
    summarize(out,protocol)
    print('Results:',out,flush=True)

if __name__=='__main__':main()
