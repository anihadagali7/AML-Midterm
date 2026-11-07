"""Refit, serialize, independently replay, and validate the four submissions."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from experiments import ROOT, OUT, PHASES, load_data, prepare, matrix, fit_linear, predict_linear, metrics
from boosted import transform_dense
from ensemble_search import collect, weighted
from refinements import interactions

SUBMISSIONS=ROOT/'submissions'


def reported_kaggle_score(filename,sha256):
    path=SUBMISSIONS/'leaderboard_results.json'
    if not path.exists():return None
    record=json.loads(path.read_text()).get('submissions',{}).get(filename,{})
    return record.get('kaggle_score') if record.get('sha256')==sha256 else None


def linear_inputs(component, df, blocks):
    c=component['config'];family=c['family']
    if family in ['lr','recency','balanced','interactions']:
        x=matrix(blocks,c['representation'],c.get('year','none'))
        if family=='interactions':x=interactions(x,df,c['weight'])
        return x
    if family=='text':
        t=component['vectorizer'].transform(df.criteria.fillna('').str.replace(r'\\[nr]',' ',regex=True))
        if c['representation']=='charcriteria':
            return sparse.hstack([blocks['criteria'],t*.7,blocks['phase']],format='csr')
        return sparse.hstack([t,blocks['phase']],format='csr')
    raise ValueError(f'Unsupported linear configuration {c}')


def predict_components(bundle, df):
    blocks=bundle['features'].transform(df)
    dense=None;result={}
    for name,c in bundle['components'].items():
        cfg=c['config'];family=cfg['family']
        if family in ['catboost','robust_catboost','histgb','lightgbm','extratrees']:
            if family=='lightgbm':
                x=matrix(blocks,'knowledge',cfg['year'])
                p=c['model'].predict_proba(x,num_iteration=cfg['iterations'])[:,1]
            else:
                if dense is None:
                    dense=transform_dense(df,blocks,bundle['dense_svds'],bundle['dense_mask'],bundle['features'].year_median)
                x=dense.drop(columns='year') if cfg.get('year','none')=='none' else dense
                if family in ['histgb','extratrees']:
                    x=x.drop(columns=x.select_dtypes('str').columns)
                    p=c['model'].predict_proba(x)[:,1]
                else:p=c['model'].predict_proba(x,ntree_end=cfg['iterations'])[:,1]
        else:
            x=linear_inputs(c,df,blocks)
            if family=='lr':
                a=cfg['phase_weight']
                p=(1-a)*predict_linear(c['pooled'],x,df)
                if a:p+=a*predict_linear(c['separate'],x,df)
            else:p=c['model'].predict_proba(x)[:,1]
        result[name]=p
    return result


def predict(bundle,df):
    components=predict_components(bundle,df)
    return {name:weighted(components,w,df.phase) for name,w in bundle['ensembles'].items()}


def fit_components(ensembles,configs):
    data=load_data();tr,ev,f,bt,be,bx=prepare(data,'validation')
    names=sorted({k for w in ensembles.values() for k in w})
    bundle={'features':f,'components':{},'ensembles':ensembles,'training_rows':len(tr),
            'training_trial_ids':sorted(tr.trial_id.unique().tolist())}
    dense_path=OUT/'dense_validation.joblib'
    if dense_path.exists():
        frames,svds,mask=joblib.load(dense_path)
        bundle.update(dense_svds=svds,dense_mask=mask)
    for name in names:
        cfg=configs[name];family=cfg['family'];component={'config':cfg}
        if family in ['catboost','robust_catboost','histgb','lightgbm','extratrees']:
            if family=='catboost':path=OUT/f"catboost_validation_d{cfg['depth']}_{cfg['year']}.joblib"
            elif family=='robust_catboost':path=OUT/f"robust_catboost_{cfg['variant']}_validation.joblib"
            elif family=='histgb':path=OUT/f"histgb_leaves{cfg['leaves']}_{cfg['year']}_validation.joblib"
            elif family=='lightgbm':path=OUT/f"lightgbm_validation_leaves{cfg['leaves']}_{cfg['year']}.joblib"
            else:path=OUT/f"extratrees_validation_leaf{cfg['leaf']}_{cfg['year']}.joblib"
            if family=='extratrees' and not path.exists():
                x=frames[0]
                if cfg['year']=='none':x=x.drop(columns='year')
                x=x.drop(columns=x.select_dtypes('str').columns)
                model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=cfg['leaf'],max_features=.5,n_jobs=4,random_state=42)
                model.fit(x,tr.label);joblib.dump(model,path,compress=3)
            component['model']=joblib.load(path)
        else:
            if family=='text':component['vectorizer']=joblib.load(OUT/f"text_{cfg['representation']}_validation.joblib")
            x=linear_inputs(component,tr,bt)
            if family=='lr':
                component['pooled']=fit_linear(x,tr,cfg['C'],False)
                if cfg['phase_weight']:component['separate']=fit_linear(x,tr,cfg['C'],True)
            else:
                w=None
                if family=='recency':
                    w=2**((tr.start_year.fillna(tr.start_year.median()).to_numpy()-tr.start_year.max())/cfg['half_life'])
                    w=w/w.mean()
                elif family=='balanced':w=tr.phase.map({p:len(tr)/(3*sum(tr.phase==p)) for p in PHASES}).to_numpy()
                m=LogisticRegression(C=cfg['C'],solver='liblinear',max_iter=1000,tol=1e-5,random_state=42)
                m.fit(x,tr.label,sample_weight=w);component['model']=m
        bundle['components'][name]=component
        print('prepared',name,flush=True)
    return bundle


def bootstrap_delta(df, candidate, baseline, seed=42, repetitions=2000):
    # Shared trial weights keep combined-phase rows together in every resample.
    ids,index=np.unique(df.trial_id.to_numpy(),return_inverse=True)
    rng=np.random.default_rng(seed)
    w=rng.poisson(1,size=(repetitions,len(ids)))[:,index]
    y=df.label.to_numpy()
    def loss(p):
        p=np.clip(p,1e-6,1-1e-6)
        return -(y*np.log(p)+(1-y)*np.log1p(-p))
    delta=loss(candidate)-loss(baseline)
    estimates=[]
    for phase in PHASES:
        m=df.phase.eq(phase).to_numpy()
        estimates.append((w[:,m]@delta[m])/w[:,m].sum(axis=1))
    d=np.mean(estimates,axis=0)
    return dict(delta_mean=float(d.mean()),delta_lower=float(np.quantile(d,.025)),delta_upper=float(np.quantile(d,.975)),fraction_lower_loss=float((d<0).mean()))


def build_and_verify():
    SUBMISSIONS.mkdir(exist_ok=True)
    selected=json.loads((OUT/'selected_ensembles.json').read_text())
    pred,tests,configs=collect();data=load_data()
    ensemble_defs=json.loads((OUT/'ensemble_configs.json').read_text())
    weights={name:ensemble_defs[source] for name,source in selected.items()}
    bundle=fit_components(weights,configs)
    val_components=predict_components(bundle,data['val'])
    test_components=predict_components(bundle,data['test'])
    verification=[]
    for name in bundle['components']:
        for split,new,old in [('validation',val_components[name],pred['validation'][name]),('test',test_components[name],tests['validation'][name])]:
            error=float(np.max(np.abs(new-old)))
            assert error<1e-8,(name,split,error)
            verification.append(dict(component=name,split=split,max_probability_difference=error))
    joblib.dump(bundle,OUT/'submission_models.joblib',compress=3)
    reloaded=joblib.load(OUT/'submission_models.joblib')
    vp=predict(reloaded,data['val']);tp=predict(reloaded,data['test'])
    sample=pd.read_csv(ROOT/'part2_sample_submission.csv')
    _,temporal,_,_,_,_=prepare(data,'temporal')
    manifest=[];validation={};historical={};bootstrap=[]
    baseline=pred['validation']['lr_criteria_none_C2_phase0']
    for name,source in selected.items():
        w=weights[name];p=vp[name];pt=tp[name]
        assert np.allclose(p,weighted(val_components,w,data['val'].phase),atol=1e-12,rtol=0)
        expected=weighted(tests['validation'],w,data['test'].phase)
        assert np.allclose(pt,expected,atol=1e-8,rtol=0)
        ph=weighted(pred['temporal'],w,temporal.phase)
        validation[name]=p;historical[name]=ph
        output=sample[['ID']].merge(pd.DataFrame({'ID':data['test'].ID,'label':pt}),on='ID',validate='one_to_one')
        assert len(output)==1500 and output.ID.equals(sample.ID)
        assert output.label.between(0,1,inclusive='neither').all()
        path=SUBMISSIONS/f'{name}.csv'
        output.to_csv(path,index=False,float_format='%.10f')
        reread=pd.read_csv(path)
        assert reread.columns.tolist()==['ID','label'] and not reread.isna().any().any()
        assert np.max(abs(reread.label-pt))<5.1e-11
        mv=metrics(data['val'],p);mt=metrics(temporal,ph)
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(dict(submission=path.name,ensemble=source,
            validation_logloss=mv['phase_mean_logloss'],temporal_logloss=mt['phase_mean_logloss'],
            joint_logloss=.5*(mv['phase_mean_logloss']+mt['phase_mean_logloss']),
            mean_test_probability=float(pt.mean()),sha256=sha256,kaggle_score=reported_kaggle_score(path.name,sha256)))
        bootstrap.append(dict(model=name,**bootstrap_delta(data['val'],p,baseline)))
    np.savez_compressed(OUT/'selected_validation_predictions.npz',**validation)
    np.savez_compressed(OUT/'selected_temporal_predictions.npz',**historical)
    pd.DataFrame(manifest).to_csv(SUBMISSIONS/'index.csv',index=False)
    (SUBMISSIONS/'manifest.json').write_text(json.dumps({'training':'outcome_train.csv only; validation never fitted',
        'selection':'Equal-weight average of official-validation and chronological-holdout phase-mean log loss',
        'ensembles':weights,'files':manifest},indent=2))
    pd.DataFrame(verification).to_csv(OUT/'model_replay_verification.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(OUT/'paired_validation_bootstrap.csv',index=False)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','rdkit','joblib','matplotlib','nbformat','nbclient','ipykernel']}
    (OUT/'versions.json').write_text(json.dumps(versions,indent=2))
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.input:
        bundle=joblib.load(OUT/'submission_models.joblib')
        df=pd.read_csv(args.input)
        destination=args.output or ROOT/'replayed_predictions'
        destination.mkdir(parents=True,exist_ok=True)
        for name,p in predict(bundle,df).items():
            pd.DataFrame({'ID':df.ID,'label':p}).to_csv(destination/f'{name}.csv',index=False,float_format='%.10f')
    else:build_and_verify()
