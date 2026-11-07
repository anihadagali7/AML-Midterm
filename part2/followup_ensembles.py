"""Two controlled prospective follow-ups after the first two Kaggle results.

Reuse the original fitted models without retraining or changing earlier CSVs.
Phase I stays exactly as in 02; change only tree fractions in Phases II/III.
"""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from experiments import ROOT,OUT,PHASES,load_data,split_data,metrics
from ensemble_search import collect,weighted
from submission_models import predict,reported_kaggle_score


def run():
    data=load_data();pred,tests,_=collect()
    original=joblib.load(OUT/'submission_models.joblib')
    prior_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'submissions').glob('0[1-4]_*.csv')}
    linear={'recency_clinical_C0.5_half3':.25,'recency_criteria_C2_half6':.375,'text_wordwide_C2':.375}
    tree={'catboost_d6_n1000_none':.5,'catboost_d8_n1000_none':.5}
    policies={'05_text_later_phases':(.25,0.,0.),'06_lighter_tree_blend':(.25,.125,.125)}
    ensembles={}
    for name,alphas in policies.items():
        ensembles[name]={k:{phase:(1-a)*linear.get(k,0)+a*tree.get(k,0) for phase,a in zip(PHASES,alphas)} for k in set(linear)|set(tree)}
    followup=dict(original,ensembles=ensembles)
    followup['components']={k:v for k,v in original['components'].items() if k in set(linear)|set(tree)}
    joblib.dump(followup,OUT/'followup_models.joblib',compress=3)
    reloaded=joblib.load(OUT/'followup_models.joblib')
    vp=predict(reloaded,data['val']);tp=predict(reloaded,data['test'])
    sample=pd.read_csv(ROOT/'part2_sample_submission.csv')
    previous=pd.read_csv(ROOT/'submissions/02_text_heavy.csv')
    assert previous.ID.equals(data['test'].ID)
    temporal=split_data(data,'temporal')[1]
    scores=[];details=[];hist={}
    for name,weights in ensembles.items():
        for split,df,new,bank in [('validation',data['val'],vp[name],pred['validation']),('test',data['test'],tp[name],tests['validation'])]:
            expected=weighted(bank,weights,df.phase)
            assert np.max(np.abs(new-expected))<1e-8
        p=tp[name]
        assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        assert np.max(np.abs(p[data['test'].phase.eq('I')]-previous.label[data['test'].phase.eq('I')]))<5.1e-11
        output=sample[['ID']].merge(pd.DataFrame({'ID':data['test'].ID,'label':p}),on='ID',validate='one_to_one')
        assert len(output)==1500 and output.ID.equals(sample.ID) and output.ID.is_unique
        path=ROOT/'submissions'/f'{name}.csv';output.to_csv(path,index=False,float_format='%.10f')
        disk=pd.read_csv(path);assert disk.columns.tolist()==['ID','label']
        assert disk.label.between(0,1,inclusive='neither').all() and np.max(abs(disk.label-p))<5.1e-11
        h=weighted(pred['temporal'],weights,temporal.phase);hist[name]=h
        a,b=metrics(data['val'],vp[name]),metrics(temporal,h)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        scores.append(dict(submission=path.name,validation_logloss=a['phase_mean_logloss'],temporal_logloss=b['phase_mean_logloss'],
            mean_test_probability=float(p.mean()),sha256=sha,kaggle_score=reported_kaggle_score(path.name,sha)))
        for phase in PHASES:details.append(dict(submission=path.name,phase=phase,tree_fraction=policies[name][PHASES.index(phase)],
            validation_logloss=a[phase+'_logloss'],temporal_logloss=b[phase+'_logloss']))
    for name,sha in prior_hashes.items():assert hashlib.sha256((ROOT/'submissions'/name).read_bytes()).hexdigest()==sha
    np.savez_compressed(OUT/'followup_validation_predictions.npz',**vp)
    np.savez_compressed(OUT/'followup_temporal_predictions.npz',**hist)
    table=pd.DataFrame(scores);table.to_csv(ROOT/'submissions/followup_index.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'followup_phase_metrics.csv',index=False)
    (ROOT/'submissions/followup_manifest.json').write_text(json.dumps({
        'motivation':'03 Kaggle 0.59562; 02 Kaggle 0.58993; reference 0.58688. Test less tree influence in later phases; no guarantee of improvement.',
        'training':'Reuse the original models fitted on outcome_train.csv only; no refitting or new label inputs.',
        'first_comparison':'05_text_later_phases.csv; reserve 06 until 05 is scored.',
        'ensembles':ensembles,'files':scores,'original_four_preserved':True},indent=2))
    print(table.to_string(index=False))

if __name__=='__main__':run()
