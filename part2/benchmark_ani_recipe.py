"""Sensitivity track: faithful Ani preprocessing on the SAME frozen splits.

Unlike the main strict benchmark, this deliberately preserves Ani's feature-only
transduction: train/validation/test text shapes vocabulary and IDF. No evaluation
outcome labels fit a model. Keep this track explicitly separate from strict scores.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import copy,json,time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ani_models import AniStack
from benchmark_models import OurLinear
from benchmark_protocol import initialize,resolve,digest
from local_benchmark import run_directory,predict_and_score,summarize
from experiments import ROOT


def run():
    universe,protocol=initialize();strict=run_directory(protocol)
    out=strict/'published_recipe_track';out.mkdir(exist_ok=True)
    space=joblib.load(ROOT/'experiments/ani_comparison/transductive_features.joblib')
    (out/'run_manifest.json').write_text(json.dumps({'protocol_sha256':digest(ROOT/'experiments/local_benchmark/protocol.json'),
        'source_commit':'1b8bd11c47428fb5453e53cdc275542b92a3b347',
        'preprocessing':'Faithful Ani: vocabulary and IDF include all available train/validation/test covariates; internal OOF uses that shared vocabulary.',
        'labels':'Evaluation labels excluded from every fit. This is a transductive sensitivity track, not strict prospective validation.'},indent=2))
    for split in protocol['splits']:
        if (out/f"scores_{split['name']}.csv").exists():continue
        start=time.time();train,ev=resolve(universe,split)
        folder=out/'models'/split['name'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/'ani_stack.joblib'
        if path.exists():model=joblib.load(path)
        else:
            model=AniStack(tau=3).fit(train,features=space);joblib.dump(model,path,compress=3)
        z=model.predict_base_matrix(ev)
        p=model.meta.predict_proba(model.meta_matrix(z,ev))[:,1]
        linear_path=strict/'models'/split['name']/'our_linear.joblib'
        if linear_path.exists():linear=joblib.load(linear_path)
        else:linear=OurLinear().fit(train)
        lp=linear.predict_proba(ev)
        trend=copy.copy(model);trend.phase3_year=True
        year=train.start_year.fillna(space.year_median).to_numpy();w=np.exp((year-year.max())/3);w=w/w.mean()
        trend.meta=LogisticRegression(C=1,max_iter=3000).fit(trend.meta_matrix(trend.oof_base_predictions,train),train.label,sample_weight=w)
        tp=trend.meta.predict_proba(trend.meta_matrix(z,ev))[:,1]
        predict_and_score(out,split,ev,{'ani_stack':p,'our_linear':lp,'ani_phase3_trend':tp,
            'blend_ani75_linear25':.75*p+.25*lp,'blend_ani50_linear50':.5*p+.5*lp})
        print(split['name'],'published-recipe sensitivity completed',round(time.time()-start,1),'seconds',flush=True)
        summarize(out,protocol,bootstrap=False)
    summarize(out,protocol)
    print('Published-recipe sensitivity results:',out,flush=True)

if __name__=='__main__':run()
