"""Verify frozen sources, retained scores and the selected export without fitting."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
for version in ['blend_refinement_v1','blend_refinement_v2']:
    p=ROOT/version
    spec=importlib.util.spec_from_file_location(version,p/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.freeze()
    z=np.load(p/'evaluation_inputs.npz');repeats=pd.read_csv(p/'results/repeat_scores.csv');labels=pd.read_csv(p/'results/per_label_scores.csv')
    for name in m.RECIPES:
        s,t=m.scores(name,z['anchored'],z['cand2'],z['lda']);pred=m.decide(s,t)
        for row in repeats[repeats.recipe==name].itertuples():
            ix=(z['pool']==row.pool)&(z['seed']==row.seed);y=z['truth'][ix];yp=pred[ix]
            value=np.mean(2*np.sum(y*yp,axis=1)/(y.sum(1)+yp.sum(1)));assert abs(value-row.sample_f1)<1e-14
            tp=(y*yp).sum(0);fp=((1-y)*yp).sum(0);fn=(y*(1-yp)).sum(0)
            div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
            pr,re,f=div(tp,tp+fp),div(tp,tp+fn),div(2*tp,2*tp+fp+fn)
            assert abs(f.mean()-row.macro_f1)<1e-14
            got=labels[(labels.recipe==name)&(labels.pool==row.pool)&(labels.seed==row.seed)].set_index('label').loc[m.LABELS]
            np.testing.assert_allclose(got[['precision','recall','f1','support']],np.column_stack([pr,re,f,y.sum(0)]),atol=1e-14,rtol=0)
    if version.endswith('v2'):
        final=np.load(p/'final_inputs.npz');r=m.read(p/'prepared_submission.json');f=m.P1/r['csv'];assert m.sha(f)==r['sha256']
        a=np.clip(final['anchored'],1e-9,1-1e-9);c=np.clip(final['cand2'],1e-9,1-1e-9)
        odds=(a/(1-a))**.8*(c/(1-c))**.2;score=odds/(1+odds);pred=m.decide(score,.3)
        out=pd.read_csv(f);expected=m.frame(final['ids'],pred);pd.testing.assert_frame_equal(out,expected)
        parent=m.frame(final['ids'],m.decide(.8*final['anchored']+.2*final['cand2'],.3));pd.testing.assert_frame_equal(parent,pd.read_csv(p/'parent_scored.csv'))
        assert (out.Predicted!=parent.Predicted).sum()==5 and out.ID.is_unique and len(out)==312
print('Both fixed batches independently rescored; selected log-odds formula, exact parent CSV, export and hashes verified.')
