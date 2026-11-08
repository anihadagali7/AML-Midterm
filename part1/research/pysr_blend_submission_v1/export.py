"""Freeze and prepare the user-authorized 80/20 PySR trial on all 235 labels."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','JULIA_NUM_THREADS','JULIA_NUM_PRECOMPILE_TASKS']:os.environ[key]='1'
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'pysr_v1';P1=HERE.parents[1];ROOT=P1.parent
sys.path.insert(0,str(STUDY))
import study
from model import SymbolicReasons,centered,decide,FEATURE_NAMES
from common import LABELS,FEATURES,sha,save
PARENT=HERE.parent/'anchored_lda_blend_submission_v1'
CSV=P1/'submissions/ready/part1_current80_pysr20_trainval235.csv'


def sources():
    paths=[Path(__file__),HERE/'verify.py',STUDY/'protocol.json',STUDY/'outcome.json',STUDY/'retention.json',PARENT/'prediction_scores.npz',PARENT/'manifest.json',P1/'submissions/results/anchored75-lda25-trainval235.json',P1/'submissions/ready/part1_anchored75_lda25_trainval235.csv']
    return study.sources()|{str(p.relative_to(ROOT)):sha(p) for p in paths}


def guard():
    study.guard();assert sources()==json.loads((HERE/'protocol.json').read_text())['sources']


def register():
    assert not (HERE/'protocol.json').exists()
    save(HERE/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User requested trying the PySR blend on Kaggle despite its weaker full local comparison.',
        evaluation_source_commit='80f713fb',recipe='current80_pysr20',
        formula='0.8 * existing anchored75_lda25 centered score + 0.2 * center(PySR sigmoid score, native cutoff=.3). Decode at .5 plus nonempty argmax.',
        component_weights=dict(existing_89850_blend=.8,pysr=.2),effective_weights=dict(anchored=.60,lda_ensemble=.20,pysr=.20),
        fitting_population='All 1171 original training rows followed by78 validation rows:235 observed gold +1014 originally masked. Fit exactly the registered PySR model. Existing75/25 component uses its already verified full235 fits.',
        access='Inductive: actual test inputs used only after fitting. No labels from test. No threshold, equation-selection, seed, search-budget or weight changes.',
        policy_exception='User-authorized manual trial. Local both-pool rule failed and all four headline checks regressed. This export does not change that finding or promote a leader.',
        selection='Exactly one requested candidate; no further search based on test outputs. Retain full final model state, equations, generated labels and predictions.',
        status='Registered preparation, not an upload or reported score.',sources=sources()))


def prepare():
    guard();assert not CSV.exists() and not (HERE/'manifest.json').exists()
    train=pd.read_csv(P1/'data/part1_train.csv');val=pd.read_csv(P1/'data/part1_val.csv');fit=pd.concat([train,val],ignore_index=True)
    x=fit[FEATURES].to_numpy(float);y=fit[LABELS].to_numpy(int);known=np.isin(y,[0,1]).all(1)
    assert known.sum()==235 and (~known).sum()==1014 and fit.ID.is_unique
    print('Fitting unchanged PySR recipe:235 gold +1014 masked rows',flush=True);start=time.monotonic()
    with threadpool_limits(limits=1):model=SymbolicReasons().fit(x,y)
    # Actual test inputs first enter after model fitting is complete.
    test=pd.read_csv(P1/'data/part1_test.csv');sample=pd.read_csv(P1/'data/part1_sample_submission.csv')
    assert set(fit.ID).isdisjoint(test.ID);np.testing.assert_array_equal(test.ID,sample.ID)
    q=test[FEATURES].to_numpy(float)
    with threadpool_limits(limits=1):raw,_=model.predict_scores(q)
    np.testing.assert_allclose(model.predict_scores(q[:1])[0],raw[:1],atol=1e-12,rtol=0)
    with np.load(PARENT/'prediction_scores.npz') as z:
        np.testing.assert_array_equal(z['ids'],test.ID);parent=z['scores'];parent_pred=z['predictions']
    record=json.loads((P1/'submissions/results/anchored75-lda25-trainval235.json').read_text())
    assert sha(ROOT/record['submission_file'])==record['submission_sha256']
    encode=lambda p:[' '.join(label for label,v in zip(LABELS,row) if v) for row in p]
    assert pd.read_csv(ROOT/record['submission_file']).Predicted.tolist()==encode(parent_pred)
    np.testing.assert_array_equal(decide(parent,.5),parent_pred)
    scores=.8*parent+.2*centered(raw);pred=decide(scores,.5)
    frame=pd.DataFrame(dict(ID=test.ID,Predicted=encode(pred)));assert len(frame)==312 and frame.ID.is_unique and frame.Predicted.str.len().gt(0).all()
    frame.to_csv(CSV,index=False);pd.testing.assert_frame_equal(frame,pd.read_csv(CSV))
    np.savez_compressed(HERE/'prediction_scores.npz',ids=test.ID.to_numpy(str),labels=np.array(LABELS),parent_scores=parent,parent_predictions=parent_pred,pysr_raw=raw,pysr_centered=centered(raw),scores=scores,predictions=pred)
    g=model.geometry
    np.savez_compressed(HERE/'model_state.npz',fit_ids=fit.ID.to_numpy(str),gold_mask=known,embedding_mean=g.scaler.mean_,embedding_scale=g.scaler.scale_,
        cluster_centers=g.cluster.cluster_centers_,fit_cluster=g.cluster.labels_,bank=g.bank,bank_y=g.y,bank_groups=g.bank_groups,bank_cluster=g.bank_cluster,
        augmented_labels=g.augmented,sample_weights=g.weights,feature_means=np.stack([s.mean_ for s in model.scales]),feature_scales=np.stack([s.scale_ for s in model.scales]))
    save(HERE/'equations.json',dict(feature_names=FEATURE_NAMES,equations=model.equations,model_audit=model.audit))
    augmented=fit[['ID']].copy();augmented['source']=np.where(known,'observed','kmeans_generated')
    for j,label in enumerate(LABELS):augmented[label]=g.augmented[:,j]
    augmented.to_csv(HERE/'augmented_labels.csv',index=False)
    changed=(pred!=parent_pred).any(1)
    pd.DataFrame(dict(ID=test.ID[changed],previous=np.array(encode(parent_pred))[changed],candidate=np.array(encode(pred))[changed])).to_csv(HERE/'changes_vs_89850.csv',index=False)
    save(HERE/'manifest.json',dict(status='Prepared for user-requested manual Kaggle trial; not uploaded',recipe='current80_pysr20_trainval235',created_utc=datetime.now(timezone.utc).isoformat(),
        submission_file=str(CSV.relative_to(ROOT)),submission_sha256=sha(CSV),rows=312,verified_training_labels=235,originally_masked_training_rows=1014,
        new_pysr_full235_fits=1,new_symbolic_heads=5,existing_parent_fits_reused=True,test_inputs_used_for_fit=False,observed_labels_preserved=True,query_batch_invariance_passed=True,
        weights=dict(existing_89850_blend=.8,pysr=.2),changed_labelsets_vs_89850=int(changed.sum()),minimum_margin_from_cutoff=float(abs(scores-.5).min()),
        upload_performed=False,reported_kaggle_score=None,preparation_seconds=time.monotonic()-start,protocol_sha256=sha(HERE/'protocol.json'),
        artifacts={p.name:sha(p) for p in [HERE/'prediction_scores.npz',HERE/'model_state.npz',HERE/'equations.json',HERE/'augmented_labels.csv',HERE/'changes_vs_89850.csv']}))
    guard();print(json.dumps(json.loads((HERE/'manifest.json').read_text()),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','prepare']);a=p.parse_args()
    register() if a.command=='register' else prepare()
