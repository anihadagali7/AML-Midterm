"""Replay the selected compact implementation on both pools, then package locally."""
from datetime import datetime,timezone
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from anchored_solution import AnchoredReasonClassifier
from simple_benchmark import ROOT,BASE,digest,metrics
from simple_models import FEATURES,REASONS
from paired_search import OUT,REFERENCE,pools


def main():
    stage=OUT/'06_final_ablations';name='selected_control'
    candidate=OUT/'candidate';candidate.mkdir(exist_ok=True)
    protocol=json.loads((stage/'protocol.json').read_text())
    spec=next(s for s in protocol['specs'] if s['name']==name)
    selected=dict(stage=stage.name,name=name,spec=spec,ani_commit=protocol['ani_commit'],
                  reason='Largest minimum improvement across the two required CV pools; macro F1 also higher on both. All four additional sample-F1 checks improve. Retains Ani C, blend, rounds and decision cutoff.',
                  caveat='Development-fold selection; neither CV pool nor the additional diagnostics is untouched. Kaggle score unknown.',
                  feature_policy=protocol['feature_policy'],test_features_used_for_fit=False)
    (OUT/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
    expected=pd.read_csv(stage/'predictions.csv');expected=expected[expected.name==name]
    diag=pd.read_csv(stage/'diagnostic_predictions.csv');diag=diag[diag.name==name]
    checks=[];diagnostic_checks=[];rescored=[]
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for pool,frame,splitdir,nrep,nfold,unknown in pools():
            known=np.flatnonzero(frame[REASONS].ge(0).all(axis=1).to_numpy())
            for repeat in range(nrep):
                assignment=pd.read_csv(splitdir/f'repeat{repeat}.csv').set_index('ID').loc[frame.iloc[known].ID,'fold'].to_numpy()
                assert digest(splitdir/f'repeat{repeat}.csv')==protocol['pools'][pool]['split_sha256'][f'repeat{repeat}.csv']
                repeat_predictions=[];repeat_targets=[]
                for fold in range(nfold):
                    fit=known[assignment!=fold];held=known[assignment==fold]
                    permitted=np.sort(np.r_[fit,unknown])
                    assert not np.intersect1d(permitted,held).size
                    if pool=='train157':assert len(frame)==1171 and (permitted<1171).all()
                    model=AnchoredReasonClassifier().fit(frame.iloc[permitted][FEATURES],frame.iloc[permitted][REASONS])
                    scores=model.predict_proba(frame.iloc[held][FEATURES]);prediction=model.predict(frame.iloc[held][FEATURES])
                    old=expected[(expected.pool==pool)&(expected.repeat==repeat)&(expected.fold==fold)].set_index('ID').loc[frame.iloc[held].ID]
                    saved=old[[f'probability_{r}' for r in REASONS]].to_numpy()
                    np.testing.assert_allclose(scores,saved,atol=1e-7,rtol=0)
                    np.testing.assert_array_equal(prediction,old[[f'predicted_{r}' for r in REASONS]])
                    np.testing.assert_array_equal(frame.iloc[held][REASONS],old[[f'actual_{r}' for r in REASONS]])
                    for audit in model.round_audit_:np.testing.assert_allclose(audit['known_weight'],len(fit),atol=1e-10)
                    generated_weight=.25*(model.generated_labels_@model.class_weights_/model.generated_labels_.sum(axis=1)).sum()
                    np.testing.assert_allclose(generated_weight,model.round_audit_[-1]['generated_weight'],atol=1e-10)
                    checks.append(dict(pool=pool,repeat=repeat,fold=fold,max_score_error=float(abs(scores-saved).max()),
                                       exact_predictions=True,fit_ids=frame.iloc[fit].ID.tolist(),held_ids=frame.iloc[held].ID.tolist(),
                                       feature_ids=frame.iloc[permitted].ID.tolist(),unknown_ids=frame.iloc[unknown].ID.tolist(),
                                       fit_weight_checks_pass=True,test_features_used=False))
                    repeat_predictions.append(prediction);repeat_targets.append(frame.iloc[held][REASONS].to_numpy())
                rescored.append(dict(pool=pool,repeat=repeat,**metrics(np.vstack(repeat_targets),np.vstack(repeat_predictions))))
            print('Independent replay:',pool,nrep*nfold,'folds passed',flush=True)
        _,frame,_,_,_,unknown=pools()[1]
        known=np.flatnonzero(frame[REASONS].ge(0).all(axis=1).to_numpy())
        for split in ['original_validation','after_2003','after_2005','after_2007']:
            if split=='original_validation':fit=known[:157];held=known[157:];u=unknown
            else:
                year=int(split.split('_')[-1]);early=frame.start_year.to_numpy()<=year
                fit=known[early[known]];held=known[~early[known]];u=unknown[early[unknown]]
            permitted=np.sort(np.r_[fit,u]);assert not np.intersect1d(permitted,held).size
            model=AnchoredReasonClassifier().fit(frame.iloc[permitted][FEATURES],frame.iloc[permitted][REASONS])
            scores=model.predict_proba(frame.iloc[held][FEATURES]);old=diag[diag.split==split].set_index('ID').loc[frame.iloc[held].ID]
            np.testing.assert_allclose(scores,old[[f'probability_{r}' for r in REASONS]],atol=1e-7,rtol=0)
            np.testing.assert_array_equal(model.predict(frame.iloc[held][FEATURES]),old[[f'predicted_{r}' for r in REASONS]])
            diagnostic_checks.append(dict(split=split,exact_predictions=True,fit_ids=frame.iloc[fit].ID.tolist(),
                                          held_ids=frame.iloc[held].ID.tolist(),feature_ids=frame.iloc[permitted].ID.tolist()))
        print('All four diagnostic replays passed.',flush=True)
        # The recipe is selected and every replay is complete before opening test inputs.
        model=AnchoredReasonClassifier().fit(frame[FEATURES],frame[REASONS])
        joblib.dump(model,candidate/'anchored_model.joblib',compress=3)
        test=pd.read_csv(ROOT/'data/part1_test.csv');template=pd.read_csv(ROOT/'data/part1_sample_submission.csv')
        assert set(test.ID).isdisjoint(frame.ID)
        scores=model.predict_proba(test[FEATURES]);prediction=model.predict(test[FEATURES])
        np.testing.assert_array_equal(joblib.load(candidate/'anchored_model.joblib').predict_proba(test[FEATURES]),scores)
        labels=[' '.join(r for r,v in zip(REASONS,row) if v) for row in prediction]
        submission=pd.DataFrame(dict(ID=test.ID,Predicted=labels)).set_index('ID').loc[template.ID].reset_index()
        assert len(submission)==312 and submission.ID.is_unique and submission.ID.tolist()==template.ID.tolist()
        assert submission.Predicted.map(lambda s:bool(s.split()) and set(s.split())<=set(REASONS)).all()
        submission.to_csv(candidate/'part1_anchored_submission.csv',index=False)
        pd.DataFrame(scores,index=test.ID,columns=REASONS).to_csv(candidate/'test_scores.csv')
        U=frame.iloc[unknown];generated=model.predict(U[FEATURES]);generated_scores=model.predict_proba(U[FEATURES])
        audit=U[['ID','start_year']].copy();audit['accepted_for_final_fit']=model.generated_mask_
        for j,r in enumerate(REASONS):
            audit[r]=generated[:,j];audit[f'score_{r}']=generated_scores[:,j]
            target=np.full(len(U),-100);target[model.generated_mask_]=model.generated_labels_[:,j];audit[f'fit_target_{r}']=target
        audit.to_csv(candidate/'generated_labels.csv',index=False)
        train=pd.read_csv(ROOT/'data/part1_train.csv');masked=train[REASONS].lt(0).all(axis=1)
        assert train.loc[masked,'ID'].tolist()==U.ID.tolist()
        augmented=train.copy();augmented.loc[masked,REASONS]=generated
        augmented['label_origin']=np.where(masked,'model_generated','verified_training')
        np.testing.assert_array_equal(augmented.loc[~masked,REASONS],train.loc[~masked,REASONS])
        augmented.to_csv(candidate/'part1_train_augmented.csv',index=False)
    (candidate/'independent_replay_checks.json').write_text(json.dumps(dict(cv=checks,diagnostics=diagnostic_checks),separators=(',',':'))+'\n')
    repeats=pd.DataFrame(rescored);repeats.to_csv(candidate/'independent_repeat_scores.csv',index=False)
    summary=pd.read_csv(stage/'summary.csv')
    for pool,g in repeats.groupby('pool'):
        old=summary[(summary.pool==pool)&(summary.name==name)].iloc[0]
        np.testing.assert_allclose(g.sample_f1.mean(),old.mean_f1,atol=1e-15)
        np.testing.assert_allclose(g.macro_f1.mean(),old.macro_f1,atol=1e-15)
    summary[summary.name.isin([name,'ani_current','trusted_two_pass'])].to_csv(OUT/'selected_comparison.csv',index=False)
    diagnostic=pd.read_csv(stage/'diagnostic_scores.csv');diagnostic[diagnostic.name.isin([name,'ani_current','trusted_two_pass'])].to_csv(OUT/'selected_diagnostics.csv',index=False)
    all_results=[];all_diagnostics=[]
    for p in sorted(OUT.glob('*/summary.csv')):
        df=pd.read_csv(p);df.insert(0,'stage',p.parent.name);all_results.append(df)
        df=pd.read_csv(p.parent/'diagnostic_scores.csv');df.insert(0,'stage',p.parent.name);all_diagnostics.append(df)
    pd.concat(all_results).to_csv(OUT/'all_results.csv',index=False)
    pd.concat(all_diagnostics).to_csv(OUT/'all_diagnostic_scores.csv',index=False)
    saved=pd.read_csv(stage/'predictions.csv');saved=saved[saved.name.isin([name,'ani_current'])].copy()
    y=saved[[f'actual_{r}' for r in REASONS]].to_numpy();pred=saved[[f'predicted_{r}' for r in REASONS]].to_numpy()
    saved['trial_f1']=2*(y*pred).sum(axis=1)/(y.sum(axis=1)+pred.sum(axis=1))
    intervals=[];rng=np.random.default_rng(20261106)
    for pool,g in saved.groupby('pool'):
        wide=g.pivot(index=['repeat','ID'],columns='name',values='trial_f1')
        diff=(wide[name]-wide.ani_current).groupby('ID').mean().to_numpy()
        draws=rng.integers(0,len(diff),(10000,len(diff)))
        lo,hi=np.quantile(diff[draws].mean(axis=1),[.025,.975])
        intervals.append(dict(pool=pool,delta=diff.mean(),lower_95=lo,upper_95=hi,
                              caveat='Descriptive paired-trial bootstrap. Not adjusted for selection or overlapping fits.'))
    pd.DataFrame(intervals).to_csv(OUT/'paired_trial_bootstrap.csv',index=False)
    protected=json.loads((BASE/'protocol.json').read_text())['protected_input_sha256']
    for path,h in protected.items():assert digest(ROOT/path)==h,path
    for old_dir in ['simple_methodical','trusted_iteration']:
        old=ROOT/'experiments'/old_dir/'candidate';manifest=json.loads((old/'manifest.json').read_text())
        for f,h in manifest['artifact_sha256'].items():assert digest(old/f)==h,f
        for f,h in manifest['source_sha256'].items():assert digest(ROOT/f)==h,f
    (candidate/'anchored_solution.py').write_bytes((ROOT/'anchored_solution.py').read_bytes())
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Unsubmitted candidate; local CV improvement over pinned current Ani, Kaggle performance unknown.',
                  selected=selected,known_labels=235,originally_unlabeled=1014,test_rows=312,
                  final_fit_audit=model.round_audit_,independent_fold_replays=55,independent_diagnostic_replays=4,
                  exact_model_reload=True,protected_originals_unchanged=True,kaggle_score=None,
                  cv_summary=summary[summary.name==name].to_dict(orient='records'),
                  source_sha256={f:digest(ROOT/f) for f in ['anchored_solution.py','paired_models.py','paired_search.py','paired_cutoffs.py','paired_finalize.py','simple_models.py','simple_benchmark.py']},
                  input_sha256={p.name:digest(p) for p in (ROOT/'data').glob('part1_*.csv')},
                  artifact_sha256={p.name:digest(p) for p in candidate.iterdir() if p.is_file() and p.name!='manifest.json'},
                  versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,joblib=joblib.__version__))
    (candidate/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary[summary.name.isin([name,'ani_current'])].to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))
    print('Saved checked 312-row CSV; no upload performed.',flush=True)


if __name__=='__main__':main()
