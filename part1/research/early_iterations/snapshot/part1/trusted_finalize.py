"""Independently replay, report and package the selected candidate; no upload."""
from datetime import datetime, timezone
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from simple_benchmark import ROOT, BASE, digest, load_data, load_references, metrics
from simple_models import FEATURES, REASONS
from trusted_benchmark import OUT
from trusted_solution import TrustedTwoPassClassifier


def main():
    OUT.mkdir(exist_ok=True)
    candidate = OUT/'candidate'
    candidate.mkdir(exist_ok=True)
    name, stage_name = 'final_r2_control', '06_final_regularization_check'
    stage = OUT/stage_name
    protocol = json.loads((stage/'protocol.json').read_text())
    spec = next(s for s in protocol['specs'] if s['name'] == name)
    selected = dict(name=name,stage=stage_name,spec=spec,
                    reason='Best single-model CV result; gains in all3chronological checks. Equal C0.1/C0.3 average gains only0.000177 and has higher repeat variability, so retain one model.',
                    caveat='Old78-row validation is lower than Ani. All local checks are reused development evidence; Kaggle performance unknown.',
                    primary_reference='ani_train_only',test_fitting_access=False)
    (OUT/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
    all_results, all_diagnostics = [], []
    for p in sorted(OUT.glob('*/summary.csv')):
        table = pd.read_csv(p); table.insert(0,'stage',p.parent.name); all_results.append(table)
        if (p.parent/'diagnostic_scores.csv').exists():
            table=pd.read_csv(p.parent/'diagnostic_scores.csv');table.insert(0,'stage',p.parent.name);all_diagnostics.append(table)
    pd.concat(all_results).to_csv(OUT/'all_results.csv',index=False)
    pd.concat(all_diagnostics).to_csv(OUT/'all_diagnostic_scores.csv',index=False)
    known,unknown = load_data()
    X,Y,U = known[FEATURES].to_numpy(),known[REASONS].to_numpy(int),unknown[FEATURES].to_numpy()
    saved=np.load(stage/'predictions.npz')
    folds=saved['folds'];expected_labels=saved[name]
    expected=np.load(stage/'probabilities.npz')[name]
    checks=[]
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for repeat in range(8):
            for fold in range(5):
                held=folds[repeat]==fold
                model=TrustedTwoPassClassifier().fit(X[~held],Y[~held],U)
                scores=model.predict_proba(X[held])
                np.testing.assert_allclose(scores,expected[repeat,held],atol=1e-10,rtol=1e-8)
                np.testing.assert_array_equal(model.predict(X[held]),expected_labels[repeat,held])
                # Class balancing remains anchored; known effective mass stays188.
                for audit in model.fit_weight_audit_:
                    np.testing.assert_allclose(audit['known'],188,atol=1e-10)
                generated_weight=.25*(model.accepted_labels_@model.class_weights_ / model.accepted_labels_.sum(axis=1)).sum()
                np.testing.assert_allclose(generated_weight,model.fit_weight_audit_[-1]['generated'],atol=1e-10)
                checks.append(dict(repeat=repeat,fold=fold,max_abs_error=float(np.abs(scores-expected[repeat,held]).max()),
                                   known_effective_mass=188,generated_weight_verified=True))
        print('Independent implementation reproduces all40folds and weight calculations.',flush=True)
        # Independently reproduce all four diagnostic fits and ID alignments.
        diagnostic=pd.read_csv(stage/'diagnostic_predictions.csv')
        diagnostic=diagnostic[diagnostic.name==name]
        diagnostic_checks=[]
        for split in ['original_validation','after_2003','after_2005','after_2007']:
            if split=='original_validation':
                fit=np.arange(len(known))<157; unlab=np.ones(len(unknown),bool)
            else:
                year=int(split.split('_')[-1]);fit=known.start_year.to_numpy()<=year;unlab=unknown.start_year.to_numpy()<=year
            model=TrustedTwoPassClassifier().fit(X[fit],Y[fit],U[unlab])
            scores=model.predict_proba(X[~fit])
            reference=diagnostic[diagnostic.split==split].set_index('ID').loc[known.loc[~fit,'ID']]
            np.testing.assert_allclose(scores,reference[[f'probability_{r}' for r in REASONS]],atol=1e-10,rtol=1e-8)
            np.testing.assert_array_equal(model.predict(X[~fit]),reference[[f'predicted_{r}' for r in REASONS]])
            diagnostic_checks.append(dict(split=split,reproduced=True))
        model=TrustedTwoPassClassifier().fit(X,Y,U)
        joblib.dump(model,candidate/'trusted_two_pass.joblib',compress=3)
        # Test inputs are opened only after the recipe is selected and fit completes.
        test=pd.read_csv(ROOT/'data/part1_test.csv')
        template=pd.read_csv(ROOT/'data/part1_sample_submission.csv')
        assert set(test.ID).isdisjoint(known.ID) and set(test.ID).isdisjoint(unknown.ID)
        scores=model.predict_proba(test[FEATURES].to_numpy())
        reloaded=joblib.load(candidate/'trusted_two_pass.joblib')
        np.testing.assert_allclose(reloaded.predict_proba(test[FEATURES].to_numpy()),scores,atol=0,rtol=0)
        predicted=model.predict(test[FEATURES].to_numpy())
        labels=[' '.join(r for r,b in zip(REASONS,row) if b) for row in predicted]
        submission=pd.DataFrame(dict(ID=test.ID,Predicted=labels)).set_index('ID').loc[template.ID].reset_index()
        assert len(submission)==312 and submission.ID.is_unique and submission.ID.tolist()==template.ID.tolist()
        assert submission.Predicted.map(lambda x:bool(x.split()) and set(x.split())<=set(REASONS)).all()
        submission.to_csv(candidate/'part1_trusted_submission.csv',index=False)
        pd.DataFrame(scores,index=test.ID,columns=REASONS).to_csv(candidate/'test_scores.csv')
        pseudo=unknown[['ID','start_year']].copy();pseudo['accepted_for_fit']=model.accepted_mask_
        p=model.predict_proba(U);labels=model.predict(U)
        for j,r in enumerate(REASONS):
            pseudo[r]=labels[:,j];pseudo[f'score_{r}']=p[:,j]
            fit_target=np.full(len(U),-100);fit_target[model.accepted_mask_]=model.accepted_labels_[:,j]
            pseudo[f'fit_target_{r}']=fit_target
        pseudo.to_csv(candidate/'generated_labels.csv',index=False)
        train=pd.read_csv(ROOT/'data/part1_train.csv');masked=train[REASONS].lt(0).all(axis=1)
        assert train.loc[masked,'ID'].tolist()==unknown.ID.tolist()
        augmented=train.copy();augmented.loc[masked,REASONS]=labels
        augmented['label_origin']=np.where(masked,'trusted_model_prediction','verified_training')
        np.testing.assert_array_equal(augmented.loc[~masked,REASONS],train.loc[~masked,REASONS])
        augmented.to_csv(candidate/'part1_train_augmented.csv',index=False)
    pd.DataFrame(checks).to_csv(candidate/'independent_40_fold_replay.csv',index=False)
    pd.DataFrame(diagnostic_checks).to_csv(candidate/'independent_diagnostic_replay.csv',index=False)
    references=load_references(known,None)
    rng=np.random.default_rng(20261105);draws=rng.integers(0,235,(10000,235))
    def trial_f1(pred):return 2*(pred*Y).sum(axis=2)/(pred.sum(axis=2)+Y.sum(axis=1))
    uncertainty=[]
    for ref_name,ref in references.items():
        diff=(trial_f1(expected_labels)-trial_f1(ref)).mean(axis=0)
        interval=np.quantile(diff[draws].mean(axis=1),[.025,.975])
        uncertainty.append(dict(name=name,reference=ref_name,delta=float(diff.mean()),lower_95=interval[0],upper_95=interval[1],
                                caveat='Descriptive paired trial bootstrap; not adjusted for model selection or correlated training fits.'))
    pd.DataFrame(uncertainty).to_csv(OUT/'paired_trial_bootstrap.csv',index=False)
    summary=pd.read_csv(stage/'summary.csv').set_index('name')
    summary.loc[[name,'ani_train_only','dbscan_train_only']].to_csv(OUT/'selected_comparison.csv')
    diagnostics=pd.read_csv(stage/'diagnostic_scores.csv')
    diagnostics[diagnostics.name.isin([name,'ani_train_only'])].to_csv(OUT/'selected_diagnostics.csv',index=False)
    # Preserve and check every protected historical artifact and prior simple candidate.
    protected=json.loads((BASE/'protocol.json').read_text())['protected_input_sha256']
    for path,expected_hash in protected.items():
        assert digest(ROOT/path)==expected_hash,path
    old=ROOT/'experiments/simple_methodical/candidate'
    old_manifest=json.loads((old/'manifest.json').read_text())
    for filename,expected_hash in old_manifest['artifact_sha256'].items():
        assert digest(old/filename)==expected_hash,filename
    for filename,expected_hash in old_manifest['source_sha256'].items():
        assert digest(ROOT/filename)==expected_hash,filename
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='unsubmitted recommended candidate; fixed benchmark reference unchanged',
                  selected=selected,known_labels=235,originally_unlabeled_rows=1014,test_rows=312,
                  final_generated_counts=model.round_counts_,final_fit_weights=model.fit_weight_audit_,
                  cv_metrics=summary.loc[name].to_dict(),kaggle_score=None,
                  independent_fold_replays=40,independent_diagnostic_replays=4,model_reload_exact=True,protected_originals_unchanged=True,
                  versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,joblib=joblib.__version__),
                  source_sha256={p.name:digest(p) for p in [ROOT/'trusted_solution.py',ROOT/'trusted_models.py',ROOT/'trusted_benchmark.py',ROOT/'trusted_average.py',ROOT/'trusted_finalize.py']},
                  input_sha256={p.name:digest(p) for p in (ROOT/'data').glob('part1_*.csv')},
                  artifact_sha256={p.name:digest(p) for p in candidate.iterdir() if p.is_file() and p.name!='manifest.json'})
    (candidate/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','final_generated_counts','cv_metrics','independent_fold_replays','independent_diagnostic_replays']},indent=2))


if __name__=='__main__':
    main()
