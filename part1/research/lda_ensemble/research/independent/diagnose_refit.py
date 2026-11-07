"""Frozen paired diagnostic of adding validation-source gold labels.

Keep the existing eight grouped pooled CV repeats and exact recipes. Compare
training on original-training gold only with training on all eligible gold,
on precisely the same held-out rows. No model/threshold search or submissions.
"""
from pathlib import Path
import argparse,json,warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_ani_improvements as study

r,b,e,models=study.r,study.b,study.e,study.models
ROOT=r.ROOT;OUT=ROOT/'experiments/refit_diagnostic_v1'
NAMES=['Ani frozen control','Ani + tri/SVM 15% each']


def protected():
    p=study.protected()
    for name in ['predictions/part1_three_way_trainval235.csv',
                 'predictions/three_way_trainval235/manifest.json',
                 'predictions/three_way_trainval235/test_scores.csv',
                 'predictions/three_way_trainval235/kaggle_result.json']:
        p[name]=b.file_hash(ROOT/name)
    return p


def sources():return {**r.sources(),**study.own_sources(),Path(__file__).name:b.file_hash(Path(__file__))}


def guard():
    study.guard();p=json.loads((OUT/'protocol.json').read_text())
    assert p['source_sha256']==sources() and p['protected_sha256']==protected()
    return p


def prepare():
    if (OUT/'protocol.json').exists():guard();print('Frozen diagnostic verified.');return
    study.guard()
    with study.pooled.pooled_context():
        first=study.stage_protocol('pooled');extra=study.stage_protocol('pooled',True)
        folds=first['folds']+extra['folds'];seeds=first['seeds']+extra['seeds']
    p=dict(created_at=b.now(),source_sha256=sources(),protected_sha256=protected(),
        candidates=[c for c in study.registry() if c['name'] in NAMES],folds=folds,seeds=seeds,
        purpose='Diagnose additional validation-source label effect, not select a new recipe.',
        comparison='Same pooled held-out groups in both arms. Reduced arm: eligible original-training gold only. Expanded arm: all eligible original-training and validation gold. Original masked pool identical, with held-out groups purged from both arms. All hyperparameters and decoding unchanged.',
        expanded_reference='Reuse frozen pooled3 and pooled5 predictions from ani_improvement_v1; verify IDs, truth, recipe and metric parity.',
        analyses='Paired F1 per repeat on all 235 gold rows, original train157 rows, and original validation78 rows; per-label F1; prediction changes. Report all eight repeats, no favorable-repeat selection.',
        fitting_audit='In reduced fits, assert every fitting ID belongs to original training file. For a representative fold, changing all validation-source labels must leave reduced-arm predictions unchanged.',
        sensitivity='Fit the unchanged three-way recipe on original train157 and compare its unlabeled-test predictions with the saved 235-label submission. Prediction changes measure sensitivity, not accuracy. Do not export another submission.',
        limitations='Within-fold sample sizes are roughly125 versus188, an analogue of157 versus235. Existing labels and splits have informed development. This cannot determine the unobserved Kaggle score of the157-label fit or isolate sample count from added-label composition.',
        settings_changed=False,submission_created=False,test_inputs_used_for_fit=False)
    r.save(OUT/'protocol.json',p);print('Frozen two-recipe paired refit diagnostic on the existing 40 pooled folds.',flush=True)


def reference(candidate):
    scores=[];files=[]
    for stage in ['pooled3','pooled5']:
        folder=study.OUT/stage
        row=next(x for x in json.loads((folder/'index.json').read_text()) if x['candidate']==candidate['name'])
        meta=json.loads((folder/(row['result_key']+'.json')).read_text())
        assert meta['config']==candidate['config'] and meta['rule']==candidate['rule']
        path=folder/(row['result_key']+'.npz');saved=np.load(path,allow_pickle=True)
        assert saved['ids'].tolist()==e.DATA.iloc[e.LAB].ID.tolist()
        np.testing.assert_array_equal(saved['truth'],e.Y)
        for i,p in enumerate(saved['scores']):
            assert abs(e.metrics(e.Y,p,candidate['rule'])['sample_f1']-row['repeat_scores'][i]['sample_f1'])<1e-12
        scores.extend(saved['scores']);files.append(dict(path=str(path.relative_to(ROOT)),sha256=b.file_hash(path)))
    return np.array(scores),files


def run():
    protocol=guard();train,_,_=r.frames();train_ids=set(train.ID);results=[]
    with study.pooled.pooled_context():
        origin=np.isin(e.DATA.iloc[e.LAB].ID,list(train_ids))
        assert origin.sum()==157 and (~origin).sum()==78
        old_features=e.features;old_native=models.native_variant
        audit=[]
        def features(indices,query,config):
            assert set(e.DATA.iloc[indices].ID)<=train_ids
            return old_features(indices,query,config)
        def native(pos,query,config,excluded=None):
            scores=old_native(pos,query,config,excluded)
            a=models.LAST_AUDIT
            assert set(a['gold_ids'])|set(a['masked_ids'])<=train_ids
            assert a['gold_rows']==len(pos)
            return scores
        e.features=features;models.native_variant=native
        try:
            for i,c in enumerate(protocol['candidates']):
                expanded,files=reference(c);small=np.full_like(expanded,np.nan);folds=[]
                for si,seed in enumerate(protocol['seeds']):
                    for f in [x for x in protocol['folds'] if x['seed']==seed]:
                        a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
                        reduced=a[origin[a]];assert len(reduced)>0 and len(reduced)<len(a)
                        query=e.DATA.iloc[e.LAB[h]].drop(columns=e.LABELS);excluded=e.GROUP[e.LAB[h]]
                        models.clear();scores=e.predict(reduced,query,c['config'],excluded)
                        assert scores.shape==(len(h),5) and np.isfinite(scores).all()
                        small[si,h]=scores
                        if si==0 and f['fold']==0:
                            y=e.Y.copy();val_rows=e.LAB[~origin];raw=e.DATA.loc[val_rows,e.LABELS].copy()
                            try:
                                e.Y[~origin]=1-e.Y[~origin]
                                e.DATA.loc[val_rows,e.LABELS]=1-e.DATA.loc[val_rows,e.LABELS].to_numpy()
                                models.clear();changed=e.predict(reduced,query,c['config'],excluded)
                                np.testing.assert_allclose(changed,scores,rtol=0,atol=1e-10)
                            finally:e.Y[:]=y;e.DATA.loc[val_rows,e.LABELS]=raw.to_numpy();models.clear()
                            audit.append(dict(candidate=c['name'],all_validation_label_mutation_invariant=True,fit_ids_original_training_only=True))
                        folds.append(dict(seed=seed,fold=f['fold'],heldout_rows=len(h),reduced_gold=len(reduced),expanded_gold=len(a),added_validation_gold=len(a)-len(reduced)))
                    print(json.dumps(dict(candidate=c['name'],seed=seed,reduced_f1=e.metrics(e.Y,small[si],c['rule'])['sample_f1'],expanded_f1=e.metrics(e.Y,expanded[si],c['rule'])['sample_f1'])),flush=True)
                assert np.isfinite(small).all()
                path=OUT/f'model{i}.npz'
                np.savez_compressed(path,reduced=small,expanded=expanded,truth=e.Y,ids=e.DATA.iloc[e.LAB].ID.to_numpy(),original_train=origin)
                rows=[]
                for scope,mask in [('all235',np.ones(len(e.Y),bool)),('original_train157',origin),('original_validation78',~origin)]:
                    reduced=[e.metrics(e.Y[mask],s[mask],c['rule']) for s in small]
                    large=[e.metrics(e.Y[mask],s[mask],c['rule']) for s in expanded]
                    delta=np.array([v['sample_f1'] for v in large])-np.array([v['sample_f1'] for v in reduced])
                    rows.append(dict(scope=scope,rows=int(mask.sum()),reduced_mean=float(np.mean([v['sample_f1'] for v in reduced])),expanded_mean=float(np.mean([v['sample_f1'] for v in large])),paired_delta=float(delta.mean()),delta_sd=float(delta.std()),expanded_wins=int((delta>1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),reduced_repeats=reduced,expanded_repeats=large))
                results.append(dict(candidate=c['name'],results=rows,fold_counts=folds,reference_files=files,predictions_file=str(path.relative_to(ROOT)),predictions_sha256=b.file_hash(path)))
                r.save(OUT/'cv_results.json',results)
        finally:e.features=old_features;models.native_variant=old_native;models.clear()
    r.save(OUT/'boundary_audit.json',dict(passed=True,checks=audit));guard()


def sensitivity():
    p=guard();candidate=next(c for c in p['candidates'] if c['name']==NAMES[1])
    train,_,_=r.frames();test=pd.read_csv(ROOT/'data/part1_test.csv')
    saved=pd.read_csv(ROOT/'predictions/three_way_trainval235/test_scores.csv')
    assert test.ID.tolist()==saved.ID.tolist()
    big=saved[['score_'+l for l in e.LABELS]].to_numpy()
    with r.data_context(train):
        assert len(e.Y)==157;models.clear()
        small=e.predict(np.arange(len(e.Y)),test,candidate['config']);models.clear()
    assert np.isfinite(small).all()
    a=e.decisions(small,candidate['rule']);bigh=e.decisions(big,candidate['rule'])
    manifest=json.loads((ROOT/'predictions/three_way_trainval235/manifest.json').read_text())
    submission=pd.read_csv(ROOT/manifest['submission_file'])
    decoded=np.array([[int(l in value.split()) for l in e.LABELS] for value in submission.Predicted])
    np.testing.assert_array_equal(decoded,bigh)
    diff=a!=bigh
    result=dict(candidate=candidate['name'],test_rows=len(test),changed_label_sets=int(diff.any(axis=1).sum()),changed_label_bits=int(diff.sum()),total_label_bits=int(diff.size),mean_absolute_score_change=float(np.abs(small-big).mean()),labels={l:dict(original157_positives=int(a[:,j].sum()),refit235_positives=int(bigh[:,j].sum()),changed_bits=int(diff[:,j].sum())) for j,l in enumerate(e.LABELS)},accuracy_comparison=None,interpretation='Prediction sensitivity only; test labels are unknown and neither direction is shown to be more accurate.',submission_created=False)
    np.savez_compressed(OUT/'test_prediction_sensitivity.npz',ids=test.ID.to_numpy(),train157_scores=small,train235_scores=big,train157_predictions=a,train235_predictions=bigh)
    r.save(OUT/'test_sensitivity.json',result);guard();print(json.dumps(result,indent=2),flush=True)


def report():
    guard();assert json.loads((OUT/'boundary_audit.json').read_text())['passed']
    cv=json.loads((OUT/'cv_results.json').read_text());sensitivity=json.loads((OUT/'test_sensitivity.json').read_text());rows=[]
    for c in cv:
        assert b.file_hash(ROOT/c['predictions_file'])==c['predictions_sha256']
        saved=np.load(ROOT/c['predictions_file'],allow_pickle=True)
        for row in c['results']:
            mask=np.ones(len(saved['truth']),bool) if row['scope']=='all235' else saved['original_train'] if row['scope']=='original_train157' else ~saved['original_train']
            for arm,key in [('reduced','reduced_repeats'),('expanded','expanded_repeats')]:
                for i,score in enumerate(saved[arm]):assert abs(e.metrics(saved['truth'][mask],score[mask],'t0.5')['sample_f1']-row[key][i]['sample_f1'])<1e-12
            rows.append(dict(candidate=c['candidate'],**{k:v for k,v in row.items() if not k.endswith('repeats')}))
    table=pd.DataFrame(rows);table.to_csv(OUT/'comparison.csv',index=False)
    r.save(OUT/'summary.json',dict(comparison=rows,test_sensitivity=sensitivity,protocol_sha256=b.file_hash(OUT/'protocol.json'),verification_passed=True,protected_sha256=protected()))
    text='<p>Frozen recipes; eight existing grouped five-fold pooled repeats. The reduced arm uses only eligible original-training gold. The expanded arm adds eligible validation-source gold. Held-out rows, masked pool policy and model settings are identical. Expanded predictions are reused from the original frozen benchmark. Delta is expanded minus reduced.</p>'
    text+=table.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
    text+=f'<p>On the 312 unlabeled test rows, fitting the three-way recipe on157 versus235 gold labels changed {sensitivity["changed_label_sets"]} complete label sets and {sensitivity["changed_label_bits"]} individual label decisions. This measures sensitivity, not accuracy. No new submission was exported.</p>'
    text+='<p>Within-fold fits use roughly125 versus188 labels, so this diagnoses an analogous addition of labels rather than measuring the unavailable Kaggle score of the157-label fit. Added sample count and label composition change together. Existing labels and splits have already informed development. The original submitted Ani pipeline used transductive fitting; the local Ani control is inductive.</p>'
    text+='<p><a href="protocol.json">Protocol</a> · <a href="cv_results.json">Every paired repeat and fitting count</a> · <a href="boundary_audit.json">Boundary audit</a> · <a href="test_sensitivity.json">Prediction changes</a> · <a href="summary.json">Verified summary</a></p>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Refit diagnostic</title><style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px}p{line-height:1.6}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>Does adding validation-source labels help?</h1>'+text)
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','sensitivity','report']);args=parser.parse_args()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);globals()[args.stage]()
