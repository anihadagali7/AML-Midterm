"""Separate, explicitly authorized transductive study of the submitted ensemble."""
from pathlib import Path
from contextlib import nullcontext
import argparse,json,warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import three_way_transductive as model

study=model.study;r,b,e=study.r,study.b,study.e
ROOT=r.ROOT;OUT=ROOT/'experiments/three_way_transductive_v1'
MODES=['ani_transductive','all_transductive']
LABELS={'inductive':'Submitted three-way, inductive','ani_transductive':'Three-way: Ani transductive','all_transductive':'Three-way: all components transductive'}


def sources():
    return {**r.sources(),**study.own_sources(),**{n:b.file_hash(ROOT/n) for n in ['three_way_transductive.py','run_three_way_transductive.py']}}


def protected():
    files=study.protected()
    for n in ['predictions/part1_three_way_trainval235.csv','predictions/three_way_trainval235/manifest.json','predictions/three_way_trainval235/kaggle_result.json']:
        files[n]=b.file_hash(ROOT/n)
    return files


def guard():
    study.guard();p=json.loads((OUT/'protocol.json').read_text())
    assert p['source_sha256']==sources() and p['protected_sha256']==protected()
    return p


def prepare():
    if (OUT/'protocol.json').exists():guard();print('Frozen transductive protocol verified.');return
    study.guard();folds={};seeds={}
    for kind in ['train','pooled']:
        with study.pooled.pooled_context() if kind=='pooled' else nullcontext():
            a=study.stage_protocol(kind);z=study.stage_protocol(kind,True)
            folds[kind]=a['folds']+z['folds'];seeds[kind]=a['seeds']+z['seeds']
    p=dict(created_at=b.now(),source_sha256=sources(),protected_sha256=protected(),
        authorization='User explicitly requested transductive learning on the submitted three-way model. Separate experimental track; original inductive conventions and artifacts remain unchanged.',
        candidate=model.CANDIDATE,modes=MODES,folds=folds,seeds=seeds,
        invariant='Exact70/15/15 weights, Ani LR C.3 and kNN10, PCA30/GMM30diag seed0, native two self-training rounds, all existing tri/SVM settings and final t0.5 rule. No tuning.',
        access='Within each CV fold only fitting gold labels are available. Purge held-out groups from original masked pool, then explicitly add held-out query inputs with all five labels erased to-100. Ani-only arm does this for Ani; full arm also permits target pseudo-labeling in tri/SVM. Only current held-out batch is the unlabeled target; actual Kaggle inputs are not included in local CV.',
        comparison='Same40 existing grouped folds per pool. Reuse exact saved inductive three-way predictions; these scores estimate different fitting-access regimes and are labeled accordingly.',
        shortlist='Positive paired mean and >=5/8 wins in BOTH train157 and pooled235 CV pools. Evaluate qualifying arms on original validation and frozen chronology. If neither qualifies, evaluate full-transductive arm as a declared diagnostic, with no improvement claim.',
        supplementary='Original validation: train157 gold plus original masked, validation78 inputs unlabeled for transductive components. Chronology: only eligible earlier gold/masked plus the explicitly supplied later evaluation batch as unlabeled targets. No later labels or other future inputs.',
        final='Fit both frozen transductive arms on235 gold plus1014 original masked and312 unlabeled test rows. Report prediction changes. Export a submission only for the CV-qualified arm with largest minimum paired gain. Never upload.',
        limitations='Repeated development labels and folds; target batch sizes in local CV differ from312-row Kaggle batch. Transductive scores describe inference on the supplied batch, not ordinary unseen-input generalization. Staff permission was not independently established; user explicitly authorized this experiment. No external trial data, deep learning, or test labels.',
        original_benchmark_modified=False,kaggle_upload=False)
    r.save(OUT/'protocol.json',p);print('Frozen two transductive arms, unchanged recipe and existing folds.',flush=True)


def reference(kind):
    scores=[];files=[]
    for suffix in ['3','5']:
        folder=study.OUT/(kind+suffix)
        row=next(x for x in json.loads((folder/'index.json').read_text()) if x['candidate']==model.CANDIDATE['name'])
        meta=json.loads((folder/(row['result_key']+'.json')).read_text())
        assert meta['config']==model.CANDIDATE['config'] and meta['rule']=='t0.5'
        path=folder/(row['result_key']+'.npz');a=np.load(path,allow_pickle=True)
        assert a['ids'].tolist()==e.DATA.iloc[e.LAB].ID.tolist();np.testing.assert_array_equal(a['truth'],e.Y)
        scores.extend(a['scores']);files.append(dict(path=str(path.relative_to(ROOT)),sha256=b.file_hash(path)))
    assert len(scores)==8
    return np.array(scores),files


def audit():
    p=guard();checks=[]
    for kind in ['train','pooled']:
        with study.pooled.pooled_context() if kind=='pooled' else nullcontext():
            f=p['folds'][kind][0];a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
            query=e.DATA.iloc[e.LAB[h]].drop(columns=e.LABELS).copy();excluded=e.GROUP[e.LAB[h]]
            model.clear();original=model.predict(a,query,'inductive',excluded)
            model.clear();repacked=model.predict(a,query,'repacked_inductive',excluded)
            np.testing.assert_allclose(original,repacked,rtol=0,atol=1e-10)
            checks.append(dict(pool=kind,check='inductive_repacking_parity',max_difference=float(np.max(np.abs(original-repacked)))))
            for mode in MODES:
                model.clear();before=model.predict(a,query,mode,excluded)
                au=model.LAST_AUDIT
                assert set(au['ani']['gold_ids'])==set(e.DATA.iloc[e.LAB[a]].ID)
                assert au['ani']['target_rows_in_fit']==len(query)
                held=e.LAB[h];masked=e.UNLAB[np.isin(e.GROUP[e.UNLAB],excluded)];forbidden=np.r_[held,masked]
                raw=e.DATA.iloc[forbidden].copy();truth=e.Y.copy()
                altered=query.copy()
                for j,label in enumerate(e.LABELS):altered[label]=1-e.Y[h,j]
                try:
                    e.Y[h]=1-e.Y[h];e.DATA.loc[held,e.LABELS]=1-e.DATA.loc[held,e.LABELS].to_numpy()
                    e.DATA.loc[forbidden,e.EMB]=1e6
                    model.clear();after=model.predict(a,altered,mode,excluded)
                finally:e.DATA.iloc[forbidden]=raw.to_numpy();e.Y[:]=truth;model.clear()
                np.testing.assert_allclose(before,after,rtol=0,atol=1e-10)
                checks.append(dict(pool=kind,mode=mode,heldout_label_mutation_invariant=True,
                    labels_supplied_in_query_ignored=True,unused_global_heldout_features_ignored=True,
                    masked_duplicates_purged=len(masked),explicit_target_rows=len(query)))
    r.save(OUT/'boundary_audit.json',dict(passed=True,checks=checks));guard();print('Parity and target-label isolation checks passed.',flush=True)


def run(kind):
    p=guard();assert json.loads((OUT/'boundary_audit.json').read_text())['passed']
    folder=OUT/kind;folder.mkdir(exist_ok=True);index=[]
    with study.pooled.pooled_context() if kind=='pooled' else nullcontext():
        native,files=reference(kind)
        for mode in ['inductive']+MODES:
            path=folder/(mode+'.json')
            if path.exists():index.append(json.loads(path.read_text()));continue
            scores=native.copy() if mode=='inductive' else np.full_like(native,np.nan)
            audits=[]
            if mode!='inductive':
                for si,seed in enumerate(p['seeds'][kind]):
                    for f in [x for x in p['folds'][kind] if x['seed']==seed]:
                        a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
                        query=e.DATA.iloc[e.LAB[h]].drop(columns=e.LABELS);excluded=e.GROUP[e.LAB[h]]
                        model.clear();scores[si,h]=model.predict(a,query,mode,excluded)
                        audit=model.LAST_AUDIT
                        assert set(audit['ani']['gold_ids'])==set(e.DATA.iloc[e.LAB[a]].ID)
                        audits.append(dict(seed=seed,fold=f['fold'],gold_rows=len(a),target_rows=len(h),
                            masked_rows=audit['ani']['masked_rows'],pseudo_history=audit['ani']['history'],target_labels_hidden=True))
                    print(json.dumps(dict(pool=kind,mode=mode,seed=seed,**e.metrics(e.Y,scores[si],'t0.5'))),flush=True)
            assert np.isfinite(scores).all()
            repeats=[e.metrics(e.Y,s,'t0.5') for s in scores]
            base=np.array([e.metrics(e.Y,s,'t0.5')['sample_f1'] for s in native]);values=np.array([s['sample_f1'] for s in repeats]);delta=values-base
            np.savez_compressed(folder/(mode+'.npz'),scores=scores,predictions=[e.decisions(s,'t0.5') for s in scores],truth=e.Y,ids=e.DATA.iloc[e.LAB].ID.to_numpy())
            row=dict(mode=mode,candidate=LABELS[mode],mean=float(values.mean()),sd=float(values.std()),paired_delta=float(delta.mean()),wins=int((delta>1e-12).sum()),repeat_scores=repeats,fit_audits=audits,reference_files=files)
            r.save(path,row);index.append(row);r.save(folder/'index.json',index)
            print(json.dumps(dict(pool=kind,mode=mode,mean=row['mean'],delta=row['paired_delta'],wins=row['wins'])),flush=True)
    r.save(folder/'index.json',index);guard()


def shortlist():
    guard();rows=[]
    for mode in ['inductive']+MODES:
        row=dict(mode=mode,candidate=LABELS[mode])
        for kind in ['train','pooled']:
            a=json.loads((OUT/kind/(mode+'.json')).read_text())
            row.update({kind+'_'+k:a[k] for k in ['mean','sd','paired_delta','wins']})
        row['minimum_gain']=min(row['train_paired_delta'],row['pooled_paired_delta'])
        row['qualifies']=bool(mode!='inductive' and row['minimum_gain']>1e-12 and row['train_wins']>=5 and row['pooled_wins']>=5)
        rows.append(row)
    table=pd.DataFrame(rows).sort_values('minimum_gain',ascending=False);qualified=table[table.qualifies]['mode'].tolist()
    selected=qualified or ['all_transductive']
    record=dict(created_at=b.now(),comparison=table.to_dict('records'),qualified=qualified,supplementary_modes=selected,diagnostic_only=not bool(qualified))
    path=OUT/'shortlist.json'
    if path.exists():assert json.loads(path.read_text())['supplementary_modes']==selected
    else:r.save(path,record)
    table.to_csv(OUT/'cv_comparison.csv',index=False);print(table.to_string(index=False),flush=True)


def supplementary():
    guard();selected=json.loads((OUT/'shortlist.json').read_text())['supplementary_modes'];train,val,full=r.frames();rows=[]
    old=json.loads((study.OUT/'supplementary/results.json').read_text())
    windows=[dict(name='original_validation',training_ids=train.ID.tolist(),heldout_ids=val.ID.tolist())]+json.loads((r.OUT/'protocol.json').read_text())['chronological_splits']
    for wi,w in enumerate(windows):
        prior=next(x for x in old if x['candidate']==model.CANDIDATE['name'] and x['window']==wi)
        rows.append(dict(prior,mode='inductive',reused_prediction_sha256=b.file_hash(ROOT/prior['prediction_file'])))
        fit=full.set_index('ID').loc[w['training_ids']].reset_index();q=full.set_index('ID').loc[w['heldout_ids']].reset_index()
        with r.data_context(fit):
            ex=None if wi==0 else pd.util.hash_pandas_object(q[e.EMB].round(10),index=False).to_numpy()
            for mode in selected:
                model.clear();scores=model.predict(np.arange(len(e.Y)),q.drop(columns=e.LABELS),mode,ex)
                path=OUT/'supplementary'/f'window{wi}_{mode}.csv';y=q[e.LABELS].to_numpy(int)
                r.persist_predictions(path,q,y,scores,'t0.5')
                rows.append(dict(mode=mode,candidate=LABELS[mode],window=wi,window_name=w['name'],prediction_file=str(path.relative_to(ROOT)),**r.detailed(y,scores,'t0.5')))
                r.save(OUT/'supplementary/results.json',rows)
                print(json.dumps(dict(mode=mode,window=wi,sample_f1=rows[-1]['sample_f1'])),flush=True)
    guard()


def final():
    guard();short=json.loads((OUT/'shortlist.json').read_text());_,_,full=r.frames()
    test=pd.read_csv(ROOT/'data/part1_test.csv');sample=pd.read_csv(ROOT/'data/part1_sample_submission.csv')
    assert test.ID.is_unique and test.ID.tolist()==sample.ID.tolist() and not set(test.ID)&set(full.ID)
    old=pd.read_csv(ROOT/'predictions/part1_three_way_trainval235.csv')
    assert old.ID.tolist()==test.ID.tolist()
    old_h=np.array([[int(label in row.split()) for label in e.LABELS] for row in old.Predicted]);rows=[]
    export=short['qualified'][0] if short['qualified'] else None
    with r.data_context(full):
        for mode in MODES:
            model.clear();scores=model.predict(np.arange(len(e.Y)),test,mode)
            hard=e.decisions(scores,'t0.5');audit=model.LAST_AUDIT
            assert len(audit['ani']['gold_ids'])==235 and audit['ani']['masked_rows']==1326
            assert set(audit['ani']['gold_ids']).isdisjoint(test.ID)
            path=OUT/'final'/f'{mode}_scores.csv';path.parent.mkdir(exist_ok=True)
            d=pd.DataFrame(scores,columns=['score_'+x for x in e.LABELS]);d.insert(0,'ID',test.ID);d.to_csv(path,index=False)
            record=dict(mode=mode,gold_rows=235,original_masked_rows=1014,target_rows=312,target_inputs_used_for_fit=True,test_labels_used=False,
                changed_test_label_sets=int((old_h!=hard).any(axis=1).sum()),label_counts=dict(zip(e.LABELS,hard.sum(axis=0).astype(int).tolist())),ani_pseudo_history=audit['ani']['history'],submission_created=False,kaggle_score=None)
            r.save(OUT/'final'/f'{mode}_fit_audit.json',audit)
            if mode==export:
                submission=pd.DataFrame(dict(ID=test.ID,Predicted=[' '.join(l for l,flag in zip(e.LABELS,row) if flag) for row in hard]))
                assert list(submission)==list(sample) and (hard.sum(axis=1)>0).all()
                destination=ROOT/'predictions'/f'part1_three_way_{mode}_trainval235.csv';payload=submission.to_csv(index=False).encode()
                if destination.exists():assert destination.read_bytes()==payload
                else:destination.write_bytes(payload)
                pd.testing.assert_frame_equal(pd.read_csv(destination),submission)
                record.update(submission_created=True,submission_file=str(destination.relative_to(ROOT)),submission_sha256=b.file_hash(destination),status='Prepared experimental candidate; not uploaded')
            rows.append(record);r.save(OUT/'final/results.json',rows);print(json.dumps(record),flush=True)
    guard()


def report():
    guard();short=json.loads((OUT/'shortlist.json').read_text());supp=json.loads((OUT/'supplementary/results.json').read_text());final=json.loads((OUT/'final/results.json').read_text());rows=[]
    for row in short['comparison']:
        row=row.copy();s=[x for x in supp if x['mode']==row['mode']]
        if s:
            row['validation']=next(x['sample_f1'] for x in s if x['window']==0)
            data=pd.concat([pd.read_csv(ROOT/x['prediction_file']) for x in s if x['window']>0],ignore_index=True)
            assert len(data)==98 and data.ID.is_unique
            row['chronological']=e.metrics(data[['actual_'+x for x in e.LABELS]].to_numpy(int),data[['score_'+x for x in e.LABELS]].to_numpy(),'t0.5')['sample_f1']
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(OUT/'final_comparison.csv',index=False)
    checks=[]
    for kind in ['train','pooled']:
        for mode in ['inductive']+MODES:
            a=np.load(OUT/kind/(mode+'.npz'),allow_pickle=True);stored=json.loads((OUT/kind/(mode+'.json')).read_text())
            assert len(a['scores'])==8 and len(set(a['ids']))==len(a['ids'])
            for i,p in enumerate(a['scores']):
                np.testing.assert_array_equal(e.decisions(p,'t0.5'),a['predictions'][i])
                assert abs(e.metrics(a['truth'],p,'t0.5')['sample_f1']-stored['repeat_scores'][i]['sample_f1'])<1e-12
            checks.append(dict(pool=kind,mode=mode,repeats_rescored=8))
    for row in supp:
        d=pd.read_csv(ROOT/row['prediction_file']);score=e.metrics(d[['actual_'+x for x in e.LABELS]].to_numpy(int),d[['predicted_'+x for x in e.LABELS]].to_numpy(),'t0.5')['sample_f1']
        assert abs(score-row['sample_f1'])<1e-12
        if 'reused_prediction_sha256' in row:assert b.file_hash(ROOT/row['prediction_file'])==row['reused_prediction_sha256']
    r.save(OUT/'verification.json',dict(passed=True,checks=checks,protected_sha256=protected(),boundary_audit_passed=json.loads((OUT/'boundary_audit.json').read_text())['passed']))
    r.save(OUT/'summary.json',dict(comparison=rows,final_prediction_runs=final,qualified=short['qualified'],diagnostic_only=short['diagnostic_only'],kaggle_upload=False,protocol_sha256=b.file_hash(OUT/'protocol.json')))
    display=['candidate','train_mean','pooled_mean','train_paired_delta','pooled_paired_delta','train_wins','pooled_wins','validation','chronological','qualifies']
    body='<p>Exact submitted70/15/15 recipe, with target-input access as the only intentional change. Two arms: Ani-only transduction, and transduction in all components. The unchanged inductive control reuses frozen predictions. Every model uses the same eight grouped five-fold repeats in each pool. Evaluation labels are erased before fitting; target embeddings are deliberately available in this separate transductive track.</p>'
    body+=table[display].to_html(index=False,float_format=lambda x:f'{x:.5f}',na_rep='Not shortlisted',border=0)
    body+='<p>Validation and chronology follow the CV shortlist. If neither arm qualifies, the full-transductive arm is evaluated as a diagnostic only. Chronological transduction uses the current later target batch as unlabeled inputs; it is a different access regime from ordinary forward prediction. Batch sizes in CV differ from the312-row competition target. All labels and folds have informed development before.</p>'
    body+='<p>Final target predictions use235 verified labels,1014 original masked rows and312 unlabeled test inputs. A prediction change is not evidence of improved Kaggle accuracy. No upload occurred; the inductive submitted score remains0.89401.</p>'+pd.DataFrame(final)[['mode','changed_test_label_sets','submission_created','kaggle_score']].to_html(index=False,border=0)
    body+='<p><a href="protocol.json">Protocol</a> · <a href="boundary_audit.json">Parity and label-isolation audit</a> · <a href="verification.json">Verification</a> · <a href="summary.json">All results</a></p>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Three-way transductive experiment</title><style>body{font:16px system-ui;max-width:1700px;margin:40px auto;padding:0 24px}table{border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.6}</style><h1>Frozen three-way ensemble: transductive experiment</h1>'+body)
    print(table[display].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audit','train','pooled','shortlist','supplementary','final','report']);args=p.parse_args()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        if args.stage in ['train','pooled']:run(args.stage)
        else:globals()[args.stage]()
