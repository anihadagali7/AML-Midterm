"""Separate repeated grouped CV on 157 training + 78 validation gold labels.

Reuses frozen recipes and scoring code, with an isolated in-memory data context.
No full-data refit, Kaggle prediction, or current-model replacement is performed.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse,json,sys
import numpy as np
from threadpoolctl import threadpool_limits
import benchmark_cv as b
e=b.e;ROOT=b.ROOT;OUT=ROOT/'experiments/pooled_benchmark_v1'

def clear_caches():
    for name,module in list(sys.modules.items()):
        if name=='experiment' or (name.startswith('phase') and name[5:].isdigit()):
            for key,value in vars(module).items():
                if 'CACHE' in key and isinstance(value,dict):value.clear()

@contextmanager
def pooled_context():
    keys=['DATA','LAB','UNLAB','Y','GROUP','make_splits'];original={k:getattr(e,k) for k in keys}
    old_out=b.OUT
    try:
        clear_caches()
        val=e.pd.read_csv(ROOT/'data/part1_val.csv')
        assert not set(e.DATA.ID)&set(val.ID)
        e.DATA=e.pd.concat([e.DATA,val],ignore_index=True)
        e.LAB=np.flatnonzero(e.DATA[e.LABELS].ge(0).all(axis=1))
        e.UNLAB=np.flatnonzero(e.DATA[e.LABELS].eq(-100).all(axis=1))
        e.Y=e.DATA.iloc[e.LAB][e.LABELS].to_numpy(int)
        e.GROUP=e.pd.util.hash_pandas_object(e.DATA[e.EMB].round(10),index=False).to_numpy()
        # The original function's default Y was bound at import; replace that
        # default for this context while preserving explicitly supplied inner folds.
        def make_splits(y=None,groups=None,seed=20261104,n_splits=5):
            if y is None:y=e.Y
            if groups is None:groups=e.GROUP[e.LAB]
            return original['make_splits'](y,groups,seed,n_splits)
        e.make_splits=make_splits;b.OUT=OUT
        assert len(e.Y)==235 and len(e.UNLAB)==1014 and len(e.DATA)==1249
        yield
    finally:
        clear_caches()
        for k,v in original.items():setattr(e,k,v)
        b.OUT=old_out

def protected_hashes():
    paths=list((ROOT/'data').glob('*.csv'))+[ROOT/'experiments/current-best.json']
    paths+=list((ROOT/'predictions/current_best').glob('*'))
    # The primary fold assignments and retained model are immutable here.
    paths += [ROOT/'experiments/benchmark_v1/protocol.json']
    return {str(p.relative_to(ROOT)):b.file_hash(p) for p in paths if p.is_file()}

def validate(p):
    assert p['training_sha256']==b.file_hash(ROOT/'data/part1_train.csv')
    assert p['validation_sha256']==b.file_hash(ROOT/'data/part1_val.csv')
    assert p['runner_sha256']==b.file_hash(Path(__file__))
    assert p['gold_ids']==e.DATA.iloc[e.LAB].ID.tolist() and p['label_order']==e.LABELS
    assert p['folds_sha256']==b.digest(p['folds']) and p['seeds']==b.SEEDS
    for seed in p['seeds']:
        folds=[f for f in p['folds'] if f['seed']==seed];assert len(folds)==5
        held=np.concatenate([f['heldout_gold_positions'] for f in folds])
        assert np.array_equal(np.sort(held),np.arange(len(e.Y)))
        for f in folds:
            a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
            assert len(np.unique(a))==len(a) and not set(a)&set(h)
            assert set(a)|set(h)==set(range(len(e.Y)))
            assert not set(e.GROUP[e.LAB[a]])&set(e.GROUP[e.LAB[h]])
            assert f['heldout_ids']==e.DATA.iloc[e.LAB[h]].ID.tolist()
            assert f['masked_duplicates_excluded']==int(np.isin(e.GROUP[e.UNLAB],e.GROUP[e.LAB[h]]).sum())

def protocol():
    path=OUT/'protocol.json'
    if path.exists():p=json.loads(path.read_text());validate(p);return p
    original_train_rows=len(e.pd.read_csv(ROOT/'data/part1_train.csv'))
    folds=[]
    for seed in b.SEEDS:
        for i,(a,h) in enumerate(e.make_splits(y=e.Y,groups=e.GROUP[e.LAB],seed=seed,n_splits=5)):
            folds.append(dict(seed=seed,fold=i,train_gold_positions=a.tolist(),heldout_gold_positions=h.tolist(),
                heldout_ids=e.DATA.iloc[e.LAB[h]].ID.tolist(),training_gold_rows=len(a),heldout_gold_rows=len(h),
                original_validation_rows_in_training=int(np.sum(e.LAB[a]>=original_train_rows)),
                original_validation_rows_heldout=int(np.sum(e.LAB[h]>=original_train_rows)),
                masked_duplicates_excluded=int(np.isin(e.GROUP[e.UNLAB],e.GROUP[e.LAB[h]]).sum())))
    p=dict(version='part1-pooled-cv-v1',created_at=b.now(),runner_sha256=b.file_hash(Path(__file__)),
        training_sha256=b.file_hash(ROOT/'data/part1_train.csv'),validation_sha256=b.file_hash(ROOT/'data/part1_val.csv'),
        gold_ids=e.DATA.iloc[e.LAB].ID.tolist(),label_order=e.LABELS,gold_rows=len(e.Y),
        gold_embedding_groups=int(len(np.unique(e.GROUP[e.LAB]))),masked_rows=len(e.UNLAB),
        seeds=b.SEEDS,folds_per_repeat=5,folds=folds,folds_sha256=b.digest(folds),
        purpose='Supplementary comparison of unchanged recipes trained with more labels; preserve training-only primary benchmark.',
        parameter_policy='Identical configs and frozen decoding rules to training-only registry. Refit learned parameters per fold. No new hyperparameter, feature, or threshold search.',
        fit_policy='Only outer-training gold (from either source) plus original masked training rows; exclude masked duplicates of held-out embedding groups. Held-out features and labels do not enter fitting. No Kaggle test inputs.',
        metric='Mean of three complete out-of-fold sample-wise F1 scores across all 235 gold rows; repeat std is split sensitivity, not a confidence interval.',
        validation_interpretation='Previously reported validation F1 applies to the original training-only fit. It is not an independent validation score for these pooled-fold models.',
        limitation='Prior model selection used both label sources, so pooled CV is a development estimate. Temporal population shift remains. It is not a third independent test.',
        final_all_label_refit=False,model_promotions=False)
    validate(p);b.save(path,p);return p

def registry():
    path=OUT/'registry.json'
    current=json.loads((ROOT/'experiments/benchmark_v1/registry.json').read_text())
    signature=lambda rs:[dict(name=r['name'],config=r['config'],rule=r['rule']) for r in rs]
    if path.exists():
        rs=json.loads(path.read_text());assert signature(rs)==signature(current),'Cohort changed; freeze a new comparison version.'
    else:rs=current;b.save(path,rs)
    assert not any(b.transductive(r['config']) for r in rs)
    return rs

def integrity(p,rs):
    path=OUT/'integrity.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['runner_sha256']==p['runner_sha256'];return old
    f=p['folds'][0];a=np.array(f['train_gold_positions']);h=np.array(f['heldout_gold_positions'])
    q=e.DATA.iloc[e.LAB[h]].drop(columns=e.LABELS);ex=e.GROUP[e.LAB[h]]
    # Check incumbent plus the nested calibrator, the two most important paths
    # affected by the enlarged gold pool. Other methods retain previous audits.
    chosen=[r for r in rs if r['incumbent'] or r['name']=='Label-frequency correction']
    checks=[]
    with threadpool_limits(limits=1):
        for r in chosen:
            clear_caches();pred=e.predict(a,q,r['config'],ex)
            y=e.Y.copy();raw=e.DATA.loc[e.LAB[h],e.LABELS].copy()
            try:
                e.Y[h]=1-e.Y[h];e.DATA.loc[e.LAB[h],e.LABELS]=1-raw.to_numpy()
                clear_caches();changed=e.predict(a,q,r['config'],ex)
            finally:
                e.Y[:]=y;e.DATA.loc[e.LAB[h],e.LABELS]=raw.to_numpy();clear_caches()
            one=e.predict(a,q.iloc[:1],r['config'],ex)
            assert np.allclose(pred,changed,atol=1e-9,rtol=0)
            assert np.allclose(pred[:1],one,atol=1e-9,rtol=0)
            checks.append(dict(model=r['name'],heldout_labels_cannot_change_predictions=True,query_batch_invariance=True))
    result=dict(passed=True,runner_sha256=p['runner_sha256'],pooled_gold_rows=235,gold_embedding_groups=222,
        no_conflicting_gold_embedding_groups=True,checks=checks,folds_verified=True,all_configs_and_rules_unchanged=True)
    b.save(path,result);clear_caches();return result

def compare(rs,outputs,p):
    base=outputs[next(i for i,r in enumerate(rs) if r['incumbent'])]
    train=e.pd.read_csv(ROOT/'experiments/benchmark_v1/comparison.csv').set_index('candidate')
    rows=[]
    for r,out in zip(rs,outputs):
        delta=np.array([x['sample_f1'] for x in out['repeat_scores']])-np.array([x['sample_f1'] for x in base['repeat_scores']])
        rows.append(dict(candidate=r['name'],incumbent=r['incumbent'],training_cv_mean=float(train.loc[r['name'],'cv_mean']),
            training_cv_std=float(train.loc[r['name'],'repeat_std']),pooled_cv_mean=out['cv_mean'],pooled_cv_std=out['repeat_std'],
            original_validation_f1=(r.get('recorded_validation') or {}).get('sample_f1'),
            pooled_paired_difference=float(delta.mean()),pooled_repeats_improved=int(np.sum(delta>1e-12)),
            pooled_result_key=out['key']))
    table=e.pd.DataFrame(rows)
    for column in ['training_cv_mean','pooled_cv_mean','original_validation_f1']:
        table[column+'_rank']=table[column].rank(method='min',ascending=False).astype(int)
    table.to_csv(OUT/'comparison.csv',index=False)
    b.save(OUT/'summary.json',dict(protocol=p['version'],protocol_sha256=b.digest(p),folds_sha256=p['folds_sha256'],
        updated_at=b.now(),model_count=len(rs),seeds=b.SEEDS,models=table.to_dict('records'),
        final_all_label_refit_performed=False,current_submission_replaced=False,
        original_validation_scores_reused=True,standalone_validation_evaluations_performed=0))
    print(table[['candidate','training_cv_mean','pooled_cv_mean','original_validation_f1']].sort_values('pooled_cv_mean',ascending=False).to_string(index=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');parser.add_argument('--integrity-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);before=protected_hashes();original_ids=e.DATA.ID.tolist()
    with pooled_context():
        p=protocol();rs=registry()
        if args.verify_only:print('Verified pooled protocol and unchanged recipe cohort.')
        elif args.integrity_only:print(json.dumps(integrity(p,rs),indent=2))
        else:
            integrity(p,rs);outputs=[b.evaluate(r,p) for r in rs];compare(rs,outputs,p)
    assert e.DATA.ID.tolist()==original_ids and len(e.Y)==157
    assert protected_hashes()==before,'A protected model, primary fold file, or input changed.'
    b.save(OUT/'preservation.json',dict(raw_inputs_and_current_model_unchanged=True,primary_folds_unchanged=True,
        original_engine_context_restored=True,protected_sha256=before))

if __name__=='__main__':main()
