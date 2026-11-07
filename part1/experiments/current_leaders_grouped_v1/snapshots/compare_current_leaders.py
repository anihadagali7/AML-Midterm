"""Frozen cross-agent comparison on the shared grouped 157/235 protocols.

No tuning, final-data refitting, new test predictions, or uploads. Existing
competitors are independently rescored; our candidate and Ani control refit.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score,precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from anchored_solution import AnchoredReasonClassifier
from paired_models import PairedModel,prepare,ANI

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/current_leaders_grouped_v1'
OTHER=ROOT.parent.parent/'AML-Part1-NewDirections'
INTEGRATION=ROOT.parent.parent/'AML-Midterm-integration/part1'
EXP=INTEGRATION/'experiments'
REASONS=['Safety','Funding','Efficacy','Recruitment','Other']
FEATURES=[f'emb_{i}' for i in range(150)]
SOURCES={}
MODELS=['original_ani','current_ani_lda','previous_ensemble','lda_ensemble',
        'recent_cutoff_ensemble','xrfm20_blend','kernel15_blend','anchored_candidate']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
def source(p):SOURCES[str(p)]=sha(p);return Path(p)
def read(p):return json.loads(source(p).read_text())
def load(p):
    with np.load(source(p),allow_pickle=True) as z:return {k:z[k] for k in z.files}
def groups(df):return pd.util.hash_pandas_object(df[FEATURES].round(10),index=False).to_numpy()
def metric(y,p):
    precision,recall,f1,support=precision_recall_fscore_support(y,p,zero_division=0)
    return dict(sample_f1=float(f1_score(y,p,average='samples',zero_division=0)),macro_f1=float(f1.mean()),
                **{f'{kind}_{r}':float(values[j]) for kind,values in [('precision',precision),('recall',recall),('f1',f1),('support',support)] for j,r in enumerate(REASONS)})
def join(parts):
    for p in parts[1:]:
        np.testing.assert_array_equal(parts[0]['ids'],p['ids'])
        np.testing.assert_array_equal(parts[0]['truth'],p['truth'])
    return dict(ids=parts[0]['ids'].astype(str),truth=parts[0]['truth'],
                scores=np.concatenate([p['scores'] for p in parts]),predictions=np.concatenate([p['predictions'] for p in parts]))
def references(pool,seeds):
    models={}
    for name,prefix in [('original_ani','ani'),('previous_ensemble','threeway')]:
        models[name]=join([load(OTHER/'references'/f'{prefix}_{pool}_{suffix}.npz') for suffix in ['first','extra']])
    for name,stage,recipe in [('current_ani_lda','ani_lda_revision_cv_v1','ani__revision_full'),
                              ('lda_ensemble','ani_lda_revision_cv_v1','ensemble__revision_full'),
                              ('recent_cutoff_ensemble','nested_decision_lda_cv_v1','ensemble__global_recent')]:
        models[name]=join([load(EXP/stage/f'{pool}{n}'/f'{recipe}.npz') for n in [3,5]])
    for name,version,recipe in [('xrfm20_blend','v10','hybrid_leaf_one_raw'),('kernel15_blend','v5','rfm_complement_all15')]:
        parts=[load(OTHER/'studies'/version/'results'/recipe/pool/f'{seed}.npz') for seed in seeds]
        for p in parts:p['scores']=p['scores'][None];p['predictions']=p['predictions'][None]
        models[name]=join(parts)
    return models


def main():
    OUT.mkdir(exist_ok=True);(OUT/'snapshots').mkdir(exist_ok=True)
    train=pd.read_csv(source(ROOT/'data/part1_train.csv'));valid=pd.read_csv(source(ROOT/'data/part1_val.csv'))
    for p in [OTHER/'data',INTEGRATION/'data']:
        for filename in ['part1_train.csv','part1_val.csv']:assert sha(p/filename)==sha(ROOT/'data'/filename)
    for p in [ROOT/'anchored_solution.py',ROOT/'paired_models.py',Path(__file__)]:
        source(p);dest=OUT/'snapshots'/p.name
        if dest.exists():assert sha(dest)==sha(p)
        else:shutil.copyfile(p,dest)
    plans={};refs={}
    for pool in ['train','pooled']:
        parts=[read(OTHER/'protocols'/f'{pool}{n}.json') for n in [3,5]]
        assert parts[0]['gold_ids']==parts[1]['gold_ids']
        plans[pool]=dict(gold_ids=parts[0]['gold_ids'],folds=parts[0]['folds']+parts[1]['folds'],seeds=parts[0]['seeds']+parts[1]['seeds'])
        for n in [3,5]:shutil.copyfile(OTHER/'protocols'/f'{pool}{n}.json',OUT/'snapshots'/f'{pool}{n}.json')
        refs[pool]=references(pool,plans[pool]['seeds'])
    chrono=read(OTHER/'protocols/chronology.json')
    shutil.copyfile(OTHER/'protocols/chronology.json',OUT/'snapshots/chronology.json')
    protocol=dict(version='current-leaders-grouped-v1',created_at=datetime.now(timezone.utc).isoformat(),models=MODELS,
                  purpose='Compare the unchanged, now submitted anchored candidate with the latest other-agent leaders.',
                  schedule='All 8 saved repeats x 5 folds in BOTH pools, then original validation and 3 saved chronological windows; no screening or tuning.',
                  access='Inductive; purge masked duplicates of held-out embedding groups; no test-input fitting.',
                  scoring='Complete OOF sample F1 per repeat; mean and population SD; paired repeat wins. Chronology pools 98 held-out rows.',
                  evidence='Refit anchored candidate and current Ani LDA control on every fold. Rescore frozen competitor predictions after exact ID/truth checks.',
                  selection='No recipe selection, threshold change, promotion, new final fit or upload.',sources_before_fit=SOURCES.copy())
    if (OUT/'protocol.json').exists():
        old=read(OUT/'protocol.json');assert old['sources_before_fit']==protocol['sources_before_fit']
    else:save(OUT/'protocol.json',protocol)
    rows=[];audits=[];all_predictions=[];control_checks=[]
    full=pd.concat([train,valid],ignore_index=True)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for pool in ['train','pooled']:
            frame=train if pool=='train' else full
            gold=frame[frame[REASONS].ge(0).all(axis=1)].reset_index(drop=True)
            masked=train[train[REASONS].eq(-100).all(axis=1)].reset_index(drop=True)
            plan=plans[pool];assert gold.ID.tolist()==plan['gold_ids']
            y=gold[REASONS].to_numpy(int);gg,ug=groups(gold),groups(masked)
            models=refs[pool];scores=np.full((8,len(gold),5),np.nan);predictions=np.full((8,len(gold),5),-1,int)
            for name,z in models.items():
                np.testing.assert_array_equal(z['ids'],gold.ID)
                np.testing.assert_array_equal(z['truth'],y)
                assert z['predictions'].shape==(8,len(gold),5)
                np.savez_compressed(OUT/'snapshots'/f'{name}_{pool}.npz',**z)
            order={id:i for i,id in enumerate(frame.ID)}
            for ri,seed in enumerate(plan['seeds']):
                coverage=np.zeros(len(gold),int)
                for fold in [f for f in plan['folds'] if f['seed']==seed]:
                    a=np.array(fold['train_gold_positions']);b=np.array(fold['heldout_gold_positions'])
                    assert set(a).isdisjoint(b) and len(a)+len(b)==len(y)
                    assert set(gg[a]).isdisjoint(gg[b]) and gold.iloc[b].ID.tolist()==fold['heldout_ids']
                    keep=~np.isin(ug,gg[b]);assert (~keep).sum()==fold['masked_duplicates_excluded']
                    fit=pd.concat([gold.iloc[a],masked[keep]],ignore_index=True)
                    fit=fit.iloc[np.argsort([order[id] for id in fit.ID])].reset_index(drop=True)
                    assert set(groups(fit)).isdisjoint(gg[b]) and set(fit.ID).isdisjoint(gold.iloc[b].ID)
                    if pool=='train':assert set(fit.ID).isdisjoint(valid.ID)
                    model=AnchoredReasonClassifier().fit(fit[FEATURES],fit[REASONS])
                    scores[ri,b]=model.predict_proba(gold.iloc[b][FEATURES]);predictions[ri,b]=model.predict(gold.iloc[b][FEATURES]);coverage[b]+=1
                    fit_known=fit[REASONS].ge(0).all(axis=1).to_numpy()
                    x=fit[FEATURES].to_numpy();labels=fit[REASONS].to_numpy(int)
                    ctrl=PairedModel(ANI).fit(x[fit_known],labels[fit_known],x[~fit_known],prepare(x))
                    raw=ctrl.predict_proba(gold.iloc[b][FEATURES].to_numpy())
                    centered=expit(logit(np.clip(raw,1e-9,1-1e-9))-logit(.3))
                    old=models['current_ani_lda']['scores'][ri,b]
                    np.testing.assert_allclose(centered,old,atol=1e-7,rtol=0)
                    np.testing.assert_array_equal(ctrl.predict(gold.iloc[b][FEATURES].to_numpy()),models['current_ani_lda']['predictions'][ri,b])
                    control_checks.append(dict(pool=pool,seed=seed,fold=fold['fold'],max_score_error=float(abs(centered-old).max()),identical_predictions=True))
                    audits.append(dict(pool=pool,seed=seed,fold=fold['fold'],gold_ids=gold.iloc[a].ID.tolist(),masked_ids=masked[keep].ID.tolist(),
                                       held_ids=gold.iloc[b].ID.tolist(),masked_purged=int((~keep).sum()),fit_audit=model.round_audit_))
                assert (coverage==1).all()
                print(pool,seed,metric(y,predictions[ri])['sample_f1'],flush=True)
            models['anchored_candidate']=dict(scores=scores,predictions=predictions,ids=gold.ID.to_numpy(str),truth=y)
            np.savez_compressed(OUT/f'anchored_{pool}.npz',**models['anchored_candidate'])
            for name,z in models.items():
                for ri,seed in enumerate(plan['seeds']):
                    rows.append(dict(pool=pool,name=name,seed=seed,**metric(y,z['predictions'][ri])))
                    out=pd.DataFrame(dict(ID=gold.ID,name=name,pool=pool,seed=seed))
                    for j,r in enumerate(REASONS):out[f'actual_{r}']=y[:,j];out[f'predicted_{r}']=z['predictions'][ri,:,j]
                    all_predictions.append(out)
        diagnostic_rows=[];diagnostic_predictions=[]
        cases=[('validation',train,valid)]
        indexed=full.set_index('ID',drop=False)
        for w in chrono['chronological_splits']:
            fit=indexed.loc[w['training_ids']];q=indexed.loc[w['heldout_ids']]
            assert fit[REASONS].ge(0).all(axis=1).sum()==w['gold_rows']
            assert fit[REASONS].eq(-100).all(axis=1).sum()==w['masked_rows']
            assert len(q)==w['heldout_rows'] and set(groups(fit)).isdisjoint(groups(q))
            cases.append((w['name'],fit,q))
        for wi,(case,fit,q) in enumerate(cases):
            model=AnchoredReasonClassifier().fit(fit[FEATURES],fit[REASONS]);p=model.predict(q[FEATURES])
            models={'anchored_candidate':dict(ids=q.ID.to_numpy(str),truth=q[REASONS].to_numpy(int),predictions=p)}
            for name,stage,recipe in [('original_ani','ani_lda_revision_cv_v1','ani'),('current_ani_lda','ani_lda_revision_cv_v1','ani__revision_full'),
                                      ('previous_ensemble','ani_lda_revision_cv_v1','ensemble'),('lda_ensemble','ani_lda_revision_cv_v1','ensemble__revision_full'),
                                      ('recent_cutoff_ensemble','nested_decision_lda_cv_v1','ensemble__global_recent')]:
                models[name]=load(EXP/stage/'supplementary'/f'window{wi}_{recipe}.npz')
            for name,version,recipe in [('xrfm20_blend','v10','hybrid_leaf_one_raw'),('kernel15_blend','v5','rfm_complement_all15')]:
                models[name]=load(OTHER/'studies'/version/'supplementary'/recipe/f'{case}.npz')
            for name,z in models.items():
                lookup={id:i for i,id in enumerate(z['ids'])};idx=[lookup[id] for id in q.ID]
                np.testing.assert_array_equal(z['truth'][idx],q[REASONS])
                pred=z['predictions'][idx]
                diagnostic_rows.append(dict(name=name,case=case,**metric(q[REASONS].to_numpy(int),pred)))
                out=pd.DataFrame(dict(ID=q.ID.to_numpy(),name=name,case=case))
                for j,r in enumerate(REASONS):out[f'actual_{r}']=q[r].to_numpy();out[f'predicted_{r}']=pred[:,j]
                diagnostic_predictions.append(out)
            print('diagnostic',case,metric(q[REASONS].to_numpy(int),p)['sample_f1'],flush=True)
    repeats=pd.DataFrame(rows);repeats.to_csv(OUT/'repeat_scores.csv',index=False)
    summary=repeats.groupby(['pool','name']).agg(mean_f1=('sample_f1','mean'),sd=('sample_f1',lambda x:np.std(x,ddof=0)),macro_f1=('macro_f1','mean'))
    pairs=[]
    for pool in ['train','pooled']:
        w=repeats[repeats.pool==pool].pivot(index='seed',columns='name',values='sample_f1')
        for name in w.columns:
            d=w.anchored_candidate-w[name]
            pairs.append(dict(pool=pool,reference=name,anchored_delta=d.mean(),wins=int((d>1e-12).sum()),ties=int((abs(d)<1e-12).sum()),losses=int((d< -1e-12).sum()),repeat_deltas=d.tolist()))
    summary.to_csv(OUT/'cv_summary.csv');pd.DataFrame(pairs).to_csv(OUT/'paired_vs_anchored.csv',index=False)
    pd.concat(all_predictions).to_csv(OUT/'cv_predictions.csv',index=False)
    dp=pd.concat(diagnostic_predictions,ignore_index=True);dp.to_csv(OUT/'diagnostic_predictions.csv',index=False)
    for name,g in dp[dp.case!='validation'].groupby('name'):
        assert len(g)==98
        diagnostic_rows.append(dict(name=name,case='chronology98',**metric(g[[f'actual_{r}' for r in REASONS]].to_numpy(),g[[f'predicted_{r}' for r in REASONS]].to_numpy())))
    ds=pd.DataFrame(diagnostic_rows);ds.to_csv(OUT/'diagnostic_scores.csv',index=False)
    comparison=summary.reset_index().pivot(index='name',columns='pool',values='mean_f1')
    for case in ['validation','chronology98']:comparison[case]=ds[ds.case==case].set_index('name').sample_f1
    comparison['reported_kaggle']=pd.Series({'anchored_candidate':.89636,'original_ani':.89423,'previous_ensemble':.89401,'lda_ensemble':.89401})
    comparison.to_csv(OUT/'comparison.csv')
    save(OUT/'fit_audits.json',audits);save(OUT/'control_checks.json',control_checks)
    for p,h in SOURCES.items():assert sha(p)==h,p
    save(OUT/'provenance.json',dict(sources=SOURCES,anchored_cv_fits=80,current_ani_control_fits=80,additional_anchored_fits=4,
                                  no_recipe_changes=True,no_new_test_predictions=True,all_source_hashes_unchanged=True,
                                  original_ani_score_access='Reported .89423 belongs to the original transductive CSV; local rows use its matched inductive adaptation.'))
    print(comparison.to_string(),flush=True)
    print(pd.DataFrame(pairs).query("reference=='lda_ensemble'").to_string(index=False),flush=True)


if __name__=='__main__':main()
