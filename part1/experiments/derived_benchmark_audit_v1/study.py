"""Retrospective audit of Dylan v2's fixed derived benchmark against scored leaders.

No model/metric tuning. All original sources and scored CSVs are read-only.
Run from repository root: .venv/bin/python part1/experiments/derived_benchmark_audit_v1/study.py
"""
import os
for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[var]='1'
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PART=HERE.parents[1]
REPO=PART.parent
ARCHIVE=PART/'research/lda_ensemble'
HISTORY=PART/'experiments/current_leaders_grouped_v1'
DYLAN=REPO.parent/'dylan/dylan-model-v2'
LABELS=['Safety','Funding','Efficacy','Recruitment','Other']
EMB=[f'emb_{i}' for i in range(150)]
MODELS=['dbscan','tri_training','original_ani','previous_ensemble','lda_ensemble','anchored']
SOURCE_MODULE=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def groups(frame):return pd.util.hash_pandas_object(frame[EMB].round(10),index=False).to_numpy()
def row_f1(y,p):return 2*(y*p).sum(axis=1)/(y.sum(axis=1)+p.sum(axis=1))
def decide(p):
    pred=(p>=.5).astype(int);empty=pred.sum(1)==0
    pred[np.flatnonzero(empty),p[empty].argmax(1)]=1
    return pred
def metrics(y,p):
    pr,re,f,s=precision_recall_fscore_support(y,p,zero_division=0)
    return dict(sample_f1=float(row_f1(y,p).mean()),macro_f1=float(f.mean()),
                **{f'{k}_{l}':float(v[j]) for k,v in [('precision',pr),('recall',re),('f1',f),('support',s)] for j,l in enumerate(LABELS)})


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def source_module():
    global SOURCE_MODULE
    if SOURCE_MODULE is None:
        sys.path.insert(0,str(PART))
        direct=import_file('historical_direct_models',ARCHIVE/'experiments/ani_direct_comparison_v1/models.py')
        anchor=import_file('submitted_anchored',PART/'anchored_solution.py')
        SOURCE_MODULE=(direct,anchor)
    return SOURCE_MODULE


def predict(fit,query):
    """Query labels are removed before any source model is invoked."""
    m,anchor=source_module();q=query.drop(columns=LABELS).copy()
    packet=m.f.train_native(fit)
    ani=m.f.native_prediction(packet,q)
    lda,_,_=m.revision.fit_addition(packet,fit,q,'revision_full')
    tri,svm=m.partners.partners_in_frame(fit,q)
    known=fit[LABELS].ge(0).all(axis=1).to_numpy()
    dbscan=m.f.native.predict(fit,np.flatnonzero(known),np.flatnonzero(~known),q,'repo_dbscan',EMB,LABELS)
    model=anchor.AnchoredReasonClassifier().fit(fit[EMB],fit[LABELS])
    scores=dict(dbscan=dbscan,tri_training=tri,original_ani=ani,
                previous_ensemble=.7*ani+.15*tri+.15*svm,lda_ensemble=.7*lda+.15*tri+.15*svm)
    predictions={name:decide(p) for name,p in scores.items()}
    scores['anchored']=model.predict_proba(q[EMB]);predictions['anchored']=model.predict(q[EMB])
    for name in MODELS:
        assert predictions[name].shape==(len(q),5) and (predictions[name].sum(1)>0).all()
        assert np.isfinite(scores[name]).all()
    return scores,predictions


def ledger():
    records=[]
    mapping={'tri_training':'tri-training-trainval235','original_ani':'ani-original',
             'previous_ensemble':'three-way-trainval235','lda_ensemble':'lda-ensemble-trainval235','anchored':'anchored-trainval235'}
    for name,stem in mapping.items():
        p=PART/'submissions/results'/f'{stem}.json';r=read(p)
        csv=REPO/r['submission_file'];assert sha(csv)==r['submission_sha256']
        records.append(dict(name=name,kaggle=r['reported_kaggle_score'],labels=235,
                            csv=str(csv.relative_to(REPO)),sha256=sha(csv),record=str(p.relative_to(REPO)),
                            fitting_access=r['fitting_access'],score_source=r['score_provenance']))
    csv=PART/'submissions/archive/submitted/dbscan_logistic_trainval235__kaggle_0.86452.csv'
    r=read(PART/'experiments/final_dbscan_trainval235/submission_manifest.json')
    assert sha(csv)=='19f29c7bcbda974f8ae9646d3ced83bad2953ead45412e88731e8c12e55d2661'
    records.append(dict(name='dbscan',kaggle=.86452,labels=235,csv=str(csv.relative_to(REPO)),sha256=sha(csv),
                        record='part1/submissions/index.csv',fitting_access='inductive',score_source='Recorded user-reported score'))
    return sorted(records,key=lambda x:MODELS.index(x['name']))


def initialize():
    (HERE/'results').mkdir(exist_ok=True);(HERE/'source_snapshot').mkdir(exist_ok=True)
    # Small source/context snapshots make the supplied benchmark reproducible after relocation.
    supplied={'evaluate.py':'midterm/part1/p1lib/evaluate.py','data.py':'midterm/part1/p1lib/data.py',
              'context.py':'midterm/part1/p1lib/context.py','benchmark_run.py':'benchmark_michael/run.py',
              'benchmark_README.md':'benchmark_michael/README.md','benchmark_results.json':'benchmark_michael/results.json',
              'methodology.md':'docs/02_methodology.md','metrics_explainer.md':'docs/explainers/02_metrics_and_what_they_prove.md'}
    for dest,src in supplied.items():
        p=HERE/'source_snapshot'/dest
        if p.exists():
            if (DYLAN/src).exists():assert sha(p)==sha(DYLAN/src)
        else:shutil.copyfile(DYLAN/src,p)
    files=[Path(__file__),PART/'anchored_solution.py',PART/'TESTING_POLICY.md',PART/'RETENTION.md',PART/'submissions/index.csv']
    files+=list((PART/'submissions/results').glob('*.json'))+list((PART/'data').glob('*.csv'))
    files+=list((ARCHIVE/'research/independent').glob('*.py'))
    files+=list((ARCHIVE/'research/independent/experiments').glob('phase*-baseline.json'))
    files+=[ARCHIVE/'research/independent/experiments/cross_repo_benchmark_v1/registry.json',ARCHIVE/'research/independent/experiments/ani_improvement_v1/registry.json']
    for stage in ['feature_gap_cv_v1','ani_lda_revision_cv_v1','ani_direct_comparison_v1']:
        files+=list((ARCHIVE/'experiments'/stage).glob('*.py'))
    files+=list((HISTORY/'snapshots').glob('*.json'))+list((HISTORY/'snapshots').glob('*.npz'))
    files+=[HISTORY/'anchored_train.npz',HISTORY/'anchored_pooled.npz',HISTORY/'diagnostic_predictions.csv',HISTORY/'comparison.csv']
    files+=list((HERE/'source_snapshot').glob('*'))
    submitted=ledger();files += [REPO/x['csv'] for x in submitted]
    hashes={str(p.relative_to(REPO)):sha(p) for p in sorted(set(files))}
    if (HERE/'protocol.json').exists():
        assert read(HERE/'protocol.json')['source_hashes']==hashes
        return
    save(HERE/'protocol.json',dict(version='derived-benchmark-audit-v1',created_utc=datetime.now(timezone.utc).isoformat(),
         base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),models=MODELS,scored_artifacts=submitted,
         hypothesis='Does Dylan v2 J rank previously scored Part 1 recipes more consistently with their reported Kaggle scores than A, B, official validation and our current grouped metrics?',
         parent='Anchored 0.89636; this is a retrospective metric audit, not model development or promotion.',
         dylan_protocol='Exact supplied make_folds: A=50 ungrouped stratified folds R10 seed1 over235; B=mean of cuts2004/2005/2006 with all1014 masked rows; C=original157-to78. J=.5*A+.5*B. Models remain inductive.',
         canonical_protocol='All80 saved grouped folds plus original validation and three saved chronological windows. Freshly refit all six recipes; compare four leaders against retained exact predictions and DBSCAN/tri against retained aggregate results.',
         strict_B_diagnostic='Use the same three Dylan query cuts; purge fitting gold and masked duplicate embedding groups; allow only masked rows earlier than the cut. Report grouped pooled A plus strict B with the same .5/.5 weight.',
         derived_scores=['Dylan_J=(Dylan_A+Dylan_B)/2','aligned_J=(grouped_pooled_CV+strict_Dylan_B)/2','current_J_diagnostic=(grouped_pooled_CV+canonical_chronology98)/2'],
         retrospective_analysis='Spearman, Kendall tau-b, pairwise ordering with ties, named historical contrasts; leave-one-model-out, exclude-transductive-Ani and top-four sensitivities. Never tune J weights, chronology cutoffs or recipes to Kaggle.',
         uncertainty='Paired bootstrap of embedding groups with shared weights across all occurrences of a row/repeat/cut; accounts for A/B overlap conditional on fitted predictions. Report original SE approximation separately. No independent holdout or exact Kaggle uncertainty claim.',
         access_caveat='Original Ani Kaggle CSV was transductive. Primary local comparison uses its established inductive adapter and explicitly reports exclusion sensitivity. Dylan candidates have no recorded Kaggle score and are excluded from correlation analysis.',
         dbscan_refit_caveat='157-label DBSCAN scored .87500 while235 scored .86452. One fixed recipe evaluated on fixed folds cannot distinguish these final fits; use only235 in same-label-count primary panel and report this limitation separately.',
         scope='All fixed cases are diagnostic, including supplementary checks. User explicitly requested evaluation of the derived score; the repository default selection policy is unchanged.',
         retention='Keep compact per-fold scores/predictions and fitting IDs because they are the evidence for this matched metric audit; no fitted model caches or duplicated datasets.',
         versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','iterative-stratification','joblib']},
         source_hashes=hashes,no_new_submission=True))


def make_cases():
    tr=pd.read_csv(PART/'data/part1_train.csv');va=pd.read_csv(PART/'data/part1_val.csv')
    full=pd.concat([tr,va],ignore_index=True);gg=groups(full)
    lab=np.flatnonzero(tr[LABELS].ge(0).all(axis=1));unl=np.flatnonzero(tr[LABELS].eq(-100).all(axis=1))
    val=np.arange(len(tr),len(full));labval=np.r_[lab,val]
    masks=full[LABELS].clip(lower=0).to_numpy(int)@np.array([1,2,4,8,16]);keys=np.unique(masks[labval])
    combo=np.full(len(full),-1);combo[labval]=np.searchsorted(keys,masks[labval])
    D=SimpleNamespace(idx=dict(lab=lab,val=val,labval=labval,unl=unl),y=combo,year=full.start_year.to_numpy())
    tree=ast.parse((HERE/'source_snapshot/evaluate.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_folds')
    env=dict(np=np,RepeatedStratifiedKFold=RepeatedStratifiedKFold,CHRONO_CUTS=(2004,2005,2006))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'supplied_make_folds','exec'),env)
    cases=[]
    def add(key,track,kind,a,b,u,**kw):
        fitrows=np.sort(np.r_[a,u]);assert len(set(fitrows)&set(b))==0
        cases.append(dict(key=key,track=track,kind=kind,train_gold=full.iloc[a].ID.tolist(),
                          fit_ids=full.iloc[fitrows].ID.tolist(),query_ids=full.iloc[b].ID.tolist(),
                          masked_ids=full.iloc[u].ID.tolist(),gold_group_overlap=len(set(gg[a])&set(gg[b])),
                          masked_group_overlap=len(set(gg[u])&set(gg[b])),future_masked=int((full.iloc[u].start_year>=full.iloc[b].start_year.min()).sum()),**kw))
    for f in env['make_folds'](D,R=10,seed=1):
        if f['protocol']=='C':continue
        a,b=f['train'],f['eval']
        add('dylan_'+f['name'],'dylan',f['protocol'],a,b,unl,repeat=f.get('repeat'),cut=f.get('cut'))
        if f['protocol']=='B':
            keepg=~np.isin(gg[a],gg[b]);keepu=(full.iloc[unl].start_year.to_numpy()<f['cut'])&~np.isin(gg[unl],gg[b])
            add('strict_'+f['name'],'strict_B','B',a[keepg],b,unl[keepu],cut=f['cut'])
    for pool,gold in [('train',lab),('pooled',labval)]:
        for n in [3,5]:
            plan=read(HISTORY/'snapshots'/f'{pool}{n}.json');assert full.iloc[gold].ID.tolist()==plan['gold_ids']
            for f in plan['folds']:
                a,b=gold[np.array(f['train_gold_positions'])],gold[np.array(f['heldout_gold_positions'])]
                assert not set(gg[a])&set(gg[b]);u=unl[~np.isin(gg[unl],gg[b])]
                assert len(unl)-len(u)==f['masked_duplicates_excluded']
                add(f'grouped_{pool}_{f["seed"]}_{f["fold"]}','canonical',pool,a,b,u,repeat=f['seed'],fold=f['fold'])
    add('validation','canonical','validation',lab,val,unl)
    ix={v:i for i,v in enumerate(full.ID)}
    for w in read(HISTORY/'snapshots/chronology.json')['chronological_splits']:
        fit=np.array([ix[v] for v in w['training_ids']]);b=np.array([ix[v] for v in w['heldout_ids']])
        known=full.iloc[fit][LABELS].ge(0).all(axis=1).to_numpy()
        add(w['name'],'canonical','chronology',fit[known],b,fit[~known])
    if (HERE/'splits.json').exists():assert read(HERE/'splits.json')==cases
    else:save(HERE/'splits.json',cases)
    return full,cases


def worker(case):
    dest=HERE/'results'/case['key']
    if dest.with_suffix('.json').exists():
        r=read(dest.with_suffix('.json'));assert sha(dest.with_suffix('.npz'))==r['sha256'];return {'key':case['key'],'cached':True}
    start=time.time()
    try:
        full=pd.concat([pd.read_csv(PART/'data/part1_train.csv'),pd.read_csv(PART/'data/part1_val.csv')],ignore_index=True).set_index('ID',drop=False)
        fit=full.loc[case['fit_ids']].reset_index(drop=True);q=full.loc[case['query_ids']].reset_index(drop=True)
        with threadpool_limits(limits=1):scores,preds=predict(fit,q)
        arrays=dict(ids=q.ID.to_numpy(str),truth=q[LABELS].to_numpy(int))
        for name in MODELS:arrays[name+'_scores']=scores[name];arrays[name+'_predictions']=preds[name]
        np.savez_compressed(dest.with_suffix('.npz'),**arrays)
        result=dict(key=case['key'],seconds=time.time()-start,sha256=sha(dest.with_suffix('.npz')),
                    metrics={name:metrics(arrays['truth'],preds[name]) for name in MODELS})
        save(dest.with_suffix('.json'),result)
        return dict(key=case['key'],seconds=round(result['seconds'],2),anchored=result['metrics']['anchored']['sample_f1'])
    except Exception:
        save(dest.with_suffix('.failure.json'),dict(traceback=traceback.format_exc()));raise


def parity(cases):
    checks=[]
    for pool in ['train','pooled']:
        files={'anchored':HISTORY/f'anchored_{pool}.npz',**{n:HISTORY/'snapshots'/f'{n}_{pool}.npz' for n in ['original_ani','previous_ensemble','lda_ensemble']}}
        for name,path in files.items():
            with np.load(path,allow_pickle=True) as z:
                lookup={v:i for i,v in enumerate(z['ids'])};seeds=sorted({c['repeat'] for c in cases if c['kind']==pool})
                for c in [c for c in cases if c['kind']==pool]:
                    if not (HERE/'results'/f'{c["key"]}.npz').exists():continue
                    with np.load(HERE/'results'/f'{c["key"]}.npz') as out:
                        ii=[lookup[v] for v in out['ids']];ri=seeds.index(c['repeat'])
                        np.testing.assert_array_equal(out['truth'],z['truth'][ii])
                        np.testing.assert_array_equal(out[name+'_predictions'],z['predictions'][ri,ii])
                        expected=z['scores'][ri,ii]
                        # The Ani snapshot is raw p at cutoff .3; its ensemble adapter
                        # centers those same scores to cutoff .5. Predictions are identical.
                        if name=='original_ani':expected=expit(logit(np.clip(expected,1e-9,1-1e-9))-logit(.3))
                        err=float(abs(out[name+'_scores']-expected).max())
                        np.testing.assert_allclose(out[name+'_scores'],expected,atol=1e-7,rtol=0)
                        checks.append(dict(case=c['key'],model=name,identical_predictions=True,max_score_error=err))
    save(HERE/'control_replay_checks.json',checks)
    return checks


def summarize(cases):
    rows=[];perrows=[]
    for c in cases:
        with np.load(HERE/'results'/f'{c["key"]}.npz') as z:
            for name in MODELS:
                y,p=z['truth'],z[name+'_predictions']
                rows.append(dict(model=name,case=c['key'],track=c['track'],kind=c['kind'],repeat=c.get('repeat'),cut=c.get('cut'),**metrics(y,p)))
                for ident,v in zip(z['ids'],row_f1(y,p)):
                    perrows.append(dict(ID=ident,model=name,case=c['key'],track=c['track'],kind=c['kind'],repeat=c.get('repeat'),cut=c.get('cut'),f1=float(v)))
    pd.DataFrame(rows).to_csv(HERE/'case_scores.csv',index=False)
    rf=pd.DataFrame(perrows);rf.to_csv(HERE/'per_trial_f1.csv.gz',index=False)
    rep=[]
    for (name,track,kind,r),g in rf[rf.kind.isin(['A','train','pooled'])].groupby(['model','track','kind','repeat']):
        assert g.ID.is_unique
        rep.append(dict(model=name,track=track,kind=kind,repeat=int(r),sample_f1=g.f1.mean()))
    pd.DataFrame(rep).to_csv(HERE/'repeat_scores.csv',index=False)
    out=[]
    for name in MODELS:
        g=rf[rf.model==name]
        mean=lambda track,kind:float(g[(g.track==track)&(g.kind==kind)].f1.mean())
        A=mean('dylan','A');B=g[(g.track=='dylan')&(g.kind=='B')].groupby('cut').f1.mean().mean()
        strict=g[(g.track=='strict_B')&(g.kind=='B')].groupby('cut').f1.mean().mean()
        train,pooled,chrono,C=[mean('canonical',k) for k in ['train','pooled','chronology','validation']]
        B2005=mean('dylan','B') # replaced with the exact single cut below
        B2005=g[(g.track=='dylan')&(g.cut==2005)].f1.mean()
        out.append(dict(model=name,kaggle=next(x['kaggle'] for x in ledger() if x['name']==name),
                        dylan_A=A,dylan_B=B,dylan_B2005=B2005,dylan_J=.5*(A+B),validation=C,
                        grouped_train=train,grouped_pooled=pooled,chronology98=chrono,
                        strict_B=strict,aligned_J=.5*(pooled+strict),current_J_diagnostic=.5*(pooled+chrono)))
    comparison=pd.DataFrame(out);comparison.to_csv(HERE/'comparison.csv',index=False)
    checks=parity(cases);assert len(checks)==320
    supplied=read(HERE/'source_snapshot/benchmark_results.json')['michael_anchored']
    our=comparison.set_index('model').loc['anchored']
    for p in ['A','B','J']:
        expected=supplied[p] if p=='J' else supplied[p]['sample_f1']
        assert abs(our['dylan_'+p]-expected)<1e-12,(p,our['dylan_'+p],expected)
    for p,h in read(HERE/'protocol.json')['source_hashes'].items():assert sha(REPO/p)==h,p
    save(HERE/'completion.json',dict(cases=len(cases),recipe_evaluations=len(cases)*len(MODELS),
         anchored_supplied_A_B_J_reproduced=True,canonical_reference_fold_prediction_checks=len(checks),protected_hashes_unchanged=True))
    print(comparison.to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    initialize();_,cases=make_cases()
    if args.smoke:
        chosen=[next(c for c in cases if c['kind']==k) for k in ['train','pooled','validation']]
        for c in chosen:print(worker(c),flush=True)
        print('Parity',len(parity(chosen)),flush=True);return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(worker,c) for c in cases]
        for i,f in enumerate(as_completed(pending),1):print(i,len(pending),json.dumps(f.result()),flush=True)
    summarize(cases)


if __name__=='__main__':main()
