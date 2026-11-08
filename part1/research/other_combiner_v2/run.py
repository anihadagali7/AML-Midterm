"""Frozen full comparison of three research directions against the .89935 blend."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

HERE=Path(__file__).resolve().parent
P1=HERE.parents[1];ROOT=P1.parent
HELPER=P1/'research/anchored_transductive_v1'
sys.path[:0]=[str(HERE/'source'),str(P1),str(HELPER),str(HERE)]
import numpy as np
import pandas as pd
import sklearn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score,precision_recall_fscore_support
from threadpoolctl import threadpool_limits
from anchored_solution import AnchoredReasonClassifier,select_reasons
from p1lib.data import Data,LABELS,EMB
from p1lib import cache
from p1lib.head import apply_head
import strict_adapter as strict
from study_models import (OtherSpecialists,PseudoWeightedAnchored,WeightedReasonLDA,
                          learned_heads,centered_logit,decide,offset_fit)
spec=importlib.util.spec_from_file_location('benchmark_helper',HELPER/'run.py')
benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
CANDIDATES=['other_rbf','other_conditional_rbf','pseudo_lda_quarter','pseudo_lda_confidence',
            'combiner_per_label','combiner_other_only','calibration_control']
NAMES=['parent','anchored']+CANDIDATES
INNER_SEED=270201


def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def idhash(v):return hashlib.sha256('\n'.join(v).encode()).hexdigest()
def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def cases():
    source=read(HERE/'source_cases.json')
    result=benchmark.cases()
    cols=['ID','start_year']+EMB+LABELS
    for c in result:
        ref=next(x for x in source if x['query_ids']==c['query'].ID.tolist()) if c['pool']=='supplementary' else next(
            x for x in source if x['kind']==c['pool'] and x['repeat']==c['seed'] and x['fold']==c['fold'])
        assert ref['fit_ids']==c['fit'].ID.tolist()
        assert ref['query_ids']==c['query'].ID.tolist()
        c['source_key']=ref['key']
        c['fit']=c['fit'][cols].reset_index(drop=True)
        c['query']=c['query'][cols].reset_index(drop=True)
    return result


def sources():
    paths=[HERE/n for n in ['run.py','study_models.py','strict_adapter.py','cand2_manifest.json',
        'parent_inputs.npz','parent_reported_result.json','source_cases.json','source_provenance.json']]
    paths+=list((HERE/'source/p1lib').glob('*.py'))
    paths+=[HELPER/'run.py',HELPER/'models.py',P1/'anchored_solution.py',P1/'TESTING_POLICY.md']
    paths+=list((P1/'data').glob('*.csv'))
    paths+=[P1/'experiments/current_leaders_grouped_v1/snapshots'/f'{name}.json'
            for name in ['train3','train5','pooled3','pooled5','chronology']]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def register():
    p=HERE/'protocol.json';assert not p.exists()
    schedule=cases();assert len(schedule)==84
    save(p,dict(version='other-combiner-v2',registered_utc=datetime.now(timezone.utc).isoformat(),
        base_main='766d16d3bc74248cb7d8f4c1e7240094ebd5a55e',
        parent='80% anchored /20% strict five-seed Cand2, user-reported Kaggle .89935; original anchored .89636 secondary control.',
        authorization='User requested all three proposed directions, prioritizing Other and learned combination.',
        observed_parent_diagnostic='On reused pooled OOF predictions: 26 Other-only positives per repeat, mean recall .7259615; nine Recruitment+Other positives, mean recall .0277778. This diagnostic informed the two fixed specialist hypotheses before scoring new recipes.',
        candidates={
            'other_rbf':'Gold-only Other vs rest SVC: standardized embeddings, RBF gamma1/150, C1, balanced classes. Its OOF margin plus intercept corrects parent Other logit only.',
            'other_conditional_rbf':'Two identical SVCs trained separately on fitting gold with/without Recruitment. Standardized inner-OOF margins gated by parent decision-centered Recruitment score; intercept and two margins correct Other only.',
            'pseudo_lda_quarter':'Anchored keeps LR, neighbor voting, cluster assistance and gold labels unchanged. At rounds1/2 use uniform-prior weighted shrinkage LDA, gold positive rows weight1, pseudo positive rows .25. Relearn pseudo labels each round, then blend altered anchored80/Cand2_unchanged20.',
            'pseudo_lda_confidence':'Same weighted LDA but each pseudo positive weight additionally multiplied by (teacher score for that reason / maximum teacher score for row)^2. Reduces uncertain secondary-label influence. Neighbors and LR still use original hard labels.',
            'combiner_per_label':'Separate regularized offset logistic correction per reason. Offset logit(clip(.8a+.2c))-logit(.3); features intercept, clipped centered anchored logit/3, clipped centered Cand2 logit/3. Zero correction exactly preserves parent.',
            'combiner_other_only':'Same fitted Other combiner, preserving the four other scores. Isolates targeted benefit from changes to stronger labels.',
            'calibration_control':'Offset logistic using intercept and clipped centered parent logit/3 for all reasons. Distinguishes recalibration from added component information.'},
        fixed_head='All corrections minimize summed binary log loss +20/2*sum(beta^2), penalizing intercept; no class weighting or threshold tuning. Predict centered score>=.5 plus argmax. Specialist margin scale learned only from inner OOF training rows, SD floor .1, clipping +/-4. Component logit features clip +/-6. No further settings search.',
        inner='Three iterative multilabel-stratified embedding-group folds, seed270201, inside each outer fitting partition. Refit preprocessing, anchored pseudo-labeling, five-seed Cand2 (including its own nested temperature fitting), and SVCs from scratch. Purge masked copies of inner-held groups. No cached outer predictions used to train the heads.',
        cand2='Exact strict adapter/source/config from cde3d500. Seeds0..4; average combo probabilities; convert to label marginals. Original manifest native transductive flags are overridden. Fixed existing eight-combination alphabet, no query label frequency used.',
        schedule='All seven fixed candidates on all 80 saved grouped folds plus validation and three chronological windows. No post-screen candidate selection; full eight-repeat comparison for every arm.',
        supplementary_exception='All candidates receive validation/chronology even if CV fails, to answer the three requested hypotheses.',
        access='Inductive, same saved gold/masked fitting IDs/order and outer purges. Train157 excludes validation-source rows. Query truth is removed from every model context. Original validation preserves historical no-outer-purge behavior. No test inputs loaded during benchmarking.',
        advancement='Positive mean sample-F1 delta and >=5/8 wins in BOTH pools against .89935 parent. Also report against anchored. Keep validation and each chronology window visible. No automatic replacement or upload. If any qualify, rank by minimum of two CV deltas then their sum, inspect supplements and prepare a final235 CSV only for a candidate with no >.005 supplementary regression.',
        scoring='Complete OOF sample F1 per repeat; population SD; paired repeat deltas and wins/ties/losses; macro F1; per-label precision/recall/F1/support. Report Other-only vs Recruitment+Other recall and false positives as diagnostics, not selection thresholds.',
        controls='Fresh anchored84 must match retained scores (atol1e-7) and predictions exactly. First-case Cand2 refit must match retained scores (atol1e-7). Pseudo control subclass equals anchored; weighted LDA at unit weights agrees with ordinary shrinkage LDA. Independent metric arithmetic and ID/group/coverage checks.',
        limitation='All labels and benchmark splits are reused development data. Only nine supplied Recruitment+Other examples. A local gain is not an independent estimate or guaranteed Kaggle gain. No J-weight fitting and no hidden-label inference.',
        retention='Keep source/config, all summaries, failure notes and verification. Keep detailed leader/parent and useful diagnostic predictions; discard bulky unsuccessful inner caches after verification. No fitted models persisted.',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        sources=sources()))
    print('Registered seven fixed candidates and 84 outer cases.',flush=True)


def guard():assert read(HERE/'protocol.json')['sources']==sources(),'Frozen sources changed'


def cand2(fit,query):
    df=pd.concat([fit,query],ignore_index=True).copy()
    df.loc[len(fit):,LABELS]=-1
    Y=df[LABELS].to_numpy(int,copy=True);Y[Y<0]=-1
    combos=np.array([1,2,4,5,8,10,16,24])
    CM=((combos[:,None]>>np.arange(5))&1).astype(int)
    known=(Y>=0).all(1);y=np.full(len(df),-1)
    bit=(Y[known]*(1<<np.arange(5))).sum(1)
    assert np.isin(bit,combos).all();y[known]=np.searchsorted(combos,bit)
    tr=np.flatnonzero(known);po=np.flatnonzero((Y[:len(fit)]<0).all(1));qu=np.arange(len(fit),len(df))
    D=Data(df=df,emb=df[EMB].to_numpy(float),Y=Y,y=y,combos=combos,CM=CM,
        idx=dict(train=np.arange(len(fit)),lab=tr,unl=po,val=np.array([],int),
                 test=np.array([],int),labval=tr),year=df.start_year.to_numpy(int))
    cfg=read(HERE/'cand2_manifest.json')['config']
    ps=[];priors=[];temps=[];n=0
    for seed in range(5):
        cache.clear();strict.AUDIT.clear()
        ctx=strict.context(D,tr,qu,po,seed=seed)
        assert (ctx.D.Y[qu]<0).all()
        result=strict.run_model(cfg['model'],ctx,qu)
        ps.append(result['P']);priors.append(result['prior']);temps.append(float(result['T']))
        assert all(a['gold_group_overlap']==a['masked_group_overlap']==0 for a in strict.AUDIT)
        n+=len(strict.AUDIT)
    _,Q,_=apply_head(np.mean(ps,axis=0),np.mean(priors,axis=0),CM,cfg['head'])
    cache.clear();strict.AUDIT.clear()
    return Q@CM,dict(temperatures=temps,internal_group_safe_contexts=n)


def inner_inputs(fit):
    gold=fit[fit[LABELS].ge(0).all(axis=1)].reset_index(drop=True)
    gg=benchmark.groups(gold);fg=benchmark.groups(fit)
    _,inv=np.unique(gg,return_inverse=True)
    target=np.zeros((inv.max()+1,5),int);np.maximum.at(target,inv,gold[LABELS].to_numpy(int))
    splitter=MultilabelStratifiedKFold(n_splits=3,shuffle=True,random_state=INNER_SEED)
    for fold,(tr,te) in enumerate(splitter.split(np.zeros((len(target),1)),target)):
        b=np.flatnonzero(np.isin(inv,te));query=gold.iloc[b].reset_index(drop=True)
        part=fit[~np.isin(fg,gg[b])].reset_index(drop=True)
        assert not set(benchmark.groups(part))&set(benchmark.groups(query))
        assert set(part.ID).isdisjoint(query.ID)
        yield fold,b,part,query


def refs(c):
    z=loadnp(HERE/'parent_inputs.npz');keep=z['case']==c['source_key']
    ids=z['ids'][keep];assert ids.tolist()==c['query'].ID.tolist()
    np.testing.assert_array_equal(z['truth'][keep],c['query'][LABELS])
    return z['anchored'][keep],z['cand2'][keep]


def evaluate(c,output):
    start=time.monotonic();fit,query=c['fit'],c['query']
    # Model functions receive query inputs only; actual labels stay with evaluator.
    clean=query.copy();clean.loc[:,LABELS]=-100
    x,y,qx=fit[EMB].to_numpy(float),fit[LABELS].to_numpy(int),clean[EMB].to_numpy(float)
    a_ref,cc=refs(c);audit={}
    with threadpool_limits(limits=1):
        parent=AnchoredReasonClassifier().fit(x,y);a=parent.predict_proba(qx)
        np.testing.assert_allclose(a,a_ref,atol=1e-7,rtol=0)
        np.testing.assert_array_equal(select_reasons(a),select_reasons(a_ref))
        audit['parent_max_error']=float(np.max(np.abs(a-a_ref)))
        if c['key']=='train_20261126_0':
            fresh,ca=cand2(fit,clean)
            np.testing.assert_allclose(fresh,cc,atol=1e-7,rtol=0)
            audit['cand2_control_max_error']=float(np.max(np.abs(fresh-cc)))
            control=PseudoWeightedAnchored('control').fit(x,y).predict_proba(qx)
            np.testing.assert_allclose(control,a,atol=1e-10,rtol=0)
            audit['pseudo_subclass_control_max_error']=float(np.max(np.abs(control-a)))
        scores=dict(parent=expit_safe(.8*a_ref+.2*cc),anchored=expit_safe(a_ref))
        for name in ['pseudo_lda_quarter','pseudo_lda_confidence']:
            m=PseudoWeightedAnchored(name).fit(x,y)
            scores[name]=expit_safe(.8*m.predict_proba(qx)+.2*cc)
            audit[name]=m.round_audit_
        gold=fit[fit[LABELS].ge(0).all(axis=1)].reset_index(drop=True)
        oa=np.zeros((len(gold),5));oc=np.zeros_like(oa);oh=np.zeros((len(gold),3));coverage=np.zeros(len(gold),int)
        inner=[]
        for fold,b,part,held in inner_inputs(fit):
            heldclean=held.copy();heldclean.loc[:,LABELS]=-100
            xx,yy,hx=part[EMB].to_numpy(float),part[LABELS].to_numpy(int),heldclean[EMB].to_numpy(float)
            oa[b]=AnchoredReasonClassifier().fit(xx,yy).predict_proba(hx)
            oc[b],ca=cand2(part,heldclean)
            oh[b]=OtherSpecialists().fit(xx,yy).predict(hx);coverage[b]+=1
            inner.append(dict(fold=fold,train_ids=part.ID.tolist(),query_ids=held.ID.tolist(),
                              group_overlap=0,**ca))
        assert (coverage==1).all()
        qh=OtherSpecialists().fit(x,y).predict(qx)
        heads,coef=learned_heads(oa,oc,oh,a_ref,cc,qh,gold[LABELS].to_numpy(int))
        scores.update(heads)
        audit['inner']=inner;audit['coefficients']=coef
        audit['inner_gold_ids']=gold.ID.tolist()
    S=np.stack([scores[n] for n in NAMES]);P=np.stack([decide(s) for s in S])
    assert np.isfinite(S).all() and (P.sum(2)>0).all()
    np.testing.assert_array_equal(P[0],select_reasons(.8*a_ref+.2*cc))
    for name in ['other_rbf','other_conditional_rbf','combiner_other_only']:
        np.testing.assert_array_equal(S[NAMES.index(name),:,:4],S[0,:,:4])
    packet=dict(ids=query.ID.to_numpy(str),truth=query[LABELS].to_numpy(int),scores=S,predictions=P,
                names=np.array(NAMES),inner_gold_ids=gold.ID.to_numpy(str),inner_anchored=oa,inner_cand2=oc,
                inner_specialists=oh,outer_specialists=qh)
    dest=Path(output)/'predictions'/(c['key']+'.npz');np.savez_compressed(dest,**packet)
    audit.update(key=c['key'],pool=c['pool'],seed=c['seed'],fold=c['fold'],source_key=c['source_key'],
        fitting_ids_sha256=idhash(fit.ID),query_ids_sha256=idhash(query.ID),gold=len(gold),
        masked=len(fit)-len(gold),seconds=time.monotonic()-start,sha256=sha(dest))
    save(dest.with_suffix('.json'),audit)
    return dict(key=c['key'],seconds=round(audit['seconds'],1))


def expit_safe(s):
    from scipy.special import expit
    return expit(centered_logit(s))


def measure(y,p):
    pr,re,f,s=precision_recall_fscore_support(y,p,zero_division=0)
    return dict(sample_f1=float(f1_score(y,p,average='samples')),macro_f1=float(f.mean())),[
        dict(label=label,precision=float(pr[j]),recall=float(re[j]),f1=float(f[j]),support=int(s[j]))
        for j,label in enumerate(LABELS)]


def collect(output,partial=False):
    output=Path(output);packets={};audits=[]
    for c in cases():
        p=output/'predictions'/(c['key']+'.npz')
        if not p.exists():
            assert partial
            continue
        z=loadnp(p);a=read(p.with_suffix('.json'));assert sha(p)==a['sha256']
        np.testing.assert_array_equal(z['ids'],c['query'].ID)
        np.testing.assert_array_equal(z['truth'],c['query'][LABELS])
        for j in range(len(NAMES)):np.testing.assert_array_equal(z['predictions'][j],decide(z['scores'][j]))
        key=(c['pool'],str(c['seed'])) if c['pool']!='supplementary' else (c['key'],'-1')
        packets.setdefault(key,[]).append(z);audits.append(a)
    rows=[];labels=[];diagnostics=[]
    for (pool,seed),parts in packets.items():
        if pool in ['train','pooled'] and len(parts)!=5:continue
        y=np.concatenate([p['truth'] for p in parts]);ids=np.concatenate([p['ids'] for p in parts])
        assert len(set(ids))==len(ids)
        for j,name in enumerate(NAMES):
            pred=np.concatenate([p['predictions'][j] for p in parts]);m,l=measure(y,pred)
            base=dict(pool=pool,seed=int(seed),recipe=name,n=len(y))
            rows.append(dict(**base,**m));labels.extend(dict(**base,**v) for v in l)
            for label,keep in [('other_alone',(y[:,4]==1)&(y.sum(1)==1)),
                               ('other_with_recruitment',(y[:,4]==1)&(y[:,3]==1)),
                               ('not_other',y[:,4]==0)]:
                diagnostics.append(dict(**base,subgroup=label,support=int(keep.sum()),
                    other_predicted=int(pred[keep,4].sum())))
    chrono=[p for (pool,_),parts in packets.items() if pool.startswith('through_') for p in parts]
    if len(chrono)==3:
        y=np.concatenate([p['truth'] for p in chrono]);assert len(y)==98
        for j,name in enumerate(NAMES):
            pred=np.concatenate([p['predictions'][j] for p in chrono]);m,l=measure(y,pred)
            base=dict(pool='chronology98',seed=-1,recipe=name,n=len(y))
            rows.append(dict(**base,**m));labels.extend(dict(**base,**v) for v in l)
    results=output/'summary';results.mkdir(exist_ok=True)
    d=pd.DataFrame(rows);ld=pd.DataFrame(labels)
    d.to_csv(results/'repeat_scores.csv',index=False);ld.to_csv(results/'per_label_scores.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(results/'other_subgroups.csv',index=False)
    s=d.groupby(['pool','recipe']).agg(mean_sample_f1=('sample_f1','mean'),
        population_sd=('sample_f1',lambda x:x.std(ddof=0)),mean_macro_f1=('macro_f1','mean'),repeats=('seed','count')).reset_index()
    s.to_csv(results/'overview.csv',index=False)
    pairs=[];deltas=[];qualification={name:True for name in CANDIDATES}
    for pool in ['train','pooled']:
        pivot=d[d.pool==pool].pivot(index='seed',columns='recipe',values='sample_f1')
        for control in ['parent','anchored']:
            for name in CANDIDATES:
                v=pivot[name]-pivot[control];wins=int((v>1e-12).sum());ties=int((v.abs()<=1e-12).sum())
                passing=bool(len(v)==8 and v.mean()>1e-12 and wins>=5)
                if control=='parent':qualification[name]&=passing
                pairs.append(dict(pool=pool,recipe=name,control=control,mean_delta=float(v.mean()),wins=wins,
                    ties=ties,losses=len(v)-wins-ties,repeats=len(v),passes=passing))
                deltas.extend(dict(pool=pool,recipe=name,control=control,seed=int(k),delta=float(x)) for k,x in v.items())
    pd.DataFrame(pairs).to_csv(results/'paired_summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(results/'paired_repeat_deltas.csv',index=False)
    save(results/'qualification.json',qualification)
    save(results/'fit_summary.json',dict(cases=len(audits),seconds=sum(a['seconds'] for a in audits),
        parent_max_error=max(a['parent_max_error'] for a in audits),
        cand2_control=[a['cand2_control_max_error'] for a in audits if 'cand2_control_max_error' in a],
        nested_contexts=sum(v['internal_group_safe_contexts'] for a in audits for v in a['inner'])))
    print(s[s.pool.isin(['train','pooled','validation','chronology98'])].to_string(index=False),flush=True)
    return d,ld


def run(output,workers):
    guard();output=Path(output);(output/'predictions').mkdir(parents=True,exist_ok=True)
    plan=cases();todo=[c for c in plan if not (output/'predictions'/(c['key']+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs=[executor.submit(evaluate,c,str(output)) for c in todo]
        for i,f in enumerate(as_completed(jobs),1):print(json.dumps(dict(done=i,total=len(todo),**f.result())),flush=True)
    guard();collect(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','collect'])
    p.add_argument('--output',type=Path,default=HERE/'results');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--partial',action='store_true');args=p.parse_args()
    if args.command=='register':register()
    elif args.command=='run':run(args.output,args.workers)
    else:guard();collect(args.output,args.partial)
