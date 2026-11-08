"""Five predeclared refinements of the user-reported .89935 blend; no new fitting."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import f1_score,precision_recall_fscore_support

HERE=Path(__file__).resolve().parent;P1=HERE.parents[1]
LABELS=['Safety','Funding','Efficacy','Recruitment','Other']
CANDIDATES=['cand2_10','cand2_30','cand2_40','parent_lda10','parent_lda25']
RECIPES=['anchored','parent','lda_control']+CANDIDATES

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())

def freeze():
    files=[Path(__file__),HERE/'evaluation_inputs.npz',HERE/'final_inputs.npz',HERE/'parent_scored.csv',HERE/'parent_reported_result.json',HERE/'provenance.json']
    hashes={f.name:sha(f) for f in files}
    p=HERE/'protocol.json'
    if p.exists():assert read(p)['hashes']==hashes
    else:save(p,dict(created_utc=datetime.now(timezone.utc).isoformat(),parent='80% anchored /20% Cand2, user-reported .89935',
        secondary_control='Original anchored .89636 and previously submitted LDA ensemble .89401',
        candidates=CANDIDATES,recipes=dict(cand2_10='90% anchored +10% Cand2, native .3 decoder',cand2_30='70% anchored +30% Cand2, native .3 decoder',cand2_40='60% anchored +40% Cand2, native .3 decoder',parent_lda10='90% centered current80/20 blend +10% existing centered LDA70/15/15 ensemble, .5 decoder',parent_lda25='75% centered current80/20 blend +25% existing centered LDA70/15/15 ensemble, .5 decoder'),
        centering='expit(logit(clip(score,1e-9,1-1e-9))-logit(.3)); nonempty argmax fallback for every decoder',
        evaluation='All saved grouped eight-repeat five-fold plans in both pools, original validation and all three chronology windows; reuse exact component predictions. No new classifier fitting.',
        supplementary_exception='All five fixed candidates receive supplementary checks as an explicitly declared diagnostic.',
        promotion_rule='Positive means and >=5/8 paired wins in BOTH pools against .89935 parent; keep supplementary regressions visible.',
        exploratory_selection='Exclude candidates losing >.005 sample F1 on original validation or pooled chronology if any survive. Rank remaining by maximum minimum(train_delta,pooled_delta), then sum of those two deltas, then registration order. If none survive, rank all and disclose regression. Skip a candidate only if final label predictions exactly duplicate the scored parent. Export one; no automatic promotion.',
        access='All component fits inductive on eligible training inputs; final components trained on 235 gold +1014 masked rows. Test inputs prediction only. IDs only alignment.',
        limitation='Selected after the parent Kaggle outcome; small fixed search on heavily reused local development folds. Not an untouched estimate. No J tuning or hidden-label inference.',hashes=hashes))

def scores(name,a,c,l):
    parent=.8*a+.2*c
    if name=='anchored':return a,.3
    if name=='parent':return parent,.3
    if name=='lda_control':return l,.5
    if name.startswith('cand2_'):
        w=int(name.split('_')[1])/100;return (1-w)*a+w*c,.3
    w=int(name.split('lda')[1])/100
    centered=expit(logit(np.clip(parent,1e-9,1-1e-9))-logit(.3))
    return (1-w)*centered+w*l,.5

def decide(s,t):
    p=(s>=t).astype(int);p[np.arange(len(p)),s.argmax(1)]=1;return p

def frame(ids,p):return pd.DataFrame(dict(ID=ids,Predicted=[' '.join(l for l,v in zip(LABELS,row) if v) for row in p]))

def main():
    freeze();z=np.load(HERE/'evaluation_inputs.npz');rows=[];labels=[];windows=[];preds={}
    for name in RECIPES:
        s,t=scores(name,z['anchored'],z['cand2'],z['lda']);p=decide(s,t);preds[name]=p
        assert np.isfinite(s).all() and (p.sum(1)>0).all()
        for pool in np.unique(z['pool']):
            for seed in np.unique(z['seed'][z['pool']==pool]):
                ix=(z['pool']==pool)&(z['seed']==seed);y=z['truth'][ix];pr=p[ix]
                assert len(set(z['ids'][ix]))==len(y)
                assert len(y)==dict(train=157,pooled=235,validation=78,chronology=98)[pool]
                value=f1_score(y,pr,average='samples');manual=float((2*(y*pr).sum(1)/(y.sum(1)+pr.sum(1))).mean());assert abs(value-manual)<1e-14
                rows.append(dict(pool=pool,seed=int(seed),recipe=name,n=len(y),sample_f1=value,macro_f1=f1_score(y,pr,average='macro',zero_division=0)))
                precision,recall,f,support=precision_recall_fscore_support(y,pr,zero_division=0)
                for j,label in enumerate(LABELS):labels.append(dict(pool=pool,seed=int(seed),recipe=name,label=label,precision=precision[j],recall=recall[j],f1=f[j],support=int(support[j])))
        for case in np.unique(z['case'][z['pool']=='chronology']):
            ix=z['case']==case;windows.append(dict(case=case,recipe=name,n=int(ix.sum()),sample_f1=f1_score(z['truth'][ix],p[ix],average='samples')))
    repeats=pd.DataFrame(rows);summary=[];paired=[]
    for (pool,name),g in repeats.groupby(['pool','recipe']):
        parent=repeats[(repeats.pool==pool)&(repeats.recipe=='parent')].set_index('seed')
        d=g.sample_f1.to_numpy()-parent.loc[g.seed,'sample_f1'].to_numpy()
        summary.append(dict(pool=pool,recipe=name,repeats=len(g),mean=g.sample_f1.mean(),population_sd=g.sample_f1.std(ddof=0),macro_f1=g.macro_f1.mean(),delta=float(d.mean()),wins=int((d>1e-12).sum()),ties=int((abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum())))
        paired.extend(dict(pool=pool,recipe=name,seed=int(seed),delta=float(v)) for seed,v in zip(g.seed,d))
    summary=pd.DataFrame(summary);r=HERE/'results';r.mkdir(exist_ok=True)
    repeats.to_csv(r/'repeat_scores.csv',index=False);summary.to_csv(r/'summary.csv',index=False);pd.DataFrame(labels).to_csv(r/'per_label_scores.csv',index=False);pd.DataFrame(windows).to_csv(r/'chronology_windows.csv',index=False);pd.DataFrame(paired).to_csv(r/'paired_repeat_deltas.csv',index=False)
    expected=dict(anchored=[.9320594479830149,.9213652482269503,.8910256410256411,.8843537414965985],parent=[.935244161358811,.9216312056737589,.8910256410256411,.8877551020408163],lda_control=[.9317940552016984,.9216312056737589,.9059829059829061,.8775510204081632])
    for name,values in expected.items():
        got=summary[summary.recipe==name].set_index('pool').loc[['train','pooled','validation','chronology'],'mean'].to_numpy()
        np.testing.assert_allclose(got,values,atol=1e-12,rtol=0)
    ranking=[]
    for order,name in enumerate(CANDIDATES):
        q=summary[summary.recipe==name].set_index('pool');d=q.delta
        eligible=d['validation']>=-.005-1e-12 and d['chronology']>=-.005-1e-12
        qualifies=bool(((q.loc[['train','pooled'],'delta']>1e-12)&(q.loc[['train','pooled'],'wins']>=5)).all())
        ranking.append(dict(recipe=name,guardrail_passed=bool(eligible),cv_min_delta=float(min(d['train'],d['pooled'])),cv_total_delta=float(d['train']+d['pooled']),order=order,full_promotion_rule=qualifies))
    anyeligible=any(v['guardrail_passed'] for v in ranking)
    ranking.sort(key=lambda v:(-(int(v['guardrail_passed']) if anyeligible else 0),-v['cv_min_delta'],-v['cv_total_delta'],v['order']))
    final=np.load(HERE/'final_inputs.npz');parent=frame(final['ids'],decide(*scores('parent',final['anchored'],final['cand2'],final['lda'])))
    pd.testing.assert_frame_equal(parent,pd.read_csv(HERE/'parent_scored.csv'))
    assert sha(HERE/'parent_scored.csv')==read(HERE/'parent_reported_result.json')['submission_sha256']
    sample=pd.read_csv(P1/'data/part1_sample_submission.csv');assert parent.ID.tolist()==sample.ID.tolist()
    selected=None
    for candidate in ranking:
        name=candidate['recipe'];s,t=scores(name,final['anchored'],final['cand2'],final['lda']);p=decide(s,t);out=frame(final['ids'],p)
        changed=int((out.Predicted!=parent.Predicted).sum());candidate['final_changed_rows_vs_parent']=changed
        if changed and selected is None:
            selected=dict(candidate,changed_vs_parent=changed);chosen_s=s;chosen_p=p;chosen_frame=out
    assert selected is not None
    dest=P1/'submissions/candidates'/('refinement_'+selected['recipe']+'.csv');dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists();chosen_frame.to_csv(dest,index=False,lineterminator='\n')
    np.savez_compressed(HERE/'selected_final_predictions.npz',ids=final['ids'],scores=chosen_s,predictions=chosen_p)
    np.savez_compressed(r/'selected_evaluation_predictions.npz',ids=z['ids'],truth=z['truth'],pool=z['pool'],seed=z['seed'],case=z['case'],predictions=preds[selected['recipe']],parent_predictions=preds['parent'])
    save(r/'selection.json',dict(ranking=ranking,selected=selected))
    save(HERE/'prepared_submission.json',dict(status='prepared; no upload or Kaggle score',selected=selected,csv=str(dest.relative_to(P1)),sha256=sha(dest),rows=312,gold=235,masked=1014,test_inputs_used_for_fitting=False,no_new_component_fits=True,parent_scored_csv_reproduced=True,protocol_sha256=sha(HERE/'protocol.json')))
    freeze();print(summary.pivot(index='recipe',columns='pool',values='mean').to_string());print(json.dumps(selected,indent=2));print(dest)

if __name__=='__main__':main()
