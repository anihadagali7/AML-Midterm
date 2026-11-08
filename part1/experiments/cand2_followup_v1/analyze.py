"""Independent metric arithmetic, fixed-blend evaluation and descriptive rank audit."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run import HERE,P1,LABELS,read,save,sha,freeze


def measure(y,p):
    row=2*(y*p).sum(1)/(y.sum(1)+p.sum(1))
    tp=(y*p).sum(0);fp=((1-y)*p).sum(0);fn=(y*(1-p)).sum(0)
    div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    pr,re,f=div(tp,tp+fp),div(tp,tp+fn),div(2*tp,2*tp+fp+fn)
    return dict(sample_f1=float(row.mean()),macro_f1=float(f.mean())),pr,re,f,y.sum(0)


def independent(output,current_only=False):
    freeze();groups={};predicted_rows=0;case_count=0
    for c in read(HERE/'case_plan.json'):
        current=c['access']=='strict_inductive'
        if current_only and not current:continue
        path=(HERE/'inputs' if current else output)/(c['key']+'.npz')
        if not current:assert sha(path)==read(path.with_suffix('.json'))['sha256']
        with np.load(path) as z:a={k:z[k].copy() for k in z.files}
        assert a['ids'].tolist()==c['query_ids']
        if current:
            s=(4*a['anchored_scores']+a['cand2_scores'])/5
            pred=(s>=.3).astype(int);pred[np.arange(len(s)),s.argmax(1)]=1
            a['blend_predictions']=pred
        else:
            np.testing.assert_allclose(a['blend_scores'],(4*a['anchored_scores']+a['cand2_scores'])/5,atol=1e-15,rtol=0)
            s=a['blend_scores'];p=(s>=.3).astype(int);p[np.arange(len(p)),s.argmax(1)]=1
            np.testing.assert_array_equal(p,a['blend_predictions'])
        kind=c['kind'];rep=c.get('repeat')
        if rep is None:rep=c['key']
        track='current' if current else 'J'
        groups.setdefault((track,kind,str(rep)),[]).append(a)
        case_count+=1
    metrics=[];labels=[];complementarity=[]
    for (track,kind,rep),parts in groups.items():
        y=np.concatenate([p['truth'] for p in parts]);ids=np.concatenate([p['ids'] for p in parts])
        assert len(ids)==len(set(ids))
        if kind in ['A','train','pooled']:assert len(parts)==5
        anchor=np.concatenate([p['anchored_predictions'] for p in parts])
        af=2*(y*anchor).sum(1)/(y.sum(1)+anchor.sum(1))
        for name in ['anchored','cand2','blend']:
            p=np.concatenate([a[name+'_predictions'] for a in parts]);assert np.isin(p,[0,1]).all() and (p.sum(1)>0).all()
            m,pr,re,f,support=measure(y,p)
            metrics.append(dict(track=track,kind=kind,repeat=rep,recipe=name,n=len(y),**m));predicted_rows+=len(y)
            for j,label in enumerate(LABELS):labels.append(dict(track=track,kind=kind,repeat=rep,recipe=name,label=label,precision=pr[j],recall=re[j],f1=f[j],support=int(support[j])))
            rf=2*(y*p).sum(1)/(y.sum(1)+p.sum(1))
            if name!='anchored':complementarity.append(dict(track=track,kind=kind,repeat=rep,recipe=name,
                changed=int(np.any(p!=anchor,axis=1).sum()),improve=int((rf>af+1e-12).sum()),worsen=int((rf<af-1e-12).sum()),
                exact_fixes=int(((p==y).all(1)&~(anchor==y).all(1)).sum()),exact_breaks=int((~(p==y).all(1)&(anchor==y).all(1)).sum())))
    metrics=pd.DataFrame(metrics);labels=pd.DataFrame(labels)
    summary=[]
    for (track,kind,name),g in metrics.groupby(['track','kind','recipe']):
        parent=metrics[(metrics.track==track)&(metrics.kind==kind)&(metrics.recipe=='anchored')].set_index('repeat')
        delta=g.sample_f1.to_numpy()-parent.loc[g['repeat'],'sample_f1'].to_numpy()
        # Chronology pools held-out rows across non-overlapping windows, unlike J's equal-window B.
        mean=np.average(g.sample_f1,weights=g.n) if kind=='chronology' else g.sample_f1.mean()
        summary.append(dict(track=track,kind=kind,recipe=name,repeats=len(g),sample_f1_mean=float(mean),
            population_sd=float(g.sample_f1.std(ddof=0)),paired_delta_mean=float(np.average(delta,weights=g.n) if kind=='chronology' else delta.mean()),
            wins=int((delta>1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum())))
    summary=pd.DataFrame(summary)
    dest=output/('current_only' if current_only else 'analysis');dest.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(dest/'repeat_metrics.csv',index=False);labels.to_csv(dest/'label_metrics.csv',index=False)
    summary.to_csv(dest/'summary.csv',index=False);pd.DataFrame(complementarity).to_csv(dest/'complementarity.csv',index=False)
    old=pd.read_csv(HERE/'prior_strict_comparison.csv').set_index('model')
    for name in ['anchored','cand2']:
        for kind in ['train','pooled','validation','chronology']:
            value=summary[(summary.track=='current')&(summary.recipe==name)&(summary.kind==kind)].sample_f1_mean.iloc[0]
            np.testing.assert_allclose(value,old.loc[name,kind],atol=1e-14,rtol=0)
    check=dict(independent_metric_arithmetic=True,prior_current_metrics_reproduced=True,cases=case_count,prediction_rows=predicted_rows)
    if not current_only:
        fresh=pd.read_csv(output/'repeat_metrics.csv')
        for row in metrics.itertuples():
            # Runner uses year keys for B; compare those in sorted order below instead.
            if row.kind in ['B','chronology']:continue
            got=fresh[(fresh.track==row.track)&(fresh.kind==row.kind)&(fresh.recipe==row.recipe)&(fresh['repeat'].astype(str)==row.repeat)]
            assert len(got)==1,(row,got)
            np.testing.assert_allclose([row.sample_f1,row.macro_f1],got[['sample_f1','macro_f1']].to_numpy()[0],atol=1e-14,rtol=0)
        for track,kind in [('J','B'),('current','chronology')]:
            for name in ['anchored','cand2','blend']:
                g=metrics[(metrics.track==track)&(metrics.kind==kind)&(metrics.recipe==name)]
                got=fresh[(fresh.track==track)&(fresh.kind==kind)&(fresh.recipe==name)]
                np.testing.assert_allclose(np.sort(g.sample_f1),np.sort(got.sample_f1),atol=1e-14,rtol=0)
        fresh_labels=pd.read_csv(output/'label_metrics.csv')
        for (track,kind,name,label),g in labels.groupby(['track','kind','recipe','label']):
            other=fresh_labels[(fresh_labels.track==track)&(fresh_labels.kind==kind)&(fresh_labels.recipe==name)&(fresh_labels.label==label)]
            assert len(g)==len(other)
            for metric in ['precision','recall','f1','support']:
                np.testing.assert_allclose(np.sort(g[metric]),np.sort(other[metric]),atol=1e-14,rtol=0)
        js=[]
        for name in ['anchored','cand2','blend']:
            s=summary[(summary.track=='J')&(summary.recipe==name)].set_index('kind').sample_f1_mean
            js.append(dict(recipe=name,A=s['A'],B=s['B'],J=(s['A']+s['B'])/2,validation=s['validation']))
        pd.DataFrame(js).to_csv(dest/'matched_J.csv',index=False)
        historical=pd.read_csv(HERE/'historical_comparison.csv')
        np.testing.assert_allclose(js[0]['J'],historical.set_index('model').loc['anchored','dylan_J'],atol=1e-14,rtol=0)
        cand2=dict(model='cand2_inductive5',kaggle=read(HERE/'cand2_reported_result.json')['reported_kaggle_score'],
            dylan_J=js[1]['J'],grouped_train=old.loc['cand2','train'],grouped_pooled=old.loc['cand2','pooled'])
        panel=pd.concat([historical,pd.DataFrame([cand2])],ignore_index=True)
        panel.to_csv(dest/'expanded_panel.csv',index=False)
        pairs=[];ranks=[]
        for cleaner in [False,True]:
            df=panel[panel.model!='original_ani'] if cleaner else panel
            for metric in ['dylan_J','grouped_train','grouped_pooled']:
                correct=wrong=tie=0
                for (_,a),(_,b) in combinations(df.iterrows(),2):
                    if abs(a.kaggle-b.kaggle)<1e-10:continue
                    sg=np.sign(a[metric]-b[metric]);truth=np.sign(a.kaggle-b.kaggle)
                    status='tie' if sg==0 else ('correct' if sg==truth else 'wrong')
                    correct+=status=='correct';wrong+=status=='wrong';tie+=status=='tie'
                    pairs.append(dict(exclude_original_ani=cleaner,metric=metric,model_a=a.model,model_b=b.model,status=status))
                ranks.append(dict(exclude_original_ani=cleaner,metric=metric,correct=correct,wrong=wrong,tied=tie,total=correct+wrong+tie))
        pd.DataFrame(ranks).to_csv(dest/'rank_agreement.csv',index=False);pd.DataFrame(pairs).to_csv(dest/'pairwise_orderings.csv',index=False)
        check['fresh_J_parent_reproduced']=True
        check['independent_vs_runner_metrics']=True
        print(pd.DataFrame(js).to_string(index=False));print(pd.DataFrame(ranks).to_string(index=False))
    save(dest/'verification.json',check)
    print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--current-only',action='store_true');args=p.parse_args()
    independent(args.output,args.current_only)
