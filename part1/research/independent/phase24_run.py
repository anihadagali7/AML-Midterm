"""Frozen grouped-CV screen for the two user-requested learning approaches."""
from pathlib import Path
import json,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import phase24 as m,experiment as e

E=Path('experiments')
BASE=json.loads((E/'phase24-baseline.json').read_text())['candidate']['config']
WEIGHTS=dict(np.load(E/'phase21-weights.npz'))
W=np.array([WEIGHTS[k]/WEIGHTS[k].sum() for k in ['propensity_logistic','propensity_tree','kmm']])
RULES=['t0.45','t0.5','t0.55','t0.65'];SCOPES=['all','other','recruitment_other']

def configurations():
    cs=[]
    views=[['criteria'],['drugs','drug_targets','drug_actions','diseases_cleaned'],
           ['criteria','drugs','drug_targets','drug_actions','diseases_cleaned']]
    for blocks in views:
        for agreement in [False,True]:
            for weight in [.1,.3]:
                for vote in [0.,.2]:
                    cs.append(dict(model='co_training',views=blocks,agreement=agreement,pseudo_weight=weight,metadata_vote=vote))
        cs.append(dict(model='co_training',views=blocks,no_exchange=True,metadata_vote=0.))
        cs.append(dict(model='co_training',views=blocks,no_exchange=True,metadata_vote=1.))
    for pca in [25,50]:
        for components in [2,5,10]:
            for weight in [.15,.4,.7]:
                cs.append(dict(model='metric_ssl',metric='nca',metric_pca=pca,components=components,metric_weight=weight))
        for subtypes in [1,2,3]:
            for weight in [.15,.4,.7]:
                cs.append(dict(model='metric_ssl',metric='fisher',metric_pca=pca,subtypes=subtypes,metric_weight=weight))
    return cs

def load_predictions(c,seed,tag):
    r=e.evaluate(c,seed=seed,tag=tag)
    if r['status']!='ok':return None
    return np.load(E/f'{tag}-{e.config_id(c)}-{seed}.npz',allow_pickle=True)['p']

def score(h):
    row=2*(h*e.Y).sum(1)/(h.sum(1)+e.Y.sum(1));ordinary=float(row.mean())
    weighted=W@row
    return dict(ordinary=ordinary,robust=float(.75*weighted.min()+.25*ordinary),weighted=weighted.tolist())

def select_policy(c,p,b):
    choices=[]
    for rule in RULES:
        for scope in SCOPES:
            s=score(m.apply_policy(p,b,rule,scope))
            pc=dict(model='policy24',learner=c,baseline=BASE,decision_rule=rule,scope=scope)
            choices.append(dict(config=pc,**s))
    return choices

def run():
    cs=configurations();e.save_json(E/'phase24-configs.json',cs)
    # Numerical equivalence when the learned metric has zero weight.
    a,b=e.make_splits(seed=20261114)[0];q=e.DATA.iloc[e.LAB[b]];ex=e.GROUP[e.LAB[b]]
    with threadpool_limits(limits=1):
        p=m.legacy_predict(a,q,BASE['base'],ex)
        z=m.metric_ssl(a,q,dict(model='metric_ssl',metric='nca',metric_weight=0.),ex)
    delta=float(abs(p-z).max());assert delta<1e-5,delta
    e.save_json(E/'phase24-equivalence.json',dict(original_metric_max_difference=delta,passed=True))
    control=load_predictions(BASE,20261114,'phase24_control')
    screen=[]
    for i,c in enumerate(cs):
        print('SCREEN',i+1,len(cs),c,flush=True)
        p=load_predictions(c,20261114,'phase24')
        if p is None:continue
        choices=select_policy(c,p,control)
        screen.extend(choices);e.save_json(E/'phase24-screen.json',screen)
        e.save_json(E/f'phase24-diagnostics-{e.config_id(c)}.json',m.LAST_DIAGNOSTICS)
    short={}
    for family in ['co_training','metric_ssl']:
        pool=[r for r in screen if r['config']['learner']['model']==family and not r['config']['learner'].get('no_exchange')]
        for metric in ['ordinary','robust']:
            r=max(pool,key=lambda r:(r[metric],r['ordinary'],r['config']['scope']=='other'))
            short[e.config_id(r['config'])]=r
    e.save_json(E/'phase24-shortlist-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=list(short.values()),repeat_seeds=[20261115,20261116,20261117]))
    baseline=[]
    for seed in [20261115,20261116,20261117]:baseline.append(score(load_predictions(BASE,seed,'phase24_control')))
    repeats=[]
    for r in short.values():
        ss=[]
        for seed in [20261115,20261116,20261117]:
            print('REPEAT',r['config']['learner'],seed,flush=True)
            p=load_predictions(r['config'],seed,'phase24_repeat')
            if p is None:break
            ss.append(score(p))
        if len(ss)<3:continue
        out={**r,'ordinary_cv':float(np.mean([s['ordinary'] for s in ss])),
            'robust_cv':float(np.mean([s['robust'] for s in ss])),
            'cv_std':float(np.std([s['ordinary'] for s in ss])),
            'robust_std':float(np.std([s['robust'] for s in ss])),
            'rule':'t0.5','per_repeat':ss}
        out['selection_score']=out['ordinary_cv']-.2*out['cv_std']
        repeats.append(out);e.save_json(E/'phase24-repeat-results.json',repeats)
    # Report both requested approaches, even if they lose to the baseline.
    # Within each family only the repeated-CV winner is evaluated on validation.
    selected=[]
    for family in ['co_training','metric_ssl']:
        pool=[r for r in repeats if r['config']['learner']['model']==family]
        if pool:
            best=max(pool,key=lambda r:(r['selection_score'],r['robust_cv']))
            selected.append({**best,'name':'phase24_'+family})
    e.save_json(E/'phase24-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_cv=baseline,candidates=selected))
    V=e.pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
    with threadpool_limits(limits=1):ps=[e.predict(np.arange(len(e.Y)),V,r['config']) for r in selected]
    np.savez_compressed(E/'phase24-validation-predictions.npz',p=ps)
    y=e.pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
    for r,p in zip(selected,ps):
        out={**r,'validation':e.metrics(y,p,r['rule'])};results.append(out)
        print('VALIDATION',json.dumps(out),flush=True)
    e.save_json(E/'phase24-validation-results.json',results)
    print('COMPLETE',len(results),'validation finalists',flush=True)

if __name__=='__main__':run()
