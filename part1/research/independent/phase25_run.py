"""Training-only selection and a frozen validation batch for three refinements."""
from pathlib import Path
import json,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import phase25 as m,experiment as e

E=Path('experiments');BASE=json.loads((E/'phase25-baseline.json').read_text())['candidate']['config']
W0=dict(np.load(E/'phase21-weights.npz'))
W=np.array([W0[k]/W0[k].sum() for k in ['propensity_logistic','propensity_tree','kmm']])
RULES=['t0.45','t0.5','t0.55','t0.65'];SCOPES=['all','other','recruitment_other','rare']
FAMILIES=['label_tuned_ssl','reliable_ssl','local_ssl']

def configs():
    cs=[dict(model='label_tuned_ssl',selection=mode,penalty=penalty,inner_splits=5)
        for mode in ['sample','binary'] for penalty in [0,.005,.02]]
    for teacher in ['supervised','ssl']:
        for weight in [.03,.1,.3]:
            for kind in ['per_label','pooled']:
                for floor in [.5,.8]:
                    cs.append(dict(model='reliable_ssl',teacher=teacher,pseudo_weight=weight,
                        reliability=kind,min_reliability=floor,inner_splits=5))
            cs.append(dict(model='reliable_ssl',teacher=teacher,pseudo_weight=weight,reliability='uniform'))
    for regions in [3,6]:
        for C in [.3,1,3]:
            for gamma in [.001,.003]:
                for mix in [.25,.5]:cs.append(dict(model='local_ssl',regions=regions,C=C,gamma=gamma,local_mix=mix))
    return cs

def score(h):
    row=2*(h*e.Y).sum(1)/(h.sum(1)+e.Y.sum(1));ordinary=float(row.mean());weighted=W@row
    return dict(ordinary=ordinary,robust=float(.75*weighted.min()+.25*ordinary),weighted=weighted.tolist())

def load_predictions(c,seed,tag):
    r=e.evaluate(c,seed=seed,tag=tag)
    if r['status']!='ok':return None
    return np.load(E/f'{tag}-{e.config_id(c)}-{seed}.npz',allow_pickle=True)['p']

def main():
    cs=configs();e.save_json(E/'phase25-configs.json',cs)
    base=load_predictions(BASE,20261118,'phase25_control');screen=[]
    for i,c in enumerate(cs):
        print('SCREEN',i+1,len(cs),c,flush=True)
        p=load_predictions(c,20261118,'phase25')
        if p is None:continue
        for rule in RULES:
            for scope in SCOPES:
                cfg=dict(model='policy25',learner=c,baseline=BASE,decision_rule=rule,scope=scope)
                screen.append(dict(config=cfg,**score(m.apply_policy(p,base,rule,scope))))
        e.save_json(E/'phase25-screen.json',screen)
        e.save_json(E/f'phase25-diagnostics-{e.config_id(c)}.json',m.LAST_DIAGNOSTICS)
    short={}
    for family in FAMILIES:
        pool=[r for r in screen if r['config']['learner']['model']==family and r['config']['learner'].get('reliability')!='uniform']
        for metric in ['ordinary','robust']:
            r=max(pool,key=lambda r:(r[metric],r['ordinary'],r['config']['scope']=='rare'))
            short[e.config_id(r['config'])]=r
    e.save_json(E/'phase25-shortlist-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=list(short.values()),repeat_seeds=[20261119,20261120,20261121]))
    baseline=[]
    for seed in [20261119,20261120,20261121]:baseline.append(score(load_predictions(BASE,seed,'phase25_control')))
    repeats=[]
    for r in short.values():
        ss=[]
        for seed in [20261119,20261120,20261121]:
            print('REPEAT',r['config']['learner'],seed,flush=True)
            p=load_predictions(r['config'],seed,'phase25_repeat')
            if p is None:break
            ss.append(score(p))
        if len(ss)<3:continue
        out={**r,'ordinary_cv':float(np.mean([s['ordinary'] for s in ss])),
             'robust_cv':float(np.mean([s['robust'] for s in ss])),
             'cv_std':float(np.std([s['ordinary'] for s in ss])),
             'robust_std':float(np.std([s['robust'] for s in ss])),
             'per_repeat':ss,'rule':'t0.5'}
        out['selection_score']=out['ordinary_cv']-.2*out['cv_std']
        repeats.append(out);e.save_json(E/'phase25-repeat-results.json',repeats)
    selected=[]
    for family in FAMILIES:
        pool=[r for r in repeats if r['config']['learner']['model']==family]
        if pool:selected.append({**max(pool,key=lambda r:(r['selection_score'],r['robust_cv'])),'name':'phase25_'+family})
    e.save_json(E/'phase25-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_cv=baseline,candidates=selected))
    V=e.pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
    with threadpool_limits(limits=1):ps=[e.predict(np.arange(len(e.Y)),V,r['config']) for r in selected]
    np.savez_compressed(E/'phase25-validation-predictions.npz',p=ps)
    y=e.pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
    for r,p in zip(selected,ps):
        out={**r,'validation':e.metrics(y,p,r['rule'])};results.append(out)
        print('VALIDATION',json.dumps(out),flush=True)
    e.save_json(E/'phase25-validation-results.json',results)
    print('COMPLETE',len(results),'validation candidates',flush=True)

if __name__=='__main__':main()
