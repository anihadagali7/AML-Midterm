"""Predeclared screen, repeated grouped CV, matched controls, frozen validation."""
from pathlib import Path
import copy,datetime,json
import numpy as np
from threadpoolctl import threadpool_limits
import phase26 as m,experiment as e
import phase25_run as prior

E=Path('experiments');BASE=m.INC
SCREEN_SEED=20261122;SEEDS=[20261123,20261124,20261125]
RULES=['t0.4','t0.5','t0.6','t0.7']
FAMILIES=['label_shift26','lap_svm26','other_subclasses26']

def configs():
    cs=[]
    for method in ['em','bbse']:
        for strength in [.25,.5,1]:
            for output in ['raw','calibrated']:
                for C in [.3,3]:cs.append(dict(model='label_shift26',method=method,strength=strength,output=output,calibration_C=C))
    for C in [.3,1,3]:
        for gamma in [.001,.003]:
            for neighbors in [10,25]:
                for weight in [.1,1,10]:cs.append(dict(model='lap_svm26',C=C,gamma=gamma,neighbors=neighbors,graph_weight=weight))
    for k in [2,3]:
        for representation in ['linear','kernel']:
            for C in [.3,1,3]:
                for mix in [.5,1]:cs.append(dict(model='other_subclasses26',subgroups=k,representation=representation,C=C,mix=mix,pseudo_weight=.03))
    return cs

def load(c,seed,tag):
    r=e.evaluate(c,seed=seed,tag=tag)
    if r['status']!='ok':return None
    return np.load(E/f'{tag}-{e.config_id(c)}-{seed}.npz',allow_pickle=True)['p']

def scopes(family):
    if family=='other_subclasses26':return ['other']
    return ['all','rare','efficacy_other','recruitment_other']

def repeat(r,tag):
    ss=[]
    for seed in SEEDS:
        p=load(r['config'],seed,tag)
        if p is None:return None
        ss.append(prior.score(p))
    out={**r,'ordinary_cv':float(np.mean([s['ordinary'] for s in ss])),
        'robust_cv':float(np.mean([s['robust'] for s in ss])),
        'cv_std':float(np.std([s['ordinary'] for s in ss])),
        'per_repeat':ss,'rule':'t0.5'}
    out['selection_score']=out['ordinary_cv']-.2*out['cv_std']
    return out

def main():
    cs=configs();e.save_json(E/'phase26-configs.json',cs)
    protocol=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        raw_configurations=len(cs),screen_seed=SCREEN_SEED,repeat_seeds=SEEDS,
        selection='Up to two policies per family nominated by ordinary and robust CV; repeat selection uses mean sample-F1 minus .2 times split standard deviation.',
        validation_cap=6,validation_fit=False,final_train_plus_validation_refit=False,
        controls='For each training-selected finalist: zero prior adjustment, zero graph penalty, or one Other subgroup; preserve remaining settings and decoder.',
        graph_approximation='192-component Nystroem RBF features with a fold-local self-tuning nearest-neighbor graph.',
        label_shift_assumption='Binary label shift assumes conditional score distributions are stable; source calibration summaries are training estimates, not an independent guarantee.',
        reported_leader='0.93xx; exact value unknown')
    e.save_json(E/'phase26-protocol.json',protocol)
    base=load(BASE,SCREEN_SEED,'phase26_control');screen=[]
    for i,c in enumerate(cs):
        print('SCREEN',i+1,len(cs),c,flush=True)
        p=load(c,SCREEN_SEED,'phase26')
        if p is None:continue
        for rule in RULES:
            for scope in scopes(c['model']):
                cfg=dict(model='policy26',learner=c,baseline=BASE,decision_rule=rule,scope=scope)
                screen.append(dict(config=cfg,**prior.score(m.apply_policy(p,base,rule,scope))))
        e.save_json(E/'phase26-screen.json',screen)
        e.save_json(E/f'phase26-diagnostics-{e.config_id(c)}.json',m.LAST_DIAGNOSTICS)
    short={}
    for family in FAMILIES:
        pool=[r for r in screen if r['config']['learner']['model']==family]
        for metric in ['ordinary','robust']:
            r=max(pool,key=lambda r:(r[metric],r['ordinary']))
            short[e.config_id(r['config'])]=r
    e.save_json(E/'phase26-shortlist-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=list(short.values()),repeat_seeds=SEEDS))
    baseline=[prior.score(load(BASE,seed,'phase26_control')) for seed in SEEDS]
    repeats=[]
    for r in short.values():
        print('REPEAT',r['config'],flush=True)
        out=repeat(r,'phase26_repeat')
        if out:repeats.append(out);e.save_json(E/'phase26-repeat-results.json',repeats)
    selected=[]
    for family in FAMILIES:
        pool=[r for r in repeats if r['config']['learner']['model']==family]
        if pool:selected.append({**max(pool,key=lambda r:(r['selection_score'],r['robust_cv'])),'name':'phase26_'+family,'matched_control':False})
    controls=[]
    for r in selected:
        c=copy.deepcopy(r['config']);family=c['learner']['model']
        if family=='label_shift26':c['learner']['strength']=0
        elif family=='lap_svm26':c['learner']['graph_weight']=0
        else:c['learner']['subgroups']=1
        out=repeat(dict(config=c,name='phase26_control_'+family,matched_control=True),'phase26_matched')
        if out:controls.append(out)
    e.save_json(E/'phase26-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_cv=baseline,candidates=selected+controls))
    # Integrity must have passed before accessing validation at all.
    integrity=json.loads((E/'phase26-integrity-checks.json').read_text())
    assert integrity['passed']
    V=e.pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
    with threadpool_limits(limits=1):
        ps=[]
        for r in selected+controls:
            ps.append(e.predict(np.arange(len(e.Y)),V,r['config']))
            e.save_json(E/f"phase26-final-diagnostics-{r['name']}.json",m.LAST_DIAGNOSTICS)
    np.savez_compressed(E/'phase26-validation-predictions.npz',p=ps)
    y=e.pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
    for r,p in zip(selected+controls,ps):
        out={**r,'validation':e.metrics(y,p,r['rule'])};results.append(out)
        print('VALIDATION',json.dumps(out),flush=True)
    e.save_json(E/'phase26-validation-results.json',results)
    print('COMPLETE',len(results),'validation comparisons',flush=True)

if __name__=='__main__':main()
