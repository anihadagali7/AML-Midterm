"""Evaluate nested specialists without hand-selecting validation errors."""
from pathlib import Path
import json,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import phase23,experiment as e
import phase22_run as prior

E=Path('experiments')
inc=json.loads((E/'phase22-baseline.json').read_text())['candidate']['config']

def run():
    cs=[]
    for kind in ['other','pair','all','patterns']:
        for C in [.1,.3,1]:
            c={**inc,'model':'joint_specialist','meta':kind,'meta_C':C,'inner_splits':15}
            cs.append(dict(bucket=kind,config=c))
    for C in [.1,.3,1]:
        cs.append(dict(bucket='other',config={**inc,'model':'joint_specialist','meta':'other',
                       'meta_C':C,'inner_splits':15,'interactions':True}))
    for strength in [.5,.75]:
        cs.append(dict(bucket='other',config={**inc,'model':'joint_specialist','meta':'other',
                       'meta_C':.3,'inner_splits':15,'strength':strength}))
    e.save_json(E/'phase23-protocol.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        candidates=cs,outer_screen_seed=20261110,repeat_seeds=[20261111,20261112,20261113],
        inner_folds=15,selection='Select strongest ordinary/robust CV candidates; at most three distinct configs receive repeats, at most two receive validation.',
        validation_fit=False,reason='Other has the most errors on baseline training OOF predictions.',
        sources=['https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html',
                 'https://proceedings.neurips.cc/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html']))
    screen=[]
    for i,r in enumerate(cs):
        print('SCREEN',i+1,len(cs),r['config']['meta'],r['config']['meta_C'],flush=True)
        s=prior.evaluate(r,20261110,'phase23')
        if s is not None:screen.append({**r,**s})
        e.save_json(E/'phase23-screen.json',screen)
    short={}
    for metric in ['ordinary','robust']:
        if screen:
            r=max(screen,key=lambda r:r[metric]);short[e.config_id(r['config'])]=r
    other=[r for r in screen if r['bucket']=='other']
    if other:
        r=max(other,key=lambda r:r['robust']);short[e.config_id(r['config'])]=r
    e.save_json(E/'phase23-shortlist-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=list(short.values())))
    repeats=[]
    for r in short.values():
        ss=[]
        for seed in [20261111,20261112,20261113]:
            print('REPEAT',r['bucket'],seed,flush=True)
            s=prior.evaluate(r,seed,'phase23_repeat')
            if s is None:break
            ss.append(s)
        if len(ss)<3:continue
        out={**r,'ordinary_cv':float(np.mean([s['ordinary'] for s in ss])),
             'robust_cv':float(np.mean([s['robust'] for s in ss])),
             'cv_std':float(np.std([s['ordinary'] for s in ss])),
             'robust_std':float(np.std([s['robust'] for s in ss])),
             'per_repeat':ss,'rule':'t0.5'}
        out['selection_score']=out['robust_cv']-.2*out['robust_std'];repeats.append(out)
        e.save_json(E/'phase23-repeat-results.json',repeats)
    baseline=next(r for r in json.loads((E/'phase22-repeat-results.json').read_text()) if r['bucket']=='control')
    eligible=[r for r in repeats if r['ordinary_cv']>=baseline['ordinary_cv']-.003 and r['robust_cv']>=baseline['robust_cv']-.003]
    picked={}
    for metric in ['selection_score','ordinary_cv']:
        if eligible:
            r=max(eligible,key=lambda r:r[metric]);picked[e.config_id(r['config'])]=r
    candidates=[{**r,'name':'phase23_'+r['bucket']} for r in picked.values()]
    e.save_json(E/'phase23-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=candidates))
    results=[]
    if candidates:
        V=e.pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
        with threadpool_limits(limits=1):
            ps=[e.predict(np.arange(len(e.Y)),V,r['config']) for r in candidates]
        np.savez_compressed(E/'phase23-validation-predictions.npz',p=ps)
        y=e.pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy()
        for r,p in zip(candidates,ps):
            out={**r,'validation':e.metrics(y,p,r['rule'])};results.append(out)
            print('VALIDATION',json.dumps(out),flush=True)
    e.save_json(E/'phase23-validation-results.json',results)
    print('COMPLETE',len(results),'validation candidates',flush=True)

if __name__=='__main__':run()
