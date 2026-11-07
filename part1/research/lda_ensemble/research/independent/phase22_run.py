"""Prespecified refinements, grouped-CV selection, then one small validation batch."""
from pathlib import Path
import copy,json,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import phase22 as m,experiment as e

E=Path('experiments')
frozen=json.loads((E/'phase22-baseline.json').read_text())
inc=frozen['candidate']['config'];base=inc['base'];original=inc['tri']
refined={**original,'model':'tri_refined'}
seeds=[210123,220123,230123,240123,250123]
W=dict(np.load(E/'phase21-weights.npz'))
w=np.array([W[n]/W[n].sum() for n in ['propensity_logistic','propensity_tree','kmm']])

def hybrid(tri,**extra):
    return dict(model='hybrid22',base=base,tri=tri,base_rule=inc['base_rule'],
                tri_rule=inc['tri_rule'],use_tri=inc['use_tri'],**extra)

def ensemble(count,**extra):
    return dict(model='tri_seed_ensemble',member={**refined,**extra},seeds=seeds[:count])

def configurations():
    out=[]
    def add(bucket,c):out.append(dict(bucket=bucket,config=c))
    add('control',hybrid(refined))
    for count in [3,5]:add('bootstrap',hybrid(ensemble(count)))
    for key,values in [('C',[.3,3]),('gamma',[.0005,.002]),
                       ('logistic_C',[.1,1]),('ridge_alpha',[3,30]),
                       ('pseudo_weight',[.1,.3]),('confidence',[.6,.75])]:
        for v in values:add('parameters',hybrid({**refined,key:v}))
    for scale in ['global','raw']:add('parameters',hybrid({**refined,'scale':scale}))
    for n in [50,100]:add('parameters',hybrid({**refined,'pca':n}))
    for block in ['criteria','criteria_char','diseases_cleaned','drugs','icdcodes',
                  'drug_targets','drug_actions','drug_pathways','fingerprint','descriptors','year','counts']:
        for weight in [.15,.4]:add('features',hybrid({**refined,'features':[block],'feature_weight':weight}))
    for tri in [refined,ensemble(5)]:
        for weight in [.25,.5,.75]:add('soft',hybrid(tri,combination='soft',tri_weight=weight))
    return out

def scores(p):
    h=e.decisions(p,'t0.5');row=2*(h*e.Y).sum(1)/(h.sum(1)+e.Y.sum(1))
    weighted=w@row;ordinary=float(row.mean())
    return dict(ordinary=ordinary,robust=float(.75*weighted.min()+.25*ordinary),
                weighted=weighted.tolist(),macro=e.metrics(e.Y,p,'t0.5')['macro_f1'])

def evaluate(r,seed,tag):
    result=e.evaluate(r['config'],seed=seed,tag=tag)
    if result['status']!='ok':return None
    p=np.load(E/f"{tag}-{e.config_id(r['config'])}-{seed}.npz",allow_pickle=True)['p']
    return scores(p)

def run():
    cs=configurations();e.save_json(E/'phase22-configs.json',cs)
    # Default-core equivalence is required before relying on the copied core.
    a,b=e.make_splits(seed=20261110)[0];q=e.DATA.iloc[e.LAB[b]];excluded=e.GROUP[e.LAB[b]]
    with threadpool_limits(limits=1):
        old=m.legacy_predict(a,q,original,excluded)
        new=m.predict_core(a,q,refined,excluded)
    assert np.allclose(old,new,atol=1e-12,rtol=0)
    e.save_json(E/'phase22-core-equivalence.json',dict(max_difference=float(abs(old-new).max()),passed=True))
    screen=[]
    for i,r in enumerate(cs):
        print('SCREEN',i+1,len(cs),r['bucket'],r['config']['tri'],flush=True)
        s=evaluate(r,20261110,'phase22')
        if s is not None:screen.append({**r,**s})
        e.save_json(E/'phase22-screen.json',screen)
    shortlist={}
    for bucket in ['bootstrap','parameters','features','soft']:
        available=[r for r in screen if r['bucket']==bucket]
        if not available:continue
        best=max(available,key=lambda r:(r['robust'],r['ordinary']))
        shortlist[e.config_id(best['config'])]=best
    selected=list(shortlist.values())
    e.save_json(E/'phase22-shortlist-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=selected,repeat_seeds=[20261111,20261112,20261113]))
    repeats=[]
    for r in [cs[0]]+selected:
        ss=[]
        for seed in [20261111,20261112,20261113]:
            print('REPEAT',r['bucket'],seed,flush=True)
            s=evaluate(r,seed,'phase22_repeat')
            if s is None:break
            ss.append(s)
        if len(ss)<3:continue
        out={**r,'ordinary_cv':float(np.mean([s['ordinary'] for s in ss])),
             'robust_cv':float(np.mean([s['robust'] for s in ss])),
             'cv_std':float(np.std([s['ordinary'] for s in ss])),
             'robust_std':float(np.std([s['robust'] for s in ss])),
             'per_repeat':ss,'rule':'t0.5'}
        out['selection_score']=out['robust_cv']-.2*out['robust_std']
        repeats.append(out);e.save_json(E/'phase22-repeat-results.json',repeats)
    baseline=next(r for r in repeats if r['bucket']=='control')
    eligible=[r for r in repeats if r['bucket']!='control' and r['ordinary_cv']>=baseline['ordinary_cv']-.003 and r['robust_cv']>=baseline['robust_cv']-.003]
    chosen={}
    if eligible:
        for metric in ['selection_score','ordinary_cv']:
            r=max(eligible,key=lambda r:r[metric]);chosen[e.config_id(r['config'])]=r
        feature=[r for r in eligible if r['bucket']=='features']
        if feature:chosen[e.config_id(feature[0]['config'])]=feature[0]
    candidates=[{**r,'name':'phase22_'+r['bucket']} for r in chosen.values()]
    assert len(candidates)<=3
    e.save_json(E/'phase22-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_cv=baseline,candidates=candidates))
    results=[]
    if candidates:
        V=e.pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
        with threadpool_limits(limits=1):
            predictions=[e.predict(np.arange(len(e.Y)),V,r['config']) for r in candidates]
        np.savez_compressed(E/'phase22-validation-predictions.npz',p=predictions)
        y=e.pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy()
        for r,p in zip(candidates,predictions):
            out={**r,'validation':e.metrics(y,p,r['rule'])};results.append(out)
            print('VALIDATION',json.dumps(out),flush=True)
    e.save_json(E/'phase22-validation-results.json',results)
    print('COMPLETE',len(results),'frozen validation candidates',flush=True)

if __name__=='__main__':run()
