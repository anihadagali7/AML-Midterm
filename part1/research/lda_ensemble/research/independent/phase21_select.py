"""Target-aware selection over existing and tri-training candidates, then a frozen audit."""
import phase21 as tri,experiment as e
from pathlib import Path
import json,datetime,hashlib
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits

RULES=['t0.25','t0.35','t0.45','t0.5','t0.55','t0.65','t0.75','expected_f1','argmax']
W=dict(np.load('experiments/phase21-weights.npz'))
NAMES=list(W);weight_matrix=np.array([W[k]/W[k].sum() for k in NAMES])
PRIMARY=NAMES.index('propensity_logistic');SHIFT=[NAMES.index(k) for k in ['propensity_logistic','propensity_tree','kmm']]
UNIFORM=NAMES.index('uniform')


def table(ps):
    # axes: rule, repeated split, weighting scheme
    out=[]
    for rule in RULES:
        rr=[]
        for p in ps:
            h=e.decisions(p,rule)
            row=2*(h*e.Y).sum(1)/(h.sum(1)+e.Y.sum(1))
            rr.append(weight_matrix@row)
        out.append(rr)
    return np.array(out)


def choice(t,objective):
    means=t.mean(1)
    if objective=='ordinary':values=t[:,:,UNIFORM]
    elif objective=='iw':values=t[:,:,PRIMARY]
    else:values=.75*t[:,:,SHIFT].min(2)+.25*t[:,:,UNIFORM]
    score=values.mean(1)-.2*values.std(1)
    best=max(range(len(RULES)),key=lambda j:(score[j],-abs(float(RULES[j][1:])-.5) if RULES[j].startswith('t') else -1))
    return dict(rule=RULES[best],selection_score=float(score[best]),ordinary_cv=float(means[best,UNIFORM]),
                weighted_cv={k:float(means[best,i]) for i,k in enumerate(NAMES)},
                split_score_std=float(values[best].std()))


def family(c):
    if c['model'] in ['lowdensity_svm','lowdensity_v2','lowdensity_v3','kernel_lowdensity']:return 'lowdensity_svm'
    return c['model']


def run():
    rows=[json.loads(s) for s in Path('experiments/trials.jsonl').read_text().splitlines()]
    first={}
    for r in rows:
        if r['status']=='ok' and r.get('seed')==20261104:first[r['id']]=r
    screen=[]
    with threadpool_limits(limits=1):
        for cid,r in first.items():
            path=Path(f"experiments/{r['tag']}-{cid}-20261104.npz")
            if not path.exists():continue
            z=np.load(path,allow_pickle=True);p=z['p']
            if p.shape!=(157,5) or not np.isfinite(p).all() or len(np.unique(z['fold']))!=5:continue
            t=table([p]);screen.append(dict(id=cid,config=r['config'],family=family(r['config']),
                                           **{objective:choice(t,objective) for objective in ['ordinary','iw','robust']}))
    e.save_json('experiments/phase21-risk-screen.json',screen)
    selected={}
    old=[r for r in screen if r['config']['model']!='tri_training']
    training=[r for r in screen if r['config']['model']=='tri_training' and not r['config'].get('no_teaching')]
    for objective in ['iw','robust']:
        seen=set()
        for r in sorted(old,key=lambda r:r[objective]['selection_score'],reverse=True):
            if r['family'] in seen:continue
            selected[r['id']]=r;seen.add(r['family'])
            if len(seen)==3:break
    for objective in ['ordinary','iw','robust']:
        r=max(training,key=lambda r:r[objective]['selection_score']);selected[r['id']]=r
    # Retain one guarded teacher variant independently of in-sample peer error scores.
    r=max([r for r in training if r['config'].get('guard')=='crossfit'],key=lambda r:r['iw']['selection_score']);selected[r['id']]=r
    incumbent=json.loads(Path('experiments/phase21-prior-incumbent.json').read_text())['candidate']['config']
    cid=e.config_id(incumbent);selected[cid]=dict(id=cid,config=incumbent,family=family(incumbent))
    # Matched bootstrap-only controls make the effect of cross-teaching assessable.
    committees={r['config']['committee'] for r in selected.values() if r['config']['model']=='tri_training'}
    for r in screen:
        if r['config']['model']=='tri_training' and r['config'].get('no_teaching') and r['config']['committee'] in committees:selected[r['id']]=r
    e.save_json('experiments/phase21-refit-shortlist.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),weight_sha256=hashlib.sha256(Path('experiments/phase21-weights.npz').read_bytes()).hexdigest(),fresh_seeds=[20261107,20261108,20261109],candidates=list(selected.values())))
    summary=[]
    with threadpool_limits(limits=1):
        for i,r in enumerate(selected.values()):
            print('REFIT',i+1,len(selected),r['config'],flush=True)
            ps=[]
            for seed in [20261107,20261108,20261109]:
                record=e.evaluate(r['config'],seed=seed,tag='phase21_refit')
                if record['status']!='ok':raise RuntimeError(record)
                ps.append(np.load(f"experiments/phase21_refit-{r['id']}-{seed}.npz",allow_pickle=True)['p'])
            t=table(ps);summary.append(dict(id=r['id'],config=r['config'],**{objective:choice(t,objective) for objective in ['ordinary','iw','robust']}))
            print('SUMMARY',json.dumps(summary[-1]),flush=True)
    e.save_json('experiments/phase21-selection.json',summary)
    candidates=[]
    def add(r,objective,purpose):
        candidate=dict(name=purpose,config=r['config'],objective=objective,**r[objective])
        key=(e.config_id(candidate['config']),candidate['rule'])
        for existing in candidates:
            if (e.config_id(existing['config']),existing['rule'])==key:
                existing['name']+=' + '+purpose;return
        candidates.append(candidate)
    teaching=[r for r in summary if r['config']['model']=='tri_training' and not r['config'].get('no_teaching')]
    ordinary=max(teaching,key=lambda r:r['ordinary']['selection_score']);add(ordinary,'ordinary','tri_ordinary_cv')
    add(max(teaching,key=lambda r:r['iw']['selection_score']),'iw','tri_target_weighted_cv')
    old=[r for r in summary if r['config']['model']!='tri_training']
    add(max(old,key=lambda r:r['iw']['selection_score']),'iw','existing_target_weighted_cv')
    add(max(summary,key=lambda r:r['robust']['selection_score']),'robust','robust_weighting_cv')
    control=next(r for r in summary if r['config'].get('no_teaching') and r['config']['committee']==ordinary['config']['committee'])
    add(control,'ordinary','matched_no_teaching_control')
    assert len(candidates)<=5
    e.save_json('experiments/phase21-validation-frozen.json',dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),weighting_file='phase21-weighting.json',candidates=candidates))
    V=pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS);preds=[]
    with threadpool_limits(limits=1):
        for candidate in candidates:
            preds.append(e.predict(np.arange(len(e.Y)),V,candidate['config']))
            if candidate['config']['model']=='tri_training':
                e.save_json('experiments/phase21-fullfit-diagnostics-'+e.config_id(candidate['config'])+'.json',tri.LAST_DIAGNOSTICS)
    np.savez_compressed('experiments/phase21-validation-predictions.npz',p=preds)
    # Validation labels are first read only after predictions and decisions are frozen.
    y=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();results=[]
    for candidate,p in zip(candidates,preds):
        r={**candidate,'validation':e.metrics(y,p,candidate['rule'])};results.append(r);print('VALIDATION',json.dumps(r),flush=True)
    e.save_json('experiments/phase21-validation-results.json',results)

if __name__=='__main__':run()
