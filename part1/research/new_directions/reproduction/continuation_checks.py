"""Disclosed diagnostics after v5/v6 screens; no automatic model promotion."""
import json,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import study as e,study_v5,study_v6,models_v5,models_v6
ROOT=Path(__file__).resolve().parent
def freeze():
    path=ROOT/'continuation_diagnostic_plan.json';assert not path.exists()
    e.save(path,dict(frozen_at=time.time(),source_sha256=e.sha(__file__),
        evidence_before_plan={'v5_screen':e.sha(study_v5.OUT/'comparison_3.csv'),'v5_confirmation':e.sha(study_v5.OUT/'comparison_8.csv'),'v6_screen':e.sha(study_v6.OUT/'comparison_3.csv')},
        rationale='v5 all15 gains both means but wins only 4/8 in each pool. v6 three-cluster teacher gains both screening means over its exact GMM parent but not the stronger ensemble. Quantify these limited signals without changing recipes or calling them confirmed.',
        extra_cv=[dict(study='v6',recipe='single_gmm'),dict(study='v6',recipe='teacher_mean3')],
        supplementary=[dict(study='v5',recipe='rfm_complement_all15'),dict(study='v6',recipe='single_gmm'),dict(study='v6',recipe='teacher_mean3'),dict(study='v5',recipe='submitted_reference')],
        gates_unchanged=True,promotion=False,validation_role='Repeatedly consulted diagnostic, not fresh heldout proof.',
        unused_test_inputs=True,external_data=False,deep_learning=False))
    print('Frozen diagnostic extension before additional fits.',flush=True)
def supplementary(version,name):
    module={'v5':study_v5,'v6':study_v6}[version];reg=module.registry()
    c=next((c for c in reg['recipes'] if c['name']==name),None)
    assert c is not None or name=='submitted_reference'
    out=module.OUT/'supplementary'/name;out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'data/part1_train.csv');val=pd.read_csv(ROOT/'data/part1_val.csv')
    gold=train[train[e.LABELS].ge(0).all(1)];masked=train[train[e.LABELS].eq(-100).all(1)]
    full=pd.concat([train,val],ignore_index=True).set_index('ID',drop=False)
    cases=[('validation',gold,masked,val)]
    for w in json.loads((ROOT/'protocols/chronology.json').read_text())['chronological_splits']:
        fit=full.loc[w['training_ids']];g=fit[fit[e.LABELS].ge(0).all(1)];u=fit[fit[e.LABELS].eq(-100).all(1)];q=full.loc[w['heldout_ids']]
        assert len(g)==w['gold_rows'] and len(u)==w['masked_rows']
        assert not set(e.groups(fit))&set(e.groups(q))
        cases.append((w['name'],g,u,q))
    chunks=[];rows=[]
    for case,g,u,q in cases:
        dest=out/f'{case}.npz'
        if not dest.exists():
            if name=='submitted_reference':
                p=models_v5.old_prediction(g,u,q.drop(columns=e.LABELS),e.LABELS,e.EMB)
                pred=models_v5.models.threshold(p,.5);audit={'decoder':.5}
            else:p,pred,audit={'v5':models_v5,'v6':models_v6}[version].predict(g,u,q.drop(columns=e.LABELS),c,e.LABELS,e.EMB)
            np.savez_compressed(dest,scores=p,predictions=pred,truth=q[e.LABELS].to_numpy(int),ids=q.ID.to_numpy(str))
            e.save(dest.with_suffix('.json'),dict(name=case,recipe=c or name,metrics=e.metric(q[e.LABELS].to_numpy(int),pred),
                sources=reg['sources'],raw_data=reg['raw_data'],protocol_hashes=reg['protocols'],gold_ids=g.ID.tolist(),masked_ids=u.ID.tolist(),query_ids=q.ID.tolist(),audit=audit,
                diagnostic_plan_sha256=e.sha(ROOT/'continuation_diagnostic_plan.json'),access='inductive'))
        z=np.load(dest);m=e.metric(z['truth'],z['predictions']);rows.append(dict(case=case,**m))
        if case!='validation':chunks.append({k:z[k] for k in z.files})
        print(json.dumps(dict(recipe=name,case=case,f1=m['sample_f1'])),flush=True)
    pooled={k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]}
    np.savez_compressed(out/'chronological98.npz',**pooled)
    e.save(out/'summary.json',dict(recipe=name,validation=rows[0],windows=rows[1:],chronology=e.metric(pooled['truth'],pooled['predictions'])))
def run():
    plan=json.loads((ROOT/'continuation_diagnostic_plan.json').read_text())
    assert e.sha(__file__)==plan['source_sha256']
    with threadpool_limits(limits=1):
        study_v6.configure()
        for item in plan['extra_cv']:
            c=next(c for c in study_v6.registry()['recipes'] if c['name']==item['recipe'])
            for pool in ['train','pooled']:e.evaluate(c,pool,8)
        e.summarize(8)
        for item in plan['supplementary']:supplementary(item['study'],item['recipe'])
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:run()
