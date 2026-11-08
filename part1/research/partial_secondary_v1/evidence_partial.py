"""Retain useful matched comparisons, then independently rescore their evidence."""
import sys
import shutil
import numpy as np
import pandas as pd
import study_run as study
from verify_partial import metrics


def verify():
    study.guard();here=study.HERE;manifest=study.read(here/'retention.json')
    for name,digest in manifest['hashes'].items():assert study.sha(here/name)==digest,name
    z=study.loadnp(here/'results/retained_comparison.npz')
    scores=pd.read_csv(here/'results/summary/repeat_scores.csv')
    labels=pd.read_csv(here/'results/summary/per_label_scores.csv')
    groups=[(p,int(s),(z['pool']==p)&(z['seed']==s)) for p,s in sorted(set(zip(z['pool'],z['seed'])))]
    groups.append(('chronology98',-1,np.char.startswith(z['pool'],'through_')))
    n=0
    for pool,seed,keep in groups:
        y=z['truth'][keep];assert len(set(z['ids'][keep]))==len(y)
        for j,name in enumerate(z['names']):
            p=z['predictions'][j,keep]
            np.testing.assert_array_equal(p,study.select_reasons(z['scores'][j,keep]))
            s,m,pr,re,f,support=metrics(y,p)
            r=scores[(scores.pool==pool)&(scores.seed==seed)&(scores.recipe==name)].iloc[0]
            np.testing.assert_allclose([s,m],[r.sample_f1,r.macro_f1],atol=1e-12,rtol=0)
            l=labels[(labels.pool==pool)&(labels.seed==seed)&(labels.recipe==name)].set_index('label').loc[study.LABELS]
            np.testing.assert_allclose(np.c_[pr,re,f,support],l[['precision','recall','f1','support']],atol=1e-12,rtol=0)
            n+=1
    assert n==63 and not any(study.read(here/'results/summary/qualification.json').values())
    print(dict(passed=True,retained_metric_records_rescored=n,all_five_variants_summary_hashes_verified=True,
        full_checks_before_cleanup=manifest['verification_before_cleanup']['metric_rows_independently_checked']))


def retain():
    here=study.HERE;out=here/'results';receipt=study.read(out/'summary/verification.json')
    assert receipt['passed'] and receipt['outer_cases']==84
    names=['parent','binary_complete','binary_unknown'];index=[study.NAMES.index(n) for n in names]
    zs=[];meta=[]
    for c in study.cases():
        path=out/'predictions'/(c['key']+'.npz');z=study.loadnp(path);audit=study.read(path.with_suffix('.json'))
        assert study.sha(path)==audit['sha256'];zs.append(z)
        pool=c['pool'] if c['pool']!='supplementary' else c['key']
        meta.append(dict(pool=np.full(len(z['ids']),pool),seed=np.full(len(z['ids']),c['seed']),case=np.full(len(z['ids']),c['key'])))
    arrays={k:np.concatenate([z[k] for z in zs]) for k in ['ids','truth']}
    arrays.update({k:np.concatenate([v[k] for v in meta]) for k in ['pool','seed','case']})
    arrays.update(names=np.array(names),scores=np.concatenate([z['scores'][index] for z in zs],axis=1),
        predictions=np.concatenate([z['predictions'][index] for z in zs],axis=1))
    np.savez_compressed(out/'retained_comparison.npz',**arrays)
    files=list((out/'summary').glob('*'))+[out/'retained_comparison.npz',here/'math_check.json']
    study.save(here/'retention.json',dict(kept_recipes=names,
        reason='Useful matched comparison isolates unknown labels from binary architecture and documents opposite CV/chronology behavior.',
        removed='84 per-case prediction packets and84 JSON fit audits after checking all summaries. No fitted estimators or generated datasets were saved. Aggregate fit-count audits remain.',
        verification_before_cleanup=receipt,hashes={str(p.relative_to(here)):study.sha(p) for p in files}))
    verify()
    shutil.rmtree(out/'predictions')
    print('Removed verified unsuccessful fold caches.')


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['retain','verify']
    retain() if sys.argv[1]=='retain' else verify()
