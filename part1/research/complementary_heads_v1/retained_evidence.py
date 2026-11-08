"""Keep useful matched fusion/head comparisons after independent full verification."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import head_study as study
from verify_heads import metrics


def verify(directory):
    directory=Path(directory);study.guard()
    manifest=study.read(directory/'retention.json')
    for name,digest in manifest['files'].items():assert study.sha(directory/name)==digest
    z=study.loadnp(directory/'matched_predictions.npz')
    table=pd.read_csv(directory/'summary/repeat_scores.csv')
    labels=pd.read_csv(directory/'summary/per_label_scores.csv')
    grouped={};checked=0
    for case in study.cases():
        keep=z['case']==case['key']
        np.testing.assert_array_equal(z['ids'][keep],case['query'].ID)
        np.testing.assert_array_equal(z['truth'][keep],case['query'][study.LABELS])
        key=(case['pool'],case['seed']) if case['pool']!='supplementary' else (case['key'],-1)
        grouped.setdefault(key,[]).extend(np.flatnonzero(keep).tolist())
        if case['key'].startswith('through_'):grouped.setdefault(('chronology98',-1),[]).extend(np.flatnonzero(keep).tolist())
    for (pool,seed),positions in grouped.items():
        y=z['truth'][positions]
        for j,name in enumerate(z['names']):
            score=z['scores'][j,positions];p=(score>=.5).astype(int)
            p[np.arange(len(p)),score.argmax(1)]=1
            np.testing.assert_array_equal(p,z['predictions'][j,positions])
            s,m,l=metrics(y,p)
            row=table[(table.pool==pool)&(table.seed==seed)&(table.recipe==name)].iloc[0]
            np.testing.assert_allclose([s,m],[row.sample_f1,row.macro_f1],atol=1e-12,rtol=0)
            lab=labels[(labels.pool==pool)&(labels.seed==seed)&(labels.recipe==name)].set_index('label').loc[study.LABELS]
            np.testing.assert_allclose(l,lab[['precision','recall','f1','support']],atol=1e-12,rtol=0)
            checked+=1
    receipt=dict(passed=True,retained_metric_rows_rescored=checked,retained_recipes=z['names'].tolist(),
        prediction_rows=len(z['ids']),outer_cases=84,protected_hashes_unchanged=True)
    study.save(directory/'retained_verification.json',receipt);print(receipt)


def compact(directory, kept=None, rationale=None):
    directory=Path(directory)
    assert study.read(directory/'summary/verification.json')['passed']
    keep=list(kept) if kept is not None else ['parent','anchored','calibration']+['head_'+n for n in study.STRENGTHS]+['fusion_'+n for n in study.STRENGTHS]
    qualifying=study.read(directory/'summary/qualification.json')
    keep += [name for name,yes in qualifying.items() if yes and name not in keep]
    indices=[study.NAMES.index(name) for name in keep]
    packets=[];case_keys=[]
    for case in study.cases():
        z=study.loadnp(directory/'predictions'/(case['key']+'.npz'))
        packets.append(z);case_keys.extend([case['key']]*len(z['ids']))
    np.savez_compressed(directory/'matched_predictions.npz',names=np.array(keep),case=np.array(case_keys),
        ids=np.concatenate([z['ids'] for z in packets]),truth=np.concatenate([z['truth'] for z in packets]),
        scores=np.concatenate([z['scores'][indices] for z in packets],axis=1),
        predictions=np.concatenate([z['predictions'][indices] for z in packets],axis=1))
    paths=[directory/'matched_predictions.npz']+sorted((directory/'summary').glob('*'))
    manifest=dict(kept=keep,discarded=[n for n in study.NAMES if n not in keep],
        rationale=rationale or 'Matched lambda0/.1/.3 heads and learned fusion diagnose diversity versus accuracy; calibration and two established controls support interpretation. Unsuccessful fixed-blend fold predictions and all optimizer/nested fold caches removed after full verification.',
        full_verification='summary/verification.json',
        files={str(p.relative_to(directory)):study.sha(p) for p in paths})
    study.save(directory/'retention.json',manifest)
    verify(directory)
    paths=list((directory/'predictions').iterdir())
    assert len(paths)==168 and all(p.suffix in ['.npz','.json'] for p in paths)
    for path in paths:path.unlink()
    (directory/'predictions').rmdir()


if __name__=='__main__':
    command=sys.argv[1] if len(sys.argv)>1 else 'verify'
    directory=Path(sys.argv[2]) if len(sys.argv)>2 else study.HERE/'results'
    if command=='compact':compact(directory)
    else:verify(directory)
