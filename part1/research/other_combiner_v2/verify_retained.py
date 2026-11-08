"""Rescore compact useful comparisons after unsuccessful fold-cache cleanup."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import run as study
spec=importlib.util.spec_from_file_location('this_study_full_check',study.HERE/'verify.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
independent=checker.independent


def main():
    study.guard();manifest=study.read(study.HERE/'retention.json')
    for relative,digest in manifest['retained_sha256'].items():
        assert study.sha(study.HERE/relative)==digest,relative
    z=study.loadnp(study.HERE/'results/retained_comparison.npz')
    scores=pd.read_csv(study.HERE/'results/summary/repeat_scores.csv')
    labels=pd.read_csv(study.HERE/'results/summary/per_label_scores.csv')
    groups=[(pool,int(seed),(z['pool']==pool)&(z['seed']==seed))
            for pool,seed in sorted(set(zip(z['pool'],z['seed'])))]
    groups.append(('chronology98',-1,np.char.startswith(z['pool'],'through_')))
    checks=0
    for pool,seed,keep in groups:
        y=z['truth'][keep]
        assert len(set(z['ids'][keep]))==len(y)
        for j,name in enumerate(z['names']):
            p=z['predictions'][j,keep]
            np.testing.assert_array_equal(p,study.decide(z['scores'][j,keep]))
            sample,macro,precision,recall,f,support=independent(y,p)
            r=scores[(scores.pool==pool)&(scores.seed==seed)&(scores.recipe==name)].iloc[0]
            np.testing.assert_allclose([sample,macro],[r.sample_f1,r.macro_f1],atol=1e-12,rtol=0)
            l=labels[(labels.pool==pool)&(labels.seed==seed)&(labels.recipe==name)].set_index('label').loc[study.LABELS]
            np.testing.assert_allclose(np.c_[precision,recall,f,support],l[['precision','recall','f1','support']],atol=1e-12,rtol=0)
            checks+=1
    assert checks==63
    assert not any(study.read(study.HERE/'results/summary/qualification.json').values())
    print(dict(passed=True,retained_metric_rows_independently_rescored=checks,
               all_seven_compact_result_hashes_verified=True,
               full_189_metric_checks_and_nested_audits='Recorded before cleanup in verification.json'))


if __name__=='__main__':main()
