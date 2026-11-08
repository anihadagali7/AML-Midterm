"""Check the supplementary averaging against Dylan's unchanged final-fit builder.

Map the builder's named final train/test sets to the validation benchmark split;
this does not create a submission or use the actual test labels.
"""
from dataclasses import replace
import json
from pathlib import Path

import run as study
import numpy as np
from threadpoolctl import threadpool_limits
from p1lib import cache
from p1lib.submit import final_fit, predict_test

HERE = Path(__file__).resolve().parent


def main():
    case = next(c for c in study.cases() if c['access']=='native_dylan' and c['kind']=='validation')
    D, ctx, _, _ = study.prepare(case)
    idx = dict(D.idx, labval=ctx.train, test=ctx.eval, unl=ctx.pool)
    mapped = replace(D, idx=idx)
    checks=[]
    with threadpool_limits(limits=1), np.load(HERE/'five_seed/results'/(case['key']+'.npz')) as z:
        for name, stem in study.CANDIDATES.items():
            cfg=study.read(HERE/'manifests'/(stem+'.manifest.json'))['config']
            cache.clear()
            fitted=final_fit(cfg,mapped,n_seeds=5)
            yp,Q,prior=predict_test(cfg,mapped,fitted)
            np.testing.assert_allclose(fitted['P_test'],z[name+'_P'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(Q,z[name+'_Q'],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(yp,z[name+'_predictions'])
            checks.append(dict(model=name,raw_average_max_error=float(abs(fitted['P_test']-z[name+'_P']).max()),
                               headed_probability_max_error=float(abs(Q-z[name+'_Q']).max()),identical_predictions=True))
    result=dict(status='passed',case=case['key'],checks=checks,
                reference='Unmodified source/p1lib/submit.py final_fit and predict_test, with named final sets mapped to the validation case. No new test CSV.')
    study.save(HERE/'five_seed/source_builder_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
