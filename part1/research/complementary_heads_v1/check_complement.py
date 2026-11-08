"""Finite-difference the new nonconvex loss and verify nested group boundaries."""
import numpy as np
from scipy.optimize import approx_fprime
from scipy.special import expit
from threadpoolctl import threadpool_limits
import head_study as study
from complement import loss_gradient, ComplementaryKernel, centered_logit, decide, fit_fusion


def check():
    rng=np.random.default_rng(1127)
    f=np.c_[np.ones(19),rng.normal(size=(19,6))]
    y=rng.integers(0,2,size=(19,5))
    error=rng.normal(size=(19,5));error-=error.mean(0)
    w=rng.normal(0,.2,size=35)
    errors=[]
    for strength in [0.,.1,.3]:
        _,g=loss_gradient(w,f,y,error,strength)
        numeric=approx_fprime(w,lambda b:loss_gradient(b,f,y,error,strength)[0],1e-7)
        delta=float(np.abs(g-numeric).max());assert delta<2e-7,delta;errors.append(delta)
    # Calibration zero-correction boundary matches anchored's exact decoder.
    scores=rng.uniform(.001,.999,size=(80,5))
    np.testing.assert_array_equal(study.select_reasons(scores),decide(expit(centered_logit(scores))))
    checked=0;retry_seeds=set()
    for case in study.cases():
        study.reference(case)
        assert set(case['fit'].ID).isdisjoint(case['query'].ID)
        if case['key']!='validation':
            assert not set(study.benchmark.groups(case['fit']))&set(study.benchmark.groups(case['query']))
        for _,_,part,held,seed in study.inner_inputs(case['fit'],270301):
            retry_seeds.add(seed);checked+=1
            for _,_,sub,subheld,subseed in study.inner_inputs(part,270302):
                assert set(sub.ID).isdisjoint(held.ID)
                assert set(subheld.ID).isdisjoint(held.ID)
                retry_seeds.add(subseed);checked+=1
    # The full head optimizer and readout can run without any query-label access.
    X=rng.normal(size=(180,150));Y=np.full((180,5),-100)
    Y[:90]=rng.integers(0,2,size=(90,5));anchor=rng.uniform(.01,.99,size=(90,5))
    with threadpool_limits(limits=1):
        model=ComplementaryKernel().fit(X,Y,anchor)
        logits=model.predict_logits(X[:90])
        for name,v in logits.items():
            assert v.shape==(90,5) and np.isfinite(v).all()
            p,_=fit_fusion(anchor,v,Y[:90],anchor[:7],v[:7]);assert p.shape==(7,5)
    result=dict(passed=True,max_gradient_absolute_errors=errors,
        checked_nested_partitions=checked,chosen_inner_seeds=sorted(retry_seeds),
        synthetic_optimizer_and_fusion=True,decoder_equivalence=True)
    print(result);return result


if __name__=='__main__':study.save(study.HERE/'math_and_boundaries.json',check())
