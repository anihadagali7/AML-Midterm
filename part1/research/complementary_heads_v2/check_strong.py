"""Check v2 gradient, threshold-space conversion, and synthetic optimizer."""
import strong_study as wrapper
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import approx_fprime
from threadpoolctl import threadpool_limits
from complement import loss_gradient,decide,centered_logit
from strong_kernel import StrongKernel


def check():
    rng=np.random.default_rng(1128)
    F=np.c_[np.ones(23),rng.normal(size=(23,4))];Y=rng.integers(0,2,size=(23,5))
    e=rng.normal(size=(23,5));e-=e.mean(0);w=rng.normal(0,.2,size=25)
    errors=[]
    for strength in [0.,.1,.3]:
        _,gradient=loss_gradient(w,F,Y,e,strength,.00005)
        numeric=approx_fprime(w,lambda b:loss_gradient(b,F,Y,e,strength,.00005)[0],1e-7)
        delta=float(np.abs(gradient-numeric).max());assert delta<2e-7;errors.append(delta)
    a=rng.uniform(.001,.999,size=(71,5));b=rng.uniform(.001,.999,size=(71,5))
    centered=expit(logit(b)-logit(.3))
    back=expit(logit(centered)+logit(.3))
    np.testing.assert_allclose(back,b,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(decide(expit(centered_logit(.8*a+.2*back))),wrapper.study.select_reasons(.8*a+.2*b))
    X=rng.normal(size=(180,150));Y=np.full((180,5),-100);Y[:90]=rng.integers(0,2,size=(90,5))
    with threadpool_limits(limits=1):
        model=StrongKernel().fit(X,Y,rng.uniform(.01,.99,size=(90,5)))
        for name,z in model.predict_logits(X[:7]).items():assert z.shape==(7,5) and np.isfinite(z).all()
    result=dict(passed=True,max_gradient_absolute_errors=errors,raw_blend_decoder_equivalence=True,synthetic_optimizer=True)
    wrapper.study.save(wrapper.HERE/'math_checks.json',result);print(result)


if __name__=='__main__':check()
