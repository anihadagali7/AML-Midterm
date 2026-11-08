"""Synthetic check that training feature construction excludes the entire own group."""
import numpy as np
from model import ReasonGeometry
rng=np.random.default_rng(9)
x=rng.normal(size=(120,150));x[1]=x[0]
y=(rng.uniform(size=(120,5))>.6).astype(int);y[np.arange(120),rng.integers(5,size=120)]=1;y[1]=y[0];y[40:]=-100
changed=y.copy();changed[:2]=1-changed[:2]
a=ReasonGeometry().fit(x,y);b=ReasonGeometry().fit(x,changed)
np.testing.assert_allclose(a.transform(x[:2],training=True),b.transform(x[:2],training=True),atol=0,rtol=0)
np.testing.assert_array_equal(a.augmented[:40],y[:40]);assert np.all(a.augmented[40:].sum(1)>0)
assert abs(a.weights[40:].sum()-4)<1e-12
np.testing.assert_array_equal(a.transform(x[:1]),a.transform(x[:7])[:1])
print('Whole-group exclusion, observed-label preservation, pseudo-weight total and query-batch invariance passed on synthetic data.')
