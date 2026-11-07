"""Direct conditional sample-F1 utility regression with matched label regression."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics.pairwise import rbf_kernel

CHOICES = ((np.arange(1, 32)[:, None] >> np.arange(5)) & 1).astype(int)


def utilities(y):
    return 2*(y @ CHOICES.T)/(y.sum(axis=1)[:, None]+CHOICES.sum(axis=1)[None, :])


def fit(x, y, alpha):
    kernel = rbf_kernel(x, gamma=.001)
    column_mean, grand_mean = kernel.mean(axis=0), float(kernel.mean())
    centered = kernel-column_mean[None, :]-column_mean[:, None]+grand_mean
    targets = np.column_stack([y, utilities(y)])
    target_mean = targets.mean(axis=0)
    coefficient = cho_solve(cho_factor(centered+alpha*np.eye(len(x)), lower=True), targets-target_mean)
    return dict(x=x.copy(), column_mean=column_mean, grand_mean=grand_mean,
                target_mean=target_mean, coefficient=coefficient)


def predict(model, x, objective):
    kernel = rbf_kernel(x, model["x"], gamma=.001)
    centered = kernel-kernel.mean(axis=1)[:, None]-model["column_mean"][None, :]+model["grand_mean"]
    regression = centered @ model["coefficient"]+model["target_mean"]
    if objective == "f1":
        prediction = CHOICES[regression[:, 5:].argmax(axis=1)]
    else:
        prediction = (regression[:, :5] >= .5).astype(int)
        empty = prediction.sum(axis=1) == 0
        prediction[np.flatnonzero(empty), regression[empty, :5].argmax(axis=1)] = 1
    return prediction.astype(float)


def check_metric():
    # With uncertain mutually exclusive labels, selecting both scores 2/3,
    # exceeding either singleton's expected F1 (.6 or .4).
    y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    expected = np.array([.6, .4]) @ utilities(y)
    np.testing.assert_array_equal(CHOICES[expected.argmax()], [1, 1, 0, 0, 0])
    assert abs(expected.max()-2/3) < 1e-12
