"""Small mathematical checks of the new voting rules; no model fitting."""
from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neighbor_models import NeighborAdjustedAnchored, embedding_groups


class ZeroClassifier:
    classes_ = np.arange(5)

    def predict_proba(self, x):
        return np.zeros((len(x), 5))


def fixture(x, y, mode, gold=None):
    m = NeighborAdjustedAnchored(mode)
    m.n_gold_ = len(x) if gold is None else gold
    m.neighbor_X_, m.neighbor_Y_ = normalize(x), y.copy()
    groups = embedding_groups(x)
    m.gold_groups_, m.masked_groups_ = groups[:m.n_gold_], groups[m.n_gold_:]
    m.generated_mask_ = np.ones(len(x)-m.n_gold_, bool)
    m.logistic_ = m.lda_ = ZeroClassifier()
    return m


def main():
    x = np.eye(150)[:3]
    y = np.eye(5, dtype=int)[:3]
    q = (x[0]+.4*x[1]+.2*x[2])[None]
    repeated_x = np.vstack([x, np.repeat(x[1:2], 71, axis=0)])
    repeated_y = np.vstack([y, np.repeat(y[1:2], 71, axis=0)])
    grouped = fixture(x, y, "grouped_neighbors")._scores(q)
    repeated = fixture(repeated_x, repeated_y, "grouped_neighbors")
    np.testing.assert_allclose(repeated._scores(q), grouped, atol=1e-14, rtol=0)
    assert repeated.neighbor_audit_["voting_bank_rows"] == 3
    assert not np.allclose(fixture(repeated_x, repeated_y, "row_control")._scores(q), grouped)
    # Unique groups preserve the original row-vote computation.
    np.testing.assert_allclose(fixture(x, y, "row_control")._scores(q), grouped, atol=1e-14, rtol=0)
    # A shared vector with conflicting labels contributes the member average.
    mixed = fixture(np.vstack([x, x[1]]), np.vstack([y, y[2]]), "grouped_neighbors")
    mean_labels = y.astype(float)
    mean_labels[1] = (y[1]+y[2])/2
    expected = fixture(x, mean_labels, "row_control")._scores(q)
    np.testing.assert_allclose(mixed._scores(q), expected, atol=1e-14, rtol=0)
    # Equal similarity: one gold and one quarter-weight generated vote.
    weighted = fixture(x[:2], y[:2], "pseudo_neighbor_quarter", gold=1)
    actual = weighted._scores((x[0]+x[1])[None])/.3
    expected = np.array([[.5, .125, 0, 0, 0]])/(.625+1e-9)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    print("Passed: duplicate invariance, unique-group parity, conflicting-label averaging, quarter-weight normalization; fits=0.")


if __name__ == "__main__":
    main()
