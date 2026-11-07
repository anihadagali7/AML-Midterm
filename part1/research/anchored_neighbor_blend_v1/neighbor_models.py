"""Two isolated changes to the frozen anchored model's neighbor votes.

The inherited fit method preserves preprocessing, logistic regression, LDA,
cluster assistance, confidence, iterations and decoder. Modified neighbor votes
are used both while generating labels and at final prediction.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.preprocessing import normalize

from anchored_solution import AnchoredReasonClassifier


def embedding_groups(x):
    # Same rounded-coordinate hash as the frozen grouped benchmark.
    return pd.util.hash_pandas_object(pd.DataFrame(np.asarray(x)).round(10), index=False).to_numpy()


def centered(scores):
    """Map native .3 decision cutoff to .5; these are not calibrated probabilities."""
    return expit(logit(np.clip(scores, 1e-9, 1-1e-9))-logit(.3))


def decide(scores):
    result = (scores >= .5).astype(int)
    result[np.arange(len(scores)), scores.argmax(axis=1)] = 1
    return result


class NeighborAdjustedAnchored(AnchoredReasonClassifier):
    def __init__(self, mode):
        if mode not in ["row_control", "grouped_neighbors", "pseudo_neighbor_quarter"]:
            raise ValueError(mode)
        self.mode = mode

    def fit(self, embeddings, labels):
        x, y = np.asarray(embeddings, float), np.asarray(labels, int)
        known = np.isin(y, [0, 1]).all(axis=1)
        group = embedding_groups(x)
        self.gold_groups_ = group[known]
        self.masked_groups_ = group[~known]
        self.n_gold_ = int(known.sum())
        return super().fit(x, y)

    def _scores(self, z):
        logistic = np.zeros((len(z), 5))
        logistic[:, self.logistic_.classes_] = self.logistic_.predict_proba(z)
        bank_x, bank_y = self.neighbor_X_, self.neighbor_Y_
        generated = len(bank_x)-self.n_gold_
        groups = self.gold_groups_
        if generated:
            assert int(self.generated_mask_.sum()) == generated
            groups = np.r_[groups, self.masked_groups_[self.generated_mask_]]
        assert len(groups) == len(bank_x)
        distinct, first, inverse, counts = np.unique(
            groups, return_index=True, return_inverse=True, return_counts=True)
        if self.mode == "grouped_neighbors":
            # One vote per group; average conflicting/member label vectors.
            # Preserve first-occurrence order to make similarity ties deterministic.
            order = np.argsort(first)
            totals = np.zeros((len(distinct), 5))
            np.add.at(totals, inverse, bank_y)
            bank_x = bank_x[first[order]]
            bank_y = (totals/counts[:, None])[order]
        similarity = normalize(z) @ bank_x.T
        neighbors = np.argsort(-similarity, axis=1)[:, :10]
        weights = np.take_along_axis(similarity, neighbors, axis=1).clip(0)**2
        if self.mode == "pseudo_neighbor_quarter":
            # Keep top-ten retrieval unchanged; modify votes and denominator.
            weights *= np.where(neighbors < self.n_gold_, 1., .25)
        local = (weights[:, :, None]*bank_y[neighbors]).sum(axis=1)
        local /= weights.sum(axis=1, keepdims=True)+1e-9
        lda = np.zeros_like(logistic)
        lda[:, self.lda_.classes_] = self.lda_.predict_proba(z)
        self.neighbor_audit_ = dict(mode=self.mode, gold_rows=self.n_gold_,
                                    generated_rows=generated, original_bank_rows=len(groups),
                                    distinct_embedding_groups=len(distinct), voting_bank_rows=len(bank_x),
                                    largest_embedding_group=int(counts.max()),
                                    generated_vote_weight=.25 if self.mode == "pseudo_neighbor_quarter" else 1.)
        return (1-.3-.2)*logistic + .3*local + .2*lda
