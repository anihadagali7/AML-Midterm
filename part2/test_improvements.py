"""Checks for paired inference, frozen selection, and outcome exclusion."""
import copy
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from improvement_models import TAUS, fit_recipes
from improvement_report import eligible, paired_interval
from improvement_text import fit_extra, predict_extra


class ComparisonTests(unittest.TestCase):
    def frames(self):
        rows = []
        for trial in range(40):
            for phase in ["I", "II", "III"]:
                y = trial % 2
                rows.append(dict(ID=f"{trial}_{phase}", trial_id=str(trial), phase=phase,
                                 label=y, probability=.8 if y else .2))
        ref = pd.DataFrame(rows)
        candidate = ref.copy()
        candidate["probability"] = np.where(candidate.label, .9, .1)
        return {"fold0": dict(split=dict(kind="group_cv"), ani_stack=ref, candidate=candidate)}

    def test_known_paired_effect(self):
        got = paired_interval(self.frames(), "candidate", "ani_stack", "group_cv", 100)
        for field in ["delta", "lower", "upper"]:
            self.assertAlmostEqual(got[field], np.log(.8/.9))

    def test_repeating_same_trials_does_not_narrow_interval(self):
        frames = self.frames()
        frames["fold0"]["candidate"]["probability"] = np.linspace(.15, .85, 120)
        original = paired_interval(frames, "candidate", "ani_stack", "group_cv", 100)
        frames["repeat1"] = copy.deepcopy(frames["fold0"])
        repeated = paired_interval(frames, "candidate", "ani_stack", "group_cv", 100)
        self.assertEqual(original, repeated)

    def test_misaligned_predictions_rejected(self):
        frames = self.frames()
        frames["fold0"]["candidate"] = frames["fold0"]["candidate"].iloc[::-1]
        with self.assertRaises(AssertionError):
            paired_interval(frames, "candidate", "ani_stack", "group_cv", 10)

    def test_selection_requires_beating_both_references(self):
        summary = pd.DataFrame([
            dict(model="ani_stack", completed_splits=10, temporal=.53, cv=.58, worst=.59),
            dict(model="ani_first", completed_splits=10, temporal=.54, cv=.57, worst=.60),
            dict(model="good", completed_splits=10, temporal=.52, cv=.5705, worst=.59),
            dict(model="cv_regression", completed_splits=10, temporal=.51, cv=.58, worst=.59),
            dict(model="partial", completed_splits=10, temporal=.535, cv=.57, worst=.59)
        ]).set_index("model")
        self.assertEqual(eligible(summary), ["good"])
        summary.loc["good", "completed_splits"] = 9
        with self.assertRaises(AssertionError):
            eligible(summary)


class OutcomeBoundaryTests(unittest.TestCase):
    def test_evaluation_labels_and_identifiers_cannot_change_predictions(self):
        rng = np.random.default_rng(12)
        train = pd.DataFrame(dict(phase=["I", "II", "III"]*40,
                                 label=rng.integers(0, 2, 120), start_year=rng.integers(2000, 2012, 120)))
        ev = train.iloc[:18].copy()
        ev["ID"] = np.arange(len(ev))
        ev["trial_id"] = np.arange(len(ev))
        packet = dict(year_median=2006, year_cap=2013, bank={key:
                      dict(oof=rng.uniform(.01, .99, (len(train), 8)),
                           eval=rng.uniform(.01, .99, (len(ev), 8))) for key in TAUS})
        before, _ = fit_recipes(packet, train, ev)
        ev["label"] = 1-ev.label
        ev["ID"] = "forbidden"
        ev["trial_id"] = "forbidden"
        after, _ = fit_recipes(packet, train, ev)
        for model in before:
            np.testing.assert_array_equal(before[model], after[model])

    def test_supervised_word_ratio_has_expected_direction_and_handles_unseen_text(self):
        # The first word belongs exclusively to successful training trials,
        # the second exclusively to failed trials. Missing/unknown text stays
        # a valid intercept/phase-only prediction.
        train = pd.DataFrame(dict(label=[1,1,1,0,0,0]))
        counts = sparse.csr_matrix([[3,0], [2,0], [1,0], [0,2], [0,1], [0,5]])
        original = counts.copy()
        features = dict(crit_counts=counts, phase=sparse.csr_matrix(np.ones((6,1))))
        model = fit_extra(features, train, np.ones(6), kind=1)
        self.assertGreater(model[1][0], 0)
        self.assertLess(model[1][1], 0)
        self.assertEqual((counts != original).nnz, 0)
        query = dict(crit_counts=sparse.csr_matrix([[1,0],[0,1],[0,0]]),
                     phase=sparse.csr_matrix(np.ones((3,1))))
        p = predict_extra(query, model, kind=1)
        self.assertGreater(p[0], p[1])
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p > 0) & (p < 1)).all())


if __name__ == "__main__":
    unittest.main()
