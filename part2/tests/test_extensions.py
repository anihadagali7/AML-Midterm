"""Behavioral checks for the new neighborhood estimator."""
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from extension_neighbors import neighbor_predictions


class NeighborTests(unittest.TestCase):
    def frames(self):
        train=pd.DataFrame(dict(phase=["I","I","II","II","III","III"],label=[0,1]*3,
                                start_year=[2000,2010]*3,ID=list("abcdef"),trial_id=list("abcdef")))
        ev=pd.DataFrame(dict(phase=["I","II","III"],label=[0,0,0],start_year=[2014]*3))
        fa={k:sparse.csr_matrix(np.ones((6,1))) for k in ["knn","disease","drug","icd"]}
        fb={k:sparse.csr_matrix(np.ones((3,1))) for k in fa}
        return train,ev,fa,fb

    def test_recency_moves_identical_neighbors_toward_newer_outcomes(self):
        train,ev,fa,fb=self.frames();p=neighbor_predictions(fa,fb,train,ev)
        np.testing.assert_allclose(p[:,:3],.5)
        self.assertTrue((p[:,3]>.8).all())
        self.assertTrue((p[:,4]>p[:,3]).all())

    def test_zero_similarity_uses_training_phase_prior(self):
        train,ev,fa,fb=self.frames()
        fb={k:sparse.csr_matrix(v.shape) for k,v in fb.items()}
        p=neighbor_predictions(fa,fb,train,ev)
        np.testing.assert_allclose(p[:,:3],.5)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p[:,3]>.7).all())

    def test_heldout_labels_and_identifiers_do_not_enter_neighbors(self):
        train,ev,fa,fb=self.frames();before=neighbor_predictions(fa,fb,train,ev)
        ev["label"]=1;ev["ID"]="forbidden";ev["trial_id"]="forbidden";ev["start_year"]=9999
        np.testing.assert_array_equal(before,neighbor_predictions(fa,fb,train,ev))


if __name__=="__main__":unittest.main()
