"""Independent mathematical controls for weighted LDA and unknown neighbor labels."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from study_run import HERE,save
from partial_model import BinaryOtherLDA,observed_other,other_neighbor_vote


def main():
    rng=np.random.default_rng(81);X=rng.normal(size=(80,12));y=np.repeat([0,1],[50,30]);Q=rng.normal(size=(21,12))
    m=BinaryOtherLDA().fit(X,y,np.ones(80))
    ref=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.array([.8,.2])).fit(X,y)
    error=float(np.abs(m.predict_proba(Q)-ref.predict_proba(Q)).max())
    np.testing.assert_allclose(m.predict_proba(Q),ref.predict_proba(Q),atol=1e-9,rtol=0)
    bank=np.array([[.8,.6],[0,1],[1,0]],float);labels=np.array([[0,0,0,1,1],[0,0,0,1,0],[0,0,0,1,0]])
    obs=observed_other(labels,2,True);np.testing.assert_array_equal(obs,[True,True,False])
    partial=other_neighbor_vote(np.array([[1.,0]]),bank,labels,obs).item()
    complete=other_neighbor_vote(np.array([[1.,0]]),bank,labels,np.ones(3,bool)).item()
    np.testing.assert_allclose(partial,1.,atol=1e-8)
    np.testing.assert_allclose(complete,.64/(1+.64),atol=1e-8)
    # Marking a selected pseudo Other positive must restore it as observed.
    labels[2,4]=1;assert observed_other(labels,2,True).all()
    result=dict(passed=True,unit_weight_lda_max_error=error,unknown_neighbor_does_not_vote_zero=True,
        gold_negatives_preserved=True,selected_pseudo_positive_observed=True,partial_vote=partial,complete_vote=complete)
    save(HERE/'math_check.json',result);print(result)


if __name__=='__main__':main()
