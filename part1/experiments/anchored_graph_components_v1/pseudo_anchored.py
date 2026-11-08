"""Current Ani with three small changes: anchored weights, binary votes, weaker GMM.

Fit receives only permitted rows, in original CSV order. Fully missing labels
are -100; query/test rows must never be passed to fit. Prediction uses only the
150 stopping-reason embedding columns, ordered emb_0 through emb_149.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize


def select_reasons(scores):
    result=(scores>=.3).astype(int)
    result[np.arange(len(scores)),scores.argmax(axis=1)]=1
    return result


class PseudoTeacherAnchored:
    def fit(self, embeddings, labels, teacher_scores=None):
        X,Y=np.asarray(embeddings,float),np.asarray(labels,int)
        if X.ndim!=2 or X.shape[1]!=150 or Y.shape!=(len(X),5):
            raise ValueError('Expected 150 embeddings and five labels per row.')
        known=np.isin(Y,[0,1]).all(axis=1)
        unknown=(Y==-100).all(axis=1)
        if not (known|unknown).all() or (Y[known].sum(axis=1)==0).any() or (Y[known].sum(axis=0)==0).any():
            raise ValueError('Use complete nonempty known labels or five -100 values; all five reasons must be represented.')
        self.scaler_=StandardScaler().fit(X)
        Z=self.scaler_.transform(X)
        self.pca_=PCA(30,random_state=0).fit(Z)
        self.gmm_=GaussianMixture(30,covariance_type='diag',n_init=3,random_state=0).fit(self.pca_.transform(Z))
        assert self.gmm_.converged_
        K,V=Z[known],Z[unknown];targets=Y[known]
        Rk=self.gmm_.predict_proba(self.pca_.transform(K))
        Ru=self.gmm_.predict_proba(self.pca_.transform(V))
        counts=(targets/targets.sum(axis=1,keepdims=True)).sum(axis=0)
        self.class_weights_=counts.mean()/counts
        Za,Ya,Ra=K,targets,Rk
        self.round_audit_=[]
        for step in range(3):
            row,reason=np.nonzero(Ya)
            weights=self.class_weights_[reason]/Ya.sum(axis=1)[row]
            weights*=np.where(row<len(K),1.,.25)
            self.logistic_=LogisticRegression(C=.1,max_iter=3000,tol=1e-4).fit(Za[row],reason,sample_weight=weights)
            self.lda_=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(5)/5).fit(Za[row],reason)
            self.neighbor_X_=normalize(Za)
            self.neighbor_Y_=Ya.copy()
            self.round_audit_.append(dict(step=step,known=len(K),generated=len(Za)-len(K),
                                          known_weight=float(weights[row<len(K)].sum()),
                                          generated_weight=float(weights[row>=len(K)].sum())))
            if step==2:break
            T=Ya/Ya.sum(axis=1,keepdims=True)
            rates=(Ra.T@T+T.mean(axis=0))/(Ra.sum(axis=0)[:,None]+1)
            scores=(.95*self._scores(V)+.05*(Ru@rates) if teacher_scores is None
                    else np.asarray(teacher_scores, float))
            assert scores.shape == (len(V), 5)
            self.generated_mask_=scores.max(axis=1)>=.8
            self.generated_labels_=select_reasons(scores[self.generated_mask_])
            Za=np.vstack([K,V[self.generated_mask_]])
            Ya=np.vstack([targets,self.generated_labels_])
            Ra=np.vstack([Rk,Ru[self.generated_mask_]])
        return self

    def _scores(self,Z):
        logistic=np.zeros((len(Z),5))
        logistic[:,self.logistic_.classes_]=self.logistic_.predict_proba(Z)
        similarity=normalize(Z)@self.neighbor_X_.T
        neighbors=np.argsort(-similarity,axis=1)[:,:10]
        weights=np.take_along_axis(similarity,neighbors,axis=1).clip(0)**2
        local=(weights[:,:,None]*self.neighbor_Y_[neighbors]).sum(axis=1)
        local/=weights.sum(axis=1,keepdims=True)+1e-9
        lda=np.zeros_like(logistic)
        lda[:,self.lda_.classes_]=self.lda_.predict_proba(Z)
        return (1-.3-.2)*logistic+.3*local+.2*lda

    def predict_proba(self, embeddings):
        """Selection scores, not calibrated marginal probabilities."""
        return self._scores(self.scaler_.transform(np.asarray(embeddings,float)))

    def predict(self, embeddings):
        return select_reasons(self.predict_proba(embeddings))
