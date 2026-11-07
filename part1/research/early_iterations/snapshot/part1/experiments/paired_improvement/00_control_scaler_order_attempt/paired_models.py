"""Small controlled changes to current Ani and the trusted two-pass recipe."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize

from simple_models import decide

ANI = dict(C=.1, pseudo_weight=1., balance_source='augmented', binary_neighbors=False,
           lda_weight=.2, lda_source='augmented', gmm_weight=.2, rounds=2)
TRUSTED = dict(C=.3, pseudo_weight=.25, balance_source='verified', binary_neighbors=True,
               lda_weight=0., gmm_weight=0., rounds=2)


def prepare(features, use_gmm=True):
    """Only features explicitly permitted by the outer/inner fitting split enter."""
    scaler=StandardScaler().fit(features)
    prep=dict(scaler=scaler)
    if use_gmm:
        Z=scaler.transform(features)
        pca=PCA(30,random_state=0).fit(Z)
        gmm=GaussianMixture(30,covariance_type='diag',n_init=3,random_state=0).fit(pca.transform(Z))
        assert gmm.converged_
        prep.update(pca=pca,gmm=gmm)
    return prep


def expand(Y):
    row, reason=np.nonzero(Y)
    weight=1./Y.sum(axis=1)[row]
    return row,reason,weight


def lda_fit(Z,Y):
    row,reason,_=expand(Y)
    k=len(np.unique(reason))
    return LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(k)/k).fit(Z[row],reason)


class PairedModel:
    def __init__(self,spec):
        self.spec=dict(spec)

    def fit(self,X,Y,U,preprocessor=None):
        s=self.spec;self.preprocessor_=preprocessor or prepare(np.vstack([X,U]),s.get('gmm_weight',0)>0)
        self.scaler_=self.preprocessor_['scaler']
        Z,V=self.scaler_.transform(X),self.scaler_.transform(U)
        row,reason,weights=expand(Y)
        verified_counts=np.bincount(reason,weights=weights,minlength=5)+1e-9
        self.known_lda_=lda_fit(Z,Y) if s.get('lda_weight',0)>0 and s.get('lda_source')=='verified' else None
        if s.get('gmm_weight',0):
            pca,gmm=self.preprocessor_['pca'],self.preprocessor_['gmm']
            Rk=gmm.predict_proba(pca.transform(Z));Ru=gmm.predict_proba(pca.transform(V))
            Ra=Rk
        Za,Ya=Z,Y
        self.audit_=[]
        self.generated_mask_=np.zeros(len(U),bool)
        self.generated_labels_=np.empty((0,5),int)
        for step in range(s.get('rounds',2)+1):
            row,reason,weights=expand(Ya)
            counts=(np.bincount(reason,weights=weights,minlength=5)+1e-9) if s.get('balance_source','augmented')=='augmented' else verified_counts
            weights*=counts.mean()/counts[reason]
            weights*=np.where(row<len(X),1.,s.get('pseudo_weight',1.))
            self.lr_=LogisticRegression(C=s.get('C',.1),max_iter=3000,tol=s.get('tol',1e-4)).fit(Za[row],reason,sample_weight=weights)
            self.lda_=None
            if s.get('lda_weight',0)>0:
                self.lda_=self.known_lda_ if self.known_lda_ is not None else lda_fit(Za,Ya)
            self.neighbor_X_=normalize(Za)
            self.neighbor_Y_=Ya.copy() if s.get('binary_neighbors',False) else Ya/Ya.sum(axis=1,keepdims=True)
            self.audit_.append(dict(step=step,known=len(X),generated=len(Za)-len(X),known_weight=float(weights[row<len(X)].sum()),generated_weight=float(weights[row>=len(X)].sum())))
            if step==s.get('rounds',2):break
            P=self._base(V)
            if s.get('gmm_weight',0):
                T=Ya/Ya.sum(axis=1,keepdims=True)
                rates=(Ra.T@T+T.mean(axis=0))/(Ra.sum(axis=0)[:,None]+1)
                P=(1-s['gmm_weight'])*P+s['gmm_weight']*(Ru@rates)
            keep=P.max(axis=1)>=s.get('confidence',.8)
            self.generated_mask_=keep
            self.generated_labels_=decide(P[keep],.3)
            Za=np.vstack([Z,V[keep]]);Ya=np.vstack([Y,self.generated_labels_])
            if s.get('gmm_weight',0):Ra=np.vstack([Rk,Ru[keep]])
        return self

    def _base(self,Z):
        lr=np.zeros((len(Z),5));lr[:,self.lr_.classes_]=self.lr_.predict_proba(Z)
        similarity=normalize(Z)@self.neighbor_X_.T
        neighbors=np.argsort(-similarity,axis=1)[:,:self.spec.get('k',10)]
        weight=np.take_along_axis(similarity,neighbors,axis=1).clip(0)**2
        local=(weight[:,:,None]*self.neighbor_Y_[neighbors]).sum(axis=1)/(weight.sum(axis=1,keepdims=True)+1e-9)
        wknn=self.spec.get('knn_weight',.3);wlda=self.spec.get('lda_weight',0.)
        P=(1-wknn-wlda)*lr+wknn*local
        if self.lda_ is not None:
            ld=np.zeros_like(P);ld[:,self.lda_.classes_]=self.lda_.predict_proba(Z)
            P+=wlda*ld
        return P

    def predict_proba(self,X):return self._base(self.scaler_.transform(X))
    def predict(self,X):return decide(self.predict_proba(X),self.spec.get('cutoff',.3))


def first_specs():
    return [dict(ANI,name='ani_exact_control'),dict(TRUSTED,name='trusted_exact_control'),
            dict(TRUSTED,name='trusted_lda_verified',lda_weight=.2,lda_source='verified'),
            dict(TRUSTED,name='trusted_lda_augmented',lda_weight=.2,lda_source='augmented'),
            dict(TRUSTED,name='trusted_lda_verified_C0.1',lda_weight=.2,lda_source='verified',C=.1),
            dict(TRUSTED,name='trusted_lda_augmented_C0.1',lda_weight=.2,lda_source='augmented',C=.1),
            dict(TRUSTED,name='trusted_lda_verified_w0.1',lda_weight=.1,lda_source='verified'),
            dict(ANI,name='ani_without_gmm',gmm_weight=0.),
            dict(ANI,name='ani_trust_and_binary',pseudo_weight=.25,balance_source='verified',binary_neighbors=True)]
