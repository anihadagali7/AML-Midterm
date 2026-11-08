"""Partial Other labels: an unselected pseudo-label need not be a negative."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler,normalize
from anchored_solution import AnchoredReasonClassifier,select_reasons


def observed_other(labels,n_gold,partial):
    """Gold labels are complete; absent pseudo Other is unknown in partial mode."""
    return np.ones(len(labels),bool) if not partial else (np.arange(len(labels))<n_gold)|(labels[:,4]==1)


def other_neighbor_vote(Z,bank,labels,observed):
    # Retrieve ten *observed* Other labels; unknown entries neither vote zero nor
    # crowd an observed neighbor out. Gold negatives remain eligible.
    bank=bank[observed];target=labels[observed,4]
    similarity=normalize(Z)@bank.T
    nearest=np.argsort(-similarity,axis=1)[:,:10]
    w=np.take_along_axis(similarity,nearest,axis=1).clip(0)**2
    return (w*target[nearest]).sum(1)/(w.sum(1)+1e-9)


class BinaryOtherLDA:
    """Two Gaussian classes; fixed .8/.2 priors and weighted shrinkage covariance."""
    def fit(self,X,y,w):
        prior=np.array([.8,.2]);means=[];cov=np.zeros((X.shape[1],X.shape[1]))
        for k in [0,1]:
            x=X[y==k];wk=w[y==k];assert len(x)>1 and wk.sum()>0
            mu=np.average(x,axis=0,weights=wk);means.append(mu)
            residual=(x-mu)*np.sqrt(wk/wk.mean())[:,None]
            sd=np.sqrt(np.mean(residual**2,axis=0));sd[sd<1e-12]=1
            C=ledoit_wolf(residual/sd,assume_centered=True)[0]
            cov+=prior[k]*(sd[:,None]*C*sd[None,:])
        means=np.asarray(means)
        self.coef_=np.linalg.lstsq(cov,means.T,rcond=None)[0].T
        self.intercept_=-.5*(self.coef_*means).sum(1)+np.log(prior)
        return self

    def predict_proba(self,X):
        s=X@self.coef_.T+self.intercept_;s-=s.max(1,keepdims=True)
        p=np.exp(s);return p/p.sum(1,keepdims=True)


class PartialOtherAnchored(AnchoredReasonClassifier):
    def __init__(self,mode):
        assert mode in ['control','neighbor_unknown','binary_complete','binary_unknown']
        self.mode=mode;self.binary=mode.startswith('binary_')
        self.partial=mode in ['neighbor_unknown','binary_unknown']

    def fit(self,X,Y):
        X,Y=np.asarray(X,float),np.asarray(Y,int)
        known=np.isin(Y,[0,1]).all(1);unknown=(Y==-100).all(1)
        assert (known|unknown).all() and (Y[known].sum(0)>0).all()
        self.n_gold_=int(known.sum());self.gold_=Y[known].copy()
        self.scaler_=StandardScaler().fit(X);Z=self.scaler_.transform(X)
        self.pca_=PCA(30,random_state=0).fit(Z)
        self.gmm_=GaussianMixture(30,covariance_type='diag',n_init=3,random_state=0).fit(self.pca_.transform(Z))
        assert self.gmm_.converged_
        K,V=Z[known],Z[unknown];targets=Y[known]
        Rk=self.gmm_.predict_proba(self.pca_.transform(K));Ru=self.gmm_.predict_proba(self.pca_.transform(V))
        counts=(targets/targets.sum(1,keepdims=True)).sum(0);self.class_weights_=counts.mean()/counts
        Za,Ya,Ra=K,targets,Rk;self.round_audit_=[]
        for step in range(3):
            np.testing.assert_array_equal(Ya[:self.n_gold_],self.gold_)
            row,reason=np.nonzero(Ya)
            weights=self.class_weights_[reason]/Ya.sum(1)[row]
            weights*=np.where(row<self.n_gold_,1.,.25)
            self.logistic_=LogisticRegression(C=.1,max_iter=3000,tol=1e-4).fit(Za[row],reason,sample_weight=weights)
            self.lda_=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(5)/5).fit(Za[row],reason)
            self.neighbor_X_=normalize(Za);self.neighbor_Y_=Ya.copy()
            self.other_observed_=observed_other(Ya,self.n_gold_,self.partial)
            audit=dict(step=step,gold=self.n_gold_,generated=len(Za)-self.n_gold_,
                generated_multilabel=int((Ya[self.n_gold_:].sum(1)>1).sum()),
                observed_other_positive=int(((Ya[:,4]==1)&self.other_observed_).sum()),
                observed_other_negative=int(((Ya[:,4]==0)&self.other_observed_).sum()),
                unknown_other=int((~self.other_observed_).sum()))
            if self.partial:
                assert audit['observed_other_negative']==int((targets[:,4]==0).sum())
                assert self.other_observed_[:self.n_gold_].all()
            if self.binary:
                obs=self.other_observed_;yy=Ya[obs,4];xx=Za[obs]
                raw=np.where(np.arange(len(Za))[obs]<self.n_gold_,1.,.25)
                mass=np.bincount(yy,weights=raw,minlength=2)
                # Hold class prior and total training weight fixed when dropping
                # unknown negatives; otherwise positive-only selection changes prior.
                w=raw*(self.n_gold_*np.array([.8,.2])/mass)[yy]
                np.testing.assert_allclose(np.bincount(yy,weights=w,minlength=2),self.n_gold_*np.array([.8,.2]),atol=1e-10)
                self.other_lr_=LogisticRegression(C=.1,max_iter=3000,tol=1e-4).fit(xx,yy,sample_weight=w)
                self.other_lda_=BinaryOtherLDA().fit(xx,yy,raw)
                audit['binary_training_rows']=len(yy)
                audit['binary_weight_by_class']=np.bincount(yy,weights=w,minlength=2).tolist()
            self.round_audit_.append(audit)
            if step==2:break
            T=Ya/Ya.sum(1,keepdims=True)
            rates=(Ra.T@T+T.mean(0))/(Ra.sum(0)[:,None]+1)
            if self.mode=='binary_unknown':
                o=self.other_observed_
                rates[:,4]=(Ra[o].T@T[o,4]+T[o,4].mean())/(Ra[o].sum(0)+1)
            s=.95*self._scores(V)+.05*(Ru@rates)
            keep=s.max(1)>=.8;generated=select_reasons(s[keep])
            self.generated_mask_=keep;self.generated_labels_=generated
            Za=np.vstack([K,V[keep]]);Ya=np.vstack([targets,generated]);Ra=np.vstack([Rk,Ru[keep]])
        return self

    def _scores(self,Z):
        logistic=np.zeros((len(Z),5));logistic[:,self.logistic_.classes_]=self.logistic_.predict_proba(Z)
        similarity=normalize(Z)@self.neighbor_X_.T
        neighbors=np.argsort(-similarity,axis=1)[:,:10]
        weights=np.take_along_axis(similarity,neighbors,axis=1).clip(0)**2
        local=(weights[:,:,None]*self.neighbor_Y_[neighbors]).sum(1)
        local/=weights.sum(1,keepdims=True)+1e-9
        lda=np.zeros_like(logistic);lda[:,self.lda_.classes_]=self.lda_.predict_proba(Z)
        if self.partial:
            local[:,4]=other_neighbor_vote(Z,self.neighbor_X_,self.neighbor_Y_,self.other_observed_)
        if self.binary:
            logistic[:,4]=self.other_lr_.predict_proba(Z)[:,1]
            lda[:,4]=self.other_lda_.predict_proba(Z)[:,1]
        return .5*logistic+.3*local+.2*lda
