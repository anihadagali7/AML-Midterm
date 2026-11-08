"""Fixed Other specialists, nested residual combiners and weighted anchored LDA."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.covariance import ledoit_wolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from anchored_solution import AnchoredReasonClassifier, select_reasons


def centered_logit(scores):
    return logit(np.clip(scores, 1e-6, 1-1e-6))-logit(.3)


def decide(scores):
    pred=(scores>=.5).astype(int)
    pred[np.arange(len(scores)), scores.argmax(1)]=1
    return pred


class OtherSpecialists:
    """Gold-only RBF heads; geometry uses eligible gold and masked inputs."""
    def fit(self, X, Y):
        gold=(Y>=0).all(1)
        self.scaler=StandardScaler().fit(X)
        Z=self.scaler.transform(X[gold]); y=Y[gold]
        self.models=[]
        for keep in [np.ones(len(y), bool), y[:,3]==0, y[:,3]==1]:
            target=y[keep,4]
            if len(np.unique(target))<2:
                self.models.append(float(2*target.mean()-1) if len(target) else -1.)
            else:
                self.models.append(SVC(C=1., kernel='rbf', gamma=1/150,
                    class_weight='balanced', probability=False, tol=1e-5).fit(Z[keep],target))
        return self

    def predict(self, X):
        Z=self.scaler.transform(X)
        return np.column_stack([np.full(len(X),m) if isinstance(m,float)
                                else m.decision_function(Z) for m in self.models])


def offset_fit(F, y, offset, penalty=20.):
    """Sum binary log loss + lambda/2 * squared coefficients, including bias."""
    def objective(b):
        z=offset+F@b
        loss=np.logaddexp(0,z).sum()-y@z+.5*penalty*(b@b)
        grad=F.T@(expit(z)-y)+penalty*b
        return loss,grad
    result=minimize(objective,np.zeros(F.shape[1]),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8))
    if not result.success and np.max(np.abs(result.jac))>1e-5:
        raise RuntimeError(result.message)
    return result.x


def learned_heads(a,c,h,qa,qc,qh,y):
    """Train only on inner-OOF inputs. Evaluate on separate outer query scores."""
    b=.8*a+.2*c; qb=.8*qa+.2*qc
    o=centered_logit(b); qo=centered_logit(qb)
    baseline=expit(qo)
    z=np.clip(np.stack([centered_logit(a),centered_logit(c)],axis=2),-6,6)/3
    qz=np.clip(np.stack([centered_logit(qa),centered_logit(qc)],axis=2),-6,6)/3
    scores={}; coefficients={}
    for name in ['calibration_control','combiner_per_label']:
        out=baseline.copy(); bs=[]
        for j in range(5):
            if name=='calibration_control':
                F=np.c_[np.ones(len(y)),np.clip(o[:,j],-6,6)/3]
                Q=np.c_[np.ones(len(qa)),np.clip(qo[:,j],-6,6)/3]
            else:
                F=np.c_[np.ones(len(y)),z[:,j]]
                Q=np.c_[np.ones(len(qa)),qz[:,j]]
            coef=offset_fit(F,y[:,j],o[:,j]);bs.append(coef.tolist())
            out[:,j]=expit(qo[:,j]+Q@coef)
        scores[name]=out;coefficients[name]=bs
    scores['combiner_other_only']=baseline.copy()
    scores['combiner_other_only'][:,4]=scores['combiner_per_label'][:,4]
    for name,idx in [('other_rbf',[0]),('other_conditional_rbf',[1,2])]:
        mean=h[:,idx].mean(0);sd=np.maximum(h[:,idx].std(0),.1)
        F=np.clip((h[:,idx]-mean)/sd,-4,4)
        Q=np.clip((qh[:,idx]-mean)/sd,-4,4)
        if len(idx)==2:
            r=expit(o[:,3]);qr=expit(qo[:,3])
            F*=np.c_[1-r,r];Q*=np.c_[1-qr,qr]
        F=np.c_[np.ones(len(y)),F];Q=np.c_[np.ones(len(qa)),Q]
        coef=offset_fit(F,y[:,4],o[:,4])
        out=baseline.copy();out[:,4]=expit(qo[:,4]+Q@coef)
        scores[name]=out
        coefficients[name]=dict(beta=coef.tolist(),mean=mean.tolist(),std=sd.tolist())
    return scores,coefficients


class WeightedReasonLDA:
    """Uniform-prior weighted class means and standardized shrinkage covariances."""
    def fit(self,X,y,w):
        self.classes_=np.arange(5);means=[];cov=np.zeros((X.shape[1],X.shape[1]))
        for k in self.classes_:
            Xk=X[y==k];wk=w[y==k]
            mu=np.average(Xk,axis=0,weights=wk);means.append(mu)
            R=(Xk-mu)*np.sqrt(wk/wk.mean())[:,None]
            # RMS, not variance around the weighted residual array's unweighted mean.
            sd=np.sqrt(np.mean(R*R,axis=0));sd[sd<1e-12]=1
            C=ledoit_wolf(R/sd,assume_centered=True)[0]
            cov+=.2*(sd[:,None]*C*sd[None,:])
        means=np.array(means)
        self.coef_=np.linalg.lstsq(cov,means.T,rcond=None)[0].T
        self.intercept_=-.5*np.sum(means*self.coef_,axis=1)+np.log(.2)
        return self

    def predict_proba(self,X):
        z=X@self.coef_.T+self.intercept_;z-=z.max(1,keepdims=True)
        p=np.exp(z);return p/p.sum(1,keepdims=True)


class PseudoWeightedAnchored(AnchoredReasonClassifier):
    def __init__(self,mode):self.mode=mode

    def fit(self,X,Y):
        # Original input geometry, class balancing, neighbor bank and LR remain fixed.
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        X,Y=np.asarray(X,float),np.asarray(Y,int);known=(Y>=0).all(1)
        self.scaler_=StandardScaler().fit(X);Z=self.scaler_.transform(X)
        self.pca_=PCA(30,random_state=0).fit(Z)
        self.gmm_=GaussianMixture(30,covariance_type='diag',n_init=3,random_state=0).fit(self.pca_.transform(Z))
        assert self.gmm_.converged_
        K,V=Z[known],Z[~known];targets=Y[known]
        Rk=self.gmm_.predict_proba(self.pca_.transform(K));Ru=self.gmm_.predict_proba(self.pca_.transform(V))
        counts=(targets/targets.sum(1,keepdims=True)).sum(0)
        self.class_weights_=counts.mean()/counts
        Za,Ya,Ra=K,targets,Rk;confidence=np.ones_like(Ya,float)
        self.round_audit_=[]
        for step in range(3):
            row,reason=np.nonzero(Ya)
            weights=self.class_weights_[reason]/Ya.sum(1)[row]
            weights*=np.where(row<len(K),1.,.25)
            self.logistic_=LogisticRegression(C=.1,max_iter=3000,tol=1e-4).fit(Za[row],reason,sample_weight=weights)
            if step==0 or self.mode=='control':
                self.lda_=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(5)/5).fit(Za[row],reason)
            else:
                lw=np.where(row<len(K),1.,.25)
                if self.mode=='pseudo_lda_confidence':lw*=confidence[row,reason]
                self.lda_=WeightedReasonLDA().fit(Za[row],reason,lw)
            self.neighbor_X_=normalize(Za);self.neighbor_Y_=Ya.copy()
            self.round_audit_.append(dict(step=step,gold=len(K),pseudo=len(Za)-len(K)))
            if step==2:break
            T=Ya/Ya.sum(1,keepdims=True)
            rates=(Ra.T@T+T.mean(0))/(Ra.sum(0)[:,None]+1)
            s=.95*self._scores(V)+.05*(Ru@rates)
            keep=s.max(1)>=.8;generated=select_reasons(s[keep])
            self.generated_mask_=keep;self.generated_labels_=generated
            confidence=np.vstack([np.ones_like(targets,float),
                (s[keep]/s[keep].max(1,keepdims=True))**2])
            Za=np.vstack([K,V[keep]]);Ya=np.vstack([targets,generated]);Ra=np.vstack([Rk,Ru[keep]])
        return self
