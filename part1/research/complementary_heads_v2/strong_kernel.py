"""Stronger matched kernel heads; same diversity loss as v1, distinct protocol."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from complement import ComplementaryKernel, STRENGTHS, centered_logit, loss_gradient


class StrongKernel(ComplementaryKernel):
    def fit(self, X, Y, anchor_oof):
        gold=(Y>=0).all(1);y=Y[gold]
        assert anchor_oof.shape==y.shape
        self.scaler=StandardScaler().fit(X)
        self.map=Nystroem(kernel='rbf',gamma=1/150,n_components=128,random_state=270303,n_jobs=1)
        transformed=self.map.fit_transform(self.scaler.transform(X))
        features=np.c_[np.ones(len(y)),transformed[gold]]
        error=expit(centered_logit(anchor_oof))-y;error-=error.mean(0)
        initial=np.zeros((features.shape[1],5));initial[0]=logit(np.clip(y.mean(0),.02,.98))
        self.weights={};self.audit={}
        for name,strength in STRENGTHS.items():
            start=initial.ravel() if name=='ordinary' else self.weights['ordinary'].ravel()
            result=minimize(loss_gradient,start,args=(features,y,error,strength,.00005),
                method='L-BFGS-B',jac=True,options={'maxiter':1000,'ftol':1e-11,'gtol':1e-7})
            assert result.success,result.message
            self.weights[name]=result.x.reshape(features.shape[1],5)
            residual=expit(features@self.weights[name])-y;residual-=residual.mean(0)
            correlation=(error*residual).mean(0)/np.sqrt((np.square(error).mean(0)+1e-8)*(np.square(residual).mean(0)+1e-8))
            self.audit[name]=dict(iterations=int(result.nit),objective=float(result.fun),
                max_gradient=float(np.abs(result.jac).max()),training_error_correlation=correlation.tolist(),
                training_mean_squared_error_correlation=float(np.square(correlation).mean()))
        return self

    def predict_logits(self,X):
        # Head cutoff .3 is represented at .5 for the unchanged fusion interface.
        return {name:z-logit(.3) for name,z in super().predict_logits(X).items()}
