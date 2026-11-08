"""Small symbolic classifiers over fold-local reason similarity signals."""
import tempfile
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

LABELS=['Safety','Funding','Efficacy','Recruitment','Other']
FEATURE_NAMES=['neighbor_logodds','centroid_margin','cluster_logodds','rival_neighbor_logodds','rival_centroid_margin']
SETTINGS=dict(clusters=12,neighbors=10,pseudo_mass=.10,cutoff=.30,seed=42)
SEARCH=dict(niterations=80,populations=8,population_size=27,ncycles_per_iteration=100,
    maxsize=15,maxdepth=5,binary_operators=['+','-','*'],unary_operators=[],
    model_selection='best',parsimony=.003,batching=True,batch_size=128,
    parallelism='serial',deterministic=True,precision=64,progress=False,verbosity=0,
    update_verbosity=0,elementwise_loss='weighted_logistic(x, y, w) = w * (max(x, zero(x)) - x*y + log1p(exp(-abs(x))))')


def groups(x):
    return pd.util.hash_pandas_object(pd.DataFrame(np.round(x,10)),index=False).to_numpy()


def decide(scores,cutoff=.3):
    p=(scores>=cutoff).astype(int)
    p[np.arange(len(p)),scores.argmax(1)]=1
    return p


def centered(s):
    s=np.clip(s,1e-9,1-1e-9)
    return .7*s/(.7*s+.3*(1-s))


class ReasonGeometry:
    def fit(self,x,y):
        self.known=np.isin(y,[0,1]).all(1)
        assert (self.known | (y==-100).all(1)).all()
        assert self.known.any() and (~self.known).any()
        self.scaler=StandardScaler().fit(x)
        z=normalize(self.scaler.transform(x))
        self.cluster=KMeans(n_clusters=SETTINGS['clusters'],n_init=10,random_state=SETTINGS['seed']).fit(z)
        self.bank=z[self.known];self.y=y[self.known].copy();self.bank_groups=groups(x[self.known])
        self.bank_cluster=self.cluster.labels_[self.known]
        prior=self.y.mean(0)
        cluster_rates=np.stack([(self.y[self.bank_cluster==k].sum(0)+prior)/(int((self.bank_cluster==k).sum())+1) for k in range(SETTINGS['clusters'])])
        generated=decide(cluster_rates[self.cluster.labels_[~self.known]])
        self.augmented=y.copy();self.augmented[~self.known]=generated
        np.testing.assert_array_equal(self.augmented[self.known],y[self.known])
        self.weights=np.where(self.known,1.,SETTINGS['pseudo_mass']*self.known.sum()/(~self.known).sum())
        return self

    def transform(self,x,training=False):
        z=normalize(self.scaler.transform(x));g=groups(x);cl=self.cluster.predict(z)
        similarities=z@self.bank.T
        result=[]
        for i,v in enumerate(z):
            # Training features exclude the row's whole raw-embedding group from
            # all label-derived signals. Query features use the frozen gold bank.
            keep=(self.bank_groups!=g[i]) if training else np.ones(len(self.y),dtype=bool)
            bank=self.bank[keep];y=self.y[keep];clusters=self.bank_cluster[keep]
            assert len(y)>0
            sim=similarities[i,keep];ix=np.argsort(-sim,kind='stable')[:SETTINGS['neighbors']]
            w=np.maximum(sim[ix],0)**2
            vote=(w@y[ix]+1e-3*y.mean(0))/(w.sum()+1e-3)
            rates=(y[clusters==cl[i]].sum(0)+y.mean(0))/(int((clusters==cl[i]).sum())+1)
            pos=normalize(y.T@bank);neg=normalize((1-y).T@bank)
            margin=(pos-neg)@v
            row=[]
            for j in range(5):
                other=np.arange(5)!=j
                row.append([logit(np.clip(vote[j],.01,.99)),margin[j],logit(np.clip(rates[j],.01,.99)),
                            logit(np.clip(vote[other].max(),.01,.99)),margin[other].max()])
            result.append(row)
        return np.asarray(result,float)


class SymbolicReasons:
    def __init__(self,search_override=None):
        self.search_override=search_override or {}

    def fit(self,x,y):
        from pysr import PySRRegressor
        x,y=np.asarray(x,float),np.asarray(y,int)
        self.geometry=ReasonGeometry().fit(x,y)
        inputs=self.geometry.transform(x,training=True)
        self.heads=[];self.linear=[];self.scales=[];self.equations=[]
        with tempfile.TemporaryDirectory(prefix='aml-pysr-') as tmp,threadpool_limits(limits=1):
            for j,label in enumerate(LABELS):
                scale=StandardScaler().fit(inputs[:,j]);features=scale.transform(inputs[:,j])
                target=self.geometry.augmented[:,j].astype(float)
                self.scales.append(scale)
                if np.unique(target).size<2:
                    prior=float((target.sum()+.5)/(len(target)+1))
                    self.heads.append(prior);self.linear.append(prior)
                    self.equations.append(dict(label=label,expression=str(logit(prior)),complexity=1,loss=None,constant_fallback=True))
                    continue
                linear=LogisticRegression(C=1.,max_iter=3000,random_state=42).fit(features,target,sample_weight=self.geometry.weights)
                sr=PySRRegressor(**(SEARCH|self.search_override),random_state=42+j,output_directory=tmp,run_id=label)
                sr.fit(features,target,weights=self.geometry.weights,variable_names=FEATURE_NAMES)
                best=sr.get_best()
                # SymPy's exported callable works after the temporary Julia/model
                # files have been deleted; retain only this small Python expression.
                func=best['lambda_format']
                test=np.asarray(func(features),float)
                assert np.isfinite(test).all()
                np.testing.assert_allclose(test,sr.predict(features),atol=1e-10,rtol=1e-10)
                self.heads.append(func);self.linear.append(linear)
                self.equations.append(dict(label=label,expression=str(best['sympy_format']),complexity=int(best['complexity']),loss=float(best['loss']),constant_fallback=False))
        self.audit=dict(gold_rows=int(self.geometry.known.sum()),masked_rows=int((~self.geometry.known).sum()),
            generated_positive_counts=self.geometry.augmented[~self.geometry.known].sum(0).tolist(),
            generated_weight=float(self.geometry.weights[~self.geometry.known].sum()),observed_labels_preserved=True,
            training_feature_group_exclusion=True,query_inputs_used_for_fit=False,search=SEARCH|self.search_override)
        return self

    def predict_scores(self,x):
        inputs=self.geometry.transform(np.asarray(x,float));symbolic=[];linear=[]
        for j,(head,control,scale) in enumerate(zip(self.heads,self.linear,self.scales)):
            f=scale.transform(inputs[:,j])
            symbolic.append(np.full(len(f),head) if isinstance(head,float) else expit(np.broadcast_to(head(f),(len(f),))))
            linear.append(np.full(len(f),control) if isinstance(control,float) else control.predict_proba(f)[:,1])
        s,l=np.column_stack(symbolic),np.column_stack(linear)
        assert np.isfinite(s).all() and np.isfinite(l).all()
        return s,l
