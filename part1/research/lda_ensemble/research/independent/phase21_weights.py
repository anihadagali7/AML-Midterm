"""Cross-fitted selection-propensity weights and KMM for model-selection risk.
No reason-label values, validation rows, or test rows are used to estimate weights.
"""
import experiment as e
import numpy as np,json,datetime
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits


def stabilized(w):
    w=np.clip(w,.2,5.)
    w=w/w.mean()
    return .5+.5*w


def run():
    X=e.DATA[e.EMB].to_numpy();year=e.DATA.start_year.to_numpy()[:,None]
    selected=np.zeros(len(X),int);selected[e.LAB]=1
    rate=selected.mean();weights={'uniform':np.ones(len(e.LAB))};diagnostics={}
    splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=210719).split(X,selected,e.GROUP))
    with threadpool_limits(limits=1):
        for kind in ['logistic','tree']:
            p=np.zeros(len(X))
            for a,b in splits:
                sc=StandardScaler().fit(X[a]);pc=PCA(n_components=20,svd_solver='full').fit(sc.transform(X[a]))
                za=np.column_stack([pc.transform(sc.transform(X[a])),year[a]])
                zb=np.column_stack([pc.transform(sc.transform(X[b])),year[b]])
                scale=StandardScaler().fit(za);za=scale.transform(za);zb=scale.transform(zb)
                if kind=='logistic':model=LogisticRegression(C=.1,max_iter=1500)
                else:model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=30,learning_rate=.04,l2_regularization=10,random_state=42)
                model.fit(za,selected[a]);p[b]=model.predict_proba(zb)[:,1]
            # P(X) / P(X | selected) = P(selected) / P(selected | X).
            raw=rate/np.maximum(p[e.LAB],.02)
            weights['propensity_'+kind]=stabilized(raw)
            diagnostics[kind]=dict(domain_oof_auc=float(roc_auc_score(selected,p)),domain_oof_logloss=float(log_loss(selected,p)),gold_propensity_quantiles=np.quantile(p[e.LAB],[0,.1,.5,.9,1]).tolist(),raw_weight_quantiles=np.quantile(raw,[0,.1,.5,.9,1]).tolist(),clipped_fraction=float(np.mean((raw<.2)|(raw>5))))
        # Match the full observed training population, not external validation/test.
        A=StandardScaler().fit_transform(X);n=len(e.LAB)
        K=rbf_kernel(A[e.LAB],gamma=.003);kappa=n*rbf_kernel(A[e.LAB],A,gamma=.003).mean(1)
        ridge=.1
        result=minimize(lambda w:.5*w@K@w-kappa@w+.5*ridge*np.sum((w-1)**2),np.ones(n),jac=lambda w:K@w-kappa+ridge*(w-1),method='SLSQP',bounds=[(.2,5)]*n,constraints=[dict(type='eq',fun=lambda w:w.sum()-n,jac=lambda w:np.ones(n))],options={'maxiter':400,'ftol':1e-9})
        if not result.success:raise RuntimeError(result.message)
        weights['kmm']=stabilized(result.x)
        diagnostics['kmm']=dict(gamma=.003,ridge=ridge,optimizer_success=bool(result.success),iterations=int(result.nit))
    for name,w in weights.items():
        diagnostics.setdefault(name,{})
        diagnostics[name].update(effective_sample_size=float(w.sum()**2/(w@w)),minimum=float(w.min()),maximum=float(w.max()),mean=float(w.mean()))
    metadata=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source='157 gold-status training rows',target='all 1171 training rows; only features and gold/masked status',validation_used=False,test_used=False,reason_labels_used=False,clipping=[.2,5],shrinkage_to_uniform=.5,diagnostics=diagnostics,limitations='Estimated covariate-shift weights are approximate. Label-dependent sampling and missing support can violate the necessary assumptions. These are development-risk weights, not guaranteed unbiased target risks.')
    e.save_json('experiments/phase21-weighting.json',metadata)
    np.savez_compressed('experiments/phase21-weights.npz',**weights)
    frame=e.pd.DataFrame({'ID':e.DATA.iloc[e.LAB].ID.to_numpy(),**weights});frame.to_csv('experiments/phase21-weights.csv',index=False)
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':run()
