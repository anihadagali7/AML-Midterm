"""Corrective audit: selected candidate vs fetched current Ani on both label pools."""
import ast
import json
import subprocess
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from simple_benchmark import ROOT, BASE, digest, metrics
from simple_models import FEATURES, REASONS, decide
from trusted_solution import TrustedTwoPassClassifier

OUT=ROOT/'experiments/current_ani_pool_audit'


def read_current():
    OUT.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','origin/ani'],cwd=ROOT,text=True).strip()
    notebook=subprocess.check_output(['git','show',f'{commit}:midterm/part1_best_model.ipynb'],cwd=ROOT)
    path=OUT/'ani_current_source.ipynb'
    if path.exists():assert path.read_bytes()==notebook,'Use a new audit folder for a changed Ani commit.'
    else:path.write_bytes(notebook)
    source='\n\n'.join(''.join(c['source']) for c in json.loads(notebook)['cells'] if c['cell_type']=='code')
    names={'decide','expand','multinomial_lr','knn','gmm_cluster_label','shrinkage_lda','base_model','final_model'}
    tree=ast.parse(source)
    functions='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
    assert len([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names])==len(names)
    (OUT/'ani_functions.py').write_text(functions+'\n')
    return commit,functions


def ani_predict(frame,fit,unknown,held,functions,legacy=False):
    X=frame[FEATURES].to_numpy()
    pool=np.sort(np.r_[fit,unknown])
    assert set(pool).isdisjoint(held)
    scaler=StandardScaler().fit(X[pool]);Z=scaler.transform(X)
    pca=PCA(30,random_state=0).fit(Z[pool])
    gmm=GaussianMixture(30,covariance_type='diag',n_init=3,random_state=0).fit(pca.transform(Z[pool]))
    assert gmm.converged_
    ns=dict(np=np,normalize=normalize,LogisticRegression=LogisticRegression,
            LinearDiscriminantAnalysis=LinearDiscriminantAnalysis,EMB=FEATURES,pca30=pca,gmm=gmm)
    exec(compile(functions,str(OUT/'ani_functions.py'),'exec'),ns)
    options=dict(w_lda=0.,C=.3) if legacy else {}
    return ns['final_model'](Z[fit],frame.iloc[fit][REASONS].to_numpy(int),Z[held],Z[unknown],**options)


def main():
    commit,functions=read_current()
    train=pd.read_csv(ROOT/'data/part1_train.csv');valid=pd.read_csv(ROOT/'data/part1_val.csv')
    unknown=np.flatnonzero(train[REASONS].lt(0).all(axis=1).to_numpy())
    pools=[('train157',train,ROOT/'experiments/classifier_cross_validation/splits',5,3),
           ('trainval235',pd.concat([train,valid],ignore_index=True),BASE/'splits',8,5)]
    protocol=dict(ani_commit=commit,ani_source='midterm/part1_best_model.ipynb',source_sha256=digest(OUT/'ani_current_source.ipynb'),
                  old_reference='no LDA, C0.3',current_ani='LDA0.2 + LR0.5 at C0.1 + KNN0.3; GMM-assisted2passes',
                  candidate_source_sha256=digest(ROOT/'trusted_solution.py'),
                  pools={name:dict(labels=157 if name=='train157' else 235,repeats=repeats,folds=nfold,
                                  split_sha256={f'repeat{r}.csv':digest(splitdir/f'repeat{r}.csv') for r in range(repeats)})
                         for name,_,splitdir,repeats,nfold in pools},
                  policy='Strict fold-local feature fitting for all models. Exclude held-out/test inputs. 157-pool fits also exclude all78validation inputs and labels.',
                  native_transductive_comparison_run=False,no_parameter_tuning=True,
                  correction='Prior claims used only235-label CV and an older Ani recipe; they did not establish superiority over current Ani on both label pools.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    scores=[];repeats=[];roles=[];prediction_frames=[]
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for pool_name,frame,splitdir,nrepeats,nfold in pools:
            verified=np.flatnonzero(frame[REASONS].ge(0).all(axis=1).to_numpy())
            ids=frame.iloc[verified].ID
            for repeat in range(nrepeats):
                assignment=pd.read_csv(splitdir/f'repeat{repeat}.csv').set_index('ID').loc[ids,'fold'].to_numpy()
                repeated={name:np.full((len(verified),5),-1,int) for name in ['ani_current','ani_legacy','trusted_two_pass']}
                for fold in range(nfold):
                    held_mask=assignment==fold;held=verified[held_mask];fit=verified[~held_mask]
                    assert len(np.intersect1d(np.r_[fit,unknown],held))==0
                    probs={}
                    for name,legacy in [('ani_current',False),('ani_legacy',True)]:
                        probs[name]=ani_predict(frame,fit,unknown,held,functions,legacy)
                    model=TrustedTwoPassClassifier().fit(frame.iloc[fit][FEATURES].to_numpy(),frame.iloc[fit][REASONS].to_numpy(int),frame.iloc[unknown][FEATURES].to_numpy())
                    probs['trusted_two_pass']=model.predict_proba(frame.iloc[held][FEATURES].to_numpy())
                    if pool_name=='trainval235':
                        old=pd.read_csv(BASE/'predictions'/f'repeat{repeat}_fold{fold}__ani_train_only.csv').set_index('ID').loc[frame.iloc[held].ID]
                        np.testing.assert_allclose(probs['ani_legacy'],old[[f'probability_{r}' for r in REASONS]],atol=1e-10,rtol=1e-8)
                        frozen=np.load(ROOT/'experiments/trusted_iteration/06_final_regularization_check/probabilities.npz')['final_r2_control']
                        np.testing.assert_allclose(probs['trusted_two_pass'],frozen[repeat,held_mask],atol=1e-10,rtol=1e-8)
                    truth=frame.iloc[held][REASONS].to_numpy(int)
                    for name,P in probs.items():
                        pred=decide(P,.3);repeated[name][held_mask]=pred
                        scores.append(dict(pool=pool_name,repeat=repeat,fold=fold,name=name,fit_rows=len(fit),held_rows=len(held),**metrics(truth,pred)))
                        df=pd.DataFrame(dict(ID=frame.iloc[held].ID,pool=pool_name,repeat=repeat,fold=fold,name=name))
                        for j,r in enumerate(REASONS):df[f'actual_{r}']=truth[:,j];df[f'predicted_{r}']=pred[:,j];df[f'probability_{r}']=P[:,j]
                        prediction_frames.append(df)
                    roles.append(dict(pool=pool_name,repeat=repeat,fold=fold,fit_ids=frame.iloc[fit].ID.tolist(),held_ids=frame.iloc[held].ID.tolist(),unlabeled_ids=frame.iloc[unknown].ID.tolist(),test_features_used=False))
                for name,pred in repeated.items():
                    assert (pred>=0).all()
                    repeats.append(dict(pool=pool_name,repeat=repeat,name=name,**metrics(frame.iloc[verified][REASONS].to_numpy(int),pred)))
                print(pool_name,'repeat',repeat,{r['name']:round(r['sample_f1'],6) for r in repeats if r['pool']==pool_name and r['repeat']==repeat},flush=True)
        # Correct the four diagnostic comparisons to use the current Ani recipe.
        frame=pd.concat([train,valid],ignore_index=True);verified=np.flatnonzero(frame[REASONS].ge(0).all(axis=1).to_numpy())
        diagnostic=[]
        for label in ['original_validation','after_2003','after_2005','after_2007']:
            if label=='original_validation':fit=verified[verified<len(train)];held=verified[verified>=len(train)];u=unknown
            else:
                year=int(label.split('_')[-1]);early=frame.start_year.to_numpy()<=year;fit=verified[early[verified]];held=verified[~early[verified]];u=unknown[early[unknown]]
            P=ani_predict(frame,fit,u,held,functions)
            model=TrustedTwoPassClassifier().fit(frame.iloc[fit][FEATURES].to_numpy(),frame.iloc[fit][REASONS].to_numpy(int),frame.iloc[u][FEATURES].to_numpy())
            for name,prob in [('ani_current',P),('trusted_two_pass',model.predict_proba(frame.iloc[held][FEATURES].to_numpy()))]:
                pred=decide(prob,.3);truth=frame.iloc[held][REASONS].to_numpy(int)
                diagnostic.append(dict(split=label,name=name,fit_rows=len(fit),unlabeled_rows=len(u),held_rows=len(held),**metrics(truth,pred)))
        pd.DataFrame(diagnostic).to_csv(OUT/'current_diagnostic_scores.csv',index=False)
    repeat_table=pd.DataFrame(repeats);fold_table=pd.DataFrame(scores)
    repeat_table.to_csv(OUT/'repeat_scores.csv',index=False);fold_table.to_csv(OUT/'fold_scores.csv',index=False)
    pd.concat(prediction_frames).to_csv(OUT/'predictions.csv',index=False)
    (OUT/'role_audit.json').write_text(json.dumps(roles,separators=(',',':'))+'\n')
    summary=repeat_table.groupby(['pool','name']).agg(mean_f1=('sample_f1','mean'),repeat_sd=('sample_f1','std'),macro_f1=('macro_f1','mean'),**{f'f1_{r}':(f'f1_{r}','mean') for r in REASONS})
    for pool,_,_,_,_ in pools:
        ref=repeat_table[(repeat_table.pool==pool)&(repeat_table.name=='ani_current')].sample_f1.to_numpy()
        fr=fold_table[(fold_table.pool==pool)&(fold_table.name=='ani_current')].sample_f1.to_numpy()
        for name in ['ani_current','ani_legacy','trusted_two_pass']:
            diff=repeat_table[(repeat_table.pool==pool)&(repeat_table.name==name)].sample_f1.to_numpy()-ref
            fd=fold_table[(fold_table.pool==pool)&(fold_table.name==name)].sample_f1.to_numpy()-fr
            summary.loc[(pool,name),'delta_current_ani']=diff.mean()
            summary.loc[(pool,name),'repeat_wins']=sum(diff>1e-12)
            summary.loc[(pool,name),'fold_wins']=sum(fd>1e-12)
            summary.loc[(pool,name),'fold_ties']=sum(abs(fd)<1e-12)
    summary.to_csv(OUT/'summary.csv')
    print(summary.to_string(),flush=True)
    print(pd.DataFrame(diagnostic)[['split','name','sample_f1','macro_f1']].to_string(index=False),flush=True)


if __name__=='__main__':main()
