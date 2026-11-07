"""Reproducible, train-only Part 1 development engine. See experiments/protocol.txt."""
import os
for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(var,'1')
import argparse, hashlib, json, time, traceback, warnings, re, ast
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, softmax
from scipy.special import logit
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering, DBSCAN, HDBSCAN, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, pairwise_distances
from sklearn.multioutput import ClassifierChain
from sklearn.cross_decomposition import PLSRegression
from scipy.special import logsumexp
from scipy.linalg import cho_factor, cho_solve
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(__file__).resolve().parent
LABELS=['Safety','Funding','Efficacy','Recruitment','Other']; EMB=[f'emb_{i}' for i in range(150)]
DATA=pd.read_csv(ROOT/'data/part1_train.csv')
LAB=np.flatnonzero(DATA[LABELS].ge(0).all(axis=1)); UNLAB=np.flatnonzero(DATA[LABELS].eq(-100).all(axis=1))
Y=DATA.iloc[LAB][LABELS].to_numpy(int)
GROUP=pd.util.hash_pandas_object(DATA[EMB].round(10),index=False).to_numpy()
assert len(LAB)+len(UNLAB)==len(DATA)
CACHE={}

def config_id(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:14]
def save_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def make_splits(y=Y,groups=None,seed=20261104,n_splits=5):
    if groups is None: groups=GROUP[LAB]
    unique,inv=np.unique(groups,return_inverse=True)
    gy=np.array([y[inv==i].max(axis=0) for i in range(len(unique))])
    return [(np.flatnonzero(np.isin(inv,a)),np.flatnonzero(np.isin(inv,b))) for a,b in MultilabelStratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(gy,gy)]

def decisions(p,rule='t0.5'):
    p=np.clip(np.asarray(p),0,1)
    if rule=='expected_f1':
        patterns=((np.arange(32)[:,None]>>np.arange(5))&1)
        prob=np.prod(np.where(patterns[None,:,:],p[:,None,:],1-p[:,None,:]),axis=2)
        choices=patterns[1:]
        f1=2*(choices@patterns.T)/(choices.sum(1)[:,None]+patterns.sum(1)[None,:])
        return choices[np.argmax(prob@f1.T,axis=1)]
    if rule=='argmax':
        out=np.zeros_like(p,dtype=int);out[np.arange(len(p)),p.argmax(1)]=1;return out
    out=(p>=float(rule[1:])).astype(int)
    empty=out.sum(1)==0
    out[np.flatnonzero(empty),p[empty].argmax(1)]=1
    return out
RULES=['t0.25','t0.35','t0.45','t0.5','t0.55','expected_f1','argmax']
def metrics(y,p,rule='t0.5'):
    pred=decisions(p,rule)
    return {'sample_f1':float(f1_score(y,pred,average='samples',zero_division=0)), 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),**{f'f1_{l}':float(v) for l,v in zip(LABELS,f1_score(y,pred,average=None,zero_division=0))}}
def choose_rule(y,p):
    scores=[metrics(y,p,r)['sample_f1'] for r in RULES]
    # Prefer default threshold when differences are negligible.
    best=max(scores); idx=min([i for i,s in enumerate(scores) if s>=best-1e-12],key=lambda i:abs(i-3))
    return RULES[idx],best

def preprocess(A,B,c):
    A=np.asarray(A,float);B=np.asarray(B,float)
    mode=c.get('scale','standard')
    if mode=='standard':
        sc=StandardScaler();A=sc.fit_transform(A);B=sc.transform(B)
    elif mode=='global':
        mean=A.mean(0);std=np.sqrt(np.mean(np.var(A,axis=0)));A=(A-mean)/std;B=(B-mean)/std
    elif mode=='raw':
        # Global scale without centering preserves cosine and original relative geometry.
        std=np.sqrt(np.mean(np.var(A,axis=0)));A=A/std;B=B/std
    n=c.get('pca',0)
    if n:
        pc=PCA(n_components=min(n,A.shape[1],A.shape[0]-1),svd_solver='full');A=pc.fit_transform(A);B=pc.transform(B)
        power=c.get('white',0)
        if power:
            scale=np.maximum(pc.explained_variance_,1e-8)**(power/2);A/=scale;B/=scale
    if c.get('norm',False):A=normalize(A);B=normalize(B)
    return A,B

def chemical(frame,kind):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    desc=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumHDonors,Descriptors.NumHAcceptors,Descriptors.NumRotatableBonds,Descriptors.RingCount,Descriptors.FractionCSP3,Descriptors.HeavyAtomCount]
    out=[]
    for field in frame.drug_smiles.fillna(''):
        mols=[m for s in field.split(';') if s.strip() and (m:=Chem.MolFromSmiles(s.strip())) is not None]
        if kind=='fingerprint':v=np.max([gen.GetFingerprintAsNumPy(m) for m in mols],axis=0) if mols else np.zeros(1024)
        else:
            z=np.array([[fn(m) for fn in desc] for m in mols])
            v=np.r_[len(mols),z.mean(0),z.max(0)] if mols else np.zeros(19)
        out.append(v)
    return np.array(out,float)

def block_features(A,B,block):
    if block in ('fingerprint','descriptors'):
        a=chemical(A,block);b=chemical(B,block)
    elif block=='year':a=A[['start_year']].to_numpy();b=B[['start_year']].to_numpy()
    elif block=='missingness':
        cols=['criteria','diseases_cleaned','drugs','icdcodes','matched_drugs','drug_targets','drug_actions','drug_pathways','drug_smiles']
        a=A[cols].isna().astype(float).to_numpy();b=B[cols].isna().astype(float).to_numpy()
    elif block=='counts':
        cols=['criteria','diseases_cleaned','drugs','icdcodes','matched_drugs','drug_targets','drug_actions','drug_pathways','drug_smiles']
        def counts(D):return np.column_stack([np.log1p(D[c].fillna('').str.len()) for c in cols]+[D[c].fillna('').str.count(';')+D[c].notna() for c in cols[4:]])
        a=counts(A);b=counts(B)
    elif block.startswith('pathway_level'):
        catalog=pd.read_csv(ROOT/'data/pathway_catalog.csv');levels=dict(zip(catalog.reactome_id,catalog.pathway_level))
        if block=='pathway_levels':
            def calc(D):return np.array([[sum(levels.get(s.strip(),-1)==i for s in str(v).split(';')) for i in range(10)] for v in D.drug_pathways.fillna('')],float)
            a=np.log1p(calc(A));b=np.log1p(calc(B))
        else:
            lev=int(block[-1]); allowed=set(catalog.loc[catalog.pathway_level<=lev,'reactome_id'])
            def clean(D):return D.drug_pathways.fillna('').map(lambda x:' '.join(t.strip() for t in x.split(';') if t.strip() in allowed))
            v=TfidfVectorizer(token_pattern=r'[^\s]+',min_df=2,max_features=1000);a=v.fit_transform(clean(A)).toarray();b=v.transform(clean(B)).toarray();return a,b
    else:
        col=block.replace('_char','');char=block.endswith('_char')
        v=TfidfVectorizer(lowercase=True,analyzer='char_wb' if char else 'word',ngram_range=(3,5) if char else (1,2),min_df=2,max_df=0.99,max_features=3000,sublinear_tf=True,token_pattern=r'(?u)\b[\w-]+\b')
        a=v.fit_transform(A[col].fillna(''));b=v.transform(B[col].fillna(''))
        # Keep exact sparse-text geometry; at this dataset size dense blocks fit in memory.
        return a.toarray(),b.toarray()
    if block!='fingerprint':
        sc=StandardScaler();a=sc.fit_transform(a);b=sc.transform(b)
    a=np.nan_to_num(a);b=np.nan_to_num(b)
    norm=np.sqrt(np.mean(np.sum(a*a,axis=1)))
    return a/max(norm,1e-8),b/max(norm,1e-8)

def features(train_idx,query_frame,c):
    fitframe=DATA.iloc[train_idx]
    A,B=preprocess(fitframe[EMB],query_frame[EMB],c)
    blocks=c.get('features',[])
    if blocks:
        base_norm=np.sqrt(np.mean(np.sum(A*A,axis=1)))
        weight=c.get('feature_weight',0.3)
        for block in blocks:
            a,b=block_features(fitframe,query_frame,block)
            A=np.column_stack([A,a*base_norm*weight]);B=np.column_stack([B,b*base_norm*weight])
    return A,B

class Graph:
    """Normalized graph diffusion of all five binary labels; independent negative mass implicit."""
    def __init__(self,c):self.c=c
    def fit(self,X,Y):
        self.X=X; n=len(X);c=self.c;k=min(c.get('k',15),n-1);self.k=k
        self.nn=NearestNeighbors(n_neighbors=k+1).fit(X);dist,ix=self.nn.kneighbors(X)
        dist=dist[:,1:];ix=ix[:,1:];self.sigma=np.maximum(dist[:,-1],1e-8)
        self.band=float(np.median(dist[:,-1]))*c.get('band',1.)
        if c.get('graph','adaptive')=='adaptive':w=np.exp(-dist**2/(self.sigma[:,None]*self.sigma[ix]+1e-12))
        elif c.get('graph')=='binary':w=np.ones_like(dist)
        else:w=np.exp(-dist**2/max(self.band**2,1e-12))
        W=sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(n),k),ix.ravel())),shape=(n,n))
        W=(W+W.T)*0.5
        degree=np.maximum(np.asarray(W.sum(1)).ravel(),1e-12)
        known=Y[:,0]>=0
        if c.get('hard',False):
            P=sparse.diags(1/degree)@W;u=np.flatnonzero(~known);l=np.flatnonzero(known)
            alpha=c.get('alpha',0.99)
            U=spsolve(sparse.eye(len(u),format='csc')-alpha*P[u][:,u].tocsc(),alpha*P[u][:,l]@Y[l]+(1-alpha)*Y[l].mean(0))
            self.F=Y.copy().astype(float);self.F[u]=U
        else:
            S=sparse.diags(degree**-0.5)@W@sparse.diags(degree**-0.5)
            Z=np.zeros((n,6));Z[known,:5]=Y[known];Z[known,5]=1
            sol=spsolve(sparse.eye(n,format='csc')-c.get('alpha',0.95)*S.tocsc(),Z)
            self.F=np.divide(sol[:,:5],sol[:,5,None],out=np.tile(Y[known].mean(0),(n,1)),where=sol[:,5,None]>1e-15)
        self.F=np.clip(self.F,0,1);return self
    def predict(self,Q):
        d,ix=self.nn.kneighbors(Q,n_neighbors=self.k)
        if self.c.get('graph','adaptive')=='adaptive':w=np.exp(-d*d/(np.maximum(d[:,-1,None],1e-8)*self.sigma[ix]+1e-12))
        elif self.c.get('graph')=='binary':w=np.ones_like(d)
        else:w=np.exp(-d*d/max(self.band**2,1e-12))
        exact=d[:,0]<1e-8
        p=np.einsum('ij,ijk->ik',w,self.F[ix])/np.maximum(w.sum(1)[:,None],1e-15)
        p[exact]=self.F[ix[exact,0]]
        return p

class Cluster:
    def __init__(self,c):self.c=c
    def fit(self,X,Y):
        c=self.c;typ=c['cluster'];n=c.get('clusters',12)
        if typ=='kmeans':model=KMeans(n_clusters=n,n_init=10,random_state=42)
        elif typ=='bisect':model=BisectingKMeans(n_clusters=n,n_init=3,random_state=42)
        elif typ=='gmm':model=GaussianMixture(n_components=n,covariance_type=c.get('cov','diag'),reg_covar=0.02,n_init=2,max_iter=200,random_state=42)
        elif typ=='agglomerative':model=AgglomerativeClustering(n_clusters=n,linkage=c.get('linkage','ward'))
        elif typ=='spectral':model=SpectralClustering(n_clusters=n,affinity='nearest_neighbors',n_neighbors=c.get('k',15),assign_labels='cluster_qr',random_state=42)
        elif typ=='birch':model=Birch(n_clusters=n,threshold=0.5)
        elif typ=='hdbscan':model=HDBSCAN(min_cluster_size=c.get('min_cluster',8),min_samples=c.get('min_samples',3))
        elif typ=='dbscan':
            ds=NearestNeighbors(n_neighbors=6).fit(X).kneighbors()[0][:,-1]
            model=DBSCAN(eps=float(np.quantile(ds,c.get('eps_q',0.5))),min_samples=4)
        else:raise ValueError(typ)
        z=model.fit_predict(X);self.model=model;self.X=X;self.z=z
        known=Y[:,0]>=0;self.prior=Y[known].mean(0)
        ids=sorted(set(z)-{-1});self.ids=ids;self.centers=np.array([X[z==i].mean(0) for i in ids])
        self.mapping={};smooth=c.get('smooth',0.2)
        for i in ids:
            mask=known&(z==i);self.mapping[i]=(Y[mask].sum(0)+smooth*self.prior)/(mask.sum()+smooth)
        self.fallback=KNeighborsRegressor(n_neighbors=min(5,known.sum()),weights='distance').fit(X[known],Y[known])
        self.known=known;self.Y=Y
        if typ=='gmm':
            R=model.predict_proba(X[known]);self.softmap=(R.T@Y[known]+smooth*self.prior)/(R.sum(0)[:,None]+smooth)
        return self
    def predict(self,Q):
        typ=self.c['cluster']
        if typ=='gmm':return self.model.predict_proba(Q)@self.softmap
        if hasattr(self.model,'predict'):z=self.model.predict(Q)
        else:
            # Inductive extension for algorithms without predict: nearest training member.
            _,ix=NearestNeighbors(n_neighbors=1).fit(self.X).kneighbors(Q);z=self.z[ix[:,0]]
        p=self.fallback.predict(Q)
        for i in self.ids:
            if np.sum(self.known&(self.z==i))>0:p[z==i]=self.mapping[i]
        return p

def fit_supervised(X,Y,Q,c,weights=None):
    typ=c['model'];param=c.get('C',1.);out=[]
    if typ=='knn':return KNeighborsRegressor(n_neighbors=min(c.get('k',5),len(X)),weights=c.get('weights','distance'),metric=c.get('metric','euclidean')).fit(X,Y).predict(Q)
    if typ=='ridge':return np.clip(Ridge(alpha=c.get('alpha',10)).fit(X,Y,sample_weight=weights).predict(Q),0,1)
    if typ=='krr':return np.clip(KernelRidge(alpha=c.get('alpha',.1),kernel='rbf',gamma=c.get('gamma',.003)).fit(X,Y,sample_weight=weights).predict(Q),0,1)
    if typ=='pls':return np.clip(PLSRegression(n_components=c.get('components',5)).fit(X,Y).predict(Q),0,1)
    if typ in ('softmax','softsvc'):
        rows,classes=np.where(Y>0);ww=1/Y.sum(1)[rows]
        if weights is not None:ww*=weights[rows]
        if typ=='softmax':m=LogisticRegression(C=param,max_iter=2000)
        else:m=SVC(C=param,gamma=c.get('gamma','scale'),probability=True,random_state=42)
        m.fit(X[rows],classes,sample_weight=ww)
        p=np.zeros((len(Q),Y.shape[1]));p[:,m.classes_]=m.predict_proba(Q);return p
    if typ=='nca':
        nca=NeighborhoodComponentsAnalysis(n_components=c.get('components',5),max_iter=100,random_state=42)
        xx=nca.fit_transform(X,Y@2**np.arange(5));qq=nca.transform(Q)
        return KNeighborsRegressor(n_neighbors=c.get('k',5),weights='distance').fit(xx,Y).predict(qq)
    if typ=='chain':
        model=ClassifierChain(LogisticRegression(C=param,max_iter=1000),order='random',random_state=c.get('seed',42),cv=3,chain_method='predict_proba')
        return model.fit(X,Y).predict_proba(Q)
    if typ=='powerset':
        # Preserve multilabel patterns instead of imposing mutual exclusivity.
        code=Y@2**np.arange(5)
        if c.get('kernel','rbf')=='linear':model=LogisticRegression(C=param,max_iter=1000)
        else:model=SVC(C=param,gamma=c.get('gamma','scale'),probability=True,random_state=42)
        model.fit(X,code,sample_weight=weights);patterns=((model.classes_[:,None]>>np.arange(5))&1)
        return model.predict_proba(Q)@patterns
    for j in range(Y.shape[1]):
        y=Y[:,j]
        if len(np.unique(y))<2:out.append(np.full(len(Q),y[0]));continue
        if typ=='logistic':m=LogisticRegression(C=param,max_iter=2000,class_weight=c.get('balanced'),solver='lbfgs')
        elif typ=='svc':m=SVC(C=param,kernel=c.get('kernel','rbf'),gamma=c.get('gamma','scale'),degree=c.get('degree',2),class_weight=c.get('balanced'),probability=True,random_state=42)
        elif typ=='linear_svc':m=CalibratedClassifierCV(LinearSVC(C=param,class_weight=c.get('balanced'),max_iter=5000),cv=3)
        elif typ=='lda':m=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=c.get('shrinkage','auto'))
        elif typ=='qda':m=QuadraticDiscriminantAnalysis(reg_param=c.get('reg',0.9))
        elif typ=='gnb':m=GaussianNB(var_smoothing=c.get('smooth',1e-2))
        elif typ=='gp':m=GaussianProcessClassifier(kernel=ConstantKernel(c.get('amplitude',1))*RBF(c.get('length',10)),optimizer=None,random_state=42)
        elif typ in ('rf','extra'):
            cls=RandomForestClassifier if typ=='rf' else ExtraTreesClassifier
            m=cls(n_estimators=c.get('trees',200),max_depth=c.get('depth'),min_samples_leaf=c.get('leaf',2),max_features=c.get('max_features','sqrt'),class_weight=c.get('balanced'),random_state=42,n_jobs=1)
        elif typ=='hist':m=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,l2_regularization=5,random_state=42)
        elif typ=='catboost':
            from catboost import CatBoostClassifier
            m=CatBoostClassifier(iterations=c.get('trees',250),depth=c.get('depth',3),learning_rate=0.04,l2_leaf_reg=5,verbose=False,thread_count=1,allow_writing_files=False,random_seed=42)
        elif typ=='xgb':
            from xgboost import XGBClassifier
            m=XGBClassifier(n_estimators=c.get('trees',150),max_depth=c.get('depth',2),learning_rate=.04,reg_lambda=5,min_child_weight=2,subsample=.9,colsample_bytree=.8,n_jobs=1,random_state=42)
        elif typ=='lightgbm':
            from lightgbm import LGBMClassifier
            m=LGBMClassifier(n_estimators=150,num_leaves=7,min_child_samples=10,learning_rate=.04,reg_lambda=5,verbosity=-1,n_jobs=1,random_state=42)
        else:raise ValueError(typ)
        if weights is not None and typ not in ('lda','qda'):m.fit(X,y,sample_weight=weights)
        else:m.fit(X,y)
        out.append(m.predict_proba(Q)[:,1])
    return np.array(out).T

def predict(train_pos,query_frame,c,excluded_groups=None,return_unlabeled=False):
    if c['model']=='ensemble':
        weights=np.asarray(c.get('weights',np.ones(len(c['members']))),float);weights/=weights.sum()
        outputs=[predict(train_pos,query_frame,m,excluded_groups,return_unlabeled) for m in c['members']]
        if return_unlabeled:
            return sum(w*x[0] for w,x in zip(weights,outputs)),sum(w*x[1] for w,x in zip(weights,outputs)),outputs[0][2]
        return sum(w*x for w,x in zip(weights,outputs))
    # Fit unsupervised transforms only on fold-training labels and legitimate masked rows.
    actual=LAB[np.asarray(train_pos)]
    u=UNLAB if excluded_groups is None else UNLAB[~np.isin(GROUP[UNLAB],excluded_groups)]
    idx=np.r_[actual,u];n=len(actual)
    A,B=features(idx,query_frame,c);y=Y[train_pos];ys=np.full((len(A),5),-1);ys[:n]=y
    typ=c['model']
    if typ=='semigmm':
        # Fractional membership handles multilabel gold rows; masked rows participate in EM.
        # Labeled responsibilities remain fixed, and receive an explicit reliability weight.
        z=y/np.maximum(y.sum(1)[:,None],1)
        means=np.array([np.average(A[:n],axis=0,weights=z[:,j]+1e-8) for j in range(5)])
        covariance=np.cov(A.T)+np.eye(A.shape[1])*.1
        prior=(z.sum(0)+1)/(z.sum()+5)
        labelweight=c.get('label_weight',5.)
        def posterior(Q):
            ll=[]
            for j in range(5):
                cov=covs[j];factor=cho_factor(cov,lower=True);diff=Q-means[j]
                ll.append(-.5*np.sum(diff*cho_solve(factor,diff.T).T,axis=1)-np.log(np.diag(factor[0])).sum()+np.log(prior[j]))
            return softmax(np.array(ll).T,axis=1)
        covs=np.array([covariance.copy() for _ in range(5)])
        for iteration in range(80):
            R=posterior(A);R[:n]=z;R[:n]*=labelweight
            counts=R.sum(0)+1e-8;newmeans=R.T@A/counts[:,None]
            delta=np.max(np.abs(newmeans-means));means=newmeans;prior=(counts+1)/(counts.sum()+5)
            covlist=[]
            for j in range(5):
                diff=A-means[j];cov=(diff.T*R[:,j])@diff/counts[j]
                shrink=c.get('shrink',.8);cov=(1-shrink)*cov+shrink*covariance+np.eye(A.shape[1])*.05;covlist.append(cov)
            if c.get('cov','shared')=='shared':
                cov=np.average(covlist,axis=0,weights=counts);covs=np.array([cov]*5)
            else:covs=np.array(covlist)
            if delta<1e-5:break
        p=posterior(B);pu=posterior(A[n:])
    elif typ=='laprls':
        # RBF kernel regression with graph-Laplacian regularization from masked data.
        from sklearn.metrics.pairwise import rbf_kernel
        gamma=c.get('gamma',.003);K=rbf_kernel(A,gamma=gamma);KB=rbf_kernel(B,A,gamma=gamma)
        nn=NearestNeighbors(n_neighbors=11).fit(A);dd,ii=nn.kneighbors(A);dd=dd[:,1:];ii=ii[:,1:]
        W=sparse.csr_matrix((np.exp(-dd.ravel()**2/np.median(dd[:,-1])**2),(np.repeat(np.arange(len(A)),10),ii.ravel())),shape=(len(A),len(A)))
        W=(W+W.T)*.5;degree=np.asarray(W.sum(1)).ravel();L=sparse.diags(degree)-W
        # Normal equations in the representer coefficients; ridge makes the system stable.
        mat=K[:n].T@K[:n]+c.get('alpha',1)*K+c.get('mu',1)*(K@(L@K))+np.eye(len(A))*1e-6
        coef=cho_solve(cho_factor(mat,lower=True),K[:n].T@y)
        p=np.clip(KB@coef,0,1);pu=np.clip(K[n:]@coef,0,1)
    elif typ=='cluster_features':
        # Unsupervised prototype dictionary from genuine training + masked rows only.
        km=KMeans(n_clusters=c.get('clusters',40),n_init=5,random_state=42).fit(A)
        da=km.transform(A);db=km.transform(B)
        band=np.median(np.min(da,axis=1))*c.get('band',1.)
        H=np.exp(-da**2/max(2*band**2,1e-8));J=np.exp(-db**2/max(2*band**2,1e-8))
        sc=StandardScaler();H=sc.fit_transform(H);J=sc.transform(J)
        mix=c.get('original_weight',.3)
        H=np.column_stack([H,A*mix]);J=np.column_stack([J,B*mix])
        out=fit_supervised(H[:n],y,np.r_[J,H[n:]],{**c,'model':'logistic'})
        p=out[:len(B)];pu=out[len(B):]
    elif typ=='metric_graph':
        # Label-informed metric fitted only to fold-training labels, followed by unsupervised graph inference.
        base={**c,'model':c.get('metric_model','logistic')}
        scores=fit_supervised(A[:n],y,np.r_[A,B],base)
        scores=logit(np.clip(scores,.005,.995))
        sc=StandardScaler();s=sc.fit_transform(scores[:len(A)]);q=sc.transform(scores[len(A):])
        residual=c.get('residual',.3)
        norm=np.sqrt(np.mean(np.sum(A*A,axis=1)))
        U=np.column_stack([s,A*residual/max(norm,1e-8)])
        V=np.column_stack([q,B*residual/max(norm,1e-8)])
        model=Graph(c).fit(U,ys);p=model.predict(V);pu=model.F[n:]
    elif typ=='graph':
        model=Graph(c).fit(A,ys);p=model.predict(B);pu=model.F[n:]
    elif typ=='cluster':
        model=Cluster(c).fit(A,ys);p=model.predict(B);pu=model.predict(A[n:])
    elif typ=='pseudo':
        teacher=c['teacher'];T,U=preprocess(DATA.iloc[idx][EMB],query_frame[EMB],teacher)
        if teacher['model']=='graph':m=Graph(teacher).fit(T,ys);pt=m.F[n:]
        else:m=Cluster(teacher).fit(T,ys);pt=m.predict(T[n:])
        student={**c,'model':c.get('student','svc')}
        conf=c.get('confidence',.9);w=c.get('pseudo_weight',.2)
        base_out=None
        if c.get('consensus',False):
            base_out=fit_supervised(A[:n],y,np.r_[B,A[n:]],student)
            base_u=base_out[len(B):]
            agreement=(pt>=.5)==(base_u>=.5)
            # Confidence must be supported by both independently constructed views.
            pt=np.where(agreement,(pt+base_u)*.5,.5)
        yp=(pt>=.5).astype(int)
        rows=np.min(np.maximum(pt,1-pt),axis=1)>=conf
        if c.get('labelwise',False):
            p=np.zeros((len(B),5));pu=np.zeros((len(u),5))
            for j in range(5):
                sel=np.maximum(pt[:,j],1-pt[:,j])>=conf
                xx=np.r_[A[:n],A[n:][sel]];yy=np.r_[y[:,j],yp[sel,j]][:,None];ww=np.r_[np.ones(n),np.full(sel.sum(),w)]
                out=fit_supervised(xx,yy,np.r_[B,A[n:]],student,ww);p[:,j]=out[:len(B),0];pu[:,j]=out[len(B):,0]
        else:
            xx=np.r_[A[:n],A[n:][rows]];yy=np.r_[y,yp[rows]];ww=np.r_[np.ones(n),np.full(rows.sum(),w)]
            out=fit_supervised(xx,yy,np.r_[B,A[n:]],student,ww);p=out[:len(B)];pu=out[len(B):]
        if c.get('base_mix',0):
            if base_out is None:base_out=fit_supervised(A[:n],y,np.r_[B,A[n:]],student)
            mix=c['base_mix'];p=(1-mix)*p+mix*base_out[:len(B)];pu=(1-mix)*pu+mix*base_out[len(B):]
    else:
        out=fit_supervised(A[:n],y,np.r_[B,A[n:]] if return_unlabeled else B,c)
        p=out[:len(B)];pu=out[len(B):] if return_unlabeled else None
    if return_unlabeled:return p,pu,u
    return np.clip(p,0,1)

def evaluate(c,splits=None,seed=20261104,tag='screen',force=False):
    if splits is None:splits=make_splits(seed=seed)
    cid=config_id(c);path=ROOT/'experiments'/f'{tag}-{cid}-{seed}.npz'
    if path.exists() and not force:return dict(np.load(path,allow_pickle=True)['record'].item())
    started=time.time();p=np.full(Y.shape,np.nan);foldids=np.full(len(Y),-1)
    try:
        with threadpool_limits(limits=1):
            for fold,(a,b) in enumerate(splits):
                p[b]=predict(a,DATA.iloc[LAB[b]],c,excluded_groups=GROUP[LAB[b]])
                foldids[b]=fold
        use=foldids>=0;rule,score=choose_rule(Y[use],p[use]);base=metrics(Y[use],p[use]);tuned=metrics(Y[use],p[use],rule)
        fs=[metrics(Y[b],p[b],rule)['sample_f1'] for a,b in splits]
        record={'id':cid,'tag':tag,'seed':seed,'model':c['model'],'config':c,'n_scored':int(use.sum()),'score':score,'rule':rule,'default_score':base['sample_f1'],'fold_scores':fs,'fold_std':float(np.std(fs)),**tuned,'seconds':round(time.time()-started,3),'status':'ok'}
        np.savez_compressed(path,p=p,fold=foldids,record=record)
    except Exception as e:
        record={'id':cid,'tag':tag,'seed':seed,'config':c,'status':'error','error':str(e),'traceback':traceback.format_exc(),'seconds':round(time.time()-started,3)}
    with open(ROOT/'experiments/trials.jsonl','a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps({k:record.get(k) for k in ['id','model','score','rule','default_score','seconds','status','error']}),flush=True)
    return record

def screen_configs():
    out=[]
    for scale in ['standard','global']:
        for C in [.01,.1,1,10,100]:out.append(dict(model='logistic',C=C,scale=scale))
        for C in [.1,1,10,100]:
            for gamma in [.001,.003,.01,.03]:out.append(dict(model='svc',C=C,gamma=gamma,scale=scale))
        for k in [1,3,5,9,15]:
            for weights in ['uniform','distance']:out.append(dict(model='knn',k=k,weights=weights,scale=scale))
        for a in [1,10,100,1000]:out.append(dict(model='ridge',alpha=a,scale=scale))
        for C in [.01,.1,1,10]:out.append(dict(model='linear_svc',C=C,scale=scale))
    for typ in ['logistic','svc']:
        for C in [.1,1,10]:out.append(dict(model=typ,C=C,balanced='balanced'))
    for pc in [10,25,50,100]:
        for white in [0,.5,1]:
            for typ in ['svc','logistic','knn']:
                out.append(dict(model=typ,pca=pc,white=white,C=10,k=3))
    for typ in ['logistic','svc','knn']:
        for scale in ['standard','global','raw']:out.append(dict(model=typ,scale=scale,norm=True,C=10,k=3,gamma='scale'))
    for typ in ['lda','qda','gnb','rf','extra','hist','catboost','xgb','lightgbm','chain','powerset']:out.append(dict(model=typ))
    for typ in ['rf','extra','catboost','xgb']:
        for pc in [15,40]:out.append(dict(model=typ,pca=pc))
    for typ in ['kmeans','bisect','gmm','agglomerative','spectral','birch']:
        for n in [5,12,25,50]:out.append(dict(model='cluster',cluster=typ,clusters=n,pca=25))
    for n in [5,12,25]:out.append(dict(model='cluster',cluster='gmm',clusters=n,pca=15,cov='full'))
    for mc in [5,10,20]:out.append(dict(model='cluster',cluster='hdbscan',min_cluster=mc,pca=25))
    for q in [.25,.5,.75]:out.append(dict(model='cluster',cluster='dbscan',eps_q=q,pca=25))
    for pc in [0,25,50]:
        for k in [5,10,20,40]:
            for alpha in [.8,.95,.99]:out.append(dict(model='graph',pca=pc,k=k,alpha=alpha))
    for k in [5,15,40]:
        for alpha in [.95,.995]:out.append(dict(model='graph',k=k,alpha=alpha,hard=True))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='screen');parser.add_argument('--configs');parser.add_argument('--seed',type=int,default=20261104);args=parser.parse_args()
    configs=json.loads(Path(args.configs).read_text()) if args.configs else screen_configs()
    save_json(ROOT/'experiments'/f'{args.stage}-configs.json',configs)
    for i,c in enumerate(configs):
        print(f'RUN {i+1}/{len(configs)} {c}',flush=True);evaluate(c,seed=args.seed,tag=args.stage)
    records=[json.loads(s) for s in (ROOT/'experiments/trials.jsonl').read_text().splitlines()]
    pd.DataFrame([{**r,'config':json.dumps(r['config'],sort_keys=True)} for r in records]).to_csv(ROOT/'experiments/trials.csv',index=False)

if __name__=='__main__':main()
