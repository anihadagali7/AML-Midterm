"""Ani's eight-model stack, made reusable for fold-local evaluation.

Adapted from Ani's midterm/part2_best_model.ipynb at commit
1b8bd11c47428fb5453e53cdc275542b92a3b347. The preserved source is in
experiments/ani_comparison/source/. Transductive reproduction passes an
explicit feature space fitted on all available covariates; the benchmark
defaults to fitting its vocabulary and IDF only on the fold's training rows.
"""
from __future__ import annotations
import ast
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

PHASES=('I','II','III')
BASES=[('criteria LR',['criteria'],2.),('disease LR',['disease'],1.),
       ('ICD LR',['icd'],1.),('drug-name LR',['drug'],1.),
       ('drug-knowledge LR',['targets','pathways','actions'],.3),
       ('metadata LR',['meta'],1.),('criteria MNB',['crit_counts'],None),
       ('criteria kNN',['criteria'],None)]


def parse_list(s):
    try:
        out=[]
        for x in ast.literal_eval(s):
            out+=ast.literal_eval(x) if isinstance(x,str) and x.startswith('[') else [x]
        return [str(x) for x in out]
    except (ValueError,SyntaxError):return []


def text_fields(df):
    icd=df.icdcodes.fillna('[]').map(parse_list)
    text={'criteria':df.criteria.fillna(''),
          'disease':df.diseases_cleaned.fillna('[]').map(lambda s:' '.join(parse_list(s))),
          'drug':df.drugs.fillna('[]').map(lambda s:' '.join(parse_list(s))),
          'icd':icd.map(lambda v:' '.join(sorted({c[:1] for c in v}|{c[:3] for c in v}|{c[:5] for c in v})))}
    for key,col in [('targets','drug_targets'),('pathways','drug_pathways'),('actions','drug_actions')]:
        text[key]=df[col].fillna('').map(lambda s:' '.join(sorted({t.strip() for t in s.split(';') if t.strip()})))
    return text


def onehot(df):return np.column_stack([df.phase.eq(p).to_numpy().astype(float) for p in PHASES])


class AniFeatures:
    def fit(self,df):
        text=text_fields(df);split=dict(tokenizer=str.split,lowercase=False,token_pattern=None)
        self.vectorizers={
            'criteria':TfidfVectorizer(min_df=5,ngram_range=(1,2),sublinear_tf=True),
            'crit_counts':CountVectorizer(min_df=5,ngram_range=(1,2)),
            'disease':TfidfVectorizer(min_df=2,ngram_range=(1,2),sublinear_tf=True),
            'drug':TfidfVectorizer(min_df=2,sublinear_tf=True),
            'icd':TfidfVectorizer(min_df=2,**split),'targets':TfidfVectorizer(min_df=3,**split),
            'pathways':TfidfVectorizer(min_df=5,**split),'actions':CountVectorizer(min_df=2,binary=True,**split)}
        for key,vec in self.vectorizers.items():vec.fit(text['criteria' if key=='crit_counts' else key])
        self.year_median=float(df.start_year.median())
        self.fit_ids=df.ID.tolist()
        return self

    def transform(self,df):
        text=text_fields(df)
        f={key:vec.transform(text['criteria' if key=='crit_counts' else key]) for key,vec in self.vectorizers.items()}
        f['meta']=sp.csr_matrix(np.column_stack([
            df.matched_drugs.notna(),np.log1p(text['drug'].str.split().str.len()),
            np.log1p(text['criteria'].str.len())/10,df.phase_detail.str.contains('/'),
            df.phase_detail.eq('early phase 1')]).astype(float))
        f['phase']=sp.csr_matrix(onehot(df));f['knn']=normalize(f['criteria'])
        return f


def matrix(f,blocks):return sp.hstack([f[k] for k in blocks]+[f['phase']],format='csr')


def fit_base(name,blocks,C,f,df,weights):
    if name=='criteria kNN':
        # Faithfully preserve Ani's implementation: recency does not weight kNN.
        return {p:(f['knn'][df.phase.eq(p).to_numpy()],df.label.to_numpy()[df.phase.eq(p).to_numpy()]) for p in PHASES}
    x=matrix(f,blocks)
    model=MultinomialNB(alpha=1.) if name=='criteria MNB' else LogisticRegression(C=C,max_iter=3000)
    return model.fit(x,df.label.to_numpy(),sample_weight=weights)


def predict_base(name,blocks,model,f,df,k=50):
    if name!='criteria kNN':return model.predict_proba(matrix(f,blocks))[:,1]
    result=np.zeros(len(df))
    for phase,(x,y) in model.items():
        mask=df.phase.eq(phase).to_numpy()
        if not mask.any():continue
        similarities=(f['knn'][mask]@x.T).toarray()
        neighbors=min(k,len(y)-1)
        idx=np.argpartition(-similarities,neighbors,axis=1)[:,:neighbors]
        weights=np.take_along_axis(similarities,idx,1)
        result[mask]=((weights*y[idx]).sum(axis=1)+1)/(weights.sum(axis=1)+2)
    return result


def subset(f,idx):return {k:v[idx] for k,v in f.items()}


class AniStack:
    def __init__(self,tau=3.,meta_C=1.,phase3_year=False,knn_k=50,inner_refit=False):
        self.tau=tau;self.meta_C=meta_C;self.phase3_year=phase3_year;self.knn_k=knn_k;self.inner_refit=inner_refit

    def meta_matrix(self,z,df):
        p=np.clip(z,1e-4,1-1e-4);ph=onehot(df)
        parts=[np.log(p/(1-p)),ph]
        if self.phase3_year:
            year=df.start_year.fillna(self.features.year_median).to_numpy()
            parts.append((ph[:,2]*(np.minimum(year,self.year_cap)-2007)/3)[:,None])
        return np.hstack(parts)

    def fit(self,df,features=None):
        df=df.reset_index(drop=True)
        self.features=features if features is not None else AniFeatures().fit(df)
        f=self.features.transform(df);year=df.start_year.fillna(self.features.year_median).to_numpy()
        self.year_cap=float(year.max()+2)
        w=np.ones(len(df)) if self.tau is None else np.exp((year-year.max())/self.tau)
        w=w/w.mean();z=np.zeros((len(df),len(BASES)));self.models=[];self.fold_audit=[]
        folds=list(GroupKFold(5).split(np.arange(len(df)),groups=df.trial_id.to_numpy()))
        if self.inner_refit:
            for fold,(a,b) in enumerate(folds):
                assert not set(df.trial_id.iloc[a])&set(df.trial_id.iloc[b])
                inner=AniFeatures().fit(df.iloc[a])
                fa,fb=inner.transform(df.iloc[a]),inner.transform(df.iloc[b])
                for j,(name,blocks,C) in enumerate(BASES):
                    m=fit_base(name,blocks,C,fa,df.iloc[a],w[a])
                    z[b,j]=predict_base(name,blocks,m,fb,df.iloc[b],self.knn_k)
                self.fold_audit.append({'fold':fold,'fit_rows':len(a),'heldout_rows':len(b),'trial_overlap':0,'preprocessing':'inner training rows only'})
            self.models=[fit_base(name,blocks,C,f,df,w) for name,blocks,C in BASES]
        else:
            for j,(name,blocks,C) in enumerate(BASES):
                for fold,(a,b) in enumerate(folds):
                    assert not set(df.trial_id.iloc[a])&set(df.trial_id.iloc[b])
                    m=fit_base(name,blocks,C,subset(f,a),df.iloc[a],w[a])
                    z[b,j]=predict_base(name,blocks,m,subset(f,b),df.iloc[b],self.knn_k)
                    if j==0:self.fold_audit.append({'fold':fold,'fit_rows':len(a),'heldout_rows':len(b),'trial_overlap':0,'preprocessing':'shared fitting-population vocabulary'})
                self.models.append(fit_base(name,blocks,C,f,df,w))
        self.meta=LogisticRegression(C=self.meta_C,max_iter=3000).fit(self.meta_matrix(z,df),df.label.to_numpy(),sample_weight=w)
        self.oof_base_predictions=z;self.training_ids=df.ID.tolist()
        return self

    def predict_base_matrix(self,df):
        f=self.features.transform(df)
        return np.column_stack([predict_base(name,blocks,m,f,df,self.knn_k) for (name,blocks,_),m in zip(BASES,self.models)])

    def predict_proba(self,df):return self.meta.predict_proba(self.meta_matrix(self.predict_base_matrix(df),df))[:,1]
