"""Explicit model adapters for the frozen local benchmark."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from experiments import Features,matrix


class OurLinear:
    """Exactly the linear component shared by submissions 02 and 05."""
    def fit(self,df):
        self.features=Features().fit(df)
        self.wide=TfidfVectorizer(dtype=np.float64,strip_accents='unicode',ngram_range=(1,2),
            min_df=2,max_features=80000,sublinear_tf=True).fit(self.features.documents(df,'criteria'))
        blocks=self.features.transform(df)
        xs=self.inputs(df,blocks)
        self.models=[]
        for x,C,half in zip(xs,[2.,2.,.5],[None,6.,3.]):
            w=None
            if half:
                w=2**((df.start_year.fillna(df.start_year.median()).to_numpy()-df.start_year.max())/half)
                w=w/w.mean()
            model=LogisticRegression(C=C,solver='liblinear',max_iter=1000,tol=1e-5,random_state=42)
            self.models.append(model.fit(x,df.label,sample_weight=w))
        self.training_ids=df.ID.tolist()
        return self

    def inputs(self,df,blocks):
        wide=sparse.hstack([self.wide.transform(self.features.documents(df,'criteria')),blocks['phase']],format='csr')
        return [wide,matrix(blocks,'criteria'),matrix(blocks,'clinical')]

    def predict_proba(self,df):
        xs=self.inputs(df,self.features.transform(df))
        return sum(w*m.predict_proba(x)[:,1] for w,m,x in zip([.375,.375,.25],self.models,xs))
