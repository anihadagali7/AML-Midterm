"""Text ablations, molecular features, shared phase effects, and NB-log ratios."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.naive_bayes import MultinomialNB
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from experiments import ROOT, OUT, PHASES, load_data, prepare, matrix, metrics

RDLogger.DisableLog('rdApp.*')


def molecule_bank(data):
    path=OUT/'molecules.joblib'
    if path.exists(): return joblib.load(path)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    descs=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumHDonors,
           Descriptors.NumHAcceptors,Descriptors.NumRotatableBonds,Descriptors.RingCount,
           Descriptors.FractionCSP3,Descriptors.HeavyAtomCount]
    bank={}
    for text in pd.concat(list(data.values())).drug_smiles.fillna('').unique():
        molecules=[Chem.MolFromSmiles(t) for t in text.split(';') if t]
        molecules=[m for m in molecules if m is not None]
        if molecules:
            fp=np.max([gen.GetFingerprintAsNumPy(m) for m in molecules],axis=0)
            ds=np.array([[f(m) for f in descs] for m in molecules])
            d=np.r_[ds.mean(axis=0),ds.max(axis=0),len(molecules),0.]
        else: fp=np.zeros(1024); d=np.r_[np.zeros(19),1.]
        bank[text]=(fp,d)
    joblib.dump(bank,path,compress=3)
    return bank


def molmat(df,bank):
    values=[bank[t] for t in df.drug_smiles.fillna('')]
    return sparse.csr_matrix(normalize(np.array([x[0] for x in values]))),np.array([x[1] for x in values])


def interactions(x, df, weight):
    return sparse.hstack([x]+[x.multiply(df.phase.eq(p).to_numpy()[:,None])*weight for p in PHASES],format='csr')


def run():
    data=load_data(); bank=molecule_bank(data)
    for split in ['temporal','validation']:
        tr, ev, f, bt, be, bx=prepare(data,split)
        dfs=[tr,ev,data['test']]; blocks=[bt,be,bx]
        pred={}; tests={}; scores=[]; configs={}

        def record(name,p,pt,config):
            pred[name]=p; tests[name]=pt; configs[name]=config
            scores.append(dict(name=name,**metrics(ev,p)))
            pd.DataFrame(scores).sort_values('phase_mean_logloss').to_csv(OUT/f'refined_{split}_scores.csv',index=False)
            print(split,name,metrics(ev,p)['phase_mean_logloss'],flush=True)

        def linear(name,xs,C,config,weights=None):
            model=LogisticRegression(C=C,solver='liblinear',max_iter=1000,tol=1e-5,random_state=42)
            model.fit(xs[0],tr.label,sample_weight=weights)
            record(name,model.predict_proba(xs[1])[:,1],model.predict_proba(xs[2])[:,1],config)

        # Phase-dependent coefficients share a common set of coefficients and shrink toward it.
        for rep in ['criteria','clinical']:
            for C in [.5,2.,8.]:
                xs=[matrix(b,rep,'none') for b in blocks]
                for weight in [.3,1.]:
                    xx=[interactions(x,d,weight) for x,d in zip(xs,dfs)]
                    name=f'interactions_{rep}_C{C:g}_weight{weight:g}'
                    linear(name,xx,C,dict(family='interactions',representation=rep,C=C,weight=weight))
                # Equal phase weights match the competition's phase-balanced objective.
                w=tr.phase.map({p:len(tr)/(3*sum(tr.phase==p)) for p in PHASES}).to_numpy()
                name=f'balanced_{rep}_C{C:g}'
                linear(name,xs,C,dict(family='balanced',representation=rep,C=C),w)

        # Molecular structures are supplied inputs, not external outcome information.
        fps,ds=zip(*(molmat(d,bank) for d in dfs))
        scaler=StandardScaler().fit(ds[0]); ds=[scaler.transform(x) for x in ds]
        for weight in [.25,1.]:
            xs=[sparse.hstack([matrix(b,'clinical'),fp*weight,sparse.csr_matrix(np.clip(d,-5,5))*.05],format='csr')
                for b,fp,d in zip(blocks,fps,ds)]
            for C in [.5,2.,8.]:
                name=f'molecular_C{C:g}_weight{weight:g}'
                linear(name,xs,C,dict(family='molecular',C=C,weight=weight))

        # Alternate eligibility text representations.
        for rep,options in [
            ('binary',dict(binary=True,use_idf=False,ngram_range=(1,2),min_df=3,max_features=60000)),
            ('wordwide',dict(ngram_range=(1,2),min_df=2,max_features=80000,sublinear_tf=True)),
            ('charcriteria',dict(analyzer='char_wb',ngram_range=(3,5),min_df=5,max_features=40000,sublinear_tf=True)),
            ('stopwords',dict(ngram_range=(1,2),min_df=3,max_features=35000,sublinear_tf=True,stop_words='english')),
        ]:
            vec=TfidfVectorizer(dtype=np.float64,strip_accents='unicode',**options).fit(f.documents(tr,'criteria'))
            texts=[vec.transform(f.documents(d,'criteria')) for d in dfs]
            for C in [.5,2.,8.]:
                if rep=='charcriteria':
                    xs=[sparse.hstack([b['criteria'],t*.7,b['phase']],format='csr') for t,b in zip(texts,blocks)]
                else: xs=[sparse.hstack([t,b['phase']],format='csr') for t,b in zip(texts,blocks)]
                name=f'text_{rep}_C{C:g}'
                linear(name,xs,C,dict(family='text',representation=rep,C=C))
            joblib.dump(vec,OUT/f'text_{rep}_{split}.joblib',compress=3)

        # NB feature ratios are estimated from fitting labels only.
        nb=MultinomialNB(alpha=1.).fit(bt['criteria'],tr.label)
        ratio=nb.feature_log_prob_[1]-nb.feature_log_prob_[0]
        for C in [.5,2.,8.,32.]:
            xs=[sparse.hstack([b['criteria'].multiply(ratio),b['phase']],format='csr') for b in blocks]
            name=f'nb_logratio_C{C:g}'
            linear(name,xs,C,dict(family='nb_logratio',C=C))
        np.savez_compressed(OUT/f'refined_{split}_predictions.npz',**pred)
        np.savez_compressed(OUT/f'refined_{split}_test_predictions.npz',**tests)
        (OUT/'refined_configs.json').write_text(json.dumps(configs,indent=2))

if __name__=='__main__': run()
