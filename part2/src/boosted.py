"""Boosted-tree and forest comparisons using training-only text projections."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from experiments import ROOT, OUT, load_data, prepare, metrics


def transform_dense(df, blocks, svds, mask, year_median):
    parts=[svds[k].transform(blocks[k]) for k in ['criteria','disease','drug','char']]
    parts.append(blocks['knowledge'][:,mask].toarray())
    text=df.criteria.fillna('')
    parts.append(np.column_stack([np.log1p(text.str.len()),np.log1p(text.str.count('\n')),
        df.drug_smiles.notna().astype(float),df.drug_targets.fillna('').str.count(';'),
        df.drugs.fillna('').str.count(','),df.diseases_cleaned.fillna('').str.count(',')]))
    f=pd.DataFrame(np.hstack(parts).astype(np.float32))
    f.columns=[f'x{i}' for i in range(f.shape[1])]
    for col in ['phase','phase_detail','diseases_cleaned','drugs','matched_drugs']:
        f[col]=df[col].fillna('missing').astype(str).to_numpy()
    f['year']=df.start_year.fillna(year_median).to_numpy()
    return f


def dense_features(tr, ev, test, bt, be, bx):
    cache = OUT / ('dense_temporal.joblib' if len(tr) < 7000 else 'dense_validation.joblib')
    if cache.exists():
        return joblib.load(cache)
    parts = [[], [], []]
    svds = {}
    for key, n in [('criteria', 100), ('disease', 24), ('drug', 24), ('char', 32)]:
        svd = TruncatedSVD(n_components=n, random_state=42, n_iter=5).fit(bt[key])
        svds[key] = svd
        for out, b in zip(parts, [bt, be, bx]):
            out.append(svd.transform(b[key]))
    mask = np.asarray((bt['knowledge'] > 0).sum(axis=0)).ravel() >= 15
    for out, b, d in zip(parts, [bt, be, bx], [tr, ev, test]):
        out.append(b['knowledge'][:, mask].toarray())
        text = d.criteria.fillna('')
        out.append(np.column_stack([np.log1p(text.str.len()), np.log1p(text.str.count('\n')),
            d.drug_smiles.notna().astype(float), d.drug_targets.fillna('').str.count(';'),
            d.drugs.fillna('').str.count(','), d.diseases_cleaned.fillna('').str.count(',')]))
    frames=[]
    for out,d in zip(parts,[tr, ev, test]):
        f=pd.DataFrame(np.hstack(out).astype(np.float32))
        f.columns=[f'x{i}' for i in range(f.shape[1])]
        for col in ['phase', 'phase_detail', 'diseases_cleaned', 'drugs', 'matched_drugs']:
            f[col]=d[col].fillna('missing').astype(str).to_numpy()
        f['year']=d.start_year.fillna(tr.start_year.median()).to_numpy()
        frames.append(f)
    packet=(frames, svds, mask)
    joblib.dump(packet, cache, compress=3)
    return packet


def run():
    data=load_data()
    for split in ['temporal','validation']:
        tr, ev, features, bt, be, bx=prepare(data, split)
        frames,svds,mask=dense_features(tr,ev,data['test'],bt,be,bx)
        scores=[]; preds={}; tests={}; configs={}
        for year in ['none','year']:
            xs=[f.drop(columns=['year']) if year=='none' else f for f in frames]
            cats=xs[0].select_dtypes('str').columns.tolist()
            for depth in [4,6,8]:
                model=CatBoostClassifier(iterations=1000, depth=depth, learning_rate=.035,
                    l2_leaf_reg=6, loss_function='Logloss', random_seed=42, thread_count=4,
                    verbose=False, allow_writing_files=False)
                model.fit(xs[0],tr.label,cat_features=cats)
                for n in [250,600,1000]:
                    name=f'catboost_d{depth}_n{n}_{year}'
                    p=model.predict_proba(xs[1],ntree_end=n)[:,1]
                    pt=model.predict_proba(xs[2],ntree_end=n)[:,1]
                    preds[name]=p; tests[name]=pt
                    configs[name]=dict(family='catboost',depth=depth,iterations=n,year=year)
                    scores.append(dict(name=name,**metrics(ev,p)))
                print(split,year,'catboost',depth, pd.DataFrame(scores).phase_mean_logloss.min(), flush=True)
                pd.DataFrame(scores).sort_values('phase_mean_logloss').to_csv(OUT/f'boosted_{split}_scores.csv',index=False)
                joblib.dump(model, OUT/f'catboost_{split}_d{depth}_{year}.joblib',compress=3)
            nums=[f.drop(columns=cats) for f in xs]
            for leaves,l2 in [(7,10),(15,20),(31,30)]:
                model=HistGradientBoostingClassifier(max_iter=350,max_leaf_nodes=leaves,l2_regularization=l2,
                    learning_rate=.04, max_bins=64,min_samples_leaf=30,early_stopping=False,random_state=42)
                model.fit(nums[0],tr.label)
                name=f'histgb_leaves{leaves}_{year}'
                p=model.predict_proba(nums[1])[:,1]; pt=model.predict_proba(nums[2])[:,1]
                preds[name]=p; tests[name]=pt
                configs[name]=dict(family='histgb',leaves=leaves,l2=l2,year=year)
                scores.append(dict(name=name,**metrics(ev,p)))
                joblib.dump(model, OUT/f'{name}_{split}.joblib',compress=3)
                print(split,name,metrics(ev,p)['phase_mean_logloss'],flush=True)
            for leaf in [5,20]:
                model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=leaf,max_features=.5,n_jobs=4,random_state=42)
                model.fit(nums[0],tr.label)
                name=f'extratrees_leaf{leaf}_{year}'
                p=model.predict_proba(nums[1])[:,1]; pt=model.predict_proba(nums[2])[:,1]
                preds[name]=p;tests[name]=pt
                configs[name]=dict(family='extratrees',leaf=leaf,year=year)
                scores.append(dict(name=name,**metrics(ev,p)))
                print(split,name,metrics(ev,p)['phase_mean_logloss'],flush=True)
        pd.DataFrame(scores).sort_values('phase_mean_logloss').to_csv(OUT/f'boosted_{split}_scores.csv',index=False)
        np.savez_compressed(OUT/f'boosted_{split}_predictions.npz',**preds)
        np.savez_compressed(OUT/f'boosted_{split}_test_predictions.npz',**tests)
        (OUT/'boosted_configs.json').write_text(json.dumps(configs,indent=2))

if __name__=='__main__':
    run()
