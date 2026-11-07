"""Reproducible Part 2 experiments. Only supplied pre-outcome features are used.

Run from any directory: ../../.venv/bin/python experiments.py --stage screen
All preprocessing is refitted on each experiment's fitting population.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "model_search"
OUT.mkdir(parents=True, exist_ok=True)
PHASES = ("I", "II", "III")
REASONS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]


def load_data():
    return {s: pd.read_csv(ROOT / f"outcome_{s}.csv") for s in ("train", "val", "test")}


def audit_data(data=None):
    data=load_data() if data is None else data
    audit={'datasets':{},'overlaps':{},
        'excluded_predictors':['ID','trial_id','label','all Part 1 reason labels and embeddings'],
        'evaluation':'Equal mean of per-phase log losses; official validation and train-only chronological holdout at 2008'}
    for split,df in data.items():
        assert not df.ID.duplicated().any()
        if 'label' in df:
            assert set(df.label)=={0,1}
            assert df.groupby('trial_id').label.nunique().max()==1
        audit['datasets'][split]={'rows':len(df),'trials':df.trial_id.nunique(),
            'min_year':float(df.start_year.min()),'max_year':float(df.start_year.max()),
            'sha256':hashlib.sha256((ROOT/f'outcome_{split}.csv').read_bytes()).hexdigest(),
            'phases':df.phase.value_counts().to_dict(),
            'matched_drug_fraction':float(df.matched_drugs.notna().mean())}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        n=len(set(data[a].trial_id)&set(data[b].trial_id));assert n==0
        audit['overlaps'][a+'_'+b]=n
    sample=pd.read_csv(ROOT/'part2_sample_submission.csv')
    assert set(sample.ID)==set(data['test'].ID)
    audit['sample_order_matches_test']=bool(sample.ID.equals(data['test'].ID))
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
    return audit


def list_values(value):
    if pd.isna(value):
        return []
    try:
        x = ast.literal_eval(value)
        return [str(t).lower().strip() for t in x] if isinstance(x, list) else [str(x).lower()]
    except (ValueError, SyntaxError):
        return [str(value).lower()]


def metadata_dict(row):
    out = {"phase=" + row.phase: 1., "phase_detail=" + row.phase_detail: 1.}
    for col in ("diseases_cleaned", "drugs"):
        for token in set(list_values(row[col])):
            out[col + "=" + token] = 1.
    for code in set(re.findall(r"[A-Z][0-9][0-9](?:\.[0-9A-Z]+)?", str(row.icdcodes))):
        out["icd=" + code[:3]] = 1.
        out["icdgroup=" + code[0]] = 1.
    for col in ("matched_drugs", "drug_targets", "drug_actions", "drug_pathways"):
        vals = [] if pd.isna(row[col]) else str(row[col]).lower().split(";")
        for token in set(vals):
            out[col + "=" + token.strip()] = 1.
        out[col + "_missing"] = float(not vals)
    return out


class Features:
    """Independent TF-IDF blocks; no IDs, stopping labels or reason embeddings."""
    def fit(self, df):
        self.vecs = {}
        specs = {
            "criteria": dict(ngram_range=(1, 2), min_df=3, max_df=.99, max_features=35000),
            "disease": dict(ngram_range=(1, 2), min_df=2, max_features=8000),
            "drug": dict(ngram_range=(1, 2), min_df=2, max_features=8000),
            "char": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_features=25000),
        }
        for name, kw in specs.items():
            self.vecs[name] = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", dtype=np.float64, **kw)
            self.vecs[name].fit(self.documents(df, name))
        self.dv = DictVectorizer()
        self.dv.fit([metadata_dict(r) for _, r in df.iterrows()])
        self.year_median = float(df.start_year.median())
        self.year_max = float(df.start_year.max())
        return self

    @staticmethod
    def documents(df, name):
        if name == "criteria":
            return df.criteria.fillna("").str.replace(r"\\[nr]", " ", regex=True)
        if name == "disease":
            return df.diseases_cleaned.fillna("")
        if name == "drug":
            return df.drugs.fillna("") + " " + df.matched_drugs.fillna("")
        return df.diseases_cleaned.fillna("") + " " + df.drugs.fillna("")

    def transform(self, df):
        b = {name: v.transform(self.documents(df, name)) for name, v in self.vecs.items()}
        b["knowledge"] = self.dv.transform([metadata_dict(r) for _, r in df.iterrows()])
        b["phase"] = sparse.csr_matrix(np.column_stack([(df.phase == p).astype(float) for p in PHASES]))
        year = df.start_year.fillna(self.year_median).to_numpy()
        b["year"] = sparse.csr_matrix(((year - 2007) / 5)[:, None])
        b["year_cap"] = sparse.csr_matrix(((np.minimum(year, self.year_max) - 2007) / 5)[:, None])
        return b


REPRESENTATIONS = {
    "criteria": {"criteria": 1., "phase": 1.},
    "clinical": {"criteria": 1., "disease": 1., "drug": .7, "phase": 1.},
    "knowledge": {"criteria": 1., "disease": 1., "drug": .7, "knowledge": .35, "phase": 1.},
    "rich": {"criteria": 1., "disease": .8, "drug": .5, "char": .5, "knowledge": .25, "phase": 1.},
    "metadata": {"disease": 1., "drug": .7, "knowledge": .35, "phase": 1.},
}


def matrix(blocks, representation, year="none"):
    weights = dict(REPRESENTATIONS[representation])
    if year != "none":
        weights[year] = 1.
    return sparse.hstack([blocks[k] * weight for k, weight in weights.items()], format="csr")


def metrics(df, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    result = {"logloss": log_loss(df.label, p)}
    for phase in PHASES:
        m = df.phase.eq(phase).to_numpy()
        result[f"{phase}_logloss"] = log_loss(df.label[m], p[m])
        result[f"{phase}_f1"] = f1_score(df.label[m], p[m] >= .5)
        result[f"{phase}_roc_auc"] = roc_auc_score(df.label[m], p[m])
        result[f"{phase}_pr_auc"] = average_precision_score(df.label[m], p[m])
    result["phase_mean_logloss"] = np.mean([result[f"{p}_logloss"] for p in PHASES])
    return result


def fit_linear(x, df, C, separate):
    models = {}
    for phase in (PHASES if separate else ("all",)):
        idx = np.ones(len(df), dtype=bool) if phase == "all" else df.phase.eq(phase).to_numpy()
        model = LogisticRegression(C=C, solver="liblinear", max_iter=1000, tol=1e-5, random_state=42)
        model.fit(x[idx], df.label.to_numpy()[idx])
        models[phase] = model
    return models


def predict_linear(models, x, df):
    if "all" in models:
        return models["all"].predict_proba(x)[:, 1]
    p = np.zeros(len(df))
    for phase, model in models.items():
        idx = df.phase.eq(phase).to_numpy()
        p[idx] = model.predict_proba(x[idx])[:, 1]
    return p


def split_data(data, split):
    if split == "validation":
        return data["train"].copy(), data["val"].copy()
    # Year is trial-invariant: combined-phase rows cannot cross this boundary.
    tr = data["train"]
    return tr[tr.start_year.fillna(2007) <= 2008].copy(), tr[tr.start_year > 2008].copy()


def prepare(data, split):
    path = OUT / f"features_{split}.joblib"
    if path.exists():
        return joblib.load(path)
    tr, ev = split_data(data, split)
    assert not set(tr.trial_id) & set(ev.trial_id)
    f = Features().fit(tr)
    packet = (tr, ev, f, f.transform(tr), f.transform(ev), f.transform(data["test"]))
    joblib.dump(packet, path, compress=3)
    return packet


def screen():
    data = load_data()
    audit_data(data)
    for split in ("temporal", "validation"):
        tr, ev, features, bt, be, btest = prepare(data, split)
        scores, predictions, tests, configs = [], {}, {}, {}
        scorepath = OUT / f"linear_{split}_scores.csv"
        for rep in REPRESENTATIONS:
            for year in ("none", "year"):
                xt, xe, xx = [matrix(b, rep, year) for b in (bt, be, btest)]
                for C in (.1, .5, 2., 8.):
                    pair, pair_test = [], []
                    for separate in (False, True):
                        start = time.time()
                        models = fit_linear(xt, tr, C, separate)
                        p = predict_linear(models, xe, ev)
                        pt = predict_linear(models, xx, data["test"])
                        pair.append(p); pair_test.append(pt)
                        alpha = float(separate)
                        name = f"lr_{rep}_{year}_C{C:g}_phase{alpha:g}"
                        config = dict(family="lr", representation=rep, year=year, C=C, phase_weight=alpha)
                        scores.append({"name": name, **metrics(ev, p), "seconds": time.time()-start})
                        predictions[name], tests[name], configs[name] = p, pt, config
                    for alpha in (.25, .5, .75):
                        name = f"lr_{rep}_{year}_C{C:g}_phase{alpha:g}"
                        p = (1-alpha)*pair[0] + alpha*pair[1]
                        predictions[name] = p
                        tests[name] = (1-alpha)*pair_test[0] + alpha*pair_test[1]
                        configs[name] = dict(family="lr", representation=rep, year=year, C=C, phase_weight=alpha)
                        scores.append({"name": name, **metrics(ev, p), "seconds": 0.})
                    pd.DataFrame(scores).sort_values("phase_mean_logloss").to_csv(scorepath, index=False)
                    print(split, rep, year, C, 'best', pd.DataFrame(scores).phase_mean_logloss.min(), flush=True)
        np.savez_compressed(OUT / f"linear_{split}_predictions.npz", **predictions)
        np.savez_compressed(OUT / f"linear_{split}_test_predictions.npz", **tests)
        (OUT / "linear_configs.json").write_text(json.dumps(configs, indent=2))
        print(pd.DataFrame(scores).sort_values("phase_mean_logloss").head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen"], default="screen")
    args = parser.parse_args()
    from experiments import screen
    screen()
