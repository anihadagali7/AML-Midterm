"""Original files 01–06, refitted without tuning on arbitrary benchmark folds."""
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from benchmark_models import OurLinear
from boosted import transform_dense
from ensemble_search import weighted
from experiments import ROOT, matrix

LINEAR_NAMES = ["text_wordwide_C2", "recency_criteria_C2_half6",
                "recency_clinical_C0.5_half3"]
CAP = "lr_criteria_yearcap_C2_phase0"


def recipes():
    result = {}
    for file in ["manifest.json", "followup_manifest.json"]:
        result.update(json.loads((ROOT / "submissions" / file).read_text())["ensembles"])
    return result


class LegacySubmissions:
    """Fit six original base components once, then apply frozen ensemble weights."""
    def fit(self, train, progress=lambda message: None):
        self.training_ids = train.ID.tolist()
        self.training_trial_ids = sorted(train.trial_id.unique().tolist())
        self.ensembles = recipes()
        self.linear = OurLinear().fit(train)
        self.features = self.linear.features
        blocks = self.features.transform(train)
        self.capped = LogisticRegression(C=2, solver="liblinear", max_iter=1000,
                                        tol=1e-5, random_state=42)
        self.capped.fit(matrix(blocks, "criteria", "year_cap"), train.label)
        progress("linear components fitted")
        self.svds = {key: TruncatedSVD(n_components=n, random_state=42, n_iter=5).fit(blocks[key])
                     for key, n in [("criteria", 100), ("disease", 24), ("drug", 24), ("char", 32)]}
        self.mask = np.asarray((blocks["knowledge"] > 0).sum(axis=0)).ravel() >= 15
        dense = transform_dense(train, blocks, self.svds, self.mask,
                                self.features.year_median).drop(columns="year")
        cats = dense.select_dtypes("str").columns.tolist()
        self.trees = {}
        for depth in [6, 8]:
            model = CatBoostClassifier(iterations=1000, depth=depth, learning_rate=.035,
                l2_leaf_reg=6, loss_function="Logloss", random_seed=42, thread_count=4,
                verbose=False, allow_writing_files=False)
            model.fit(dense, train.label, cat_features=cats)
            self.trees[f"catboost_d{depth}_n1000_none"] = model
            progress(f"CatBoost depth {depth} fitted")
        return self

    def components(self, df):
        blocks = self.features.transform(df)
        xs = self.linear.inputs(df, blocks)
        bank = {name: model.predict_proba(x)[:, 1]
                for name, model, x in zip(LINEAR_NAMES, self.linear.models, xs)}
        bank[CAP] = self.capped.predict_proba(matrix(blocks, "criteria", "year_cap"))[:, 1]
        dense = transform_dense(df, blocks, self.svds, self.mask,
                                self.features.year_median).drop(columns="year")
        bank.update({name: model.predict_proba(dense, ntree_end=1000)[:, 1]
                     for name, model in self.trees.items()})
        return bank

    def predict(self, df):
        bank = self.components(df)
        result = {name: weighted(bank, weights, df.phase) for name, weights in self.ensembles.items()}
        result["our_linear"] = sum(w * bank[k] for w, k in zip([.375, .375, .25], LINEAR_NAMES))
        return result
