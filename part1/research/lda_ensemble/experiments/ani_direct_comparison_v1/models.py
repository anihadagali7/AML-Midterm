"""Unchanged original Ani and current LDA recipes under matched input access."""
from pathlib import Path
import ast
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parent
GAP = HERE.parent/"feature_gap_cv_v1"
REVISION = HERE.parent/"ani_lda_revision_cv_v1"
sys.path.insert(0, str(GAP))
import features as f
import run as runner
originals = {name:getattr(runner, name) for name in ["OUT", "m", "prepare", "guard", "parity", "shortlist", "report"]}
spec = importlib.util.spec_from_file_location("direct_lda_source", REVISION/"study.py")
revision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revision)
for name, value in originals.items():
    setattr(runner, name, value)
sys.path.insert(0, str(f.RESEARCH))
import three_way_transductive as partners

NOTEBOOKS = {"ani": f.PART1/"research/ani_snapshots/a548721/part1_best_model.ipynb",
             "lda_source": f.PART1/"research/ani_snapshots/8a8c3fc/part1_best_model.ipynb"}
ACCESS = ["inductive", "target_inputs"]
RECIPES = ["ani", "lda_source", "lda_ensemble"]
NAMES = [recipe+"__"+access for access in ACCESS for recipe in RECIPES]


def fitting_frame(fit, query, access):
    query = query.drop(columns=f.LABELS, errors="ignore").copy()
    assert not set(fit.ID)&set(query.ID)
    assert fit.ID.is_unique and query.ID.is_unique
    assert fit[f.LABELS].ge(0).all(axis=1).any()
    if access == "inductive":
        return fit.copy().reset_index(drop=True), query
    assert access == "target_inputs"
    target = query.assign(**{label:-100 for label in f.LABELS})
    result = pd.concat([fit, target], ignore_index=True)
    assert result.iloc[len(fit):][f.LABELS].eq(-100).all().all()
    return result, query


def notebook_probability(packet, query, recipe):
    """Run only original function definitions, with this partition's transforms."""
    notebook = NOTEBOOKS[recipe]
    cells = json.loads(notebook.read_text())["cells"]
    code = next("".join(c["source"]) for c in cells if c["cell_type"]=="code" and "def final_model(" in "".join(c["source"]))
    tree = ast.parse(code)
    assert all(isinstance(node, ast.FunctionDef) for node in tree.body)
    geometry = revision.geometry(packet)
    scope = dict(np=np, normalize=normalize, LogisticRegression=LogisticRegression,
        LinearDiscriminantAnalysis=LinearDiscriminantAnalysis, EMB=f.EMB,
        pca30=geometry["pca"], gmm=geometry["gmm"])
    exec(compile(tree, str(notebook)+":function_definitions_only", "exec"), scope)
    unknown = np.setdiff1d(np.arange(len(packet["z"])), packet["gold"])
    return scope["final_model"](packet["z"][packet["gold"]], packet["y"],
        packet["scaler"].transform(query[f.EMB].to_numpy()), packet["z"][unknown])


def predict(fit, query, access, source_parity=False):
    fitting, query = fitting_frame(fit, query, access)
    packet = f.train_native(fitting)
    ani = f.native_prediction(packet, query)
    lda, _, _ = revision.fit_addition(packet, fitting, query, "revision_full")
    # Refit both unchanged partner recipes on exactly the same permitted frame.
    tri, svm = partners.partners_in_frame(fitting, query)
    ensemble = .7*lda+.15*tri+.15*svm
    original_ensemble = .7*ani+.15*tri+.15*svm
    parity = {}
    if source_parity:
        for name, score in [("ani", ani), ("lda_source", lda)]:
            expected = f.centered(notebook_probability(packet, query, name))
            parity[name] = float(np.max(abs(score-expected)))
            np.testing.assert_allclose(score, expected, atol=1e-10, rtol=0)
    latest = revision.fitted_variant(packet, revision.SETTINGS["revision_full"])
    known = fitting[f.LABELS].ge(0).all(axis=1)
    assert not set(fitting.loc[known, "ID"])&set(query.ID)
    audit = dict(access=access, permitted_base_fit_ids=fit.ID.tolist(),
        fitting_ids=fitting.ID.tolist(), gold_ids=fitting.loc[known, "ID"].tolist(),
        masked_ids=fitting.loc[~known, "ID"].tolist(), query_ids=query.ID.tolist(),
        target_input_rows=len(query) if access=="target_inputs" else 0,
        target_labels_erased=True, identical_input_access_all_components=True,
        original_pseudo_round_counts=packet["history"], latest_pseudo_history=latest["history"],
        original_final_pseudo_query_rows=int(fitting.iloc[packet["rows"]].ID.isin(query.ID).sum()),
        latest_final_pseudo_query_rows=int(fitting.iloc[latest["rows"]].ID.isin(query.ID).sum()),
        notebook_function_max_differences=parity)
    scores = {"ani__"+access:ani, "lda_source__"+access:lda, "lda_ensemble__"+access:ensemble}
    for score in scores.values():
        assert score.shape==(len(query), 5) and np.isfinite(score).all()
        assert score.min()>=0 and score.max()<=1+1e-10
    return scores, audit, original_ensemble
