"""Author the report-focused notebooks with narrative, live analysis and visible source."""

import json
import textwrap
import nbformat as nbf

SETUP = """from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from threadpoolctl import threadpool_limits

# Works from the extracted part1 directory or from the repository root.
options = [Path.cwd(), Path.cwd() / "submission/midterm/part1", Path.cwd() / "midterm/part1"]
HOME = next((p for p in options if (p / "resources/handin.py").is_file()), None)
if HOME is None:
    raise RuntimeError("Extract the complete Part 1 ZIP and open this notebook beside its resources folder.")
RES = HOME / "resources"
OUT = HOME / "generated"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(RES))
import handin as h
P1, LABELS, EMB = h.P1, h.LABELS, h.EMB
# Stable numerical behavior; release automatically when the kernel exits.
thread_limit = threadpool_limits(limits=1)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
display(h.verify_resources())
"""


def md(s):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def code(s):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())


def write(home, name, cells):
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version="3.12"),
        ),
    )
    nbf.validate(nb)
    nbf.write(nb, home / name)


def build(repo, entries):
    home = repo / "submission/midterm/part1"
    common = """Run with Python 3.12 and the pinned `requirements.txt` beside this notebook. In Colab, extract the complete ZIP, change into `midterm/part1`, and install those dependencies before running. The supplied inputs and selected evidence are included; no Git checkout, private path, external trial data, or model cache is required. Notebook 05 contains the complete packaged Python source."""
    cells = [
        md(
            """# Part 1 — final anchored/Cand2 model

**Selected submission: .89935 user-reported Kaggle sample-wise F1.** This notebook explains and reconstructs the exact 80% anchored / 20% strict-inductive, five-seed Cand2 model.

Run All recomputes retained validation scores and the exact final CSV, then **freshly fits both components from the supplied data**, including all five Cand2 seeds, for validation and final235. Stored scores enter only the post-fit checks. Set `RUN_FRESH_TRAINING=False` for a quicker retained-evidence review; prior fresh-fit receipts are also included.

Read in order: data → anchored pseudo-labeling → Cand2 → fixed blend → evaluation → exact export → fresh training. This notebook replaces the older anchored-only entry point; it does not change the frozen recipe.
"""
            + common
        ),
        code(SETUP),
        md("""## Data and fitting boundaries

The original training data has 157 complete gold label rows and 1,014 rows with all five labels masked as −100. The separate validation set has 78 gold rows. The final submission uses 235 gold rows plus the original masked rows. Test has 312 rows and enters prediction only. IDs align records; year is used for chronological checks, not as a feature. Supplied labels are preserved.
"""),
        code("""train, val, test = h.data()
known = train[LABELS].ge(0).all(axis=1)
assert known.sum() == 157 and (~known).sum() == 1014 and len(val) == 78 and len(test) == 312
assert train.loc[~known, LABELS].eq(-100).all().all()
display(pd.DataFrame({"rows": [len(train), len(val), len(test)], "supplied labels": [157,78,0]}, index=["training","validation","test"]))
label_rates = pd.DataFrame({"training gold": train.loc[known,LABELS].mean(), "validation": val[LABELS].mean()})
display(label_rates)
"""),
        md("""## Anchored: clustering-assisted self-training

1. Standardize all permitted fitting embeddings. Fit 30-dimensional PCA and a 30-component diagonal GMM, three restarts, seed zero. Query inputs never fit these transformations.
2. Combine multiclass logistic regression (`C=.1`), cosine 10-neighbor voting and shrinkage LDA with weights **.5/.3/.2**. Classifiers use all 150 coordinates. Positive labels are expanded into multiclass examples; logistic weights normalize each gold row and remain anchored to gold class weights. Neighbor votes retain every positive label.
3. Build a teacher from **95% classifier scores + 5% GMM label profiles**. Accept masked rows with maximum score ≥.8; select labels at .3 plus argmax. Two refreshes give three classifier fits. Only logistic pseudo-rows receive weight .25; LDA and neighbor votes use their original weighting.

The final anchored fit accepted 757 masked rows. These are generated labels, not verified answers. The exact implementation is included in `resources/part1/anchored_solution.py` and notebook 05.
"""),
        md("""## Cand2: a distinct graph/feature component

Cand2 models eight label combinations and converts their probabilities into five label marginals. It adds soft k-means-8 memberships and 20-neighbor label-vote features. A Correct-and-Smooth labeler uses a symmetric 10-neighbor graph on standardized, PCA-64, L2-normalized embeddings. Its generated-label weight is .0630724910, confidence cutoff .5, correction coefficient .8 and smoothing coefficient .419961977. Classification uses balanced weighted shrinkage LDA.

Five seeds (0–4) are averaged. Temperature fitting and label-derived features use inner grouped refits and masked-duplicate purges. The original supplied manifest's transductive flag is provenance; the strict adapter overrides fitting access. Standalone Cand2 used expected-F1 decoding and scored .89102. The winner consumes its marginals instead.

The implementation follows the supplied Dylan pipeline, with the separately documented strict adapter. Correct-and-Smooth: Huang et al., [arXiv:2010.13993](https://arxiv.org/abs/2010.13993). Source provenance is retained with the frozen studies.
"""),
        code("""manifest = json.loads((P1 / "experiments/cand2_submission_v1/original_manifest.json").read_text())
display(manifest["config"]["model"])
print("Actual execution: strict-inductive adapter; seeds 0–4; test inputs prediction only.")
"""),
        md("""## Fixed decision rule and original validation

`score = .8 * anchored_score + .2 * Cand2_label_marginal`.

Select every label scoring at least .3 and always include argmax. These are selection scores, not established calibrated probabilities. This is arithmetic pooling, with no learned final combiner. The validation below belongs to a separate 157-label fit; it is not independent validation of the final 235-label artifact.
"""),
        code("""validation = h.arrays(P1 / "experiments/cand2_followup_v1/inputs/strict_inductive__validation.npz")
np.testing.assert_array_equal(validation["ids"], val.ID.to_numpy(str))
np.testing.assert_array_equal(validation["truth"], val[LABELS].to_numpy())
scores = .8 * validation["anchored_scores"] + .2 * validation["cand2_scores"]
prediction = h.select_reasons(scores)
headline, labels = h.metrics(validation["truth"], prediction)
assert abs(headline["sample_f1"] - .8910256410256411) < 1e-12
assert np.array_equal(prediction, validation["anchored_predictions"])
display(pd.Series(headline), labels)
labels.to_csv(OUT / "validation_per_label.csv", index=False)
ax = labels.set_index("label").f1.plot.bar(color="#336b87", ylim=(0,1.04), rot=20, title="Original validation: per-label F1")
ax.set_ylabel("F1"); plt.tight_layout(); plt.savefig(OUT / "validation_per_label.png"); plt.show()
"""),
        md("""## Reconstruct the exact scored CSV

No hidden-test labels are used. Hash equality identifies the intended exact file; the .89935 score was reported by the user, not queried from Kaggle. Outputs go to `generated/`, so the immutable scored copy is preserved.
"""),
        code("""final = h.arrays(P1 / "experiments/cand2_followup_v1/blend_final_predictions.npz")
np.testing.assert_array_equal(final["ids"], test.ID.to_numpy(str))
final_scores = .8 * final["anchored_scores"] + .2 * final["cand2_scores"]
np.testing.assert_allclose(final_scores, final["scores"], atol=1e-12, rtol=0)
final_prediction = h.select_reasons(final_scores)
np.testing.assert_array_equal(final_prediction, final["predictions"])
submission = h.frame(final["ids"], final_prediction)
submission.to_csv(OUT / "reconstructed_submission.csv", index=False, lineterminator="\\n")
assert h.sha(OUT / "reconstructed_submission.csv") == h.WINNER_SHA
assert h.sha(HOME / "part1_best_submission.csv") == h.WINNER_SHA
print("312 rows; exact winner SHA-256:", h.WINNER_SHA)
display(submission.head())
"""),
        md("""## Fresh fitting from the supplied data

The switch below defaults to `True`: fit anchored and all five Cand2 seeds for both original validation and final235. Stored scores are used **only after fitting for verification**. The final CSV must reproduce the immutable winner hash. The generated label export keeps supplied labels, identifies accepted anchored pseudo-labels, and leaves unaccepted rows masked. Runtime depends on hardware; the included receipts record the preparation machine's actual timings.

Disabling the switch skips fitting and only displays the earlier preparation receipts. With the default enabled, the outputs below document fresh fits performed by this notebook execution.
"""),
        code("""RUN_FRESH_TRAINING = True
if RUN_FRESH_TRAINING:
    validation_receipt, _ = h.fit_winner(validation=True, output=OUT / "fresh_validation")
    final_receipt, _ = h.fit_winner(validation=False, output=OUT / "fresh_final")
    display(validation_receipt, final_receipt)
else:
    for name in ["fresh_validation.json", "fresh_final.json"]:
        path = HOME / "results" / name
        if path.exists():
            display(json.loads(path.read_text()))
    print("Fresh training is disabled in this execution. Enable the switch to reproduce it.")
"""),
        md("""## Interpretation

The final blend's .89935 improves the reported score by .00299 over anchored and .00085 over the earlier anchored/LDA blend. It does not win every local comparison: the earlier LDA blend has higher Pooled235 mean. Our winner also missed the original requirement of five repeat wins in both pools. Its exploratory manual submission determined the final report choice. Notebook 03 reconstructs that evidence and preserves the selection limitation. Funding and Other have only four and ten validation positives; their metrics are particularly uncertain.
"""),
    ]
    write(home, "01_final_model.ipynb", cells)

    # Reuse visible historical code, with only data-path and evaluation-scope changes.
    legacy = json.loads(
        (repo / "submission/midterm/part1_best_model.ipynb").read_text()
    )
    src = lambda i: "".join(legacy["cells"][i]["source"])
    imports = src(1).replace("DATA = '../part1/'", 'DATA = str(P1 / "data") + "/"')
    baseline_def = src(9).split("comparison =")[0]
    feature_def = src(11).split("rows = [evaluate")[0]
    cells = [
        md(
            """# Part 1 — baselines and feature experiments

This notebook presents representative algorithms and feature choices from the report. It **freshly reruns the original validation comparisons** for the historical GMM-assisted model, then reads compact records for the separate K-means/DBSCAN and feature-screen studies. It does not rerun every grid setting.

**Historical boundary:** the first model below predates anchored. Its unsupervised scaler/PCA/GMM and feature transforms see unlabeled validation/test inputs; its pseudo-label pool includes test inputs. Validation labels stay outside fitting. This is an explicitly historical, transductive comparison, not the strict-inductive winning model and not our current grouped benchmark. The original notebook is preserved in `resources/legacy_feature_model.ipynb`.

The code is adapted from that notebook only to use packaged paths and to run validation without its expensive historical CV loop. Notebook 05 contains the full original source. The older CV averages and current grouped CV must not be mixed.
"""
            + common
        ),
        code(SETUP),
        md("## Historical preprocessing and model definitions"),
        code(imports),
        code(src(3)),
        code(src(5)),
        code("""X_pool, Y_pool = np.vstack([X_tr[is_lab], X_va]), np.vstack([Y_tr[is_lab], Y_va])
X_unl = np.vstack([X_tr[~is_lab], X_te])
def validation_result(name, fn, Xl=X_tr[is_lab], Xq=X_va, Xu=X_unl):
    score = fn(Xl, Y_tr[is_lab], Xq, Xu)
    metric, _ = h.metrics(Y_va, decide(score))
    return {"model": name, **metric}
"""),
        code(baseline_def),
        md("""## Representative classifier and self-training comparisons

These are fresh validation fits with the historical access boundary stated above. Every row shares that boundary. The label-only GMM is weak; adding supervised scores and conservative self-training improved this earlier model. The table does not imply that these values belong to the final anchored model.
"""),
        code("""comparison = pd.DataFrame([validation_result(name, fn) for name, fn in candidates.items()])
display(comparison)
comparison.to_csv(OUT / "historical_baselines_validation.csv", index=False)
expected = [.868, .872, .780, .850, .885, .889, .893]
np.testing.assert_allclose(comparison.sample_f1, expected, atol=.00051, rtol=0)
"""),
        md("""## Feature ablations from the report

Add each block separately to the same earlier model: start year; eligibility TF-IDF reduced to 50 SVD coordinates; disease TF-IDF reduced to 20; and 1,024-bit Morgan fingerprints reduced to 20. Missing/unmatched drugs receive zero fingerprints; multiple matched drugs use the elementwise maximum. Feature transforms below retain the historical transductive boundary for an exact comparison.

This experiment supports the narrower claim that these encodings did not improve this model's validation score. It does not prove those fields cannot help other models.
"""),
        code(feature_def),
        code("""rows = [validation_result("embeddings only", final_model)]
for name, block in extra.items():
    a, b, c = split_feats(np.hstack([X_all, block]))
    rows.append(validation_result(name, final_model, Xl=a[is_lab], Xq=b, Xu=np.vstack([a[~is_lab], c])))
ablation = pd.DataFrame(rows)
display(ablation)
np.testing.assert_allclose(ablation.sample_f1, [.893,.891,.876,.876,.889], atol=.00051, rtol=0)
ablation.to_csv(OUT / "historical_feature_validation.csv", index=False)
ax = ablation.set_index("model").sample_f1.plot.barh(color="#407088", title="Earlier model: validation feature comparison")
ax.set_xlim(.84,.92); ax.set_xlabel("Sample-wise F1"); plt.tight_layout()
plt.savefig(OUT / "feature_comparison.png"); plt.show()
"""),
        md("""## Separate clustering and feature-screen families

The following tables are **retained historical search results**, not fresh fits in this cell. They include K-means, diagonal/tied GMM, DBSCAN, graph labeling, nonlinear classifiers, pathways, drug targets/actions and missingness indicators. Their thresholds, pseudo-label weighting and parents differ from the model above, so compare within each study. The original notebook source, fixed settings and complete compact result tables are packaged; omitted fitted estimators and generated matrices are not needed by these finalized notebooks.
"""),
        code("""clustering = pd.read_csv(P1 / "experiments/method_comparison/clustering_comparison.csv")
display(clustering.loc[clustering.groupby("family").sample_f1.idxmax(), ["family","k","generated_trials","sample_f1","macro_f1"]])
classical = pd.read_csv(P1 / "experiments/additional_methods/new_comparison.csv")
display(classical.sort_values("sample_f1", ascending=False).head(8))
for folder in ["classifier_features", "remaining_classifier_features"]:
    filename = "comparison.csv" if folder == "classifier_features" else "best_by_group.csv"
    table = pd.read_csv(P1 / "experiments" / folder / filename)
    print(folder, "— retained study table")
    display(table[[c for c in ["experiment","feature_group","sample_f1","macro_f1","delta_vs_current_leader"] if c in table]])
"""),
        md("""## Why this did not select the final winner

The DBSCAN/logistic recipe reached .910256 on the original validation set, but the original157 and final235 submissions scored .87500 and .86452. That difference does not isolate a cause of refit degradation. More extensive feature sweeps also reused the same 78 labels, so their best values are development results. The stronger grouped and chronological comparisons in notebook 03 were introduced to make those limitations visible. Current-model boundary handling must be read from notebook 01, not inferred from these older studies.
"""),
    ]
    write(home, "02_baselines_and_features.ipynb", cells)

    cells = [
        md(
            """# Part 1 — evaluation, complementary blends and J

This notebook recomputes the current winner comparison from retained held-out predictions on **all 84 cases**: 40 Train157 folds, 40 Pooled235 folds, original validation and three chronological windows. It does not train those components again. Repeated-fold evidence is development evidence, not independent significance tests.
"""
            + common
        ),
        code(SETUP),
        md("""## Complete repeats, paired comparisons and chronology

Each grouped repeat contains every gold row exactly once. Compute sample-wise F1 over the complete out-of-fold predictions, then average the eight repeats; do not take an unweighted mean of fold F1 values. Chronology pools 98 predictions across the three fixed earlier-to-later windows. The original validation retains the original no-outer-purge boundary; grouped CV purges matching embedding groups from masked fitting rows.
"""),
        code("""repeat = h.winner_repeats()
summary = repeat.groupby(["pool","recipe"]).agg(mean_sample_f1=("sample_f1","mean"), population_sd=("sample_f1",lambda v:np.std(v,ddof=0)), mean_macro_f1=("macro_f1","mean")).reset_index()
display(summary)
repeat.to_csv(OUT / "winner_repeat_scores.csv",index=False)
summary.to_csv(OUT / "winner_benchmark.csv",index=False)
for pool, expected in [("train",(5,1,2)),("pooled",(4,1,3))]:
    paired=repeat[repeat.pool==pool].pivot(index="seed",columns="recipe",values="sample_f1")
    delta=paired.winner-paired.anchored
    counts=(int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum()),int((delta<-1e-12).sum()))
    assert counts==expected
    print(pool,"wins/ties/losses:",counts,"mean delta:",delta.mean())
"""),
        md("""The winner did not meet the original five-wins-in-both-pools rule. An explicitly authorized exploratory submission subsequently scored .89935. We preserve both facts. Its validation predictions match anchored; Pooled235 improvement is small. The prior 75/25 anchored/LDA blend remains the stronger Pooled235 average.
"""),
        code("""comparison = pd.read_csv(P1 / "reporting/benchmark_comparison.csv")
for recipe, model in [("winner","Anchored80/Cand2_20 (winner)"),("anchored","Anchored"),("cand2","Cand2 strict five-seed")]:
    current=comparison.set_index("model").loc[model]
    actual=summary[summary.recipe==recipe].set_index("pool")
    np.testing.assert_allclose(actual.loc[["train","pooled","validation","chronology98"],"mean_sample_f1"], current[["train","pooled","validation","chronology98"]].to_numpy(float),atol=1e-12,rtol=0)
display(comparison[["model","train","pooled","validation","chronology98","kaggle"]])
ax=comparison.set_index("model")[["train","pooled","kaggle"]].plot.barh(figsize=(9,6),xlim=(.85,.95),title="Different evaluation populations; Kaggle values are user-reported")
ax.set_xlabel("Sample-wise F1");plt.tight_layout();plt.savefig(OUT / "model_comparison.png");plt.show()
"""),
        md("""## What J measures, and what its ordering means

J is `(A+B)/2`: A is the original ten-repeat ordinary (ungrouped) CV score over 235 labels; B is the mean of three original chronological cuts. B retains all originally masked training years. Our current chronology uses earlier eligible rows and duplicate-group purges. These are different stress tests.

For each pair with unequal reported Kaggle scores, a metric orders the pair correctly when it assigns the higher local value to the higher Kaggle result. Pairs share models and are not independent observations. Exclude original Ani when asking about matched submitted recipes because its local adaptation differs from the historical scored artifact. The original six-recipe and later expanded panels have different denominators.
"""),
        code("""from itertools import combinations
panel=pd.read_csv(P1 / "experiments/cand2_followup_v1/results/expanded_panel.csv")
matched=panel[panel.model!="original_ani"]
rows=[]
for metric in ["dylan_J","grouped_train","grouped_pooled"]:
    correct=wrong=tied=0
    for (_,a),(_,b) in combinations(matched.iterrows(),2):
        kg=a.kaggle-b.kaggle
        if abs(kg)<1e-12:continue
        local=a[metric]-b[metric]
        if abs(local)<1e-12:tied+=1
        elif kg*local>0:correct+=1
        else:wrong+=1
    rows.append(dict(metric=metric,correct=correct,wrong=wrong,tied=tied,total=correct+wrong+tied))
rank=pd.DataFrame(rows)
assert rank.correct.tolist()==[10,14,13] and rank.total.tolist()==[14,14,14]
display(rank)
rank.to_csv(OUT / "J_pair_ordering.csv",index=False)
display(panel[panel.model.isin(["anchored","cand2_inductive5"])])
"""),
        md("""## Interpretation

The exact strict Cand2 recipe scored lower than anchored on Kaggle while J preferred it. Its small fixed contribution still helped the arithmetic blend. This supports testing complementary combinations; it does not validate J as a universal ranker or prove that decorrelation alone improves performance. Local improvements from log-odds pooling failed on Kaggle, as notebook 04 shows. No replacement J formula was fitted to these outcomes.
"""),
    ]
    write(home, "03_benchmarks_and_blends.ipynb", cells)

    cells = [
        md(
            """# Part 1 — promising directions that did not improve the winner

We keep several distinct failed directions rather than every sweep: log-odds pooling, Other-label specialists and learned combination, partial-label treatment, a diversity-penalized kernel head, and symbolic regression. These explain what we tried to add and why the .89935 model remained selected.

This notebook independently rescores retained useful predictions and PySR sufficient statistics. It does not claim new full fitting of all historical variants. Full compact result tables retain unfavorable results, and notebook 05 exposes the model/loss/training source. Fresh winning-model training is in notebook 01; symbolic search requires its separately recorded Python/Julia environment.
"""
            + common
        ),
        code(SETUP),
        md("""## Arithmetic versus log-odds pooling

The arithmetic parent mixes raw anchored scores and Cand2 marginals at .8/.2. The refinement kept those weights but pooled log-odds. Small local gains did not translate: the reported score was **.88888**. Changing score combination is not guaranteed to preserve useful calibration or decision margins.
"""),
        code("""refinement=pd.read_csv(P1 / "experiments/blend_refinement_v2/results/summary.csv")
display(refinement[refinement.recipe.isin(["parent","log_pool"])])
"""),
        md("""## Other specialists, pseudo-label weighting and learned combination

Other is a weak label: only ten validation positives. We tested nonlinear Other specialists, altered pseudo-label weighting, and nested learned heads. These recipes had to outperform the parent across both grouped pools. None qualified. Learned fusion used inner held-out inputs; it was not fitted to the same outer predictions used to evaluate it.

Partial-label experiments treated absent pseudo-Other as unknown rather than negative, while preserving supplied negative labels. A matched complete-label binary control distinguishes the architecture change from the unknown-label hypothesis. The submitted variant scored **.86645**. Its improved chronology did not justify a general claim of better performance.
"""),
        code("""checks=[]
for family in ["other_combiner_v2","partial_secondary_v1"]:
    rescored=h.compact_rescore(f"research/{family}/results/retained_comparison.npz")
    original=pd.read_csv(P1 / "research" / family / "results/summary/repeat_scores.csv")
    merged=rescored.merge(original,on=["pool","seed","recipe"],suffixes=("_fresh","_recorded"),validate="one_to_one")
    assert len(merged)==63
    np.testing.assert_allclose(merged.sample_f1_fresh,merged.sample_f1_recorded,atol=1e-12,rtol=0)
    np.testing.assert_allclose(merged.macro_f1_fresh,merged.macro_f1_recorded,atol=1e-12,rtol=0)
    checks.append({"family":family,"rescored_rows":len(merged)})
    print(family,"— all compact variant summaries")
    overview=pd.read_csv(P1 / "research" / family / "results/summary/overview.csv")
    display(overview[overview.pool.isin(["train","pooled","validation","chronology98"])].pivot(index="recipe",columns="pool",values="mean_sample_f1"))
"""),
        md("""## Explicitly complementary kernel head and learned fusion

The second head is a classical Nyström RBF feature model trained to predict labels while penalizing squared correlation between its residuals and the anchor's inner out-of-fold residuals. The ordinary-head control separates kernel capacity from the diversity term. We tested fixed score mixing and learned fusion. This was not joint neural-head training or removal of neural layers.

A stronger penalty improved the fixed blend relative to its ordinary-head control in both grouped pools, but no variant displaced the .89935 parent. The submitted fixed blend scored **.89316**. Because the ordinary-kernel blend was not submitted, this does not isolate the penalty's Kaggle effect.
"""),
        code("""z=h.arrays(P1 / "research/complementary_heads_v2/results/matched_predictions.npz")
reference=pd.read_csv(P1 / "research/complementary_heads_v2/results/summary/repeat_scores.csv")
records=[]
for pool,seed in reference[["pool","seed"]].drop_duplicates().itertuples(index=False,name=None):
    if pool in ["train","pooled"]:
        keep=np.char.startswith(z["case"],f"{pool}_{int(seed)}_")
    elif pool=="chronology98":keep=np.char.startswith(z["case"],"through_")
    else:keep=z["case"]==pool
    assert keep.any(),pool
    for j,name in enumerate(z["names"]):
        metric,_=h.metrics(z["truth"][keep],z["predictions"][j,keep])
        records.append({"pool":pool,"seed":seed,"recipe":name,**metric})
rescored=pd.DataFrame(records)
merged=rescored.merge(reference,on=["pool","seed","recipe"],suffixes=("_fresh","_recorded"),validate="one_to_one")
assert len(merged)==189
np.testing.assert_allclose(merged.sample_f1_fresh,merged.sample_f1_recorded,atol=1e-12,rtol=0)
np.testing.assert_allclose(merged.macro_f1_fresh,merged.macro_f1_recorded,atol=1e-12,rtol=0)
checks.append({"family":"complementary_heads_v2","rescored_rows":len(merged)})
overview=pd.read_csv(P1 / "research/complementary_heads_v2/results/summary/overview.csv")
display(overview.pivot(index="recipe",columns="pool",values="mean_sample_f1"))
"""),
        md("""## Bounded symbolic regression (PySR)

PySR learned small expressions over neighbor, centroid and cluster signals, with unsupervised label generation. The fixed search covered all 84 cases and five label heads per case. Its declared parent was the earlier **.89850 anchored75/LDA25** blend. Both standalone PySR and its fixed 20% contribution regressed on all four headline checks. The exploratory blend scored **.89423**.

The following code rebuilds repeat sample-F1 and per-label/macro F1 from retained sums and confusion counts. Symbolic searches are stochastic software-dependent computations; this package retains the search code, fixed settings, equations and Python/Julia locks, while default notebook execution does not install Julia or launch 420 searches.
"""),
        code("""stats=pd.read_csv(P1 / "research/pysr_v1/results/sufficient_statistics.csv")
records=[]
for (mode,pool,seed),g in stats[stats.pool.isin(["train","pooled"])].groupby(["mode","pool","seed"]):
    sums=g.select_dtypes(include=[np.number]).sum()
    fs=[]
    for label in LABELS:
        tp,fp,fn=[sums[f"{key}_{label}"] for key in ["tp","fp","fn"]]
        fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    records.append({"mode":mode,"pool":pool,"seed":seed,"sample_f1":sums.sample_f1_sum/sums.rows,"macro_f1":np.mean(fs)})
repeats=pd.DataFrame(records)
summary=repeats.groupby(["mode","pool"]).sample_f1.mean().unstack()
reference=pd.read_csv(P1 / "research/pysr_v1/results/cv_summary.csv")
for row in reference.itertuples():
    assert abs(summary.loc[row.mode,row.pool]-row.mean_sample_f1)<1e-12
display(summary, pd.read_csv(P1 / "research/pysr_v1/results/supplementary_scores.csv"))
display(pd.read_csv(P1 / "research/pysr_v1/results/equations.csv").head(10))
checks.append({"family":"pysr_v1","rescored_rows":len(repeats)})
pd.DataFrame(checks).to_csv(OUT / "negative_evidence_checks.csv",index=False)
"""),
        md("""## What we learned

A small component can help despite being worse alone, as Cand2 did. More complexity, explicit diversity, a learned combiner or better chronology is not sufficient evidence of a better hidden-test model. Preserve matched controls, look across metrics without combining incompatible protocols, and retain negative outcomes. These comparisons repeatedly reuse development labels; no overlapping-repeat significance claim is made.
"""),
    ]
    write(home, "04_unsuccessful_directions.ipynb", cells)

    cells = [
        md(
            """# Part 1 — packaged source code

This notebook exposes the Python implementations behind the four reader notebooks, including frozen training, graph/pseudo-label methods, loss functions, calibration, experiment runners, and verification. The packaged files retain their original hashes. `%%writefile` cells write review copies under `_source_export/`; they do not execute every historical experiment or overwrite the frozen resources.

The four finalized notebooks are portable. Historical runners and source notebooks remain provenance: some refer to omitted intermediate models/matrices or historical worktrees and may need the full research checkout or regeneration of those inputs. They are not presented as portable one-click replays. The winner's fresh training and the representative feature experiments have dedicated, tested portable paths in notebooks 01 and 02.
"""
            + common
        ),
        code(SETUP),
    ]
    source_names = [name for name in entries if name.endswith(".py")]
    cells.append(
        code(
            """source_names = """
            + repr(source_names)
            + """
EXPORT=HOME / "_source_export"
for name in source_names:
    (EXPORT/name).parent.mkdir(parents=True,exist_ok=True)
# Relative writefile destinations below are rooted beside this notebook.
import os
os.chdir(HOME)
"""
        )
    )
    for name in source_names:
        cells.extend(
            [
                md("## " + name),
                code(
                    "%%writefile _source_export/"
                    + name
                    + "\n"
                    + (home / "resources" / name).read_text()
                ),
            ]
        )
    cells.append(
        code("""for name in source_names:
    assert h.sha(EXPORT/name)==h.sha(RES/name),name
print("Verified visible source copies:",len(source_names))
""")
    )
    cells.append(
        md("""## Original notebook-only study source

The historical baseline/feature studies below were authored directly as notebooks. Their full code is reproduced in Markdown code blocks for review; executing this source-export notebook does not replay their searches. The original `.ipynb` files, settings and compact results are also included as source records.
""")
    )
    for name in [n for n in entries if n.endswith(".ipynb")]:
        nb = json.loads((home / "resources" / name).read_text())
        cells.append(md("## " + name))
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                cells.append(md("```python\n" + "".join(c["source"]) + "\n```"))
    write(home, "05_source_code.ipynb", cells)
