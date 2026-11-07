"""Two fixed blends isolating the existing complementary classifier partners."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("fixed_blend_followup", HERE.parent/"blend_followup_v1/run.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SAVED = base.PART1/"experiments/current_leaders_grouped_v1/snapshots"
SUPP = base.PART1/"research/lda_ensemble/experiments/ani_lda_revision_cv_v1/supplementary"
FINAL = base.PART1/"research/lda_ensemble/experiments/final_lda_ensemble_trainval235_v1/prediction_scores.npz"
WEIGHTS = {"anchored90_partners10": .10, "anchored85_partners15": .15}


def sources():
    paths = [Path(__file__), *[SAVED/f"current_ani_lda_{pool}.npz" for pool in ["train", "pooled"]],
             *[SUPP/f"window{i}_ani__revision_full.npz" for i in range(4)], FINAL,
             base.PART1/"research/lda_ensemble/experiments/ani_direct_comparison_v1/models.py"]
    return base.sources() | {str(p.relative_to(ROOT)):base.sha(p) for p in paths}


def guard():
    assert sources() == json.loads((HERE/"protocol.json").read_text())["sources"]


def register():
    assert not (HERE/"protocol.json").exists()
    base.save(HERE/"protocol.json", dict(registered_at=datetime.now(timezone.utc).isoformat(),
        hypothesis="The current 75/25 blend adds both a similar classifier and two complementary partners. Isolating the latter may preserve diversity while avoiding the first batch's chronological regression.",
        history="Adaptive second batch after seven fixed weight/decoder changes; no clean all-check improvement. All first-batch results retained.",
        candidates=WEIGHTS, formula="P=(existing_LDA_ensemble - .7*matched_Ani_LDA_source)/.3; P is equal-weight tri-training/SVM hard-vote mixture. Score=(1-w)*centered_anchored+w*P. Cutoff .5 plus nonempty argmax.",
        actual_weights={n:dict(anchored=1-w,tri_training=w/2,svm_hard_vote=w/2) for n,w in WEIGHTS.items()},
        controls="Both scored 75/25 blend and policy parent anchored; standalone LDA ensemble retained.",
        evaluation="All 80 saved CV cases and four supplementary cases for both candidates. All component scores reused from identical inductive fitting partitions; zero new fits.",
        strict_rule="Positive mean paired delta and >=5/8 wins in both pools, separately against each control. Preserve supplementary tradeoffs.",
        shortlist="Choose at most one candidate from this batch using all local checks; no automatic promotion. Record selection before opening final235 test score arrays; avoid duplicate CSVs only after selection.",
        limitation="Adaptive development on reused labels; source score algebra requires exact ID/truth alignment and reconstruction checks. No guarantee of Kaggle improvement.",
        sources=sources()))


def partners(ensemble, source):
    p = (ensemble-.7*source)/.3
    assert p.min() >= -1e-9 and p.max() <= 1+1e-9
    p = p.clip(0, 1)
    np.testing.assert_allclose(.7*source+.3*p, ensemble, atol=1e-12, rtol=0)
    return p


def cv_components():
    data = base.load_cv()
    source = np.empty_like(data["scores"][0])
    jobs = base.reference.study.benchmark.cases()
    supplementary = [c["key"] for c in jobs if c["pool"] == "supplementary"]
    for c in jobs:
        mask = data["case"] == c["key"]
        if c["pool"] != "supplementary":
            with np.load(SAVED/f"current_ani_lda_{c['pool']}.npz") as z:
                ids, truth = z["ids"][c["positions"]], z["truth"][c["positions"]]
                score = z["scores"][c["repeat_index"], c["positions"]]
        else:
            with np.load(SUPP/f"window{supplementary.index(c['key'])}_ani__revision_full.npz") as z:
                order = {v:i for i,v in enumerate(z["ids"])}
                positions = [order[v] for v in data["ids"][mask]]
                ids, truth, score = z["ids"][positions], z["truth"][positions], z["scores"][positions]
        np.testing.assert_array_equal(ids, data["ids"][mask])
        np.testing.assert_array_equal(truth, data["truth"][mask])
        source[mask] = score
    partner = partners(data["scores"][2], source)
    np.testing.assert_allclose(.75*data["scores"][0]+.175*source+.075*partner, data["scores"][1], atol=1e-12, rtol=0)
    return data, partner


def evaluate():
    guard()
    assert not (HERE/"results").exists()
    data, partner = cv_components()
    predictions = {n:data["predictions"][i] for i,n in enumerate(["anchored", "current75_25", "lda_control"])}
    predictions.update({n:base.blend(data["scores"][0],partner,dict(lda_weight=w,cutoff=.5))[1] for n,w in WEIGHTS.items()})
    records, supp, errors = [], [], []
    for name, pred in predictions.items():
        masks = [(dict(pool=pool,seed=int(seed)),(data["pool"]==pool)&(data["seed"]==seed))
                 for pool in ["train","pooled"] for seed in np.unique(data["seed"][data["pool"]==pool])]
        masks += [(dict(case=case),data["case"]==case) for case in np.unique(data["case"][data["pool"]=="supplementary"])]
        masks += [(dict(case="chronology98"),(data["pool"]=="supplementary")&(data["case"]!="validation"))]
        for metadata, mask in masks:
            assert len(np.unique(data["ids"][mask])) == mask.sum()
            m=base.reference.study.benchmark.metrics(data["truth"][mask],pred[mask])
            independent=base.reference.metrics(data["truth"][mask],pred[mask])
            errors.extend(abs(m[k]-independent[k]) for k in m)
            (records if "pool" in metadata else supp).append(dict(mode=name,**metadata,**m))
    assert len(errors)==2310 and max(errors)<1e-12
    records,supp=pd.DataFrame(records),pd.DataFrame(supp)
    summary=records.groupby(["pool","mode"]).agg(mean_sample_f1=("sample_f1","mean"),population_sd=("sample_f1",lambda s:s.std(ddof=0)),mean_macro_f1=("macro_f1","mean")).reset_index()
    pairs,deltas=[],[]
    for baseline in ["anchored","current75_25"]:
        for pool in ["train","pooled"]:
            p=records[records.pool==pool].pivot(index="seed",columns="mode",values="sample_f1")
            for name in WEIGHTS:
                d=p[name]-p[baseline]
                wins,ties,losses=int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d< -1e-12).sum())
                pairs.append(dict(baseline=baseline,pool=pool,mode=name,mean_delta=float(d.mean()),wins=wins,ties=ties,losses=losses,meets_pool_rule=bool(d.mean()>0 and wins>=5)))
                deltas.extend(dict(baseline=baseline,pool=pool,mode=name,seed=int(s),sample_f1_delta=float(v)) for s,v in d.items())
    out=HERE/"results";out.mkdir()
    for filename,frame in [("repeat_scores",records),("cv_summary",summary),("supplementary_scores",supp),("paired_summary",pd.DataFrame(pairs)),("paired_repeat_deltas",pd.DataFrame(deltas))]:frame.to_csv(out/(filename+".csv"),index=False)
    base.save(out/"verification.json",dict(passed=True,matched_cases=84,new_component_fits=0,independent_metric_values_checked=len(errors),max_metric_error=max(errors),component_reconstruction_checked=True))
    guard()
    print(summary.pivot(index="mode",columns="pool",values="mean_sample_f1").to_string())
    print(supp.pivot(index="mode",columns="case",values="sample_f1").to_string())
    print(pd.DataFrame(pairs).query("baseline=='current75_25'").to_string(index=False))


def prepare(name):
    guard();assert name in WEIGHTS and not (HERE/"selection.json").exists()
    base.save(HERE/"selection.json",dict(selected=name,weight=WEIGHTS[name],selected_at=datetime.now(timezone.utc).isoformat(),actual_test_scores_opened_before_selection=False))
    data,partner=cv_components()
    score,pred=base.blend(data["scores"][0],partner,dict(lda_weight=WEIGHTS[name],cutoff=.5))
    np.savez_compressed(HERE/"finalist_cv.npz",scores=score,predictions=pred,**{k:data[k] for k in ["ids","truth","pool","seed","case"]})
    with np.load(base.FINAL/"prediction_scores.npz") as z:f={k:z[k] for k in z.files}
    with np.load(FINAL) as z:
        np.testing.assert_array_equal(z["ids"],f["ids"])
        np.testing.assert_array_equal(z["scores"],f["lda_ensemble"])
        partner=partners(z["scores"],z["lda_source__inductive"])
    score,pred=base.blend(f["anchored_centered"],partner,dict(lda_weight=WEIGHTS[name],cutoff=.5))
    assert not np.array_equal(pred,f["predictions"])
    sample=pd.read_csv(base.PART1/"data/part1_sample_submission.csv")
    np.testing.assert_array_equal(f["ids"],sample.ID)
    frame=pd.DataFrame(dict(ID=sample.ID,Predicted=[" ".join(label for label,flag in zip(base.LABELS,row) if flag) for row in pred]))
    path=base.PART1/"submissions/ready"/f"part1_{name}_trainval235.csv";assert not path.exists()
    frame.to_csv(path,index=False);pd.testing.assert_frame_equal(frame,pd.read_csv(path))
    np.savez_compressed(HERE/"finalist_test.npz",ids=f["ids"],scores=score,predictions=pred)
    record=dict(recipe=name,file=str(path.relative_to(ROOT)),sha256=base.sha(path),rows=312,changed_labelsets_vs_current=int((pred!=f["predictions"]).any(axis=1).sum()),
        component_weights=dict(anchored=1-WEIGHTS[name],tri_training=WEIGHTS[name]/2,svm_hard_vote=WEIGHTS[name]/2),
        all_components_verified_labels=235,original_masked_rows=1014,new_component_fits=0,test_inputs_used_for_fit=False,uploaded=False,kaggle_score=None)
    base.save(HERE/"submission.json",record);guard();print(json.dumps(record,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["register","evaluate","prepare"]);p.add_argument("--name");a=p.parse_args()
    if a.command=="register":register()
    elif a.command=="evaluate":evaluate()
    else:prepare(a.name)
