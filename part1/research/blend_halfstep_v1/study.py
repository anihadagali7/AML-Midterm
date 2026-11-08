"""One fixed half-step between the scored blend and the complementary blend."""
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("partner_followup",HERE.parent/"blend_partners_v1/study.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
b=p.base
NAME="blend_halfstep"


def sources():
    return p.sources() | {str(Path(__file__).relative_to(p.ROOT)):b.sha(Path(__file__))}


def guard():
    assert sources()==json.loads((HERE/"protocol.json").read_text())["sources"]


def score(a,ensemble,partner):
    # Half current (.75 A + .25 ensemble), half (.90 A + .10 partner).
    s=.5*(.75*a+.25*ensemble)+.5*(.9*a+.1*partner)
    pred=(s>=.5).astype(int);pred[np.arange(len(s)),s.argmax(axis=1)]=1
    return s,pred


def register():
    assert not (HERE/"protocol.json").exists()
    b.save(HERE/"protocol.json",dict(registered_at=datetime.now(timezone.utc).isoformat(),
        hypothesis="The 90% anchored/10% complementary-partners blend improves both CV means but loses chronology. A fixed half-step toward it may reduce this loss while preserving some CV gain.",
        history="Adaptive third batch after seven weight/decoder settings and two partner-only blends. All prior results retained; no unobserved-holdout claim.",
        recipe="Exactly half current75_25, half anchored90_partners10; cutoff .5 plus argmax. Equivalent weights: anchored .825, Ani-LDA source .0875, tri-training .04375, SVM hard vote .04375.",
        controls="Scored 75/25 blend and original anchored policy parent.",
        evaluation="All 80 saved CV cases plus original validation and all three chronological windows, irrespective of CV outcome; exact saved components, zero fresh fits.",
        strict_rule="Positive mean paired difference and >=5/8 wins in both pools, reported separately from manual trial recommendation.",
        selection="One candidate only. Export if locally supported after displaying all tradeoffs. Freeze selection before opening final test score arrays. No upload.",
        sources=sources()))


def evaluate():
    guard();assert not (HERE/"results").exists()
    data,partner=p.cv_components();s,pred=score(data["scores"][0],data["scores"][2],partner)
    rows,supp,errors=[],[],[]
    masks=[(dict(pool=pool,seed=int(seed)),(data["pool"]==pool)&(data["seed"]==seed)) for pool in ["train","pooled"] for seed in np.unique(data["seed"][data["pool"]==pool])]
    masks +=[(dict(case=case),data["case"]==case) for case in np.unique(data["case"][data["pool"]=="supplementary"])]
    masks +=[(dict(case="chronology98"),(data["pool"]=="supplementary")&(data["case"]!="validation"))]
    predictions={"anchored":data["predictions"][0],"current75_25":data["predictions"][1],NAME:pred}
    for name,q in predictions.items():
        for meta,mask in masks:
            values=b.reference.study.benchmark.metrics(data["truth"][mask],q[mask]);ind=b.reference.metrics(data["truth"][mask],q[mask])
            errors.extend(abs(values[k]-ind[k]) for k in values)
            (rows if "pool" in meta else supp).append(dict(mode=name,**meta,**values))
    assert len(errors)==1386 and max(errors)<1e-12
    rows,supp=pd.DataFrame(rows),pd.DataFrame(supp)
    summary=rows.groupby(["pool","mode"]).agg(mean_sample_f1=("sample_f1","mean"),population_sd=("sample_f1",lambda x:x.std(ddof=0)),mean_macro_f1=("macro_f1","mean")).reset_index()
    pairs,deltas=[],[]
    for parent in ["anchored","current75_25"]:
        for pool in ["train","pooled"]:
            pivot=rows[rows.pool==pool].pivot(index="seed",columns="mode",values="sample_f1");d=pivot[NAME]-pivot[parent]
            w,t,l=int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d< -1e-12).sum())
            pairs.append(dict(baseline=parent,pool=pool,mode=NAME,mean_delta=float(d.mean()),wins=w,ties=t,losses=l,meets_pool_rule=bool(d.mean()>0 and w>=5)))
            deltas.extend(dict(baseline=parent,pool=pool,seed=int(seed),sample_f1_delta=float(v)) for seed,v in d.items())
    out=HERE/"results";out.mkdir()
    for name,frame in [("repeat_scores",rows),("cv_summary",summary),("supplementary_scores",supp),("paired_summary",pd.DataFrame(pairs)),("paired_repeat_deltas",pd.DataFrame(deltas))]:frame.to_csv(out/(name+".csv"),index=False)
    b.save(out/"verification.json",dict(passed=True,matched_cases=84,new_component_fits=0,independent_metric_values_checked=len(errors),max_metric_error=max(errors)))
    guard();print(summary.to_string(index=False));print(supp[["mode","case","sample_f1"]].to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))


def prepare():
    guard();assert not (HERE/"selection.json").exists()
    b.save(HERE/"selection.json",dict(selected=NAME,selected_at=datetime.now(timezone.utc).isoformat(),test_scores_opened_before_selection=False))
    data,partner=p.cv_components();s,pred=score(data["scores"][0],data["scores"][2],partner)
    np.savez_compressed(HERE/"finalist_cv.npz",scores=s,predictions=pred,**{k:data[k] for k in ["ids","truth","case","pool","seed"]})
    with np.load(b.FINAL/"prediction_scores.npz") as z:f={k:z[k] for k in z.files}
    with np.load(p.FINAL) as z:
        np.testing.assert_array_equal(z["ids"],f["ids"]);np.testing.assert_array_equal(z["scores"],f["lda_ensemble"])
        partner=p.partners(z["scores"],z["lda_source__inductive"])
    s,pred=score(f["anchored_centered"],f["lda_ensemble"],partner)
    assert not np.array_equal(pred,f["predictions"])
    sample=pd.read_csv(b.PART1/"data/part1_sample_submission.csv");np.testing.assert_array_equal(f["ids"],sample.ID)
    frame=pd.DataFrame(dict(ID=sample.ID,Predicted=[" ".join(label for label,flag in zip(b.LABELS,row) if flag) for row in pred]))
    path=b.PART1/"submissions/ready/part1_blend_halfstep_trainval235.csv";assert not path.exists()
    frame.to_csv(path,index=False);pd.testing.assert_frame_equal(frame,pd.read_csv(path))
    np.savez_compressed(HERE/"finalist_test.npz",ids=f["ids"],scores=s,predictions=pred)
    record=dict(recipe=NAME,file=str(path.relative_to(p.ROOT)),sha256=b.sha(path),rows=312,changed_labelsets_vs_current=int((pred!=f["predictions"]).any(axis=1).sum()),
        component_weights=dict(anchored=.825,ani_lda_source=.0875,tri_training=.04375,svm_hard_vote=.04375),all_components_verified_labels=235,
        original_masked_rows=1014,new_component_fits=0,test_inputs_used_for_fit=False,uploaded=False,kaggle_score=None)
    b.save(HERE/"submission.json",record);guard();print(json.dumps(record,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["register","evaluate","prepare"]);a=parser.parse_args()
    {"register":register,"evaluate":evaluate,"prepare":prepare}[a.command]()
