"""Independently audit saved temporal feature, expert and nested-decision fits."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PART1 = HERE.parents[1]
LABELS = ["Safety","Funding","Efficacy","Recruitment","Other"]
EMB = [f"emb_{j}" for j in range(150)]
STUDIES = ["temporal_lda_cv_v1","recent_era_lda_cv_v1","nested_decision_lda_cv_v1"]
GRID = [.3,.25,.35,.2,.4]


def read(path):
    return json.loads(path.read_text())


def evaluate(threshold,audit):
    raw = np.clip(np.array(audit["inner_source_probabilities"]),1e-9,1-1e-9)
    partner = np.array(audit["inner_partner_scores"])
    t = np.asarray(threshold)
    score = .7*raw*(1-t)/(raw*(1-t)+(1-raw)*t)+partner
    predicted = score>=.5
    for row in np.flatnonzero(~predicted.any(1)):
        predicted[row,np.argmax(score[row])] = True
    truth = np.array(audit["inner_truth"])
    f1 = (2*(predicted*truth).sum(1)/(predicted.sum(1)+truth.sum(1)))
    years = np.array(audit["inner_years"])
    recent = audit["settings"]["recent"]
    weights = np.maximum(.1,2**((years-years.max())/5)) if recent else np.ones(len(years))
    mean = float((f1*weights).sum()/weights.sum())
    penalty = .01*((t-.3)/.1)**2
    penalty = float(penalty.sum() if audit["settings"]["per_label"] else penalty.mean())
    return mean-penalty,mean,penalty


def thresholds(audit):
    selected = np.repeat(.3,5)
    objective = evaluate(selected,audit)[0]
    labels = list(range(5))*2 if audit["settings"]["per_label"] else [None]
    for label in labels:
        best = selected.copy()
        for cutoff in GRID:
            trial = selected.copy()
            if label is None:
                trial[:] = cutoff
            else:
                trial[label] = cutoff
            score = evaluate(trial,audit)[0]
            if score>objective+1e-12:
                objective,best = score,trial.copy()
        selected = best
    np.testing.assert_allclose(selected,audit["thresholds"],atol=1e-12,rtol=0)
    last = audit["history"][-1]
    np.testing.assert_allclose(evaluate(selected,audit),
        [last["objective"],last["inner_weighted_sample_f1"],last["penalty"]],atol=1e-12,rtol=0)


def main():
    data = pd.concat([pd.read_csv(PART1/"data/part1_train.csv"),pd.read_csv(PART1/"data/part1_val.csv")]).set_index("ID")
    assert data.index.is_unique
    groups = pd.util.hash_pandas_object(data[EMB].round(10),index=False)
    totals = dict(feature_records=0,expert_records=0,decision_records=0,inner_fold_records=0)
    study_records = []
    artifact_hashes = {}
    for name in STUDIES:
        directory = HERE.parent/name
        verification = read(directory/"independent_verification.json")
        assert verification["passed"]
        protocol = read(directory/"protocol.json")
        for path,digest in protocol["source_sha256"].items():
            assert hashlib.sha256((PART1/path).read_bytes()).hexdigest()==digest,path
        records = []
        for stage in ["train3","pooled3","train5","pooled5"]:
            for path in sorted((directory/stage).glob("seed*_fold*.json")):
                outer = read(path)
                for feature in outer["features"]:
                    if feature["block"]!="revision_full":
                        records.append((feature,outer["fit_ids"],outer["query_ids"]))
        for path in sorted((directory/"supplementary").glob("window*_audit.json")):
            feature = read(path)
            if feature["block"]=="revision_full":
                continue
            window = path.name.split("_")[0]
            with np.load(directory/"supplementary"/(window+"_ensemble__"+feature["block"]+".npz")) as saved:
                records.append((feature,feature["fitting_ids"],saved["ids"].tolist()))
        for audit,fit_ids,query_ids in records:
            fitting,query = set(fit_ids),set(query_ids)
            assert set(audit["fitting_ids"])==fitting
            assert not fitting&query
            assert not audit["query_inputs_used_for_fitting"]
            gold = set(data.loc[fit_ids].index[data.loc[fit_ids,LABELS].ge(0).all(1)])
            if name=="temporal_lda_cv_v1":
                totals["feature_records"]+=1
                for key in ["drift_fit_ids","classifier_fit_ids","neighbor_fit_ids"]:
                    if key in audit:
                        assert set(audit[key])<=fitting
                if audit["block"]=="within_reason_half":
                    assert set(audit["drift_fit_ids"])==gold
                if "class_weight_mass_original" in audit:
                    np.testing.assert_allclose(audit["class_weight_mass_original"],audit["class_weight_mass_new"],atol=1e-9,rtol=0)
            elif name=="recent_era_lda_cv_v1":
                totals["expert_records"]+=1
                expert = set(audit["expert_fitting_ids"])
                selected = set(audit["expert_gold_ids"])
                excluded = set(audit["excluded_gold_ids"])
                assert expert<=fitting and selected<=expert
                assert not expert&excluded and selected|excluded==gold
                assert not set(groups.loc[list(expert)])&set(groups.loc[list(excluded)])
                counts = data.loc[audit["expert_gold_ids"],LABELS].sum().to_numpy(int)
                np.testing.assert_array_equal(counts,audit["expert_reason_support"])
                assert (counts>=audit["minimum_reason_support"]).all()
            else:
                totals["decision_records"]+=1
                ids = audit["inner_gold_ids"]
                assert set(ids)==gold and len(ids)==len(gold)
                np.testing.assert_array_equal(audit["inner_truth"],data.loc[ids,LABELS].to_numpy(int))
                np.testing.assert_array_equal(audit["inner_years"],data.loc[ids,"start_year"].to_numpy(float))
                seen = []
                for fold in audit["inner_folds"]:
                    a,b = set(fold["fit_ids"]),set(fold["query_ids"])
                    assert a<=fitting and b<=gold and not a&b
                    assert not set(groups.loc[list(a)])&set(groups.loc[list(b)])
                    seen.extend(fold["query_ids"])
                    totals["inner_fold_records"]+=1
                assert sorted(seen)==sorted(ids)
                thresholds(audit)
        study_records.append(dict(study=name,records_checked=len(records),independent_rescore=verification))
        for filename in ["protocol.json","summary.json","verification.json","independent_verification.json"]:
            path=directory/filename
            artifact_hashes[str(path.relative_to(PART1))]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(passed=True,**totals,studies=study_records,all_protected_hashes_unchanged=True,
        nested_cutoffs_independently_reconstructed=True,part1_only=True,test_inputs_loaded=False,
        submissions_created=False,artifact_sha256=artifact_hashes,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["studies","artifact_sha256"]}))


if __name__=="__main__":
    main()
