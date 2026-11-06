"""Independently check all scored extensions, fitting boundaries and exports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_protocol import initialize,resolve,digest
from experiments import ROOT
from improvement_report import audit_track
from extension_common import current,OUT,PREVIOUS,REFERENCES,previous_packet
from extension_meta import RECIPES as META_RECIPES
from extension_neighbors import RECIPES as NN_RECIPES
from extension_followthrough import DECISION,selection


def verify():
    universe,protocol=initialize();decision=selection()
    assert digest(ROOT/"EXTENSION_PROTOCOL.md")==decision["policy_sha256"]
    checked=0
    for batch,registry in [("meta",META_RECIPES),("neighbors",NN_RECIPES)]:
        root=current(batch);manifest=json.loads((root/"manifest.json").read_text())
        for source,expected in manifest["source_sha256"].items():assert digest(ROOT/source)==expected,source
        for track in ["strict","published"]:
            assert digest(root/track/"summary.csv")==decision["input_hashes"][batch+"/"+track]
            audit_track(root/track,universe,protocol,names=REFERENCES+list(registry))
            checked+=len(protocol["splits"])*len(registry)
            for split in protocol["splits"]:
                train,ev=resolve(universe,split)
                for name in REFERENCES:
                    original=pd.read_csv(PREVIOUS/track/"predictions"/split["name"]/f"{name}.csv")
                    new=pd.read_csv(root/track/"predictions"/split["name"]/f"{name}.csv")
                    assert original.ID.equals(new.ID)
                    np.testing.assert_allclose(original.probability,new.probability,rtol=0,atol=1e-14)
                if batch=="neighbors":
                    audits=json.loads((root/track/"bases"/f"{split['name']}_audit.json").read_text())
                    old=previous_packet(track,split)
                    for fold in audits:
                        ix=np.flatnonzero(old["fold_ids"]==fold["fold"])
                        assert fold["eval_ids"]==train.ID.iloc[ix].tolist()
                        assert not set(fold["fit_ids"])&set(fold["eval_ids"])
                        held_trials=set(train.set_index("ID").loc[fold["eval_ids"],"trial_id"])
                        fit_trials=set(train.set_index("ID").loc[fold["fit_ids"],"trial_id"])
                        assert not held_trials&fit_trials
                        if track=="strict":
                            assert set(fold["preprocessing_fit_ids"])==set(fold["fit_ids"])
                            assert not set(fold["preprocessing_fit_ids"])&set(ev.ID)
    names=REFERENCES+[x["model"] for x in decision["shortlist"]]
    gap_protocol=json.loads((DECISION/"gaps/protocol.json").read_text())
    assert gap_protocol["selection_sha256"]==digest(DECISION/"selection.json")
    for track in ["strict","published"]:
        audit_track(DECISION/track,universe,protocol,names=names)
        audit_track(DECISION/"gaps"/track,universe,gap_protocol,names=names)
    manifest=json.loads((DECISION/"final_manifest.json").read_text())
    assert digest(ROOT/manifest["bundle_path"])==manifest["bundle_sha256"]
    assert max(manifest["native_replay_errors"].values())<1e-12
    sample=pd.read_csv(ROOT/"part2_sample_submission.csv")
    for item in manifest["submissions"]:
        path=ROOT/"submissions"/item["file"]
        assert digest(path)==item["sha256"] and item["kaggle_score"] is None
        data=pd.read_csv(path)
        assert data.ID.equals(sample.ID) and data.ID.is_unique and len(data)==1500
        assert data.label.between(0,1,inclusive="neither").all()
    result=dict(extension_recipes=len(META_RECIPES)+len(NN_RECIPES),new_recipe_evaluations=checked,
        original_splits_per_track=10,preprocessing_tracks=2,controls_reproduce=True,
        source_and_input_hashes_match=True,grouped_inner_preprocessing_verified=True,
        independently_rescored=True,gapped_splits_per_track=3,exported_submissions=len(manifest["submissions"]),
        native_replay_max_error=max(manifest["native_replay_errors"].values()),
        labels_and_identifiers_do_not_change_inference=True,kaggle_scores_unknown=True)
    (DECISION/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":verify()
