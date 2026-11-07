"""Independent metric rescore plus exact fitting-boundary and notebook audit."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import study as s

OUT, r, f, m = s.OUT, s.r, s.f, s.m
sys.path.insert(0, str(m.GAP))
import verify_saved


def check_fit(record, fit, query):
    assert record["fit_ids"]==fit.ID.tolist()
    assert record["query_ids"]==query.ID.tolist()
    assert record["protocol_sha256"]==r.sha(OUT/"protocol.json")
    for access, audit in record["fits"].items():
        expected, clean = m.fitting_frame(fit, query, access)
        known = expected[f.LABELS].ge(0).all(axis=1)
        assert audit["fitting_ids"]==expected.ID.tolist()
        assert audit["gold_ids"]==expected.loc[known, "ID"].tolist()
        assert audit["masked_ids"]==expected.loc[~known, "ID"].tolist()
        assert audit["query_ids"]==clean.ID.tolist()
        assert not set(audit["gold_ids"])&set(query.ID)
        assert audit["target_input_rows"]==(0 if access=="inductive" else len(query))
        assert audit["identical_input_access_all_components"] and audit["target_labels_erased"]
        assert all(value<1e-10 for value in audit["notebook_function_max_differences"].values())
        assert set(audit["notebook_function_max_differences"])=={"ani", "lda_source"}
        pd.testing.assert_frame_equal(expected.iloc[:len(fit)][f.LABELS], fit[f.LABELS])
        if access=="inductive":
            assert not set(audit["fitting_ids"])&set(query.ID)
            assert audit["original_final_pseudo_query_rows"]==audit["latest_final_pseudo_query_rows"]==0
        else:
            assert set(query.ID)<=set(audit["masked_ids"])
            assert expected.iloc[len(fit):][f.LABELS].eq(-100).all().all()


def main():
    s.guard()
    verify_saved.HERE = OUT
    verify_saved.main()
    count = 0
    for stage in r.SPLITS:
        frame = r.frame_for("pooled" if stage.startswith("pooled") else "train")
        plan, pos, groups, masked = r.plan(stage, frame)
        assert set(row["candidate"] for row in r.read(OUT/stage/"index.json"))==set(m.NAMES)
        for fold in plan["folds"]:
            fit, query, held = r.fold_inputs(frame, pos, groups, masked, fold)
            record = r.read(OUT/stage/f"seed{fold['seed']}_fold{fold['fold']}.json")
            check_fit(record, fit, query)
            assert record["all_models_freshly_refitted"]
            assert record["control_max_difference"]<1e-10
            count += 1
    train, full = r.frame_for("train"), r.frame_for("pooled").set_index("ID", drop=False)
    val = pd.read_csv(r.PART1/"data/part1_val.csv")
    windows = [dict(training_ids=train.ID.tolist(), heldout_ids=val.ID.tolist())]
    windows += r.read(r.ROOT/"experiments/cross_repo_benchmark_v1/protocol.json")["chronological_splits"]
    for wi, window in enumerate(windows):
        fit = full.loc[window["training_ids"]].reset_index(drop=True)
        query = full.loc[window["heldout_ids"]].reset_index(drop=True)
        check_fit(r.read(OUT/"supplementary"/f"window{wi}_audit.json"), fit, query)
        for name in m.NAMES:
            with np.load(OUT/"supplementary"/f"window{wi}_{name}.npz") as saved:
                assert saved["ids"].tolist()==query.ID.tolist()
                np.testing.assert_array_equal(saved["truth"], query[f.LABELS].to_numpy(int))
        if wi:
            assert not set(pd.util.hash_pandas_object(fit[f.EMB].round(10), index=False))&set(pd.util.hash_pandas_object(query[f.EMB].round(10), index=False))
    result = dict(passed=True, grouped_folds_verified=count, supplementary_windows_verified=len(windows),
        source_function_parity_checks=(count+len(windows))*4,
        fresh_inductive_controls_verified=True, all_six_recipes_evaluated_everywhere=True,
        identical_permitted_gold_and_original_masked_inputs=True, target_labels_erased=True,
        input_access_regimes_separate=True, original_gold_unchanged=True,
        actual_test_inputs_loaded=False, submissions_created=False, source_hashes_unchanged=True,
        independent_rescore=r.read(OUT/"independent_verification.json"))
    r.save(OUT/"verification.json", result)
    print(result)


if __name__=="__main__":
    main()
