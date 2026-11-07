"""Direct comparison tables; no model selection or fitting."""
import json
import numpy as np
import pandas as pd
import study as s

OUT, r, f, m = s.OUT, s.r, s.f, s.m
TITLES = {"ani":"Original Ani (.89423 source recipe)", "lda_source":"Latest Ani LDA source alone", "lda_ensemble":"Current LDA 70/15/15 ensemble"}


def main():
    s.guard()
    assert r.read(OUT/"verification.json")["passed"]
    results, repeats, labels, paired, trial_changes = [], {}, [], [], []
    supplementary = r.read(OUT/"supplementary/results.json")
    for name in m.NAMES:
        recipe, access = name.split("__")
        row = dict(candidate=name, recipe=TITLES[recipe], access=access)
        for pool in ["train", "pooled"]:
            records = [next(x for x in r.read(OUT/(pool+suffix)/"index.json") if x["candidate"]==name) for suffix in ["3", "5"]]
            metrics = [metric for record in records for metric in record["repeat_scores"]]
            values = np.array([metric["sample_f1"] for metric in metrics])
            repeats[(name, pool)] = values
            row[pool+"_mean"], row[pool+"_sd"] = float(values.mean()), float(values.std())
            row[pool+"_repeat_scores"] = values.tolist()
            for label in f.LABELS:
                labels.append(dict(candidate=name, pool=pool, label=label,
                    **{key:float(np.mean([metric["reasons"][label][key] for metric in metrics])) for key in ["precision", "recall", "f1"]},
                    support=metrics[0]["reasons"][label]["support"]))
        row["validation"] = next(x["sample_f1"] for x in supplementary if x["candidate"]==name and x["window"]==0)
        chunks = [np.load(OUT/"supplementary"/f"window{wi}_{name}.npz") for wi in [1, 2, 3]]
        truth = np.concatenate([part["truth"] for part in chunks])
        score = np.concatenate([part["scores"] for part in chunks])
        assert len(truth)==98
        row["chronological"] = r.metrics(truth, score)["sample_f1"]
        results.append(row)
    for candidate, control, purpose in s.CONTRASTS:
        row = dict(candidate=candidate, control=control, purpose=purpose)
        for pool in ["train", "pooled"]:
            delta = repeats[(candidate, pool)]-repeats[(control, pool)]
            row[pool+"_delta"] = float(delta.mean())
            row[pool+"_wins"] = int((delta>1e-12).sum())
            row[pool+"_ties"] = int((abs(delta)<=1e-12).sum())
            row[pool+"_losses"] = int((delta<-1e-12).sum())
            row[pool+"_paired_deltas"] = delta.tolist()
        row["passes_cv_rule"] = bool(all(row[pool+"_delta"]>1e-12 and row[pool+"_wins"]>=5 for pool in ["train", "pooled"]))
        a = next(x for x in results if x["candidate"]==candidate)
        b = next(x for x in results if x["candidate"]==control)
        row["validation_delta"] = a["validation"]-b["validation"]
        row["chronological_delta"] = a["chronological"]-b["chronological"]
        paired.append(row)
    for access in m.ACCESS:
        for stage in ["pooled3", "pooled5"]:
            with np.load(OUT/stage/("ani__"+access+".npz")) as original, np.load(OUT/stage/("lda_ensemble__"+access+".npz")) as current:
                assert original["ids"].tolist()==current["ids"].tolist()
                truth = original["truth"]
                before = np.array([r.sample_values(truth, pred) for pred in original["predictions"]])
                after = np.array([r.sample_values(truth, pred) for pred in current["predictions"]])
                for ri in range(len(before)):
                    for j, trial_id in enumerate(original["ids"]):
                        trial_changes.append(dict(access=access, stage=stage, repeat=ri, ID=trial_id,
                            actual=" ".join(label for label, value in zip(f.LABELS, truth[j]) if value),
                            original_f1=float(before[ri,j]), current_f1=float(after[ri,j]), delta=float(after[ri,j]-before[ri,j])))
    pd.DataFrame(results).to_csv(OUT/"comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT/"paired_comparisons.csv", index=False)
    pd.DataFrame(labels).to_csv(OUT/"per_label_metrics.csv", index=False)
    pd.DataFrame(trial_changes).to_csv(OUT/"per_trial_changes.csv", index=False)
    summary = dict(comparison=results, paired_comparisons=paired,
        original_recipe_commit="a548721", original_reported_kaggle=.89423, current_ensemble_kaggle=None,
        source_notebook_parity=True, all_components_freshly_refitted=True,
        verification=r.read(OUT/"verification.json"), protocol_sha256=r.sha(OUT/"protocol.json"),
        interpretation="Matched access comparisons isolate recipe effects. Cross-access comparison describes the two existing policies. All local results use shared grouped development folds, not the original notebook's CV procedure or actual Kaggle targets.",
        part1_only=True, actual_test_inputs_loaded=False, final235_refit=False, submissions_created=False)
    r.save(OUT/"summary.json", summary)
    def table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x:f"{x:.6f}")
    columns = ["recipe", "access", "train_mean", "train_sd", "pooled_mean", "pooled_sd", "validation", "chronological"]
    contrast_columns = ["candidate", "control", "train_delta", "train_wins", "pooled_delta", "pooled_wins", "validation_delta", "chronological_delta", "passes_cv_rule"]
    highlights = []
    for access, row in zip(m.ACCESS, paired[:2]):
        highlights.append(f"<li>{access}: current ensemble minus original Ani is {row['train_delta']:+.6f} train and {row['pooled_delta']:+.6f} pooled, with {row['train_wins']}/8 and {row['pooled_wins']}/8 wins. Validation {row['validation_delta']:+.6f}; chronology {row['chronological_delta']:+.6f}. Two-pool CV rule: {'passed' if row['passes_cv_rule'] else 'not passed'}.</li>")
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Original Ani versus current LDA ensemble</title>
    <style>body{{font:16px/1.55 system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#192a3b}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border:1px solid #d8e0e8;text-align:left}}th{{background:#eaf0f6}}.scroll{{overflow-x:auto}}a{{color:#075c9b}}</style>
    <h1>Original Ani versus current LDA ensemble: identical tests</h1>
    <p>Both recipes are freshly fitted on all 80 saved grouped CV partitions, plus original validation and three chronological windows. The original Ani recipe comes directly from a548721, associated with the reported .89423 submission artifact. Every fit verifies original and later notebook function outputs on identical inputs. No settings, folds, weights or cutoffs were selected from these results.</p>
    <ul>{''.join(highlights)}</ul>
    <h2>Results on the same labeled populations</h2><div class="scroll">{table(pd.DataFrame(results)[columns])}</div>
    <p>Train CV covers 157 labeled rows; pooled CV covers 235. Each is eight repeats of five grouped folds, scored on complete OOF predictions. Validation covers the original 78 rows. Chronology pools 98 predictions from the three fixed later-year windows. SD measures repeat variation, not an independent confidence interval.</p>
    <h2>What is held equal</h2>
    <p><strong>Inductive:</strong> both pipelines fit only eligible training gold and original masked rows, with held embedding groups purged. Query inputs enter prediction only. This reproduces the earlier Ani control and current LDA ensemble exactly while freshly refitting the tri-training/SVM partners.</p>
    <p><strong>Target inputs:</strong> both pipelines additionally receive only the current held-out batch with every label erased. Scaling, PCA, GMM, pseudo-labeling and all ensemble partners use that same permitted input pool. This models original Ani's final submission access, in which target rows can receive pseudo-labels. It is separately labeled and is not inductive generalization. Chronological targets are visible as unlabeled inputs only in this arm.</p>
    <p>The original notebook's historical CV used globally fitted transforms, different KFold assignments and actual Kaggle test inputs. Those historical CV results are not used here. The shared local experiment uses each fold's target batch; actual Kaggle test data is never loaded. Its smaller target batches need not reproduce a final 312-row transductive fit.</p>
    <h2>Paired comparisons and attribution</h2><div class="scroll">{table(pd.DataFrame(paired)[contrast_columns])}</div>
    <p>The first two rows are primary matched-access comparisons. Source-only rows separate the LDA revision from the contribution of the two ensemble partners. Access contrasts measure the input-policy effect. The final row compares the current inductive pipeline with original Ani's target-input policy; that contrast combines model and access differences.</p>
    <h2>Original code and checks</h2>
    <p>Only function definitions from the two preserved notebooks are executed. Original loading, exploratory evaluation and export cells are not run. Verification checks source hashes, labels, row alignment, group purges, target-label erasure and manual rescoring of every saved prediction. Gold labels remain unchanged. All six recipe/access combinations receive all predeclared checks.</p>
    <p>These are reused development labels. Passing the paired CV rule supports a local improvement; it does not establish a new Kaggle score. Original Ani's reported .89423 remains attached to its exact historical artifact. No actual test prediction, final235 refit, submission or promotion was performed.</p>
    <p><a href="comparison.csv">Full comparison</a> · <a href="paired_comparisons.csv">All repeat deltas</a> · <a href="per_label_metrics.csv">Per-label metrics</a> · <a href="per_trial_changes.csv">Per-trial changes</a> · <a href="protocol.json">Frozen protocol</a> · <a href="verification.json">Verification</a> · <a href="summary.json">Summary</a></p></html>'''
    (OUT/"report.html").write_text(body)
    print(pd.DataFrame(results)[columns].to_string(index=False))
    print(pd.DataFrame(paired)[contrast_columns].to_string(index=False))


if __name__=="__main__":
    main()
