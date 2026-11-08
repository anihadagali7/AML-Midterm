"""Verify the compact scored record after nonwinning model caches are removed."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LABELS=['Safety','Funding','Efficacy','Recruitment','Other']
REMOVED={'model_state.npz','prediction_scores.npz','augmented_labels.csv'}

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def verify():
    retained=read(HERE/'retention.json')
    for path,digest in retained['files'].items():assert sha(ROOT/path)==digest,path
    protocol=read(HERE/'protocol.json')
    for path,digest in protocol['sources'].items():assert sha(ROOT/path)==digest,path
    manifest=read(HERE/'manifest.json');result_path='part1/submissions/results/current80-pysr20-trainval235.json';result=read(ROOT/result_path)
    assert result['reported_kaggle_score']==.89423
    assert result['submission_sha256']==manifest['submission_sha256']
    assert sha(ROOT/result['submission_file'])==result['submission_sha256']
    for name,digest in manifest['artifacts'].items():
        if name in REMOVED:assert not (HERE/name).exists()
        else:assert sha(HERE/name)==digest,name
    sample=rows(ROOT/'part1/data/part1_sample_submission.csv');submitted=rows(ROOT/result['submission_file'])
    ids=[r['ID'] for r in submitted];assert ids==[r['ID'] for r in sample] and len(ids)==len(set(ids))==312
    for row in submitted:
        labels=row['Predicted'].split();assert labels and len(labels)==len(set(labels)) and set(labels)<=set(LABELS)
        assert row['Predicted']==' '.join(label for label in LABELS if label in labels)
    parent_result=read(ROOT/'part1/submissions/results/anchored75-lda25-trainval235.json')
    parent=rows(ROOT/parent_result['submission_file']);assert sha(ROOT/parent_result['submission_file'])==parent_result['submission_sha256']
    assert [r['ID'] for r in parent]==ids
    changes=[dict(ID=a['ID'],previous=a['Predicted'],candidate=b['Predicted']) for a,b in zip(parent,submitted) if a['Predicted']!=b['Predicted']]
    assert changes==rows(HERE/'changes_vs_89850.csv') and len(changes)==manifest['changed_labelsets_vs_89850']==9
    eq=read(HERE/'equations.json');assert [e['label'] for e in eq['equations']]==LABELS
    assert eq['model_audit']['gold_rows']==235 and eq['model_audit']['masked_rows']==1014
    assert eq['model_audit']['observed_labels_preserved'] and not eq['model_audit']['query_inputs_used_for_fit']
    audit=read(HERE/'retention_audit.json');assert audit['pre_cleanup_verification']['full_prediction_reconstruction_passed']
    assert audit['candidate_artifacts_removed']==3 and not retained['candidate_model_or_score_arrays_retained']
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args])
    tag=result['tag'];assert git('cat-file','-t','refs/tags/'+tag).strip()==b'tag'
    assert git('rev-parse',tag+'^{commit}').decode().strip()==audit['immutable_result_commit']
    assert git('show',tag+':'+result_path)==(ROOT/result_path).read_bytes()
    assert git('show',tag+':'+result['submission_file'])==(ROOT/result['submission_file']).read_bytes()
    report=dict(passed=True,retained_files_checked=len(retained['files']),protected_sources_checked=len(protocol['sources']),
        scored_csv_rows=312,changed_labelsets_vs_parent=len(changes),score_tag_and_result_unchanged=True,
        candidate_arrays_present=False,full_prediction_reconstruction_performed=False,
        limitation='Full equation/model reconstruction passed before pruning. Current verification checks compact records, sources, exact CSV and tag; a refit or scored checkout is required for full reconstruction.')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':verify()
