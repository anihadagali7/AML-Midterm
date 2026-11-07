"""Reproduce the frozen current-best candidate; never selects using validation labels.
Run from the project or portable midterm directory. All training data stay unchanged.
"""
from pathlib import Path
import argparse,hashlib,json,datetime
import numpy as np,pandas as pd
import phase21,experiment as e
from threadpoolctl import threadpool_limits

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='predictions/current_best');args=parser.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 frozen=json.loads(Path('experiments/current-best.json').read_text());candidate=frozen['candidate'];config=candidate['config'];rule=candidate['rule']
 # Freeze/test predictions first, then read validation labels solely for reporting.
 test=pd.read_csv('data/part1_test.csv');V=pd.read_csv('data/part1_val.csv',usecols=lambda col:col not in e.LABELS)
 with threadpool_limits(limits=1):
  pt,pu,ix=e.predict(np.arange(len(e.Y)),test,config,return_unlabeled=True)
  pv=e.predict(np.arange(len(e.Y)),V,config)
 assert np.array_equal(ix,e.UNLAB)
 ht=e.decisions(pt,rule);hu=e.decisions(pu,rule);hv=e.decisions(pv,rule)
 sub=pd.DataFrame({'ID':test.ID,'Predicted':[' '.join(label for label,v in zip(e.LABELS,row) if v) for row in ht]});sub.to_csv(out/'part1_submission.csv',index=False)
 aug=e.DATA.copy();aug.loc[e.UNLAB,e.LABELS]=hu;aug['label_source']='original_gold';aug.loc[e.UNLAB,'label_source']='predicted_missing';aug.to_csv(out/'part1_train_augmented.csv',index=False)
 for name,ids,prob in [('test',test.ID,pt),('unlabeled',e.DATA.iloc[e.UNLAB].ID,pu),('validation',V.ID,pv)]:
  frame=pd.DataFrame(prob,columns=['score_'+l for l in e.LABELS]);frame.insert(0,'ID',ids.to_numpy());frame.to_csv(out/f'{name}_scores.csv',index=False)
 yv=pd.read_csv('data/part1_val.csv')[e.LABELS].to_numpy();val=e.metrics(yv,pv,rule)
 assert abs(val['sample_f1']-frozen['best_validation_f1'])<1e-12,(val,frozen['best_validation_f1'])
 audit=pd.DataFrame({'ID':V.ID,'true':[' '.join(l for l,v in zip(e.LABELS,row) if v) for row in yv],'predicted':[' '.join(l for l,v in zip(e.LABELS,row) if v) for row in hv]})
 audit['sample_f1']=2*np.sum(yv*hv,axis=1)/(yv.sum(1)+hv.sum(1));audit.to_csv(out/'validation_audit.csv',index=False)
 assert sub.ID.tolist()==pd.read_csv('data/part1_sample_submission.csv').ID.tolist()
 assert len(sub)==312 and sub.ID.is_unique and sub.Predicted.map(lambda s:bool(s) and set(s.split())<=set(e.LABELS)).all()
 assert np.array_equal(aug.loc[e.LAB,e.LABELS].to_numpy(),e.Y)
 assert aug[e.LABELS].isin([0,1]).all().all() and len(aug)==len(e.DATA)
 raw=json.loads(Path('experiments/data-audit.json').read_text())['files']
 for name,record in raw.items():assert hashlib.sha256((Path('data')/name).read_bytes()).hexdigest()==record['sha256']
 source_files=['experiment.py']+[f'phase{i}.py' for i in [3,4,5,6,7,8,9,10,11,14,16,17,18,20,21]]
 manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate=candidate,validation=val,target=.92735,target_met=val['sample_f1']>.92735,kaggle_score=None,gold_rows=157,predicted_missing_rows=1014,test_rows=312,submission_sha256=hashlib.sha256((out/'part1_submission.csv').read_bytes()).hexdigest(),raw_data_hashes_verified=True,gold_labels_preserved=True,validation_labels_used_for_training=0,final_train_plus_validation_refit_performed=False,source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_files},score_interpretation='Validation is repeatedly used for selection; CV values are development scores. Neither establishes a Kaggle leaderboard result. Final hybrid scores are binary decisions; component margin/vote scores are not calibrated probabilities.')
 e.save_json(out/'manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['validation','target_met','submission_sha256','gold_labels_preserved']},indent=2))
if __name__=='__main__':main()
