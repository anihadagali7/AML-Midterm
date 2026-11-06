"""Training-size matched nested calibration; 15-fold inner fits retain ~93% of gold."""
import phase18,experiment as e
cs=[dict(model='calibrated_ssl',calibration='thresholds',macro_weight=w,inner_splits=15) for w in [0,.25,.5]]
cs+=[dict(model='calibrated_ssl',calibration=kind,meta_C=C,inner_splits=15) for kind,C in [('independent',1),('joint',10),('patterns',1)]]
e.save_json('experiments/phase19-configs.json',cs)
for i,c in enumerate(cs):print('RUN',i+1,len(cs),c,flush=True);e.evaluate(c,tag='phase19')
