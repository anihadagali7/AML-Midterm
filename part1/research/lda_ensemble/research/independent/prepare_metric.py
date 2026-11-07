import experiment as e
configs=[]
for model in ['logistic','svc']:
 for k in [5,15,40]:
  for residual in [0,.5,2]:
   configs.append(dict(model='metric_graph',metric_model=model,C=.3 if model=='logistic' else 10,gamma=.001,scale='global',k=k,alpha=.8,residual=residual))
for n in [2,3,5,10,20,40]:configs.append(dict(model='pls',components=n))
e.save_json('experiments/metric-input.json',configs)
