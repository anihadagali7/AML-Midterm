import experiment as e
configs=[]
teachers=[dict(model='graph',pca=25,k=5,alpha=.8),dict(model='cluster',cluster='hdbscan',min_cluster=5,pca=25)]
for t in teachers:
 for model in ['logistic','svc']:
  for c in ([.1,.3] if model=='logistic' else [3,10]):
   for w in [.05,.2]:
    for conf in [.8,.95]:
     configs.append(dict(model='pseudo',teacher=t,student=model,C=c,gamma=.001,scale='global',pseudo_weight=w,confidence=conf,labelwise=True,consensus=True))
e.save_json('experiments/consensus-input.json',configs)
