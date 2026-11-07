"""Counts exact-duplicate reason embeddings. Run from part1/."""
import pandas as pd

LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
EMB = [f"emb_{i}" for i in range(150)]
train, val, test = (pd.read_csv(f"part1_{s}.csv") for s in ["train", "val", "test"])
key = lambda d: d[EMB].round(6).astype(str).agg("|".join, axis=1)
kt, kv, ke = key(train), key(val), key(test)
counts = kt.value_counts()
print("Train rows in a duplicate group:", int(kt.isin(counts[counts > 1].index).sum()), "of", len(train))
print("Distinct train embeddings:", kt.nunique(), "| largest groups:", counts.head(8).tolist())
print("Val rows matching a train row:", int(kv.isin(kt).sum()), "of", len(val))
print("Test rows matching a train row:", int(ke.isin(kt).sum()), "of", len(test))
labeled = train["Safety"] != -100
known = pd.concat([train.loc[labeled, LABELS].assign(key=kt[labeled].values), val[LABELS].assign(key=kv.values)])
groups = known.groupby("key").filter(lambda g: len(g) > 1)
conflicts = int((groups.groupby("key")[LABELS].nunique().max(axis=1) > 1).sum())
print("Known-label rows in a duplicate group:", len(groups), "of", len(known), "| groups:", groups["key"].nunique(), "| conflicting:", conflicts)
