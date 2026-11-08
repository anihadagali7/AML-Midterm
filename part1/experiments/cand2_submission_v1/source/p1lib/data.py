"""Data loading, label-powerset encoding and metrics for Part 1.

Global row order used everywhere in this package:
    [ part1_train (1171 rows, file order) | part1_val (78) | part1_test (312) ]  -> 1561 rows.
`Data.y` holds the label-combo id (0..7) for labeled rows and -1 for masked/test rows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
EMB = [f"emb_{i}" for i in range(150)]
BITS = np.array([1, 2, 4, 8, 16])

DEFAULT_DATA_DIR = os.environ.get(
    "P1_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "part1", "data"),
)


@dataclass
class Data:
    df: pd.DataFrame                 # all 1561 rows, global order, with a 'split' column
    emb: np.ndarray                  # (1561, 150) raw embeddings
    Y: np.ndarray                    # (1561, 5) labels, -1 where unknown
    y: np.ndarray                    # (1561,) combo id, -1 where unknown
    combos: np.ndarray               # (K,) bitmask of each allowed combo
    CM: np.ndarray                   # (K, 5) binary label matrix of each combo
    idx: dict = field(default_factory=dict)   # named index groups
    year: np.ndarray = None
    catalog: pd.DataFrame = None

    @property
    def K(self) -> int:
        return len(self.combos)

    @property
    def n(self) -> int:
        return len(self.df)

    def combo_names(self):
        return ["+".join(LABELS[j][0] for j in range(5) if self.CM[k, j]) for k in range(self.K)]


def load(data_dir: str | None = None) -> Data:
    d = os.path.abspath(data_dir or DEFAULT_DATA_DIR)
    tr = pd.read_csv(os.path.join(d, "part1_train.csv"))
    va = pd.read_csv(os.path.join(d, "part1_val.csv"))
    te = pd.read_csv(os.path.join(d, "part1_test.csv"))
    cat = pd.read_csv(os.path.join(d, "pathway_catalog.csv"))
    te = pd.concat([te, pd.DataFrame(-1, index=te.index, columns=LABELS)], axis=1)
    tr = tr.assign(split="train")
    va = va.assign(split="val")
    te = te.assign(split="test")
    df = pd.concat([tr, va, te], ignore_index=True).copy()
    Y = df[LABELS].to_numpy().astype(int)
    Y[Y == -100] = -1
    known = (Y >= 0).all(1)
    mask = (Y.clip(0) * BITS).sum(1)
    combos = np.unique(mask[known])
    y = np.full(len(df), -1)
    y[known] = np.searchsorted(combos, mask[known])
    CM = np.array([[(c >> j) & 1 for j in range(5)] for c in combos])
    split = df["split"].to_numpy()
    idx = {
        "train": np.where(split == "train")[0],
        "lab": np.where((split == "train") & known)[0],
        "unl": np.where((split == "train") & ~known)[0],
        "val": np.where(split == "val")[0],
        "test": np.where(split == "test")[0],
    }
    idx["labval"] = np.r_[idx["lab"], idx["val"]]
    D = Data(df=df, emb=df[EMB].to_numpy(float), Y=Y, y=y, combos=combos, CM=CM, idx=idx,
             year=df["start_year"].to_numpy(int), catalog=cat)
    assert len(idx["lab"]) == 157 and len(idx["unl"]) == 1014 and len(idx["val"]) == 78 and len(idx["test"]) == 312
    assert (Y[idx["lab"]].sum(1) >= 1).all(), "labeled rows are never all-zero"
    return D


# ----------------------------------------------------------------------------- metrics
def row_f1(Ytrue: np.ndarray, Ypred: np.ndarray) -> np.ndarray:
    """Per-row F1 between binary label sets (both non-empty here)."""
    inter = (Ytrue * Ypred).sum(1)
    denom = Ytrue.sum(1) + Ypred.sum(1)
    out = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), 1.0)
    return out


def label_f1(Ytrue: np.ndarray, Ypred: np.ndarray) -> np.ndarray:
    """Per-label (column) binary F1; F1 = 0 when a label has no positives and none predicted."""
    tp = (Ytrue * Ypred).sum(0)
    fp = ((1 - Ytrue) * Ypred).sum(0)
    fn = (Ytrue * (1 - Ypred)).sum(0)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def metrics(Ytrue: np.ndarray, Ypred: np.ndarray) -> dict:
    rf = row_f1(Ytrue, Ypred)
    lf = label_f1(Ytrue, Ypred)
    return {
        "sample_f1": float(rf.mean()),
        "sample_f1_se": float(rf.std(ddof=1) / np.sqrt(len(rf))),
        "macro_f1": float(lf.mean()),
        **{f"f1_{LABELS[j]}": float(lf[j]) for j in range(5)},
        "n": int(len(rf)),
    }


def bootstrap_se(Ytrue: np.ndarray, Ypreds: list[np.ndarray], B: int = 300, seed: int = 0) -> dict:
    """Bootstrap SE (over rows) of macro and per-label F1, averaged over a list of prediction
    matrices for the same rows (e.g. CV repeats) using shared resamples."""
    rng = np.random.default_rng(seed)
    n = len(Ytrue)
    macro, per = [], []
    for _ in range(B):
        b = rng.integers(0, n, n)
        lfs = np.mean([label_f1(Ytrue[b], P[b]) for P in Ypreds], axis=0)
        per.append(lfs)
        macro.append(lfs.mean())
    per = np.array(per)
    return {"macro_f1_se": float(np.std(macro, ddof=1)),
            **{f"f1_{LABELS[j]}_se": float(per[:, j].std(ddof=1)) for j in range(5)}}


def combos_to_labels(D: Data, kidx: np.ndarray) -> np.ndarray:
    return D.CM[kidx]


def write_submission(D: Data, Ypred: np.ndarray, path: str, ids=None) -> pd.DataFrame:
    """Kaggle format: ID,Predicted with space-separated reason names (always >=1)."""
    assert (Ypred.sum(1) >= 1).all(), "every test row needs at least one predicted reason"
    ids = D.df["ID"].to_numpy()[D.idx["test"]] if ids is None else ids
    pred = [" ".join(LABELS[j] for j in range(5) if row[j]) for row in Ypred]
    sub = pd.DataFrame({"ID": ids, "Predicted": pred})
    sub.to_csv(path, index=False, lineterminator="\n")
    return sub
