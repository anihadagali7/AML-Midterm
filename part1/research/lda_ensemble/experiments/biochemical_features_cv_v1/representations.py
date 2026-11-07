"""Deterministic chemistry and supplied-catalog features; no external lookup."""
from pathlib import Path
from functools import lru_cache
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


def values(value):
    return sorted({s.strip() for s in value.split(";") if s.strip()}) if isinstance(value, str) else []


@lru_cache(maxsize=4096)
def fingerprint(smiles):
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return GENERATOR.GetFingerprintAsNumPy(molecule).astype(float)


class BioBlock:
    def __init__(self, view, catalog):
        self.view = view
        self.catalog = catalog

    def tokens(self, value):
        result = values(value)
        if self.view.startswith("pathway"):
            if self.view == "pathway_specific":
                result = [s for s in result if s in self.catalog and self.catalog[s] >= 3]
        else:
            result = [s.lower() for s in result]
        return result

    def fit(self, frame):
        self.column = "drug_smiles" if self.view == "morgan" else "drug_actions" if self.view == "actions" else "drug_pathways"
        if self.view == "morgan":
            self.names = [f"morgan_mean_{j}" for j in range(2048)] + [f"morgan_any_{j}" for j in range(2048)]
            self.names += ["log_valid_molecules", "invalid_structure_present"]
        else:
            counts = Counter(s for value in frame[self.column] for s in self.tokens(value))
            self.vocabulary = sorted(s for s, n in counts.items() if n >= 3)
            self.names = self.vocabulary
        self.fitting_ids = frame.ID.tolist()
        return self

    def transform(self, frame):
        x = np.zeros((len(frame), len(self.names)))
        missing = np.zeros(len(frame))
        invalid = 0
        if self.view == "morgan":
            for i, value in enumerate(frame[self.column]):
                encoded = [fingerprint(s) for s in values(value)]
                valid = [v for v in encoded if v is not None]
                invalid_here = len(valid) != len(encoded)
                invalid += int(invalid_here)
                missing[i] = not bool(valid)
                if valid:
                    stack = np.array(valid)
                    x[i, :2048] = stack.mean(axis=0)
                    x[i, 2048:4096] = stack.max(axis=0)
                x[i, -2:] = [np.log1p(len(valid)), float(invalid_here)]
        else:
            lookup = {s: j for j, s in enumerate(self.vocabulary)}
            for i, value in enumerate(frame[self.column]):
                tokens = self.tokens(value)
                missing[i] = not bool(tokens)
                for token in tokens:
                    if token in lookup:
                        x[i, lookup[token]] = 1
        return csr_matrix(x), missing, invalid


def catalog_from(path):
    frame = pd.read_csv(path)
    assert not frame.reactome_id.duplicated().any()
    return dict(zip(frame.reactome_id, frame.pathway_level))
