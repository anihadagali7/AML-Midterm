"""Optional non-embedding feature blocks (each a toggle in the features config).

Every block is label-free: vocabularies / IDF / SVD are fit only on the rows the fold allows
(`fit_rows`), then applied to all rows.  Output is a dense (n_all, d) array, standardized on
`fit_rows`.

Blocks
  criteria  : TF-IDF of eligibility criteria  -> TruncatedSVD(d)
  diseases  : TF-IDF of diseases_cleaned      -> TruncatedSVD(d)
  drugs     : multi-hot matched_drugs         -> TruncatedSVD(d)
  pathways  : multi-hot drug_pathways (optionally only Reactome level <= max_level) -> SVD(d)
  morgan    : RDKit Morgan fingerprint (radius 2, 1024 bits, max over drugs) -> SVD(d)
  drugmeta  : [has_matched_drug, log1p(#matched drugs)]
  year      : standardized start_year  (sampling artifact; ablation only)
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .cache import memo

_MORGAN = None


def _split_semicolon(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [t.strip().lower() for t in s.split(";") if t.strip()]


def _morgan_matrix(D):
    global _MORGAN
    if _MORGAN is not None:
        return _MORGAN
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    out = np.zeros((D.n, 1024), dtype=np.float32)
    for i, s in enumerate(D.df["drug_smiles"].to_numpy()):
        if not isinstance(s, str):
            continue
        fps = [gen.GetFingerprintAsNumPy(m) for t in s.split(";") if (m := Chem.MolFromSmiles(t)) is not None]
        if fps:
            out[i] = np.max(fps, axis=0)
    _MORGAN = out
    return out


def _svd(M, fit_rows, d, seed=0):
    d = int(min(d, M.shape[1] - 1, len(fit_rows) - 1))
    svd = TruncatedSVD(n_components=max(d, 1), random_state=seed).fit(M[fit_rows])
    return svd.transform(M)


def _standardize(Z, fit_rows):
    mu = Z[fit_rows].mean(0)
    sd = Z[fit_rows].std(0)
    sd[sd < 1e-12] = 1.0
    return (Z - mu) / sd


def build_block(D, name: str, cfg: dict, fit_rows: np.ndarray) -> np.ndarray:
    cfg = dict(cfg or {})
    key = {"block": name, **cfg}
    return memo("block", key, fit_rows, lambda: _build(D, name, cfg, fit_rows))


def _build(D, name, cfg, fit_rows):
    d = int(cfg.get("d", 32))
    df = D.df
    if name == "criteria" or name == "diseases":
        col = "criteria" if name == "criteria" else "diseases_cleaned"
        text = df[col].fillna("").astype(str).to_numpy()
        vec = TfidfVectorizer(max_features=int(cfg.get("max_features", 5000)), min_df=3,
                              sublinear_tf=True, ngram_range=(1, int(cfg.get("ngram", 1))),
                              stop_words="english")
        vec.fit(text[fit_rows])
        M = vec.transform(text)
        Z = _svd(M, fit_rows, d)
    elif name in ("drugs", "pathways"):
        col = "matched_drugs" if name == "drugs" else "drug_pathways"
        toks = [_split_semicolon(s) for s in df[col].to_numpy()]
        if name == "pathways" and cfg.get("max_level") is not None:
            lvl = dict(zip(D.catalog["reactome_id"].str.lower(), D.catalog["pathway_level"]))
            ml = int(cfg["max_level"])
            toks = [[t for t in ts if lvl.get(t, 99) <= ml] for ts in toks]
        docs = [" ".join(t.replace(" ", "_") for t in ts) for ts in toks]
        vec = CountVectorizer(binary=True, min_df=3, token_pattern=r"[^ ]+")
        try:
            vec.fit([docs[i] for i in fit_rows])
            M = vec.transform(docs).astype(float)
        except ValueError:            # empty vocabulary
            M = sparse.csr_matrix((D.n, 1))
        Z = _svd(M, fit_rows, d) if M.shape[1] > 2 else M.toarray()
    elif name == "morgan":
        M = _morgan_matrix(D)
        Z = _svd(sparse.csr_matrix(M), fit_rows, d)
    elif name == "drugmeta":
        toks = [_split_semicolon(s) for s in df["matched_drugs"].to_numpy()]
        Z = np.c_[[len(t) > 0 for t in toks], np.log1p([len(t) for t in toks])].astype(float)
    elif name == "year":
        Z = df["start_year"].to_numpy(float)[:, None]
    else:
        raise ValueError(f"unknown block {name}")
    return _standardize(np.asarray(Z, float), fit_rows) * float(cfg.get("weight", 1.0))
