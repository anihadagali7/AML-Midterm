"""Fold-local, Part-1-only representations and fixed Ani classifier additions."""
import ast
from collections import Counter
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PART1 = Path(__file__).resolve().parents[2]
RESEARCH = PART1 / "research/independent"
sys.path.insert(0, str(RESEARCH))
import cross_repo_models as native

LABELS = ["Safety", "Funding", "Efficacy", "Recruitment", "Other"]
EMB = [f"emb_{i}" for i in range(150)]
BLOCKS = {
    "inclusion_text": "new: inclusion section word/bigram TF-IDF",
    "exclusion_text": "new: exclusion section word/bigram TF-IDF",
    "eligibility_structure": "new: section, bullet, restriction and formatting counts",
    "icd_full_category": "new: full ICD codes plus three-character categories",
    "criteria_text": "retry: whole eligibility text TF-IDF",
    "diseases": "retry: parsed disease-name indicators",
    "icd_category": "retry/control: parsed three-character ICD categories",
    "drug_targets": "retry: protein-target indicators and missingness",
    "matched_drugs": "retry: matched drug-name indicators and missingness",
}
SECTION = re.compile(r"(?im)^\s*(inclusion|exclusion)(?:\s+criteria)?\s*:\s*")
OTHER_HEADING = re.compile(
    r"(?im)^\s*(?:disease characteristics|patient characteristics|prior concurrent therapy)\s*:\s*"
)


def text(value):
    return value if isinstance(value, str) else ""


def sections(value):
    value = text(value)
    markers = [(m.start(), m.end(), m.group(1).lower()) for m in SECTION.finditer(value)]
    markers += [(m.start(), m.end(), "unassigned") for m in OTHER_HEADING.finditer(value)]
    markers.sort()
    result = {"inclusion": [], "exclusion": []}
    for i, (_, end, name) in enumerate(markers):
        if name in result:
            stop = markers[i + 1][0] if i + 1 < len(markers) else len(value)
            result[name].append(value[end:stop].strip())
    return {key: "\n".join(items) for key, items in result.items()}


def unpack(value):
    if isinstance(value, list):
        return [item for part in value for item in unpack(part)]
    if not isinstance(value, str) or not value.strip():
        return []
    value = value.strip()
    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError("Expected a list in supplied categorical field")
        return unpack(parsed)
    return [value]


def categories(value, block):
    if block.startswith("icd_"):
        codes = {re.sub(r"\s", "", code).upper() for code in unpack(value)}
        assert all(re.fullmatch(r"[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]+)?", c) for c in codes)
        result = {"category:" + c[:3] for c in codes}
        if block == "icd_full_category":
            result |= {"code:" + c for c in codes}
        return result
    raw = unpack(value) if block == "diseases" else text(value).split(";")
    return {re.sub(r"\s+", " ", v.strip()).lower() for v in raw if v.strip()}


def structural(frame):
    result = []
    for value in frame.criteria:
        whole = text(value)
        split = sections(value)
        inc, exc = split["inclusion"], split["exclusion"]
        counts = []
        for view in [whole, inc, exc]:
            counts += [len(view), len(view.split()),
                       len(re.findall(r"(?m)^\s*[-•]\s+", view)),
                       len(re.findall(r"\b(?:no|not|must|excluding|excluded|prohibited|required)\b", view, re.I))]
        counts += [len(re.findall(r"[<>≤≥]", whole)),
                   len(re.findall(r"\bnot specified\b", whole, re.I)),
                   len(re.findall(r"\b(?:and|or)\b", whole, re.I))]
        result.append([*np.log1p(counts), float(bool(inc)), float(bool(exc)),
                       float(not bool(inc or exc))])
    return np.asarray(result)


class Block:
    def __init__(self, name):
        assert name in BLOCKS
        self.name = name

    def documents(self, frame):
        if self.name == "criteria_text":
            return frame.criteria.map(text).tolist()
        section = self.name.split("_")[0]
        return [sections(v)[section] for v in frame.criteria]

    def fit(self, frame):
        if self.name.endswith("_text"):
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=.95,
                                            max_features=2000, sublinear_tf=True)
            self.vectorizer.fit(self.documents(frame))
            self.feature_names = self.vectorizer.get_feature_names_out().tolist() + ["__empty_view__"]
        elif self.name == "eligibility_structure":
            self.scaler = StandardScaler().fit(structural(frame))
            self.feature_names = [f"{view}_{kind}" for view in ["whole", "inclusion", "exclusion"]
                                  for kind in ["log_characters", "log_words", "log_bullets", "log_restrictions"]]
            self.feature_names += ["log_comparators", "log_not_specified", "log_and_or",
                                   "inclusion_present", "exclusion_present", "unsectioned"]
        else:
            self.column = ("icdcodes" if self.name.startswith("icd_") else
                           "diseases_cleaned" if self.name == "diseases" else self.name)
            documents = [categories(v, self.name) for v in frame[self.column]]
            counts = Counter(item for values in documents for item in values)
            self.vocabulary = sorted(item for item, count in counts.items() if count >= 3)
            self.feature_names = self.vocabulary + ["__annotation_missing__"]
        self.fitting_ids = frame.ID.tolist()
        return self

    def transform(self, frame):
        if self.name.endswith("_text"):
            docs = self.documents(frame)
            return np.column_stack([self.vectorizer.transform(docs).toarray(),
                                    [float(not d.strip()) for d in docs]])
        if self.name == "eligibility_structure":
            return self.scaler.transform(structural(frame))
        documents = [categories(v, self.name) for v in frame[self.column]]
        result = np.zeros((len(frame), len(self.feature_names)))
        lookup = {value: j for j, value in enumerate(self.vocabulary)}
        for i, values in enumerate(documents):
            for value in values:
                if value in lookup:
                    result[i, lookup[value]] = 1
            result[i, -1] = float(not values)
        return result


def centered(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return 7 * p / (3 + 4 * p)


def train_native(frame):
    """Reproduce native embedding-only pseudo-labeling, preserving row order."""
    known = frame[LABELS].ge(0).all(axis=1).to_numpy()
    assert frame.loc[~known, LABELS].eq(-100).all().all()
    y = frame.loc[known, LABELS].to_numpy(int)
    scaler = StandardScaler().fit(frame[EMB].to_numpy())
    z = scaler.transform(frame[EMB].to_numpy())
    pca = PCA(30, random_state=0).fit(z)
    gmm = GaussianMixture(30, covariance_type="diag", n_init=3, random_state=0).fit(pca.transform(z))
    assert gmm.converged_
    responsibility = gmm.predict_proba(pca.transform(z))
    gold, unlabeled = np.flatnonzero(known), np.flatnonzero(~known)
    rows, targets = gold.copy(), y.copy()
    history = []
    for _ in range(2):
        base = native.fit_ani_base(z[rows], targets)
        t = targets / targets.sum(axis=1, keepdims=True)
        rates = (responsibility[rows].T @ t + t.mean(axis=0)) / (responsibility[rows].sum(axis=0)[:, None] + 1)
        prob = .8 * native.ani_base_predict(base, z[unlabeled]) + .2 * (responsibility[unlabeled] @ rates)
        keep = prob.max(axis=1) >= .8
        rows = np.r_[gold, unlabeled[keep]]
        targets = np.vstack([y, native.decide(prob, .3)[keep]])
        history.append(int(keep.sum()))
    base = native.fit_ani_base(z[rows], targets)
    return dict(scaler=scaler, z=z, rows=rows, targets=targets, base=base,
                gold=gold, y=y, history=history)


def native_prediction(packet, query):
    zq = packet["scaler"].transform(query[EMB].to_numpy())
    return centered(native.ani_base_predict(packet["base"], zq))


def fit_addition(packet, fit, query, name):
    block = Block(name).fit(fit)
    a, q = block.transform(fit), block.transform(query)
    rms = np.sqrt(np.mean(np.sum(a * a, axis=1)))
    embedding_rms = np.sqrt(np.mean(np.sum(packet["z"] ** 2, axis=1)))
    multiplier = .3 * embedding_rms / max(rms, 1e-12)
    a, q = a * multiplier, q * multiplier
    zq = packet["scaler"].transform(query[EMB].to_numpy())
    x = np.column_stack([packet["z"], a])
    query_x = np.column_stack([zq, q])
    model = native.fit_ani_base(x[packet["rows"]], packet["targets"])[0]
    lr = np.zeros((len(query), len(LABELS)))
    lr[:, model.classes_] = model.predict_proba(query_x)
    old_model = packet["base"][0]
    old_lr = np.zeros_like(lr)
    old_lr[:, old_model.classes_] = old_model.predict_proba(zq)
    raw = native.ani_base_predict(packet["base"], zq)
    result = centered(np.clip(raw + .7 * (lr - old_lr), 0, 1))
    # Independent diagnostic: can metadata predict any labels using gold alone?
    meta = np.empty_like(lr)
    for j in range(len(LABELS)):
        y = packet["y"][:, j]
        if len(np.unique(y)) == 1:
            meta[:, j] = y[0]
        else:
            head = LogisticRegression(C=.3, class_weight="balanced", max_iter=3000)
            head.fit(a[packet["gold"]], y)
            meta[:, j] = head.predict_proba(q)[:, list(head.classes_).index(1)]
    audit = dict(block=name, dimensions=a.shape[1], multiplier=float(multiplier),
                 feature_names_sha256=hashlib.sha256(json.dumps(block.feature_names).encode()).hexdigest(),
                 fit_rows=len(fit), query_rows=len(query), iterations=int(model.n_iter_.max()),
                 training_block_rms=float(rms), added_to="final multinomial LR only")
    return result, meta, audit
