# Final comparison: each labeling method + classifier, scored on validation and repeated 5-fold CV
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

SEEDS = range(5)
N_REPEATS = 5

is_labeled = (df_train1["Safety"] != -100).values
labeled_rows = np.where(is_labeled)[0]
Y_labeled = df_train1.loc[is_labeled, LABELS].values
Y_valid = df_valid1[LABELS].values
Y_known = np.vstack([Y_labeled, Y_valid])     # all 235 rows with true labels, used for CV
X_val_pca = pca.transform(X_val)

# Clusterings never see labels, so each is fit once on the train embeddings
gmm_probs = {}
for k in [2, 4]:
    for seed in SEEDS:
        fitted = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed).fit(X_pca)
        gmm_probs[k, seed] = fitted.predict_proba(X_pca), fitted.predict_proba(X_val_pca)

# DBSCAN: eps is the median distance to the 3rd neighbor. New rows join the nearest core
# point's cluster if it is within eps, and otherwise count as noise.
neighbor_dist = NearestNeighbors(n_neighbors=3, metric="cosine").fit(X_scaled).kneighbors(X_scaled)[0][:, -1]
dbscan_eps = float(np.quantile(neighbor_dist, 0.5))
dbscan = DBSCAN(eps=dbscan_eps, min_samples=3, metric="cosine").fit(X_scaled)
core_rows = dbscan.core_sample_indices_
core_dist, core_index = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_scaled[core_rows]).kneighbors(X_val)
dbscan_val = dbscan.labels_[core_rows][core_index[:, 0]]
dbscan_val[core_dist[:, 0] > dbscan_eps] = -1
n_dbscan = dbscan.labels_.max() + 1
print(f"DBSCAN: eps={dbscan_eps:.3f}, {n_dbscan} clusters, {(dbscan.labels_ == -1).mean():.0%} of rows are noise")

# Eligibility criteria as 20 standardized TF-IDF components
tfidf = TfidfVectorizer(min_df=5, max_features=20000, stop_words="english")
svd = TruncatedSVD(n_components=20, random_state=0)
criteria_scaler = StandardScaler()
X_criteria = criteria_scaler.fit_transform(svd.fit_transform(tfidf.fit_transform(df_train1["criteria"].fillna(""))))
X_criteria_val = criteria_scaler.transform(svd.transform(tfidf.transform(df_valid1["criteria"].fillna(""))))


def rate_based_probs(cluster_probs, known_rows, Y_rows):
    """P(label | row) from smoothed cluster label rates, computed from known_rows only."""
    R = cluster_probs[known_rows]
    overall = Y_rows.mean(axis=0)
    rates = (R.T @ Y_rows + 5 * overall) / (R.sum(axis=0, keepdims=True).T + 5)
    return cluster_probs @ rates, overall


def apply_cutoffs(p, overall, no_cluster=None):
    """1 at 0.7 or above, 0 at half the overall rate or below, NaN (no label) in between."""
    labels = np.full(p.shape, np.nan)
    labels[p >= 0.7] = 1
    labels[p <= overall / 2] = 0
    if no_cluster is not None:
        labels[no_cluster] = np.nan
    return labels


def generate_labels(method, seed, known_rows, Y_rows):
    """Labels for the unlabeled train rows: NaN means none, and soft methods return probabilities."""
    if method == "none":
        return None
    if method.startswith("gmm"):
        p, overall = rate_based_probs(gmm_probs[int(method[3:]), seed][0], known_rows, Y_rows)
        return apply_cutoffs(p, overall)[~is_labeled]
    if method.startswith("soft"):
        return rate_based_probs(gmm_probs[int(method[4:]), seed][0], known_rows, Y_rows)[0][~is_labeled]
    if method == "dbscan":
        one_hot = np.eye(n_dbscan + 1)[dbscan.labels_][:, :-1]   # noise (-1) gets an all-zero row
        p, overall = rate_based_probs(one_hot, known_rows, Y_rows)
        return apply_cutoffs(p, overall, dbscan.labels_ == -1)[~is_labeled]
    if method == "knn":
        dist, index = NearestNeighbors(n_neighbors=10).fit(X_scaled[known_rows]).kneighbors(X_scaled)
        weights = 1 / (dist + 1e-9)
        weights = weights / weights.sum(axis=1, keepdims=True)
        p = (weights[:, :, None] * Y_rows[index]).sum(axis=1)
        return apply_cutoffs(p, Y_rows.mean(axis=0))[~is_labeled]
    raise ValueError(method)


def logistic_probs(X_true, Y_true, X_unlabeled, generated, soft_weight, X_query, C):
    """One balanced logistic regression per label, using that label's generated rows if any."""
    out = np.zeros((len(X_query), len(LABELS)))
    for j in range(len(LABELS)):
        weights = None
        if generated is None:
            X_fit, y_fit = X_true, Y_true[:, j]
        elif soft_weight is None:
            has_label = ~np.isnan(generated[:, j])
            X_fit = np.vstack([X_true, X_unlabeled[has_label]])
            y_fit = np.r_[Y_true[:, j], generated[has_label, j]]
        else:
            # Each unlabeled row appears twice: positive with weight p, negative with weight 1 - p
            p = generated[:, j]
            X_fit = np.vstack([X_true, X_unlabeled, X_unlabeled])
            y_fit = np.r_[Y_true[:, j], np.ones(len(p)), np.zeros(len(p))]
            weights = np.r_[np.ones(len(Y_true)), soft_weight * p, soft_weight * (1 - p)]
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        out[:, j] = clf.fit(X_fit, y_fit.astype(int), sample_weight=weights).predict_proba(X_query)[:, 1]
    return out


def knn_probs(X_true, Y_true, X_query, k):
    """Distance-weighted cosine kNN probabilities for each label, from true-label rows."""
    dist, index = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_true).kneighbors(X_query)
    weights = 1 / (dist + 1e-9)
    return (weights[:, :, None] * Y_true[index]).sum(axis=1) / weights.sum(axis=1, keepdims=True)


def predict_labels(model, X_true, Y_true, X_unlabeled, generated, X_query):
    probs_lr = logistic_probs(X_true, Y_true, X_unlabeled, generated, model.get("soft_weight"), X_query, model["C"])
    if "ensemble_k" in model:
        return ((probs_lr + knn_probs(X_true, Y_true, X_query, model["ensemble_k"])) / 2 >= 0.5).astype(int)
    return (probs_lr >= 0.5).astype(int)


def score_model(model):
    add_criteria = model.get("criteria", False)
    X_tr = np.hstack([X_scaled, X_criteria]) if add_criteria else X_scaled
    X_va = np.hstack([X_val, X_criteria_val]) if add_criteria else X_val
    X_true_all = np.vstack([X_tr[is_labeled], X_va])
    X_unlabeled = X_tr[~is_labeled]

    # Validation: train on labeled train rows. GMM methods are averaged across seeds.
    uses_gmm = model["labels"].startswith(("gmm", "soft"))
    val_scores = pd.DataFrame([
        evaluate(Y_valid, predict_labels(model, X_tr[is_labeled], Y_labeled, X_unlabeled,
                                         generate_labels(model["labels"], seed, labeled_rows, Y_labeled), X_va),
                 verbose=False)
        for seed in (SEEDS if uses_gmm else [0])
    ]).mean()

    # CV across all 235 known rows. Generated labels only use training-fold train rows.
    cv_scores = []
    for repeat in range(N_REPEATS):
        out_of_fold = np.zeros_like(Y_known)
        for fit_rows, test_rows in KFold(5, shuffle=True, random_state=repeat).split(X_true_all):
            from_train = fit_rows[fit_rows < len(labeled_rows)]
            generated = generate_labels(model["labels"], repeat % 5, labeled_rows[from_train], Y_known[from_train])
            out_of_fold[test_rows] = predict_labels(model, X_true_all[fit_rows], Y_known[fit_rows], X_unlabeled,
                                                    generated, X_true_all[test_rows])
        cv_scores.append(evaluate(Y_known, out_of_fold, verbose=False))
    return val_scores, pd.DataFrame(cv_scores)


final_models = [
    {"name": "Baseline: true labels, C=1", "labels": "none", "C": 1},
    {"name": "True labels, C=0.01", "labels": "none", "C": 0.01},
    {"name": "GMM k=2 labels, C=1", "labels": "gmm2", "C": 1},
    {"name": "GMM k=2 labels, C=10", "labels": "gmm2", "C": 10},
    {"name": "DBSCAN labels, C=1", "labels": "dbscan", "C": 1},
    {"name": "Soft GMM k=4 labels, weight 0.02, C=0.1", "labels": "soft4", "soft_weight": 0.02, "C": 0.1},
    {"name": "kNN labels, C=0.1", "labels": "knn", "C": 0.1},
    {"name": "kNN labels + criteria, C=0.01", "labels": "knn", "criteria": True, "C": 0.01},
    {"name": "Ensemble: LR C=0.01 + kNN k=10, true labels", "labels": "none", "C": 0.01, "ensemble_k": 10},
]

baseline_cv = None
comparison = []
for model in final_models:
    val_scores, cv_scores = score_model(model)
    if baseline_cv is None:
        baseline_cv = cv_scores
    comparison.append({
        "model": model["name"],
        "val_sample_f1": val_scores["sample_f1"],
        "val_macro_f1": val_scores["macro_f1"],
        **{f"val_{label}": val_scores[label] for label in LABELS},
        "cv_sample_change": (cv_scores["sample_f1"] - baseline_cv["sample_f1"]).mean(),
        "cv_sample_change_sd": (cv_scores["sample_f1"] - baseline_cv["sample_f1"]).std(),
        "cv_macro_change": (cv_scores["macro_f1"] - baseline_cv["macro_f1"]).mean(),
        "cv_macro_change_sd": (cv_scores["macro_f1"] - baseline_cv["macro_f1"]).std(),
    })

final_comparison = pd.DataFrame(comparison).set_index("model").round(3)
final_comparison
