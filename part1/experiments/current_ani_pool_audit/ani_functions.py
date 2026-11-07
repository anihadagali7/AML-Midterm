def decide(P, thr=0.3):
    Y = (P >= thr).astype(int)
    Y[np.arange(len(P)), P.argmax(1)] = 1
    return Y

def expand(X, Y, w):
    rows, ys, ws = [], [], []
    for i, y in enumerate(Y):
        pos = np.flatnonzero(y)
        rows += [i] * len(pos); ys += list(pos); ws += [w[i] / len(pos)] * len(pos)
    return X[rows], np.array(ys), np.array(ws)

def multinomial_lr(Xl, Yl, Xq, C=0.3, balanced=True):
    X, y, sw = expand(Xl, Yl, np.ones(len(Yl)))
    if balanced:
        cnt = np.bincount(y, weights=sw, minlength=5) + 1e-9
        sw = sw * cnt.mean() / cnt[y]
    m = LogisticRegression(C=C, max_iter=3000).fit(X, y, sample_weight=sw)
    P = np.zeros((len(Xq), 5))
    P[:, m.classes_] = m.predict_proba(Xq)
    return P

def knn(Xl, Yl, Xq, k=10):
    S = normalize(Xq) @ normalize(Xl).T
    idx = np.argsort(-S, 1)[:, :k]
    w = np.take_along_axis(S, idx, 1).clip(0) ** 2
    T = Yl / Yl.sum(1, keepdims=True)
    return (w[:, :, None] * T[idx]).sum(1) / (w.sum(1, keepdims=True) + 1e-9)

def gmm_cluster_label(Xl, Yl, Xq, a=1.0):
    Rl, Rq = (gmm.predict_proba(pca30.transform(M[:, :len(EMB)])) for M in (Xl, Xq))
    T = Yl / Yl.sum(1, keepdims=True)
    cluster_dist = (Rl.T @ T + a * T.mean(0)) / (Rl.sum(0)[:, None] + a)
    return Rq @ cluster_dist

def shrinkage_lda(Xl, Yl, Xq):
    X, y, _ = expand(Xl, Yl, np.ones(len(Yl)))
    k = len(np.unique(y))
    m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.ones(k) / k).fit(X, y)
    P = np.zeros((len(Xq), 5))
    P[:, m.classes_] = m.predict_proba(Xq)
    return P

def base_model(Xl, Yl, Xq, w_knn=0.3, w_lda=0.2, C=0.1):
    P = (1 - w_knn - w_lda) * multinomial_lr(Xl, Yl, Xq, C=C) + w_knn * knn(Xl, Yl, Xq)
    return P + w_lda * shrinkage_lda(Xl, Yl, Xq) if w_lda else P

def final_model(Xl, Yl, Xq, Xu, w_gmm=0.2, conf=0.8, thr=0.3, rounds=2, return_aug=False, **base_kw):
    XL, YL = Xl, Yl
    for _ in range(rounds):
        Pu = (1 - w_gmm) * base_model(XL, YL, Xu, **base_kw) + w_gmm * gmm_cluster_label(XL, YL, Xu)
        keep = Pu.max(1) >= conf
        XL, YL = np.vstack([Xl, Xu[keep]]), np.vstack([Yl, decide(Pu, thr)[keep]])
    P = base_model(XL, YL, Xq, **base_kw)
    return (P, keep.sum()) if return_aug else P
