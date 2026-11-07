"""New classical models. Fit sees gold + permitted masked rows; query is transform-only."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import softmax
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

PATTERNS = ((np.arange(1, 32)[:, None] >> np.arange(5)) & 1)
UTILITY = 2 * PATTERNS @ PATTERNS.T / (PATTERNS.sum(1)[:, None] + PATTERNS.sum(1)[None, :])


def threshold(p, cutoff=.3):
    out = (p >= cutoff).astype(int)
    out[np.arange(len(p)), p.argmax(1)] = 1
    return out


def soft_classifier(a, y, q, C=.3):
    rows, classes = np.where(y > 0)
    w = 1. / y.sum(1)[rows]
    counts = np.bincount(classes, weights=w, minlength=5) + 1e-9
    w *= counts.mean() / counts[classes]
    m = LogisticRegression(C=C, max_iter=3000).fit(a[rows], classes, sample_weight=w)
    p = np.zeros((len(q), 5)); p[:, m.classes_] = m.predict_proba(q)
    return p


def poisson(a, y, q, c):
    """Regularized Poisson equation, class-balanced sources, inductive local extension.

    This is an adaptation of Calder et al., not a claimed exact reproduction.
    Query nodes never affect the graph or its solve.
    """
    n, k = len(a), c['neighbors']
    nn = NearestNeighbors(n_neighbors=k+1, n_jobs=1).fit(a)
    dist, ix = nn.kneighbors(a)
    dist, ix = dist[:, 1:], ix[:, 1:]
    bandwidth = np.maximum(dist[:, -1], 1e-6)
    weights = np.exp(-dist**2 / (bandwidth[:, None] * bandwidth[ix]))
    w = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(n), k), ix.ravel())), shape=(n,n))
    w = w.maximum(w.T)
    degree = np.asarray(w.sum(1)).ravel()
    lap = sparse.diags(degree) - w
    targets = y / y.sum(1)[:, None]
    counts = targets.sum(0)
    sources = np.zeros((n, 5))
    sources[:len(y)] = targets / np.maximum(counts, 1e-8) - (counts > 0)[None, :] / len(y)
    potential = spsolve((lap + sparse.eye(n)*1e-3).tocsc(), sources)
    potential -= np.average(potential, weights=degree, axis=0)
    # One shared scale preserves relative class evidence.
    scale = max(np.sqrt(np.mean(potential[len(y):]**2)), 1e-10)
    potential /= scale
    d, neighbors = nn.kneighbors(q, n_neighbors=k)
    log_qw = -d**2 / (np.maximum(d[:, -1, None], 1e-6) * bandwidth[neighbors])
    # Normalize in log space: repeated embeddings can make all direct weights underflow.
    qw = np.exp(log_qw - log_qw.max(axis=1,keepdims=True))
    logits = np.einsum('ij,ijk->ik', qw, potential[neighbors]) / qw.sum(1)[:, None]
    logits[:, counts == 0] = -1e3
    p = softmax(logits / c['temperature'], axis=1)
    pu = softmax(potential[len(y):] / c['temperature'], axis=1)
    return p, threshold(p), dict(graph_nodes=n, graph_edges=w.nnz//2,
                                generated=threshold(pu).sum(0).tolist(), scale=scale)


def compositional(a, y, q, c):
    """A generative mixture over 31 label sets with means tied through five reasons.

    Gold pattern responsibilities stay fixed. Missing labels are inferred with EM.
    A shrinkage covariance and anchored semantic prototypes limit variance.
    Joint posterior decoding directly maximizes predicted sample-wise F1.
    """
    n, d = len(y), a.shape[1]
    h = PATTERNS / PATTERNS.sum(1)[:, None]
    gold_h = y / y.sum(1)[:, None]
    codes = y @ (2**np.arange(5)) - 1
    reg = 2.
    initial = np.linalg.solve(gold_h.T @ gold_h + reg*np.eye(5), gold_h.T @ a[:n])
    beta = initial.copy()
    target_cov = np.cov(a.T)
    counts = np.bincount(codes, minlength=31)
    # Weak support for unseen combinations without overwhelming observed singles.
    prior_anchor = counts + .05 / PATTERNS.sum(1)
    prior = prior_anchor / prior_anchor.sum()
    covariance = target_cov.copy()
    total_u_weight = c['unlabeled_mass'] * n
    row_weights = np.r_[np.ones(n), np.full(len(a)-n, total_u_weight/max(len(a)-n,1))]
    responsibilities = np.zeros((len(a),31)); responsibilities[np.arange(n), codes] = 1
    def posterior(x):
        means = h @ beta
        ch = cho_factor(covariance + np.eye(d)*1e-6, lower=True)
        ix = cho_solve(ch, x.T).T; im = cho_solve(ch, means.T).T
        logits = x @ im.T - .5*np.sum(means*im, axis=1)[None, :]
        return softmax(logits / c.get('temperature',1.) + np.log(prior)[None,:], axis=1)
    history = []
    for step in range(1 if not c['unlabeled_mass'] else 20):
        if step:
            responsibilities[n:] = posterior(a[n:])
        weighted = responsibilities * row_weights[:,None]
        component_counts = weighted.sum(0)
        eh = weighted @ h
        lhs = h.T @ (component_counts[:,None]*h) + reg*np.eye(5)
        rhs = eh.T @ a + reg*initial
        beta = np.linalg.solve(lhs,rhs)
        means = h @ beta
        fitted_mass = weighted.sum()
        second = (a.T * weighted.sum(1)) @ a
        cross = a.T @ weighted @ means
        residual = (second - cross - cross.T + (means.T*component_counts)@means) / fitted_mass
        covariance = .65*residual + .35*target_cov
        covariance = (covariance+covariance.T)/2
        if step:
            prior = (prior_anchor + weighted[n:].sum(0))
            prior /= prior.sum()
        history.append(float(np.trace(covariance)))
    joint = posterior(q); p = joint @ PATTERNS
    pred = PATTERNS[np.argmax(joint @ UTILITY.T,axis=1)]
    pu = posterior(a[n:])
    return p, pred, dict(em_steps=len(history), residual_trace=history,
                         generated=PATTERNS[pu.argmax(1)].sum(0).tolist(),
                         decoder='joint expected sample F1', prior=prior.tolist())


def fisher(a, y, q, c):
    """Unsupervised Gaussian score features, including local residual directions."""
    pc = PCA(n_components=30, svd_solver='full').fit(a)
    z, v = pc.transform(a), pc.transform(q)
    gm = GaussianMixture(c['clusters'], covariance_type='diag', reg_covar=.05,
                         random_state=1701, n_init=2, max_iter=200).fit(z)
    def mapping(x):
        r = gm.predict_proba(x)
        residual = (x[:,None,:]-gm.means_[None,:,:]) / np.sqrt(gm.covariances_)[None,:,:]
        first = r[:,:,None] * residual / np.sqrt(gm.weights_)[None,:,None]
        blocks = [first.reshape(len(x),-1)]
        if c.get('second_order'):
            second = r[:,:,None]*(residual**2-1)/np.sqrt(2*gm.weights_)[None,:,None]
            blocks.append(second.reshape(len(x),-1))
        f = np.column_stack(blocks)
        return normalize(np.sign(f)*np.sqrt(np.abs(f)))
    f, g = mapping(z), mapping(v)
    w = c['residual_weight']
    f = np.column_stack([normalize(a),w*f])*np.sqrt(150/(1+w*w))
    g = np.column_stack([normalize(q),w*g])*np.sqrt(150/(1+w*w))
    all_p = soft_classifier(f[:len(y)], y, np.r_[g,f[len(y):]], C=.3)
    p = all_p[:len(q)]
    return p, threshold(p), dict(gmm_converged=bool(gm.converged_),
                                  gmm_iterations=int(gm.n_iter_), feature_count=f.shape[1],
                                  generated=threshold(all_p[len(q):]).sum(0).tolist())


def predict(gold, masked, query, c, labels, emb):
    assert not set(gold.ID)&set(query.ID)
    assert not set(masked.ID)&set(query.ID)
    assert not any(l in query.columns for l in labels)
    assert masked[labels].eq(-100).all().all()
    y = gold[labels].to_numpy(int)
    assert np.isin(y,[0,1]).all()
    x = np.r_[gold[emb].to_numpy(),masked[emb].to_numpy()]
    scaler = StandardScaler().fit(x)
    a, q = scaler.transform(x), scaler.transform(query[emb].to_numpy())
    if c['family']=='compositional':
        pc = PCA(n_components=c['dimensions'],svd_solver='full').fit(a)
        a, q = pc.transform(a),pc.transform(q)
    elif c['family']=='poisson':
        a, q = normalize(a),normalize(q)
    p, pred, audit = globals()[c['family']](a,y,q,c)
    assert p.shape == pred.shape == (len(query),5)
    assert np.isfinite(p).all() and np.isin(pred,[0,1]).all() and (pred.sum(1)>0).all()
    return p,pred,audit
