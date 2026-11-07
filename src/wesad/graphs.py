"""G1 - coupling graphs between physiological signals, one graph per window.

Dependency matrix T (n x n): maximum absolute cross-correlation of the z-scored node
series within +-max_lag samples (diagonal = 1).

Three edge estimators:
  corr    : the off-diagonal of T itself (direct + indirect coupling mixed)
  pcorr   : partial correlation from a Ledoit-Wolf shrunk covariance (lag 0)
  rw      : random-walk diffusion-kernel inference (Noda et al. 2014; Al-Ali et al. 2019;
            Karfoul & Almuhammad Alali 2026):
                min_{w >= 0, alpha}  || T - alpha * expm(-L(w)) ||_F^2 + lam * sum(w)
            with L(w) the Laplacian of the weighted undirected graph. The diffusion kernel
            expm(-L) contains all multipath (indirect) trajectories, so the recovered w keeps
            mainly the direct couplings. Solved by projected proximal gradient, batched over
            windows, with the exact gradient of the matrix exponential (Daleckii-Krein).
            The diffusion time beta is absorbed into w (scale identifiability).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import ledoit_wolf


def zscore_rows(X, eps=1e-8):
    """X: (..., n, m). Rows with ~zero variance become all-zero."""
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return np.where(sd > eps, (X - mu) / np.maximum(sd, eps), 0.0)


def dependency_matrix(X, max_lag=20):
    """X: (W, n, m) z-scored windows -> T: (W, n, n), max |xcorr| over lags."""
    W, n, m = X.shape
    T = np.zeros((W, n, n))
    for lag in range(0, max_lag + 1):
        C = np.matmul(X[:, :, lag:], np.swapaxes(X[:, :, :m - lag], 1, 2)) / (m - lag)
        A = np.abs(C)
        T = np.maximum(T, np.maximum(A, np.swapaxes(A, 1, 2)))
    idx = np.arange(n)
    T[:, idx, idx] = 1.0
    return np.clip(T, 0.0, 1.0)


def normalize_dependency(T):
    """Symmetric normalisation D^-1/2 T D^-1/2 (D = row sums).

    The random-walk kernel expm(-L) has unit row sums, while a correlation-type matrix with a
    unit diagonal does not; without this step alpha absorbs the row sums and forces dense,
    poorly fitting graphs (relative residual ~0.7 on WESAD vs ~0.05 after normalisation).
    """
    d = T.sum(axis=-1)
    return T / np.sqrt(d[..., :, None] * d[..., None, :])


def partial_correlation(X):
    """X: (W, n, m) z-scored -> (W, n, n) partial correlations (Ledoit-Wolf shrinkage)."""
    W, n, _ = X.shape
    out = np.zeros((W, n, n))
    for k in range(W):
        cov, _ = ledoit_wolf(X[k].T)
        P = np.linalg.pinv(cov)
        d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
        out[k] = -P / np.outer(d, d)
        np.fill_diagonal(out[k], 1.0)
    return out


def _laplacian(w, iu, n):
    W = np.zeros((w.shape[0], n, n))
    W[:, iu[0], iu[1]] = w
    W = W + np.swapaxes(W, 1, 2)
    return np.einsum("bij->bi", W)[:, :, None] * np.eye(n)[None] - W


def rw_graph_inference(T, lam=0.05, n_iter=400, step=0.2, w0=None, tol=1e-7):
    """Batched random-walk graph inference.

    T: (W, n, n) symmetric dependency matrices.
    Returns w (W, n_edges) >= 0 (upper-triangle order), alpha (W,), residual (W,).
    """
    T = np.asarray(T, dtype=np.float64)
    Wn, n, _ = T.shape
    iu = np.triu_indices(n, 1)
    w = (0.5 * T[:, iu[0], iu[1]] if w0 is None else w0.copy())
    z, t_k = w.copy(), 1.0                      # FISTA momentum
    prev = np.full(Wn, np.inf)
    for it in range(n_iter):
        L = _laplacian(z, iu, n)
        lam_e, V = np.linalg.eigh(L)            # L = V diag(lam_e) V^T
        g = np.exp(-lam_e)                      # eigenvalues of expm(-L)
        E = np.einsum("bij,bj,bkj->bik", V, g, V)
        alpha = np.clip(np.einsum("bij,bij->b", T, E) / np.einsum("bij,bij->b", E, E), 0, None)
        R = T - alpha[:, None, None] * E
        loss = np.einsum("bij,bij->b", R, R)
        GE = -2.0 * alpha[:, None, None] * R    # d loss / d E
        # Daleckii-Krein: d loss / d L = V ((V^T GE V) o Gamma) V^T
        dl = lam_e[:, :, None] - lam_e[:, None, :]
        dg = g[:, :, None] - g[:, None, :]
        same = np.abs(dl) < 1e-9
        Gamma = np.where(same, -0.5 * (g[:, :, None] + g[:, None, :]), dg / np.where(same, 1.0, dl))
        M = np.einsum("bji,bjk,bkl->bil", V, GE, V) * Gamma
        GL = np.einsum("bij,bjk,blk->bil", V, M, V)
        grad = GL[:, iu[0], iu[0]] + GL[:, iu[1], iu[1]] - GL[:, iu[0], iu[1]] - GL[:, iu[1], iu[0]]
        w_new = np.maximum(0.0, z - step * (grad + lam))          # proximal step for lam*sum(w), w>=0
        t_new = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        z = w_new + ((t_k - 1) / t_new) * (w_new - w)
        w, t_k = w_new, t_new
        if np.all(np.abs(prev - loss) < tol):
            break
        prev = loss
    L = _laplacian(w, iu, n)
    lam_e, V = np.linalg.eigh(L)
    E = np.einsum("bij,bj,bkj->bik", V, np.exp(-lam_e), V)
    alpha = np.clip(np.einsum("bij,bij->b", T, E) / np.einsum("bij,bij->b", E, E), 0, None)
    resid = np.sqrt(np.einsum("bij,bij->b", T - alpha[:, None, None] * E, T - alpha[:, None, None] * E))
    return w, alpha, resid


def graph_descriptors(w, n):
    """Total strength and algebraic connectivity of weighted graphs (W, n_edges)."""
    iu = np.triu_indices(n, 1)
    L = _laplacian(np.abs(w), iu, n)
    ev = np.linalg.eigvalsh(L)
    return np.abs(w).sum(axis=1), ev[:, 1]


def edge_names(nodes):
    iu = np.triu_indices(len(nodes), 1)
    return [f"{nodes[i]}~{nodes[j]}" for i, j in zip(*iu)]
