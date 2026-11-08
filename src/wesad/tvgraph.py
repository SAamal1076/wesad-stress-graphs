"""M6-B - joint graph-TV denoising and graph learning (nodes = signals of one window).

    min_{X, W}  1/2 ||X - Y||_F^2 + lam * sum_{s<t} W_st [ (1 - eta) ||y_s - y_t|| + eta ||x_s - x_t|| ]
                - beta * sum_s log d_s + gamma * ||W||_F^2,      W = W^T >= 0, d = W 1

eta = 1 is the original joint model (B1 showed it over-fuses through a feedback loop); eta = 0 is convex
graph learning from the total variation of the data Y (X = Y); 0 < eta < 1 keeps the alternating updates
while the distance seen by the graph keeps the floor (1 - eta) ||y_s - y_t||  (amendment 1, 8 Nov 2026).
With beta = 1 the solution depends on lam and gamma only through theta = lam / sqrt(2 gamma).

Alternating minimisation (each block is convex, so the objective decreases monotonically):
  X-step (W fixed): network-lasso / graph-TV denoising with edge weights lam * eta * w_e, solved by ADMM
      with V = B X
      X <- (I + rho B^T B)^-1 (Y + rho B^T (V - U))
      v_e <- max(0, 1 - (tau_e / rho) / ||(BX + U)_e||) (BX + U)_e      (group soft-threshold)
      U <- U + B X - V
  W-step (X fixed): z_e = (1 - eta) ||y_s - y_t|| + eta ||x_s - x_t|| and
      min_w  sum_e lam z_e w_e - beta * sum_s log (S w)_s + 2 gamma ||w||^2,  w >= 0
      (w = upper triangle of W, so ||W||_F^2 = 2 ||w||^2), solved by the primal-dual algorithm of
      Kalofolias (2016), "How to learn a graph from smooth signals", AISTATS.
Everything is batched over windows: arrays have a leading batch axis.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ graph operators
def edge_index(n):
    return np.triu_indices(n, 1)


def incidence(n):
    """B (E x n): row e = (s, t) has +1 at s and -1 at t, so (B X)_e = x_s - x_t."""
    s, t = edge_index(n)
    B = np.zeros((len(s), n))
    B[np.arange(len(s)), s] = 1.0
    B[np.arange(len(s)), t] = -1.0
    return B


def degree_operator(n):
    """S (n x E): (S w)_s = sum of the weights of the edges touching s."""
    return np.abs(incidence(n)).T


# ------------------------------------------------------------------ objective
def edge_distances(B, X):
    """||x_s - x_t|| for every edge, batched: (batch, E)."""
    return np.linalg.norm(np.einsum("en,bnm->bem", B, X), axis=2)


def objective(X, Y, w, lam, beta, gamma, eta=1.0):
    n = Y.shape[1]
    B, S = incidence(n), degree_operator(n)
    diff = (1.0 - eta) * edge_distances(B, Y) + eta * edge_distances(B, X)
    d = np.einsum("ne,be->bn", S, w)
    return (0.5 * ((X - Y) ** 2).sum((1, 2)) + lam * (w * diff).sum(1)
            - beta * np.log(np.maximum(d, 1e-300)).sum(1) + 2.0 * gamma * (w ** 2).sum(1))


# ------------------------------------------------------------------ X-step
def x_step(Y, tau, rho=1.0, n_iter=500, tol=1e-7, state=None):
    """Network lasso: min_X 1/2||X - Y||^2 + sum_e tau_e ||x_s - x_t||_2 (batched). tau: (batch, E)."""
    nb, n, m = Y.shape
    B = incidence(n)
    M = np.linalg.inv(np.eye(n) + rho * B.T @ B)
    if state is None:
        X = Y.copy()
        V = np.einsum("en,bnm->bem", B, X)
        U = np.zeros_like(V)
    else:
        X, V, U = (a.copy() for a in state)
    thr = (tau / rho)[:, :, None]
    for _ in range(n_iter):
        X = np.einsum("kn,bnm->bkm", M, Y + rho * np.einsum("en,bem->bnm", B, V - U))
        BX = np.einsum("en,bnm->bem", B, X)
        Z = BX + U
        nz = np.linalg.norm(Z, axis=2, keepdims=True)
        V_old = V
        V = np.maximum(0.0, 1.0 - thr / np.maximum(nz, 1e-12)) * Z
        U = U + BX - V
        r = np.abs(BX - V).max()
        s = rho * np.abs(V - V_old).max()
        if r < tol and s < tol:
            break
    return X, (X, V, U)


# ------------------------------------------------------------------ W-step
def w_step(c, beta, gamma, n, n_iter=2000, tol=1e-9, w0=None):
    """Kalofolias primal-dual for  min_{w>=0} c^T w - beta 1^T log(S w) + 2 gamma ||w||^2  (batched).

    In Kalofolias' notation (min 2 w^T z - alpha 1^T log(Sw) + b ||w||^2): z = c/2, alpha = beta, b = 2 gamma.
    """
    S = degree_operator(n)
    z, alpha, b = c / 2.0, beta, 2.0 * gamma
    mu = 2.0 * b + np.sqrt(2.0 * (n - 1))
    g = 0.9 / mu
    w = np.ones_like(c) if w0 is None else w0.copy()
    dv = np.einsum("ne,be->bn", S, w)                       # dual variable, degree space
    for _ in range(n_iter):
        y = w - g * (2 * b * w + np.einsum("ne,bn->be", S, dv))
        yb = dv + g * np.einsum("ne,be->bn", S, w)
        p = np.maximum(0.0, y - 2 * g * z)
        pb = (yb - np.sqrt(yb ** 2 + 4 * alpha * g)) / 2.0
        q = p - g * (2 * b * p + np.einsum("ne,bn->be", S, pb))
        qb = pb + g * np.einsum("ne,be->bn", S, p)
        w_new = w - y + q
        dv = dv - yb + qb
        done = np.abs(w_new - w).max() < tol
        w = w_new
        if done:
            break
    return np.maximum(w, 0.0)


# ------------------------------------------------------------------ alternating solver
def fit(Y, lam, beta=1.0, gamma=0.5, eta=1.0, n_outer=15, x_iter=500, w_iter=2000, track=False):
    """Joint graph-TV denoising + graph learning. Returns dict with X, w, V (B X at the ADMM optimum), history."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    nb, n, m = Y.shape
    B = incidence(n)
    zy = edge_distances(B, Y)
    w = w_step(lam * zy, beta, gamma, n, n_iter=w_iter)        # at X = Y the mixed distance equals zy
    X, hist = Y.copy(), []
    V = np.einsum("en,bnm->bem", B, X)
    if track:
        hist.append(objective(X, Y, w, lam, beta, gamma, eta))
    if eta == 0.0:                                              # convex case: X = Y, a single W-step
        return {"X": X, "w": w, "V": V, "history": np.array(hist)}
    state = None
    for _ in range(n_outer):
        X, state = x_step(Y, lam * eta * w, n_iter=x_iter, state=state)
        if track:
            hist.append(objective(X, Y, w, lam, beta, gamma, eta))
        z = (1.0 - eta) * zy + eta * edge_distances(B, X)
        w = w_step(lam * z, beta, gamma, n, n_iter=w_iter, w0=w)
        if track:
            hist.append(objective(X, Y, w, lam, beta, gamma, eta))
    return {"X": X, "w": w, "V": state[1], "history": np.array(hist)}


# ------------------------------------------------------------------ per-window descriptors
def n_clusters(V, n, tol=1e-6):
    """Number of fused groups: nodes joined by edges whose ADMM split variable is exactly ~0."""
    s, t = edge_index(n)
    fused = np.linalg.norm(V, axis=2) < tol
    out = np.empty(V.shape[0], dtype=int)
    for b in range(V.shape[0]):
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for e in np.flatnonzero(fused[b]):
            ra, rb = find(s[e]), find(t[e])
            if ra != rb:
                parent[ra] = rb
        out[b] = len({find(a) for a in range(n)})
    return out


def fusion_degree(Y, X, w, n):
    """F = 1 - TV_W(X) / TV_W(Y), with TV_W(X) = sum_e w_e ||x_s - x_t||; 0 = no fusion, 1 = all fused."""
    B = incidence(n)
    tv_y = (w * edge_distances(B, Y)).sum(1)
    tv_x = (w * edge_distances(B, X)).sum(1)
    return np.where(tv_y > 1e-12, 1.0 - tv_x / np.maximum(tv_y, 1e-12), 0.0)


def descriptors(Y, res, n):
    """Per-window features from a fitted batch: edges, degrees, algebraic connectivity, TV(Y), fusion degree."""
    B = incidence(n)
    w = res["w"]
    s, t = edge_index(n)
    W = np.zeros((w.shape[0], n, n))
    W[:, s, t] = w
    W = W + np.swapaxes(W, 1, 2)
    L = np.einsum("bij->bi", W)[:, :, None] * np.eye(n)[None] - W
    ev = np.linalg.eigvalsh(L)
    tv_y = (w * edge_distances(B, Y)).sum(1)
    return {"w": w, "degree": W.sum(2), "algconn": ev[:, 1], "tv_y": tv_y,
            "fusion": fusion_degree(Y, res["X"], w, n)}


def prepare_node_signals(Xw, smooth=8):
    """(batch, n, m) raw node series -> 2 s moving average, z-score per node, divide by sqrt(m)."""
    k = np.ones(smooth) / smooth
    sm = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 2, np.nan_to_num(Xw))
    mu = sm.mean(axis=2, keepdims=True)
    sd = sm.std(axis=2, keepdims=True)
    Z = np.where(sd > 1e-8, (sm - mu) / np.maximum(sd, 1e-8), 0.0)
    return Z / np.sqrt(Z.shape[2])
