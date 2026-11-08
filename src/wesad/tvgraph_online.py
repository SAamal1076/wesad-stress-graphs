"""M6-C - online (causal) time-varying graph learning from the total variation of window signals.

At each window t, given the previous graph w_{t-1}:

    w_t = argmin_{w >= 0}  theta * z_t^T w  -  beta * sum_s log (S w)_s  +  2 gamma ||w||^2
                           + mu * ||w - w_{t-1}||_1,          z_t,e = ||y_t,s - y_t,r||_2

The first three terms are the graph-learning step of M6-B with eta = 0 (the graph is learned from the total
variation of the prepared signals Y_t on it; convex, unique solution). The last term is the total variation
of the graph along time, solved causally: edges move only when the data push harder than mu, so coupling
changes appear as steps. mu = 0 gives the independent per-window graph.

Solver: the primal-dual forward-backward-forward algorithm of Kalofolias (2016) with a second dual variable
for the l1 attraction (operator K = [S; I], ||K|| = sqrt(2n - 1)):
    prox of the conjugate of mu||. - w_prev||_1 at step g:  v -> clip(v - g w_prev, -mu, mu).
Everything is batched over independent sequences (leading axis).
"""
from __future__ import annotations

import numpy as np

from .tvgraph import degree_operator, edge_distances, incidence, w_step


def w_step_prox(c, beta, gamma, n, w_prev, mu, n_iter=10000, tol=1e-9, w0=None):
    """min_{w>=0} c^T w - beta 1^T log(S w) + 2 gamma ||w||^2 + mu ||w - w_prev||_1   (batched)."""
    if mu == 0:
        return w_step(c, beta, gamma, n, n_iter=n_iter, tol=tol, w0=w0)
    S = degree_operator(n)
    z, alpha, b = c / 2.0, beta, 2.0 * gamma
    g = 0.9 / (2.0 * b + np.sqrt(2.0 * n - 1.0))
    w = (w_prev if w0 is None else w0).copy()
    w = np.where(w > 0, w, 1e-3)
    dv = np.einsum("ne,be->bn", S, w)
    v = np.zeros_like(w)
    for _ in range(n_iter):
        y = w - g * (2 * b * w + np.einsum("ne,bn->be", S, dv) + v)
        yb = dv + g * np.einsum("ne,be->bn", S, w)
        yv = v + g * w
        p = np.maximum(0.0, y - 2 * g * z)
        pb = (yb - np.sqrt(yb ** 2 + 4 * alpha * g)) / 2.0
        pv = np.clip(yv - g * w_prev, -mu, mu)
        q = p - g * (2 * b * p + np.einsum("ne,bn->be", S, pb) + pv)
        qb = pb + g * np.einsum("ne,be->bn", S, p)
        qv = pv + g * p
        w_new = w - y + q
        dv = dv - yb + qb
        v_new = v - yv + qv
        done = max(np.abs(w_new - w).max(), np.abs(v_new - v).max()) < tol   # primal and l1 dual settled
        w, v = w_new, v_new
        if done:
            break
    return np.maximum(w, 0.0)


def online_objective(w, c, beta, gamma, n, w_prev, mu):
    d = np.einsum("ne,be->bn", degree_operator(n), w)
    return ((c * w).sum(1) - beta * np.log(np.maximum(d, 1e-300)).sum(1) + 2 * gamma * (w ** 2).sum(1)
            + mu * np.abs(w - w_prev).sum(1))


def online_fit(Yseq, theta, mu, beta=1.0, gamma=0.5, n_iter=10000, tol=1e-9):
    """Yseq: (batch, T, n, m) prepared window signals in time order -> w (batch, T, E), z (batch, T, E)."""
    nb, T, n, m = Yseq.shape
    B = incidence(n)
    E = B.shape[0]
    W = np.zeros((nb, T, E))
    Z = np.zeros((nb, T, E))
    w = None
    for t in range(T):
        z = edge_distances(B, Yseq[:, t])
        Z[:, t] = z
        if w is None:
            w = w_step(theta * z, beta, gamma, n, n_iter=n_iter, tol=tol)
        else:
            w = w_step_prox(theta * z, beta, gamma, n, w_prev=w, mu=mu, n_iter=n_iter, tol=tol, w0=w)
        W[:, t] = w
    return W, Z
