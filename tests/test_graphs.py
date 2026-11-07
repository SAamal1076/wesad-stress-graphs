"""Tests for the random-walk graph inference (G1)."""
import pathlib
import sys

import numpy as np
from scipy.linalg import expm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from wesad.graphs import _laplacian, dependency_matrix, rw_graph_inference, zscore_rows  # noqa: E402


def _loss(w, T, iu, n):
    E = expm(-_laplacian(w[None], iu, n)[0])
    a = max(0.0, (T * E).sum() / (E * E).sum())
    return ((T - a * E) ** 2).sum()


def test_gradient_matches_finite_differences():
    """One step from w0 with lam=0 must move along the exact negative gradient."""
    rng = np.random.default_rng(0)
    n = 5
    iu = np.triu_indices(n, 1)
    T = rng.uniform(0.1, 0.9, (n, n)); T = (T + T.T) / 2; np.fill_diagonal(T, 1)
    w0 = rng.uniform(0.2, 0.8, len(iu[0]))
    num = np.array([(_loss(w0 + 1e-6 * e, T, iu, n) - _loss(w0 - 1e-6 * e, T, iu, n)) / 2e-6
                    for e in np.eye(len(w0))])
    step = 1e-3
    w1, _, _ = rw_graph_inference(T[None], lam=0.0, n_iter=1, step=step, w0=w0[None])
    analytic = (w0 - w1[0]) / step
    assert np.allclose(analytic, num, rtol=1e-4, atol=1e-6)


def _random_problem(rng, n=7, density=0.3, noise=0.01):
    iu = np.triu_indices(n, 1)
    edges = rng.random(len(iu[0])) < density
    w = edges * rng.uniform(0.3, 1.0, len(iu[0]))
    E = expm(-_laplacian(w[None], iu, n)[0])
    T = 0.9 * E + noise * rng.standard_normal((n, n))
    T = (T + T.T) / 2
    np.fill_diagonal(T, 0.9 * np.diag(E))
    return T, edges


def test_recovers_sparse_graphs():
    rng = np.random.default_rng(1)
    Ts, truth = zip(*[_random_problem(rng) for _ in range(60)])
    w, _, _ = rw_graph_inference(np.array(Ts), lam=0.002, n_iter=800)
    auc_rw = roc_auc_score(np.concatenate(truth), w.ravel())
    iu = np.triu_indices(7, 1)
    auc_t = roc_auc_score(np.concatenate(truth), np.array([T[iu] for T in Ts]).ravel())
    assert auc_rw > 0.9
    assert auc_rw > auc_t  # removing multipath beats reading the dependency matrix


def test_chain_indirect_edge_is_suppressed():
    """a-b-c chain: T_ac is high (indirect) but the inferred direct edge a-c must be ~0."""
    iu = np.triu_indices(3, 1)  # (a,b), (a,c), (b,c)
    w_true = np.array([1.0, 0.0, 1.0])
    T = expm(-_laplacian(w_true[None], iu, 3)[0])
    assert T[0, 2] > 0.1  # clear indirect dependency through b
    w, _, _ = rw_graph_inference(T[None], lam=1e-3, n_iter=2000)
    assert w[0, 1] < 0.1 * min(w[0, 0], w[0, 2])


def test_dependency_matrix_detects_lagged_coupling():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(240)
    y = np.r_[np.zeros(8), x[:-8]] + 0.1 * rng.standard_normal(240)   # y = x delayed by 2 s
    z = rng.standard_normal(240)
    X = zscore_rows(np.array([[x, y, z]]))
    T = dependency_matrix(X, max_lag=20)[0]
    assert T[0, 1] > 0.9 and T[0, 2] < 0.3
