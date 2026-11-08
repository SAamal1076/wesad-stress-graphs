"""Tests for the joint graph-TV denoising + graph learning solver (M6-B)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from wesad.tvgraph import degree_operator, edge_index, fit, objective, w_step, x_step  # noqa: E402


def _two_node_closed_form(y1, y2, tau):
    d = y1 - y2
    nd = np.linalg.norm(d)
    if nd <= 2 * tau:                       # fused: both at the mean
        c = (y1 + y2) / 2
        return c, c
    u = d / nd                              # otherwise the difference shrinks by 2 tau
    return y1 - tau * u, y2 + tau * u


def test_x_step_matches_two_node_closed_form():
    rng = np.random.default_rng(0)
    for tau in (0.05, 0.4, 2.0):            # not fused, borderline, fused
        Y = rng.standard_normal((1, 2, 30)) * 0.2
        X, _ = x_step(Y, np.array([[tau]]), n_iter=5000, tol=1e-12)
        x1, x2 = _two_node_closed_form(Y[0, 0], Y[0, 1], tau)
        assert np.allclose(X[0, 0], x1, atol=1e-6) and np.allclose(X[0, 1], x2, atol=1e-6)


def test_w_step_satisfies_kkt():
    rng = np.random.default_rng(1)
    n, beta, gamma = 5, 1.0, 0.25
    c = rng.uniform(0.0, 2.0, (40, len(edge_index(n)[0])))
    w = w_step(c, beta, gamma, n, n_iter=20000, tol=1e-12)
    S = degree_operator(n)
    d = w @ S.T
    s, t = edge_index(n)
    grad = c - beta * (1 / d[:, s] + 1 / d[:, t]) + 4 * gamma * w
    active = w > 1e-6
    assert np.abs(grad[active]).max() < 1e-4                  # stationarity on positive edges
    assert grad[~active].min() > -1e-4                         # non-negative gradient on zero edges


def test_alternating_objective_is_monotone():
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((25, 5, 60))
    Y = (Y - Y.mean(2, keepdims=True)) / Y.std(2, keepdims=True) / np.sqrt(60)
    res = fit(Y, lam=0.3, n_outer=8, x_iter=3000, w_iter=5000, track=True)
    h = res["history"]                                         # (steps, batch), every half-step
    assert np.all(np.diff(h, axis=0) <= 1e-6 * (1 + np.abs(h[:-1])))
    assert np.allclose(objective(res["X"], Y, res["w"], 0.3, 1.0, 0.25), h[-1])


@pytest.mark.xfail(reason='B1 finding: with the pre-registered scale (beta=1, gamma=0.25) weights stay near-uniform, and with the log-degree prior the joint model over-fuses (independent signals collapse into clusters); fix proposed to Salman, not yet approved', strict=False)
def test_recovers_planted_clusters():
    rng = np.random.default_rng(3)
    m = 240
    t = np.linspace(0, 6 * np.pi, m)
    src = [np.sin(t + rng.uniform(0, 6)) + 0.5 * np.sin(2.3 * t), np.cos(1.7 * t) + rng.standard_normal(m) * 0.3]
    within, between = [], []
    for _ in range(20):
        Y = np.stack([src[0] + 0.6 * rng.standard_normal(m) for _ in range(3)]
                     + [src[1] + 0.6 * rng.standard_normal(m) for _ in range(3)])[None]
        Y = (Y - Y.mean(2, keepdims=True)) / Y.std(2, keepdims=True) / np.sqrt(m)
        w = fit(Y, lam=0.3, n_outer=10)["w"][0]
        s, tt = edge_index(6)
        same = (s < 3) == (tt < 3)
        within.append(w[same].mean())
        between.append(w[~same].mean())
    assert np.mean(within) > 3 * np.mean(between)
