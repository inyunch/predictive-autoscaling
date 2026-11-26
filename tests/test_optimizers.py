"""Optimizers on a quadratic with a known minimizer, and on a Poisson problem."""
import numpy as np
import pytest

import optimizers as O
from model import PoissonLoss


class Quadratic:
    """f(w) = 0.5 w^T A w - b^T w, minimizer A^{-1} b."""

    def __init__(self, cond=50.0, d=8, seed=0):
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        eig = np.geomspace(1.0, cond, d)
        self.A = (Q * eig) @ Q.T
        self.b = rng.standard_normal(d)
        self.w_star = np.linalg.solve(self.A, self.b)
        self.L = eig[-1]
        self.nfev = self.ngev = self.nhev = 0

    def f(self, w):
        self.nfev += 1
        return 0.5 * w @ self.A @ w - self.b @ w

    def grad(self, w):
        self.ngev += 1
        return self.A @ w - self.b

    def hess(self, w):
        self.nhev += 1
        return self.A


def all_methods(prob, w0, L):
    return [O.gd_fixed(prob, w0, 1.0 / L, max_iter=20000),
            O.gd_wolfe(prob, w0, max_iter=20000),
            O.momentum(prob, w0, 1.0 / L),
            O.nesterov(prob, w0, 1.0 / L),
            O.ncg_pr(prob, w0),
            O.newton(prob, w0),
            O.bfgs(prob, w0)]


@pytest.mark.parametrize("cond", [1.0, 10.0, 100.0])
def test_quadratic_known_answer(cond):
    q = Quadratic(cond=cond)
    for r in all_methods(q, np.zeros(len(q.b)), q.L):
        assert r.converged, r.method
        assert np.allclose(r.w, q.w_star, atol=1e-5), r.method


def test_newton_one_step_on_quadratic():
    q = Quadratic(cond=1e4)
    assert O.newton(q, np.zeros(len(q.b))).iters == 1


def test_bfgs_and_ncg_finite_termination_scale():
    # exact-arithmetic CG needs <= d steps on a quadratic; allow slack for inexact line search
    q = Quadratic(cond=100.0, d=8)
    assert O.ncg_pr(q, np.zeros(8)).iters <= 40
    assert O.bfgs(q, np.zeros(8)).iters <= 40


def test_divergence_is_flagged():
    q = Quadratic(cond=10.0)
    r = O.gd_fixed(q, np.ones(len(q.b)), 2.5 / q.L, max_iter=5000)
    assert r.diverged and not r.converged


def test_poisson_methods_agree():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(300), rng.standard_normal((300, 4))])
    y = rng.poisson(np.exp(X @ np.array([0.5, 0.3, -0.2, 0.1, 0.0]))).astype(float)
    prob = PoissonLoss(X, y)
    w0 = np.zeros(5)
    ref = O.newton(prob, w0)
    L = max(np.linalg.eigvalsh(prob.hess(w0))[-1], np.linalg.eigvalsh(prob.hess(ref.w))[-1])
    for r in all_methods(prob, w0, L):
        assert r.converged, r.method
        assert abs(r.f - ref.f) < 1e-9, r.method
