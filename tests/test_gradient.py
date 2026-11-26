"""Analytic gradient/Hessian vs central finite differences (Day 5)."""
import numpy as np

from model import PoissonLoss, build_dataset, load_series


def central_diff_grad(f, w, h=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g


def test_gradient_and_hessian_on_real_data():
    s = load_series()
    rng = np.random.default_rng(0)
    for k in (0, len(s["series"]) - 1):          # one periodic, one bursty app
        ds = build_dataset(s["series"][k])
        prob = PoissonLoss(ds.X_train, ds.y_train)
        for _ in range(3):
            w = 0.1 * rng.standard_normal(prob.d)
            g = prob.grad(w)
            gfd = central_diff_grad(prob.f, w)
            assert np.linalg.norm(g - gfd) / np.linalg.norm(g) < 1e-6
            Hfd = np.column_stack([central_diff_grad(lambda v: prob.grad(v)[j], w)
                                   for j in range(prob.d)])
            assert np.allclose(prob.hess(w), Hfd, rtol=1e-4, atol=1e-6)


def test_hessian_is_positive_definite():
    s = load_series()
    for y in s["series"]:
        ds = build_dataset(y)
        prob = PoissonLoss(ds.X_train, ds.y_train)
        assert np.linalg.eigvalsh(prob.hess(np.zeros(prob.d)))[0] > 0
