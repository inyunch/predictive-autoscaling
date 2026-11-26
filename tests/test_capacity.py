"""KKT/bisection allocation vs SciPy SLSQP, and the normal-quantile root finder."""
import numpy as np
import pytest
from scipy.stats import norm

from capacity import expected_shortfall, norm_ppf, solve_kkt, solve_slsqp


@pytest.mark.parametrize("q", [1e-10, 0.01, 0.3, 0.5, 0.77, 0.999, 1 - 1e-10])
def test_norm_ppf_matches_scipy(q):
    assert abs(norm_ppf(q) - norm.ppf(q)) < 1e-8


def test_expected_shortfall_matches_numeric_integral():
    mu, sigma, x = 100.0, 20.0, 110.0
    d = np.linspace(mu - 12 * sigma, mu + 12 * sigma, 400001)
    num = np.trapezoid(np.maximum(d - x, 0) * norm.pdf(d, mu, sigma), d)
    assert abs(expected_shortfall(x, mu, sigma) - num) < 1e-6


@pytest.mark.parametrize("ratio", [1.0, 1.05, 1.3])
def test_kkt_matches_slsqp(ratio):
    rng = np.random.default_rng(3)
    T = 24
    mu = 1000 + 600 * np.sin(np.linspace(0, 2 * np.pi, T)) + rng.uniform(0, 50, T)
    sigma = np.sqrt(5.0 * mu)
    c, p = mu.mean() / 10, 1.0
    B = ratio * mu.sum() / c * p
    a = solve_kkt(mu, sigma, c, p, B, n_min=1.0)
    r = solve_slsqp(mu, sigma, c, p, B, n_min=1.0)
    assert abs(a.cost - B) < 1e-8
    assert a.objective <= r.fun * (1 + 1e-7)
    assert np.max(np.abs(a.n - r.x)) < 1e-3
    assert a.kkt["stationarity_free"] < 1e-8


def test_lower_bound_active_has_nonnegative_multiplier():
    mu = np.array([5000.0, 5000.0, 10.0, 10.0, 5000.0])
    sigma = np.sqrt(mu)
    a = solve_kkt(mu, sigma, c=500.0, p=1.0, B=32.0, n_min=1.0)
    assert a.kkt["n_at_min"] == 2
    assert a.kkt["dual_nu_min"] >= 0
    r = solve_slsqp(mu, sigma, 500.0, 1.0, 32.0, 1.0)
    assert abs(a.objective - r.fun) <= 1e-6 * r.fun


def test_infeasible_budget_raises():
    with pytest.raises(ValueError):
        solve_kkt(np.ones(10), np.ones(10), 1.0, 1.0, B=5.0, n_min=1.0)
