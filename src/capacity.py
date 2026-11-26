"""Budget-constrained capacity allocation (Part 2, RQ2).

Demand in interval t:  D_t ~ Normal(mu_t, sigma_t^2),  sigma_t^2 = phi * mu_t.
Instances n_t each serve c requests per interval and cost p per interval.

    minimize    sum_t E[(D_t - c n_t)^+]
    subject to  p * sum_t n_t = B,   n_t >= n_min

KKT (derived in report/report.md): for every interval with n_t > n_min
    c * (1 - F_t(c n_t)) = lambda * p
so all such intervals are provisioned at the same quantile
    c n_t = mu_t + sigma_t * z,   z = Phi^{-1}(1 - lambda p / c),
and n_t = max(n_min, (mu_t + sigma_t z) / c). lambda is found by bisection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SQRT2 = math.sqrt(2.0)
_erfc = np.vectorize(math.erfc, otypes=[float])


def norm_pdf(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / math.sqrt(2 * math.pi)


def norm_cdf(z):
    return 0.5 * _erfc(-np.asarray(z, float) / _SQRT2)


def norm_ppf(q: float, tol: float = 1e-14) -> float:
    """Phi^{-1}(q) by safeguarded Newton's method on Phi(z) - q (Day 5 root finding).

    Solved in the lower tail (symmetry for q > 0.5) so that erfc keeps full
    relative precision; stops when the Newton step is below tol.
    """
    if not 0.0 < q < 1.0:
        return math.copysign(math.inf, q - 0.5)
    if q > 0.5:
        return -norm_ppf(1.0 - q, tol)
    lo, hi, z = -40.0, 0.0, -1.0
    for _ in range(200):
        r = float(norm_cdf(z)) - q
        if r > 0:
            hi = z
        else:
            lo = z
        dz = r / max(float(norm_pdf(z)), 1e-300)
        z_new = z - dz
        if not lo < z_new < hi:               # safeguard: bisect the bracket
            z_new = 0.5 * (lo + hi)
        if abs(z_new - z) < tol * max(1.0, abs(z)):
            return z_new
        z = z_new
    return z


def expected_shortfall(x, mu, sigma):
    """E[(D - x)^+] for D ~ N(mu, sigma^2) (unserved requests in an interval)."""
    sigma = np.maximum(sigma, 1e-12)
    z = (x - mu) / sigma
    return sigma * norm_pdf(z) + (mu - x) * (1.0 - norm_cdf(z))


def objective(n, mu, sigma, c):
    return float(np.sum(expected_shortfall(c * np.asarray(n), mu, sigma)))


@dataclass
class Allocation:
    n: np.ndarray
    lam: float
    z: float
    objective: float
    cost: float
    bisect_iters: int
    kkt: dict


def allocation_for_lambda(lam, mu, sigma, c, p, n_min):
    z = norm_ppf(1.0 - lam * p / c)
    return np.maximum(n_min, (mu + sigma * z) / c), z


def kkt_residuals(n, lam, mu, sigma, c, p, n_min, B):
    """Return primal/dual feasibility, stationarity and complementarity residuals."""
    marg = c * (1.0 - norm_cdf((c * n - mu) / np.maximum(sigma, 1e-12)))   # -dE/dn_t
    nu = lam * p - marg                       # multiplier of n_t >= n_min
    free = n > n_min + 1e-9
    return dict(
        budget_residual=float(p * n.sum() - B),
        min_violation=float(max(0.0, np.max(n_min - n))),
        stationarity_free=float(np.max(np.abs(nu[free]))) if free.any() else 0.0,
        dual_nu_min=float(nu[~free].min()) if (~free).any() else 0.0,  # must be >= 0
        complementarity=float(np.max(np.abs(nu[~free] * (n[~free] - n_min)))) if (~free).any() else 0.0,
        lam=float(lam), n_at_min=int((~free).sum()))


def solve_kkt(mu, sigma, c, p, B, n_min=1.0, tol=1e-10, max_iter=200) -> Allocation:
    """Solve the allocation problem via the KKT conditions + bisection on lambda.

    Spend S(lambda) = p * sum_t n_t(lambda) is non-increasing in lambda on (0, c/p),
    so the root of S(lambda) = B is bracketed by [0, c/p].
    """
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    T = len(mu)
    if B < p * n_min * T - 1e-9:
        raise ValueError(f"infeasible: budget {B} < p*n_min*T = {p * n_min * T}")
    spend = lambda lam: p * allocation_for_lambda(lam, mu, sigma, c, p, n_min)[0].sum()
    lo, hi = 0.0, c / p                       # S(lo) = +inf, S(hi) = p*n_min*T
    if B <= p * n_min * T + 1e-12:
        n = np.full(T, n_min)
        lam = hi
        z = -math.inf
        it = 0
    else:
        it = 0
        while hi - lo > tol * (c / p) and it < max_iter:
            mid = 0.5 * (lo + hi)
            if spend(mid) > B:
                lo = mid
            else:
                hi = mid
            it += 1
        lam = 0.5 * (lo + hi)
        n, z = allocation_for_lambda(lam, mu, sigma, c, p, n_min)
        # Bisection matches B only to tolerance; rescale the free part to hit it exactly.
        free = n > n_min
        excess = (B / p - n.sum())
        if free.any():
            n[free] += excess / free.sum()
    return Allocation(n, lam, z, objective(n, mu, sigma, c), p * n.sum(), it,
                      kkt_residuals(n, lam, mu, sigma, c, p, n_min, B))


def solve_slsqp(mu, sigma, c, p, B, n_min=1.0, x0=None):
    """Reference solution with SciPy SLSQP (cross-check only)."""
    from scipy.optimize import minimize
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    T = len(mu)
    x0 = np.full(T, B / (p * T)) if x0 is None else x0
    fun = lambda n: objective(n, mu, sigma, c)
    jac = lambda n: -c * (1.0 - norm_cdf((c * n - mu) / np.maximum(sigma, 1e-12)))
    res = minimize(fun, x0, jac=jac, method="SLSQP",
                   bounds=[(n_min, None)] * T,
                   constraints=[{"type": "eq", "fun": lambda n: p * n.sum() - B,
                                 "jac": lambda n: np.full(T, p)}],
                   options={"ftol": 1e-14, "maxiter": 1000})
    return res
