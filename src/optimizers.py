"""Unconstrained optimizers, implemented from scratch (numpy only).

All optimizers take a problem object exposing f(w), grad(w), hess(w) and return
an OptResult. They stop when ||grad|| < tol, after max_iter iterations, or when
an iterate becomes non-finite (flagged as diverged).

References: Nocedal & Wright, Numerical Optimization, 2nd ed. (NW).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptResult:
    method: str
    w: np.ndarray
    f: float
    gnorm: float
    iters: int
    converged: bool
    diverged: bool
    time: float
    f_hist: list = field(default_factory=list)
    g_hist: list = field(default_factory=list)
    nfev: int = 0
    ngev: int = 0
    nhev: int = 0


class _Tracker:
    """Shared bookkeeping: history, stopping test, divergence test, timing."""

    def __init__(self, name, prob, w0, tol, max_iter):
        self.name, self.prob, self.tol, self.max_iter = name, prob, tol, max_iter
        self.f_hist, self.g_hist = [], []
        self.t0 = time.perf_counter()
        self.c0 = (prob.nfev, prob.ngev, prob.nhev)

    def record(self, f, g) -> str | None:
        gn = float(np.linalg.norm(g)) if np.all(np.isfinite(g)) else np.inf
        self.f_hist.append(f)
        self.g_hist.append(gn)
        if not (np.isfinite(f) and np.isfinite(gn)):
            return "diverged"
        if gn < self.tol:
            return "converged"
        return None

    def result(self, w, status) -> OptResult:
        p = self.prob
        return OptResult(self.name, w, self.f_hist[-1], self.g_hist[-1],
                         len(self.f_hist) - 1, status == "converged",
                         status == "diverged" or not np.all(np.isfinite(w)),
                         time.perf_counter() - self.t0, self.f_hist, self.g_hist,
                         p.nfev - self.c0[0], p.ngev - self.c0[1], p.nhev - self.c0[2])


# ---------------------------------------------------------------- line searches
def backtracking(prob, w, f, g, p, alpha0=1.0, rho=0.5, c1=1e-4, max_steps=60):
    """Armijo backtracking (NW Alg. 3.1)."""
    slope = g @ p
    a = alpha0
    for _ in range(max_steps):
        fn = prob.f(w + a * p)
        if np.isfinite(fn) and fn <= f + c1 * a * slope:
            return a, fn
        a *= rho
    return a, prob.f(w + a * p)


def _cubic_min(a0, f0, d0, a1, f1, d1):
    """Minimizer of the cubic interpolating (a0,f0,d0),(a1,f1,d1) (NW eq. 3.59)."""
    d_1 = d0 + d1 - 3 * (f0 - f1) / (a0 - a1)
    disc = d_1 * d_1 - d0 * d1
    if disc < 0:
        return None
    d_2 = np.sign(a1 - a0) * np.sqrt(disc)
    den = d1 - d0 + 2 * d_2
    if den == 0:
        return None
    return a1 - (a1 - a0) * (d1 + d_2 - d_1) / den


def strong_wolfe(prob, w, f0, g0, p, c1=1e-4, c2=0.9, alpha0=1.0, alpha_max=1e6,
                 max_iter=30):
    """Line search satisfying the strong Wolfe conditions (NW Alg. 3.5 + 3.6).

    Returns (alpha, f_new, g_new). If no acceptable step is found the best
    Armijo-satisfying point seen is returned (or alpha = 0 as a last resort).
    """
    d0 = g0 @ p
    phi = lambda a: prob.f(w + a * p)

    def dphi(a):
        gg = prob.grad(w + a * p)
        return gg @ p, gg

    def zoom(lo, f_lo, d_lo, hi, f_hi, d_hi):
        for _ in range(max_iter):
            a = None
            if np.isfinite(f_hi) and np.isfinite(d_hi):
                a = _cubic_min(lo, f_lo, d_lo, hi, f_hi, d_hi)
            lo_b, hi_b = min(lo, hi), max(lo, hi)
            span = hi_b - lo_b
            if a is None or not (lo_b + 0.1 * span <= a <= hi_b - 0.1 * span):
                a = 0.5 * (lo + hi)              # safeguarded bisection
            fa = phi(a)
            if not np.isfinite(fa) or fa > f0 + c1 * a * d0 or fa >= f_lo:
                hi, f_hi = a, fa
                d_hi = dphi(a)[0] if np.isfinite(fa) else np.nan
            else:
                da, ga = dphi(a)
                if abs(da) <= -c2 * d0:
                    return a, fa, ga
                if da * (hi - lo) >= 0:
                    hi, f_hi, d_hi = lo, f_lo, d_lo
                lo, f_lo, d_lo = a, fa, da
            if abs(hi - lo) < 1e-16:
                break
        return lo, f_lo, prob.grad(w + lo * p)

    a_prev, f_prev, d_prev = 0.0, f0, d0
    a = alpha0
    for i in range(max_iter):
        fa = phi(a)
        if not np.isfinite(fa) or fa > f0 + c1 * a * d0 or (i > 0 and fa >= f_prev):
            d_a = dphi(a)[0] if np.isfinite(fa) else np.nan
            return zoom(a_prev, f_prev, d_prev, a, fa, d_a)
        da, ga = dphi(a)
        if abs(da) <= -c2 * d0:
            return a, fa, ga
        if da >= 0:
            return zoom(a, fa, da, a_prev, f_prev, d_prev)
        a_prev, f_prev, d_prev = a, fa, da
        a = min(2.0 * a, alpha_max)
    return a, fa, ga


# ---------------------------------------------------------------- methods
def gd_fixed(prob, w0, alpha, tol=1e-6, max_iter=5000):
    """Gradient descent with constant step (Days 7-8)."""
    tr = _Tracker("GD (fixed)", prob, w0, tol, max_iter)
    w = w0.copy()
    for _ in range(max_iter + 1):
        f, g = prob.f(w), prob.grad(w)
        if (st := tr.record(f, g)) or len(tr.f_hist) > max_iter:
            break
        w = w - alpha * g
    return tr.result(w, st)


def gd_wolfe(prob, w0, tol=1e-6, max_iter=5000, c2=0.9):
    """Steepest descent with strong-Wolfe line search (Day 9)."""
    tr = _Tracker("GD + Wolfe", prob, w0, tol, max_iter)
    w = w0.copy()
    f, g = prob.f(w), prob.grad(w)
    a_prev, slope_prev = 1.0, None
    while True:
        if (st := tr.record(f, g)) or len(tr.f_hist) > max_iter:
            break
        p = -g
        slope = g @ p
        # NW eq. 3.60: initial step from previous first-order change
        a0 = 1.0 if slope_prev is None else min(1.0, 1.01 * a_prev * slope_prev / slope)
        a, f, g_new = strong_wolfe(prob, w, f, g, p, c2=c2, alpha0=max(a0, 1e-8))
        w = w + a * p
        g, a_prev, slope_prev = g_new, a, slope
    return tr.result(w, st)


def momentum(prob, w0, alpha, beta=0.9, tol=1e-6, max_iter=5000):
    """Polyak heavy-ball momentum (Day 10)."""
    tr = _Tracker("Momentum", prob, w0, tol, max_iter)
    w, v = w0.copy(), np.zeros_like(w0)
    for _ in range(max_iter + 1):
        f, g = prob.f(w), prob.grad(w)
        if (st := tr.record(f, g)) or len(tr.f_hist) > max_iter:
            break
        v = beta * v - alpha * g
        w = w + v
    return tr.result(w, st)


def nesterov(prob, w0, alpha, beta=0.9, tol=1e-6, max_iter=5000):
    """Nesterov accelerated gradient, constant beta (Day 10)."""
    tr = _Tracker("Nesterov", prob, w0, tol, max_iter)
    w, w_prev = w0.copy(), w0.copy()
    for _ in range(max_iter + 1):
        f, g = prob.f(w), prob.grad(w)
        if (st := tr.record(f, g)) or len(tr.f_hist) > max_iter:
            break
        y = w + beta * (w - w_prev)
        w_prev, w = w, y - alpha * prob.grad(y)
    return tr.result(w, st)


def ncg_pr(prob, w0, tol=1e-6, max_iter=5000, c2=0.1, restart=None):
    """Nonlinear CG, Polak-Ribiere+ with restarts (Day 12, NW Alg. 5.4)."""
    tr = _Tracker("NCG (PR+)", prob, w0, tol, max_iter)
    restart = restart or len(w0)
    w = w0.copy()
    f, g = prob.f(w), prob.grad(w)
    p = -g
    k = 0
    while True:
        if (st := tr.record(f, g)) or len(tr.f_hist) > max_iter:
            break
        if g @ p >= 0:                       # not a descent direction: restart
            p = -g
        a0 = 1.0 if k == 0 else min(1.0, 2 * (f - f_prev) / (g @ p)) if f_prev > f else 1.0
        f_prev = f
        a, f, g_new = strong_wolfe(prob, w, f, g, p, c2=c2, alpha0=max(a0, 1e-8))
        w = w + a * p
        beta = max(0.0, g_new @ (g_new - g) / (g @ g))
        k += 1
        if k % restart == 0:
            beta = 0.0
        p = -g_new + beta * p
        g = g_new
    return tr.result(w, st)


def newton(prob, w0, tol=1e-6, max_iter=5000):
    """Newton's method with Armijo backtracking (Days 13-14)."""
    tr = _Tracker("Newton", prob, w0, tol, max_iter)
    w = w0.copy()
    f, g = prob.f(w), prob.grad(w)
    while True:
        if (st := tr.record(f, g)) or len(tr.f_hist) > max_iter:
            break
        Hm = prob.hess(w)
        try:
            L = np.linalg.cholesky(Hm)
            p = -np.linalg.solve(L.T, np.linalg.solve(L, g))
        except np.linalg.LinAlgError:       # not PD: fall back to steepest descent
            p = -g
        a, f = backtracking(prob, w, f, g, p)
        w = w + a * p
        g = prob.grad(w)
    return tr.result(w, st)


def bfgs(prob, w0, tol=1e-6, max_iter=5000, c2=0.9):
    """BFGS on the inverse Hessian, strong Wolfe, H0 scaling NW eq. 6.20."""
    tr = _Tracker("BFGS", prob, w0, tol, max_iter)
    n = len(w0)
    I = np.eye(n)
    w = w0.copy()
    f, g = prob.f(w), prob.grad(w)
    Hinv = I.copy()
    first = True
    while True:
        if (st := tr.record(f, g)) or len(tr.f_hist) > max_iter:
            break
        p = -Hinv @ g
        if g @ p >= 0:
            p, Hinv = -g, I.copy()
        a, f, g_new = strong_wolfe(prob, w, f, g, p, c2=c2, alpha0=1.0)
        s = a * p
        yv = g_new - g
        sy = s @ yv
        if sy > 1e-12:
            if first:                        # NW eq. 6.20
                Hinv = (sy / (yv @ yv)) * I
                first = False
            rho = 1.0 / sy
            V = I - rho * np.outer(s, yv)
            Hinv = V @ Hinv @ V.T + rho * np.outer(s, s)
        w, g = w + s, g_new
    return tr.result(w, st)
