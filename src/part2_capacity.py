"""Part 2 (RQ2): optimal allocation of instances across time under a budget.

Uses the 20-minute-ahead forecasts (mu_t, sigma_t) from Part 1 on the first test
day (day 10, 144 intervals). Capacity c is set so that mean forecast demand needs
K = 10 instances; price p = 1 per instance-interval; n_min = 1.

Outputs
  results/part2_kkt_vs_slsqp.csv   KKT/bisection solution vs SciPy SLSQP
  results/part2_frontier.csv       expected overload vs budget: KKT vs uniform vs proportional
  results/part2_allocation.png     n_t* against mu_t / c for a periodic and a bursty app
  results/part2_frontier.png       expected overload vs budget
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from capacity import objective, solve_kkt, solve_slsqp, kkt_residuals
from model import ROOT, PER_DAY
from plotstyle import COLORS, setup

RES = ROOT / "results"
K, P, N_MIN = 10, 1.0, 1.0
DAY = slice(0, PER_DAY)                 # first test day
BUDGET_RATIOS = [0.9, 0.95, 1.0, 1.02, 1.05, 1.1, 1.15, 1.2, 1.3]
CHECK_RATIO = 1.05


def baseline_uniform(mu, B):
    return np.full(len(mu), B / (P * len(mu)))


def baseline_proportional(mu, c, B):
    """n_t proportional to mu_t (water-filled at n_min), same spend."""
    n = np.maximum(N_MIN, mu / mu.sum() * B / P)
    for _ in range(50):
        free = n > N_MIN
        n[free] *= (B / P - N_MIN * (~free).sum()) / n[free].sum()
        n = np.maximum(N_MIN, n)
    return n


def main():
    fc = np.load(RES / "forecasts.npz", allow_pickle=True)
    check, front, keep = [], [], {}
    for name, cls, mu_all, sg_all in zip(fc["names"], fc["cls"], fc["mu"], fc["sigma"]):
        mu, sg = mu_all[DAY], sg_all[DAY]
        c = mu.mean() / K
        base = mu.sum() / c * P                # spend that exactly covers the mean

        # --- cross-check against SLSQP: small instance (first 4 h) and the full day
        for T in (24, PER_DAY):
            m, sd = mu[:T], sg[:T]
            B = CHECK_RATIO * m.sum() / c * P
            t0 = time.perf_counter(); a = solve_kkt(m, sd, c, P, B, N_MIN); t_kkt = time.perf_counter() - t0
            t0 = time.perf_counter(); r = solve_slsqp(m, sd, c, P, B, N_MIN); t_sq = time.perf_counter() - t0
            check.append(dict(app=name, cls=cls, T=T, budget=B, z=a.z, lam=a.lam,
                              bisect_iters=a.bisect_iters, obj_kkt=a.objective, obj_slsqp=r.fun,
                              rel_gap=(r.fun - a.objective) / a.objective,
                              max_abs_dn=float(np.max(np.abs(r.x - a.n))),
                              slsqp_success=bool(r.success), slsqp_iters=r.nit,
                              time_kkt_ms=1e3 * t_kkt, time_slsqp_ms=1e3 * t_sq,
                              n_at_min=a.kkt["n_at_min"],
                              kkt_stationarity=a.kkt["stationarity_free"],
                              kkt_dual_nu_min=a.kkt["dual_nu_min"],
                              kkt_budget_residual=a.kkt["budget_residual"]))
        keep[name] = (mu, sg, c, a)  # full-day KKT solution at CHECK_RATIO

        # --- frontier: optimal vs naive allocations with the same budget
        for ratio in BUDGET_RATIOS:
            B = ratio * base
            if B < P * N_MIN * len(mu):
                continue
            a2 = solve_kkt(mu, sg, c, P, B, N_MIN)
            for pol, n in (("KKT optimal", a2.n), ("uniform", baseline_uniform(mu, B)),
                           ("proportional to mu", baseline_proportional(mu, c, B))):
                front.append(dict(app=name, cls=cls, budget_ratio=ratio, policy=pol,
                                  z=a2.z, spend=P * n.sum(),
                                  exp_overload=objective(n, mu, sg, c) / mu.sum()))

    check = pd.DataFrame(check)
    check.to_csv(RES / "part2_kkt_vs_slsqp.csv", index=False, float_format="%.6g")
    front = pd.DataFrame(front)
    front.to_csv(RES / "part2_frontier.csv", index=False, float_format="%.6g")
    print(check[["app", "T", "z", "bisect_iters", "obj_kkt", "obj_slsqp", "rel_gap", "max_abs_dn",
                 "slsqp_success", "time_kkt_ms", "time_slsqp_ms", "n_at_min"]].to_string(index=False))
    print(front[front.budget_ratio == CHECK_RATIO].pivot(index="app", columns="policy",
                                                 values="exp_overload").to_string())
    plot(keep, front, fc)


def plot(keep, front, fc):
    setup()
    picks = [fc["names"][list(fc["cls"]).index(c)] for c in ("periodic", "bursty")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    hours = np.arange(PER_DAY) / 6
    for ax, name in zip(axes, picks):
        mu, sg, c, a = keep[name]
        ax.plot(hours, mu / c, color="#222222", lw=1, label=r"$\mu_t / c$ (mean need)")
        ax.fill_between(hours, np.maximum(mu - sg, 0) / c, (mu + sg) / c, color="#888888",
                        alpha=0.2, lw=0, label=r"$(\mu_t \pm \sigma_t)/c$")
        ax.plot(hours, a.n, color=COLORS["optimal"], lw=1.8,
                label=fr"KKT $n_t^\ast$ ($z$ = {a.z:.2f})")
        ax.axhline(a.n.mean(), color=COLORS["reactive"], lw=1.2, ls="--", label="uniform, same budget")
        ax.set_title(f"{name}: day 10, budget = {CHECK_RATIO} x mean need")
        ax.set_xlabel("hour of day")
    axes[0].set_ylabel("instances")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "part2_allocation.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = {"KKT optimal": (COLORS["optimal"], "-"), "uniform": (COLORS["reactive"], "--"),
              "proportional to mu": (COLORS["predictive"], ":")}
    for ax, name in zip(axes, picks):
        d = front[front.app == name]
        for pol, (col, ls) in styles.items():
            dd = d[d.policy == pol]
            ax.plot(dd.budget_ratio, 100 * dd.exp_overload, ls, color=col, marker="o", ms=3, label=pol)
        ax.set_yscale("log")
        ax.set_xlabel(r"budget / $(p \sum_t \mu_t / c)$")
        ax.set_title(name)
    axes[0].set_ylabel("expected overload (% of demand)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Expected unserved demand at equal budget (forecast distribution, day 10)")
    fig.tight_layout()
    fig.savefig(RES / "part2_frontier.png", dpi=150)


if __name__ == "__main__":
    main()
