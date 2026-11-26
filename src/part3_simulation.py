"""Part 3 (RQ3): predictive vs reactive autoscaling replayed on held-out days.

Capacity c is chosen from TRAINING data so that mean demand needs K instances,
K in {5, 10, 20, 50}. Sweep reactive target utilization u in [0.4, 0.95] and
predictive quantile z in [-1, 3] (extended to 6 so that the predictive frontier
reaches the cost range of the reactive policy); compare cost-vs-overload frontiers.

For each reactive operating point (u in {0.6, 0.7, 0.8}) we report
  - predictive overload at equal cost (interpolated; NaN if out of range),
  - predictive cost at equal overload, as a saving 1 - C_pred / C_react,
  - whether some predictive point Pareto-dominates it (cost <= and overload <=).

Outputs
  results/part3_sweep.csv          every (app, K, policy, parameter) point
  results/part3_equal_cost.csv     predictive overload at the reactive policy's cost
  results/part3_summary.csv        per K and class: wins and median overload ratio
  results/part3_frontiers.png      frontiers for all 10 apps (K = 10)
Extension beyond the plan: a hybrid policy max(predictive, reactive u=0.9).
  results/part3_timeseries.png     instances over one day, periodic and bursty example
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from model import ROOT, PER_DAY, TRAIN, load_series
from plotstyle import COLORS, setup
from simulate import metrics, overload_at_cost, predictive_policy, reactive_policy

RES = ROOT / "results"
KS = (5, 10, 20, 50)
US = np.round(np.linspace(0.40, 0.95, 12), 3)
ZS_SPEC = np.round(np.linspace(-1.0, 3.0, 17), 3)       # range from the research plan
ZS_EXT = np.array([3.5, 4.0, 5.0, 6.0])                  # extension: reach reactive's cost
ZS = np.concatenate([ZS_SPEC, ZS_EXT])
U_REF = (0.6, 0.7, 0.8)
U_SAFETY = 0.9    # hybrid: predictive plus a lean reactive safety net (extension)


def main():
    s = load_series()
    fc = np.load(RES / "forecasts.npz", allow_pickle=True)
    t_idx = fc["t_test"]
    rows, eq = [], []
    for k, (y_full, name, cls) in enumerate(zip(s["series"], s["names"], s["cls"])):
        y_full = y_full.astype(float)
        y = y_full[t_idx]
        mu, sg = fc["mu"][k], fc["sigma"][k]
        for K in KS:
            c = y_full[TRAIN].mean() / K
            oracle = np.maximum(1, np.ceil(y / c)).sum()     # perfect-foresight cost
            for u in US:
                m = metrics(y, reactive_policy(y_full, t_idx, c, u), c)
                rows.append(dict(app=name, cls=cls, K=K, policy="reactive", param=u,
                                 rel_cost=m["cost"] / oracle, **m))
            for z in ZS:
                m = metrics(y, predictive_policy(mu, sg, c, z), c)
                rows.append(dict(app=name, cls=cls, K=K, policy="predictive", param=z,
                                 rel_cost=m["cost"] / oracle, **m))
                n_h = np.maximum(predictive_policy(mu, sg, c, z),
                                 reactive_policy(y_full, t_idx, c, U_SAFETY))
                m = metrics(y, n_h, c)
                rows.append(dict(app=name, cls=cls, K=K, policy="hybrid", param=z,
                                 rel_cost=m["cost"] / oracle, **m))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "part3_sweep.csv", index=False, float_format="%.6g")

    for (name, cls, K), d in df.groupby(["app", "cls", "K"], sort=False):
        pr = d[d.policy == "predictive"]
        for u in U_REF:
            r = d[(d.policy == "reactive") & np.isclose(d.param, u, atol=0.03)]
            r = r.iloc[(r.param - u).abs().argmin()]
            o_p = overload_at_cost(pr.cost.to_numpy(), pr.overload.to_numpy(), r.cost)
            ok = pr[pr.overload <= r.overload + 1e-12]
            c_p = ok.cost.min() if len(ok) else np.nan
            dom = bool(((pr.cost <= r.cost) & (pr.overload <= r.overload + 1e-12)).any())
            hy = d[d.policy == "hybrid"]
            okh = hy[hy.overload <= r.overload + 1e-12]
            c_h = okh.cost.min() if len(okh) else np.nan
            eq.append(dict(app=name, cls=cls, K=K, u=r.param, reactive_cost=r.cost,
                           reactive_overload=r.overload, predictive_overload_eq_cost=o_p,
                           overload_ratio=o_p / r.overload if r.overload > 0 else np.nan,
                           predictive_cost_eq_overload=c_p,
                           cost_saving=1 - c_p / r.cost, predictive_dominates=dom,
                           hybrid_cost_saving=1 - c_h / r.cost,
                           hybrid_dominates=bool(((hy.cost <= r.cost) &
                                                  (hy.overload <= r.overload + 1e-12)).any())))
    eq = pd.DataFrame(eq)
    eq.to_csv(RES / "part3_equal_cost.csv", index=False, float_format="%.6g")

    summ = (eq.groupby(["K", "cls"])
            .agg(comparisons=("app", "size"),
                 predictive_dominates=("predictive_dominates", "sum"),
                 median_cost_saving=("cost_saving", "median"),
                 min_cost_saving=("cost_saving", "min"),
                 median_overload_ratio=("overload_ratio", "median"),
                 hybrid_dominates=("hybrid_dominates", "sum"),
                 hybrid_median_saving=("hybrid_cost_saving", "median"),
                 hybrid_min_saving=("hybrid_cost_saving", "min"))
            .reset_index())
    summ.to_csv(RES / "part3_summary.csv", index=False, float_format="%.4g")
    print(summ.to_string(index=False))
    print(eq[(eq.K == 10) & np.isclose(eq.u, 0.7, atol=0.03)].to_string(index=False))
    plot_frontiers(df, s)
    plot_timeseries(s, fc, t_idx)


def plot_frontiers(df, s, K=10):
    setup()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
    for ax, name in zip(axes.flat, s["names"]):
        d = df[(df.app == name) & (df.K == K)]
        for pol, mk in (("reactive", "s"), ("predictive", "o")):
            dd = d[d.policy == pol].sort_values("rel_cost")
            ax.plot(dd.rel_cost, 100 * dd.overload, marker=mk, ms=3, lw=1.3,
                    color=COLORS[pol], label=pol + (r" ($z \leq 3$)" if pol == "predictive" else ""))
        hy = d[d.policy == "hybrid"].sort_values("rel_cost")
        ax.plot(hy.rel_cost, 100 * hy.overload, lw=1.0, ls=":", color=COLORS["optimal"],
                label=f"hybrid (max with reactive u={U_SAFETY})")
        ext = d[(d.policy == "predictive") & (d.param > 3)]
        ax.plot(ext.rel_cost, 100 * ext.overload, "o", ms=5, mfc="white",
                color=COLORS["predictive"], label=r"predictive, $z > 3$")
        ref = d[(d.policy == "reactive") & np.isclose(d.param, 0.7, atol=0.03)]
        ax.plot(ref.rel_cost, 100 * ref.overload, "k*", ms=9, label="reactive u=0.7")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(f"{name} ({d.cls.iloc[0]})")
    for ax in axes[:, 0]:
        ax.set_ylabel("overload (% of requests)")
    for ax in axes[1]:
        ax.set_xlabel("cost / perfect-foresight cost")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Cost vs overload on held-out days 10-13 (K = {K} instances at mean load). "
                 "Lower-left is better.")
    fig.tight_layout()
    fig.savefig(RES / "part3_frontiers.png", dpi=150)


def plot_timeseries(s, fc, t_idx, K=10, u=0.7, z=1.0):
    setup()
    picks = [list(s["cls"]).index(c) for c in ("periodic", "bursty")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8))
    day = slice(PER_DAY, 2 * PER_DAY)                       # day 11
    hours = np.arange(PER_DAY) / 6
    for ax, k in zip(axes, picks):
        y_full = s["series"][k].astype(float)
        c = y_full[TRAIN].mean() / K
        y = y_full[t_idx]
        nr = reactive_policy(y_full, t_idx, c, u)
        npred = predictive_policy(fc["mu"][k], fc["sigma"][k], c, z)
        ax.fill_between(hours, 0, y[day] / c, color="#bbbbbb", lw=0, label="actual demand / c")
        ax.step(hours, nr[day], where="post", color=COLORS["reactive"], lw=1.4,
                label=f"reactive (u = {u})")
        ax.step(hours, npred[day], where="post", color=COLORS["predictive"], lw=1.4,
                label=f"predictive (z = {z})")
        ax.set_title(f"{s['names'][k]} ({s['cls'][k]}), day 11, K = {K}")
        ax.set_ylabel("instances")
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_xlabel("hour of day")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(RES / "part3_timeseries.png", dpi=150)


if __name__ == "__main__":
    main()
