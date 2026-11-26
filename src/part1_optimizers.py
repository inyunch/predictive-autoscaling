"""Part 1 (RQ1): compare 7 optimizers on fitting the Poisson forecast model.

Outputs
  results/part1_benchmark.csv        iterations, time, convergence per app x method
  results/part1_conditioning.csv     L0, L*, cond(H*), 1/L0 divergence check, f* vs SciPy
  results/part1_convergence.png      f(w_k) - f* for one periodic and one bursty app
  results/part1_forecast_accuracy.csv  test WAPE vs naive same-time-yesterday
  results/forecasts.npz              mu_t, sigma_t on test days (input to Parts 2-3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import optimizers as O
from model import (ROOT, PER_DAY, TEST, PoissonLoss, build_dataset, dispersion,
                   load_series, wape)
from plotstyle import COLORS, setup

np.seterr(all="ignore")
TOL, MAX_ITER, BETA = 1e-6, 5000, 0.9
RES = ROOT / "results"


def lam_max(Hm):
    return float(np.linalg.eigvalsh(Hm)[-1])


def run_methods(prob, w0, L):
    a = 1.0 / L
    return [O.gd_fixed(prob, w0, a, TOL, MAX_ITER),
            O.gd_wolfe(prob, w0, TOL, MAX_ITER),
            O.momentum(prob, w0, a, BETA, TOL, MAX_ITER),
            O.nesterov(prob, w0, a, BETA, TOL, MAX_ITER),
            O.ncg_pr(prob, w0, TOL, MAX_ITER, c2=0.1),
            O.newton(prob, w0, TOL, MAX_ITER),
            O.bfgs(prob, w0, TOL, MAX_ITER, c2=0.9)]


def main():
    s = load_series()
    bench, cond_rows, acc_rows, hist = [], [], [], {}
    fc = {k: [] for k in ("mu", "sigma", "y", "naive")}
    for y, name, cls in zip(s["series"], s["names"], s["cls"]):
        ds = build_dataset(y)
        prob = PoissonLoss(ds.X_train, ds.y_train)
        w0 = np.zeros(prob.d)

        ref = O.newton(prob, w0, tol=1e-12, max_iter=200)   # reference w*
        sp = minimize(prob.f, w0, jac=prob.grad, method="BFGS", options={"gtol": 1e-10})
        f_star = min(ref.f, sp.fun)
        H0, Hs = prob.hess(w0), prob.hess(ref.w)
        L0, Ls = lam_max(H0), lam_max(Hs)
        ev = np.linalg.eigvalsh(Hs)
        L = max(L0, Ls)

        gd_L0 = O.gd_fixed(prob, w0, 1.0 / L0, TOL, MAX_ITER)
        runs = run_methods(prob, w0, L)
        for r in runs:
            bench.append(dict(app=name, cls=cls, method=r.method, iters=r.iters,
                              time_ms=1e3 * r.time, converged=r.converged,
                              diverged=r.diverged, f_gap=r.f - f_star, gnorm=r.gnorm,
                              nfev=r.nfev, ngev=r.ngev, nhev=r.nhev))
        hist[name] = (runs, f_star, gd_L0)
        cond_rows.append(dict(app=name, cls=cls, L0=L0, L_star=Ls, L_used=L,
                              cond_H0=float(np.linalg.cond(H0)), cond_Hstar=ev[-1] / ev[0],
                              gd_1_over_L0_iters=gd_L0.iters,
                              gd_1_over_L0_converged=gd_L0.converged,
                              gd_1_over_L0_diverged=gd_L0.diverged,
                              f_star_newton=ref.f, f_star_scipy=sp.fun,
                              f_star_gap=ref.f - sp.fun,
                              w_dist=float(np.linalg.norm(ref.w - sp.x))))

        # forecast on held-out days (counts)
        mu_tr = ds.y_scale * prob.predict(ref.w)
        phi = dispersion(ds.y_train * ds.y_scale, mu_tr, prob.d)
        mu = ds.y_scale * prob.predict(ref.w, ds.X_test)
        yt = y[TEST].astype(float)
        naive = y[ds.t_test - PER_DAY].astype(float)
        acc_rows.append(dict(app=name, cls=cls, wape_model=wape(yt, mu),
                             wape_naive=wape(yt, naive),
                             wape_train=wape(ds.y_train * ds.y_scale, mu_tr), phi=phi))
        fc["mu"].append(mu); fc["sigma"].append(np.sqrt(phi * mu))
        fc["y"].append(yt); fc["naive"].append(naive)
        print(f"{name}: cond(H*)={ev[-1] / ev[0]:.1f}  L0={L0:.3f} L*={Ls:.3f}  "
              f"1/L0 GD {'DIVERGED' if gd_L0.diverged else 'ok'}  "
              f"WAPE {acc_rows[-1]['wape_model']:.3f} vs naive {acc_rows[-1]['wape_naive']:.3f}")

    bench = pd.DataFrame(bench)
    bench.to_csv(RES / "part1_benchmark.csv", index=False, float_format="%.6g")
    pd.DataFrame(cond_rows).to_csv(RES / "part1_conditioning.csv", index=False, float_format="%.6g")
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(RES / "part1_forecast_accuracy.csv", index=False, float_format="%.4f")
    np.savez_compressed(RES / "forecasts.npz", names=s["names"], cls=s["cls"],
                        t_test=np.arange(2016)[TEST], **{k: np.array(v) for k, v in fc.items()})

    summary = (bench.groupby(["cls", "method"], sort=False)
               .agg(median_iters=("iters", "median"), max_iters=("iters", "max"),
                    median_time_ms=("time_ms", "median"), converged=("converged", "sum"),
                    diverged=("diverged", "sum")).reset_index())
    summary.to_csv(RES / "part1_summary.csv", index=False, float_format="%.3g")
    print(summary.to_string(index=False))
    print(acc.to_string(index=False))
    plot_convergence(hist, s)
    plot_forecast(s, fc)


def plot_convergence(hist, s):
    setup()
    picks = [s["names"][list(s["cls"]).index(c)] for c in ("periodic", "bursty")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name in zip(axes, picks):
        runs, f_star, _ = hist[name]
        for r in runs:
            gap = np.maximum(np.array(r.f_hist) - f_star, 1e-16)
            ax.semilogy(gap, label=f"{r.method} ({r.iters})", color=COLORS[r.method], lw=1.6)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("iteration k")
        ax.set_title(f"{name} ({'periodic' if name[0] == 'P' else 'bursty'})")
        ax.set_ylim(1e-16, None)
    axes[0].set_ylabel(r"$f(w_k) - f^\ast$")
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Convergence of seven optimizers on the Poisson forecasting loss")
    fig.tight_layout()
    fig.savefig(RES / "part1_convergence.png", dpi=150)


def plot_forecast(s, fc):
    setup()
    picks = [list(s["cls"]).index(c) for c in ("periodic", "bursty")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5))
    days = np.arange(len(fc["y"][0])) / PER_DAY + 10
    for ax, k in zip(axes, picks):
        ax.plot(days, fc["y"][k], color="#222222", lw=0.8, label="actual")
        ax.plot(days, fc["mu"][k], color=COLORS["predictive"], lw=1.2, label=r"forecast $\mu_t$")
        ax.fill_between(days, np.maximum(fc["mu"][k] - 2 * fc["sigma"][k], 0), fc["mu"][k] + 2 * fc["sigma"][k],
                        color=COLORS["predictive"], alpha=0.2, lw=0, label=r"$\pm 2\sigma_t$")
        ax.set_title(f"{s['names'][k]} ({s['cls'][k]}): 20-minute-ahead forecast, test days 10-13")
        ax.set_ylabel("invocations / 10 min")
    axes[1].set_xlabel("day")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(RES / "part1_forecast.png", dpi=150)


if __name__ == "__main__":
    main()
