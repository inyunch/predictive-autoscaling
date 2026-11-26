"""Replay real traffic against an autoscaling policy (Part 3, RQ3).

Each policy picks n_t (instances for interval t) at time t - H using only
information available then. Cost and overload are measured on the actual trace.
"""
from __future__ import annotations

import numpy as np

H = 2  # decision lead time, intervals (20 min)


def reactive_policy(y_full: np.ndarray, t_idx: np.ndarray, c: float, u: float) -> np.ndarray:
    """Target-tracking style: size for the last observed load at utilization u.

    n_t = ceil(y_{t-H} / (c u)), at least 1.
    """
    return np.maximum(1, np.ceil(y_full[t_idx - H] / (c * u)))


def predictive_policy(mu: np.ndarray, sigma: np.ndarray, c: float, z: float) -> np.ndarray:
    """Provision the forecast quantile mu + z sigma (the KKT form from Part 2).

    n_t = ceil((mu_t + z sigma_t) / c), at least 1.
    """
    return np.maximum(1, np.ceil((mu + z * sigma) / c))


def metrics(y: np.ndarray, n: np.ndarray, c: float) -> dict:
    served_cap = c * n
    unserved = np.maximum(y - served_cap, 0.0)
    return dict(cost=float(n.sum()),
                overload=float(unserved.sum() / y.sum()),
                overloaded_intervals=float(np.mean(unserved > 0)),
                mean_util=float(np.mean(np.minimum(y, served_cap) / served_cap)))


def lower_envelope(cost: np.ndarray, over: np.ndarray):
    """Pareto frontier (min overload for each cost), sorted by cost."""
    o = np.argsort(cost)
    c, v = cost[o], over[o]
    keep = v <= np.minimum.accumulate(v) + 1e-15
    return c[keep], np.minimum.accumulate(v)[keep]


def overload_at_cost(cost: np.ndarray, over: np.ndarray, target: float) -> float:
    """Interpolate a policy's frontier at a given cost (NaN outside its range)."""
    c, v = lower_envelope(cost, over)
    if target < c[0] or target > c[-1]:
        return np.nan
    return float(np.interp(target, c, v))
