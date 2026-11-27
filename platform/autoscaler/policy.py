"""Scaling decisions. Same formulas as the offline study (src/simulate.py).

predictive : n = ceil((mu + z*sigma) / c)          (KKT equal-quantile rule)
reactive   : n = ceil(y_{t-2} / (c*u))             (target tracking)
hybrid     : n = max(predictive, reactive)         (AWS: the larger policy wins)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    z: float = 1.0           # forecast quantile (Part 2: equal-quantile rule)
    u_safety: float = 0.9    # target utilization of the reactive safety net
    n_min: int = 1


def predictive(mu: float, sigma: float, c: float, cfg: PolicyConfig) -> int:
    return max(cfg.n_min, math.ceil((mu + cfg.z * sigma) / c))


def reactive(y_lag2: float, c: float, cfg: PolicyConfig) -> int:
    return max(cfg.n_min, math.ceil(y_lag2 / (c * cfg.u_safety)))


def decide(mode: str, mu: float, sigma: float, y_lag2: float, c: float,
           cfg: PolicyConfig) -> dict:
    n_r = reactive(y_lag2, c, cfg)
    if mode == "reactive_only":
        return dict(n_predictive=None, n_reactive=n_r, n=n_r)
    n_p = predictive(mu, sigma, c, cfg)
    return dict(n_predictive=n_p, n_reactive=n_r, n=max(n_p, n_r))
