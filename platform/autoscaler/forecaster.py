"""Train / serve the 20-minute-ahead Poisson forecaster as a deployable artifact.

Reuses the course code (src/model.py, src/optimizers.py) so the platform and the
offline study run exactly the same model. An artifact is plain JSON so that the
streaming scaler can load it without any training dependencies.

Time is measured in absolute 10-minute intervals since the start of the trace
(144 per day). `fit_artifact(y, as_of)` trains on the 8 days before `as_of`,
which for as_of = 1440 (day 10) is exactly the course's train split (days 2-9).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from autoscaler import _paths  # noqa: F401  (puts repo/src on sys.path)
from model import LAGS, LAM, PER_DAY, PoissonLoss, dispersion, raw_features, wape
from optimizers import newton

TRAIN_DAYS = 8
HISTORY = max(LAGS)                     # intervals of history needed to build x_t


@dataclass
class Artifact:
    app: str
    name: str
    trained_through: int                # first interval NOT used for training
    w: list
    feat_mean: list
    feat_sd: list
    y_scale: float
    phi: float
    capacity: float                     # requests one instance serves per interval
    wape_holdout: float
    wape_train: float
    newton_iters: int
    mode: str                           # "hybrid" or "reactive_only" (validation gate)
    history_tail: list = field(default_factory=list)   # y[as_of-HISTORY : as_of]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Artifact":
        return Artifact(**d)


def _fit(y: np.ndarray, start: int, end: int):
    """Fit on rows [start, end); return (w, mean, sd, scale, phi, iters, train WAPE)."""
    X = raw_features(y[:end].astype(float))[start:end]
    mean, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
    sd[sd == 0] = 1.0
    X[:, 1:] = (X[:, 1:] - mean) / sd
    scale = float(y[start:end].mean())
    prob = PoissonLoss(X, y[start:end] / scale, LAM)
    res = newton(prob, np.zeros(X.shape[1]), tol=1e-10, max_iter=100)
    mu = scale * prob.predict(res.w)
    phi = dispersion(y[start:end].astype(float), mu, X.shape[1])
    return res.w, mean, sd, scale, phi, res.iters, wape(y[start:end], mu)


def features_at(history, hist_start: int, t: int, mean, sd) -> np.ndarray:
    """Standardized feature vector for interval t.

    history[i] is y[hist_start + i]; it must cover y[t-288] .. y[t-2]
    (only lags >= 2 are read, so the interval being decided need not be known)."""
    if t - max(LAGS) < hist_start or t - min(LAGS) >= hist_start + len(history):
        raise IndexError(f"history [{hist_start}, {hist_start + len(history)}) "
                         f"does not cover lags of t={t}")
    get = lambda lag: history[t - lag - hist_start]
    hour = (t % PER_DAY) / PER_DAY * 24.0
    raw = [math.log1p(get(l)) for l in LAGS]
    for k in (1, 2):
        raw += [math.sin(2 * math.pi * k * hour / 24), math.cos(2 * math.pi * k * hour / 24)]
    x = (np.asarray(raw) - np.asarray(mean)) / np.asarray(sd)
    return np.concatenate([[1.0], x])


def predict(art: Artifact, history, hist_start: int, t: int) -> tuple[float, float]:
    """(mu_t, sigma_t) in requests per interval."""
    x = features_at(history, hist_start, t, art.feat_mean, art.feat_sd)
    mu = art.y_scale * math.exp(float(x @ np.asarray(art.w)))
    return mu, math.sqrt(art.phi * mu)


def fit_artifact(y: np.ndarray, as_of: int, app: str, name: str = "",
                 k_instances: float = 10.0, wape_gate: float = 0.25,
                 capacity: float | None = None) -> Artifact:
    """Train on the TRAIN_DAYS days before `as_of` with a holdout validation gate.

    1. Fit on the first TRAIN_DAYS-1 days, measure WAPE on the last day (holdout).
    2. Refit on all TRAIN_DAYS days; this is the model that gets deployed.
    3. If holdout WAPE > wape_gate the forecast is not trusted: mode = reactive_only.

    `capacity` (requests per instance per interval) is a property of the instance
    type, so callers should pass a fixed value; the default (training mean / K)
    reproduces the course study for as_of = day 10.
    """
    y = np.asarray(y, float)
    start = as_of - TRAIN_DAYS * PER_DAY
    if start < HISTORY:
        raise ValueError(f"need {HISTORY + TRAIN_DAYS * PER_DAY} intervals before as_of")
    # 1. holdout
    w, mean, sd, scale, *_ = _fit(y, start, as_of - PER_DAY)
    Xh = raw_features(y[:as_of])[as_of - PER_DAY:as_of]
    Xh[:, 1:] = (Xh[:, 1:] - mean) / sd
    wape_h = wape(y[as_of - PER_DAY:as_of], scale * np.exp(Xh @ w))
    # 2. refit on the full window
    w, mean, sd, scale, phi, iters, wape_tr = _fit(y, start, as_of)
    return Artifact(
        app=app, name=name, trained_through=as_of, w=w.tolist(), feat_mean=mean.tolist(),
        feat_sd=sd.tolist(), y_scale=scale, phi=phi,
        capacity=float(capacity if capacity is not None else y[start:as_of].mean() / k_instances),
        wape_holdout=float(wape_h), wape_train=float(wape_tr), newton_iters=int(iters),
        mode="hybrid" if wape_h <= wape_gate else "reactive_only",
        history_tail=y[as_of - HISTORY:as_of].tolist())
