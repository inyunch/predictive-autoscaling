"""Poisson log-linear forecasting model.

Forecast target: invocations in 10-minute interval t, using only data up to
t - H (H = 2 intervals = 20 minutes, the instance start-up lead time).

Features x_t:  1, log1p(y_{t-2}), log1p(y_{t-3}), log1p(y_{t-4}),
               log1p(y_{t-144}), log1p(y_{t-288}),
               sin/cos(2*pi*k*hour/24), k = 1, 2
Non-intercept columns are standardized with training statistics, and the
target is divided by its training mean so that w = 0 is a sensible start.

Loss (negative Poisson log-likelihood up to a constant, plus ridge):
    f(w) = mean(exp(Xw) - y * Xw) + (lam/2) * ||w[1:]||^2
    grad = X^T (mu - y) / n + lam * w[1:]
    hess = X^T diag(mu) X / n + lam * I[1:]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PER_DAY = 144
H = 2                      # decision lead time (intervals)
LAGS = (2, 3, 4, 144, 288)
TRAIN = slice(2 * PER_DAY, 10 * PER_DAY)    # days 2..9  (0-indexed)
TEST = slice(10 * PER_DAY, 14 * PER_DAY)    # days 10..13
LAM = 1e-3
FEATURE_NAMES = ["intercept", *[f"log1p(y[t-{l}])" for l in LAGS],
                 "sin(1h)", "cos(1h)", "sin(2h)", "cos(2h)"]


def load_series(path: Path | None = None) -> dict:
    d = np.load(path or ROOT / "data" / "series.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def raw_features(y: np.ndarray) -> np.ndarray:
    """Unstandardized feature matrix for every t (rows with t < 288 contain junk)."""
    T = len(y)
    t = np.arange(T)
    ly = np.log1p(y)
    cols = [np.ones(T)]
    for lag in LAGS:
        c = np.full(T, np.nan)
        c[lag:] = ly[:-lag]
        cols.append(c)
    hour = (t % PER_DAY) / PER_DAY * 24.0
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * hour / 24), np.cos(2 * np.pi * k * hour / 24)]
    return np.column_stack(cols)


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray      # scaled by y_scale
    X_test: np.ndarray
    y_test: np.ndarray       # scaled by y_scale
    y_scale: float
    t_test: np.ndarray       # absolute interval index of test rows


def build_dataset(y: np.ndarray) -> Dataset:
    X = raw_features(y.astype(float))
    Xtr, Xte = X[TRAIN].copy(), X[TEST].copy()
    mu, sd = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0)
    sd[sd == 0] = 1.0
    Xtr[:, 1:] = (Xtr[:, 1:] - mu) / sd
    Xte[:, 1:] = (Xte[:, 1:] - mu) / sd
    scale = float(y[TRAIN].mean())
    return Dataset(Xtr, y[TRAIN] / scale, Xte, y[TEST] / scale, scale,
                   np.arange(len(y))[TEST])


class PoissonLoss:
    """f, gradient and Hessian of the penalized Poisson loss."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lam: float = LAM):
        self.X, self.y, self.lam = X, y, lam
        self.n, self.d = X.shape
        self.pen = np.ones(self.d)
        self.pen[0] = 0.0          # intercept is not penalized
        self.nfev = self.ngev = self.nhev = 0

    def f(self, w):
        self.nfev += 1
        z = self.X @ w
        return float(np.mean(np.exp(z) - self.y * z) + 0.5 * self.lam * np.sum(self.pen * w * w))

    def grad(self, w):
        self.ngev += 1
        mu = np.exp(self.X @ w)
        return self.X.T @ (mu - self.y) / self.n + self.lam * self.pen * w

    def hess(self, w):
        self.nhev += 1
        mu = np.exp(self.X @ w)
        return (self.X.T * mu) @ self.X / self.n + self.lam * np.diag(self.pen)

    def predict(self, w, X=None):
        return np.exp((self.X if X is None else X) @ w)


def dispersion(y: np.ndarray, mu: np.ndarray, d: int) -> float:
    """Quasi-Poisson dispersion phi from Pearson residuals (counts, not scaled)."""
    return float(np.sum((y - mu) ** 2 / mu) / (len(y) - d))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum(np.abs(y - yhat)) / np.sum(y))
