"""Build per-app 10-minute invocation series from the Azure Functions 2019 trace.

Pipeline
--------
1. For each of the 14 daily files, sum per-minute invocation counts over all
   functions of an app.
2. Keep apps with mean >= 50 invocations/min on day 1 ("busy") and present in
   all 14 daily files ("active").
3. Classify each app by
     acf  = autocorrelation of the 10-minute series at lag 144 (one day)
     burst = p99 / mean of the per-minute series
   periodic: acf > 0.8,  burst < 3, mean > 100/min
   bursty:   |acf| < 0.15, burst > 8, mean > 100/min
4. Keep the top 5 of each class (periodic by acf, bursty by burst ratio) and
   write their 10-minute series (2016 points each) to data/series.npz.

Source: Shahrad et al., "Serverless in the Wild", USENIX ATC'20 (CC BY 4.0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
N_DAYS, MIN_PER_DAY, BIN = 14, 1440, 10
BUSY_MIN_RATE = 50.0
MIN_COLS = [str(i) for i in range(1, MIN_PER_DAY + 1)]


def load_day(day: int, keep: set[str] | None = None) -> pd.DataFrame:
    """Return an (apps x 1440) frame of per-minute invocations for one day."""
    path = RAW / f"invocations_per_function_md.anon.d{day:02d}.csv"
    df = pd.read_csv(path, usecols=["HashApp", *MIN_COLS],
                     dtype={c: np.float32 for c in MIN_COLS})
    if keep is not None:
        df = df[df["HashApp"].isin(keep)]
    return df.groupby("HashApp", sort=False)[MIN_COLS].sum()


def acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    denom = np.dot(x, x)
    return float(np.dot(x[:-lag], x[lag:]) / denom) if denom > 0 else 0.0


def classify(minute: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for app, m in minute.items():
        ten = m.reshape(-1, BIN).sum(axis=1)
        mean = m.mean()
        rows.append(dict(app=app, mean_per_min=mean,
                         acf144=acf(ten, 144),
                         p99_over_mean=np.percentile(m, 99) / mean if mean > 0 else np.inf))
    df = pd.DataFrame(rows)
    big = df.mean_per_min > 100
    df["cls"] = "other"
    df.loc[big & (df.acf144 > 0.8) & (df.p99_over_mean < 3), "cls"] = "periodic"
    df.loc[big & (df.acf144.abs() < 0.15) & (df.p99_over_mean > 8), "cls"] = "bursty"
    return df


def main(n_per_class: int = 5) -> None:
    day1 = load_day(1)
    busy = day1.index[day1.mean(axis=1) >= BUSY_MIN_RATE]
    print(f"apps on day 1: {len(day1)}, busy (>= {BUSY_MIN_RATE}/min): {len(busy)}")
    keep = set(busy)
    days = [day1.loc[busy]]
    for d in range(2, N_DAYS + 1):
        days.append(load_day(d, keep))
        print(f"  loaded day {d}", flush=True)
    active = set(busy)
    for df in days[1:]:
        active &= set(df.index)
    active = sorted(active)
    print(f"active all {N_DAYS} days: {len(active)}")

    minute = {a: np.concatenate([df.loc[a].to_numpy(np.float64) for df in days])
              for a in active}
    stats = classify(minute)
    stats.sort_values("app").to_csv(ROOT / "results" / "app_classification.csv", index=False)
    print(stats.cls.value_counts().to_string())

    per = stats[stats.cls == "periodic"].nlargest(n_per_class, "acf144")
    bur = stats[stats.cls == "bursty"].nlargest(n_per_class, "p99_over_mean")
    chosen = pd.concat([per, bur])
    series = np.stack([minute[a].reshape(-1, BIN).sum(axis=1) for a in chosen.app])
    names = [f"{c[0].upper()}{i + 1}" for c, i in
             zip(chosen.cls, list(range(len(per))) + list(range(len(bur))))]
    np.savez_compressed(ROOT / "data" / "series.npz", series=series,
                        apps=chosen.app.to_numpy(), cls=chosen.cls.to_numpy(),
                        names=np.array(names))
    out = chosen.assign(name=names)[["name", "app", "cls", "mean_per_min",
                                     "acf144", "p99_over_mean"]]
    out.to_csv(ROOT / "results" / "selected_apps.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"series shape: {series.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-per-class", type=int, default=5)
    main(ap.parse_args().n_per_class)
