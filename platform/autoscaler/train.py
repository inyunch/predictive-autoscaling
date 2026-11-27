"""Train one model per selected app and publish them as a registry version.

Used by the Airflow DAG (one mapped task per app) and runnable stand-alone:
    python -m autoscaler.train --as-of-day 10 --registry platform/data/registry
"""
from __future__ import annotations

import argparse
import json

from autoscaler.data import load_selected
from autoscaler.forecaster import fit_artifact
from autoscaler.registry import Registry

PER_DAY = 144
CAPACITY_WINDOW = slice(2 * PER_DAY, 10 * PER_DAY)    # fixed: c is a property of the instance


def capacity_for(series, k_instances: float) -> float:
    """Requests per instance-interval such that mean load needs K instances (days 2-9)."""
    return float(series[CAPACITY_WINDOW].mean() / k_instances)


def version_for(as_of_day: int) -> str:
    return f"day{as_of_day:02d}"


def train_one(app: str, as_of_day: int, registry: str, k_instances: float = 10.0,
              wape_gate: float = 0.25) -> dict:
    df = load_selected().set_index("app")
    row = df.loc[app]
    art = fit_artifact(row.series, as_of_day * PER_DAY, app=app, name=row["name"],
                       k_instances=k_instances, wape_gate=wape_gate,
                       capacity=capacity_for(row.series, k_instances))
    Registry(registry).write(version_for(as_of_day), art)
    return dict(app=app, name=row["name"], mode=art.mode, wape_holdout=round(art.wape_holdout, 4),
                newton_iters=art.newton_iters)


def train_all(as_of_day: int, registry: str, publish: bool = True) -> list[dict]:
    apps = list(load_selected().app)
    out = [train_one(a, as_of_day, registry) for a in apps]
    if publish:
        Registry(registry).publish(version_for(as_of_day), expected_apps=apps)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--as-of-day", type=int, default=10)
    ap.add_argument("--registry", default="platform/data/registry")
    a = ap.parse_args()
    for r in train_all(a.as_of_day, a.registry):
        print(json.dumps(r))
