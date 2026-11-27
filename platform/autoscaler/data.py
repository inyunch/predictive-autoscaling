"""Load the per-app series from the Spark lake, falling back to the committed npz."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autoscaler._paths import REPO

LAKE = REPO / "platform" / "data" / "lake"


def load_selected(lake: Path = LAKE) -> pd.DataFrame:
    """DataFrame with columns name, app, cls, series (np.ndarray of 10-min counts)."""
    if (lake / "series_10min").exists() and (lake / "selected_apps.csv").exists():
        sel = pd.read_csv(lake / "selected_apps.csv")
        ten = pd.read_parquet(lake / "series_10min", filters=[("app", "in", list(sel.app))])
        by_app = {a: g.sort_values("interval")["count"].to_numpy(float)
                  for a, g in ten.groupby("app")}
        sel["series"] = [by_app[a] for a in sel.app]
        sel.attrs["source"] = str(lake)
        return sel[["name", "app", "cls", "series"]]
    s = np.load(REPO / "data" / "series.npz", allow_pickle=True)
    df = pd.DataFrame(dict(name=s["names"], app=s["apps"], cls=s["cls"],
                           series=list(s["series"].astype(float))))
    df.attrs["source"] = "data/series.npz"
    return df


def minute_stream(lake: Path = LAKE) -> pd.DataFrame:
    """Per-minute (app, minute, count) for the selected apps.

    Falls back to spreading each 10-minute count onto its first minute when the
    Spark lake is not available (same 10-minute aggregates, fewer messages)."""
    p = lake / "minute_selected"
    if p.exists():
        return pd.read_parquet(p)[["app", "minute", "count"]]
    rows = [(a, t * 10, c) for a, y in zip(load_selected().app, load_selected().series)
            for t, c in enumerate(y)]
    return pd.DataFrame(rows, columns=["app", "minute", "count"])
