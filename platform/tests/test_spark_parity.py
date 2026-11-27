"""Spark ETL output must equal the pandas pipeline used in the course study.

Requires `python platform/spark/aggregate_trace.py` to have been run (needs the raw
trace); skipped otherwise.
"""
import numpy as np
import pandas as pd
import pytest

from autoscaler._paths import REPO

LAKE = REPO / "platform" / "data" / "lake"
pytestmark = pytest.mark.skipif(not (LAKE / "app_stats").exists(), reason="run the Spark job first")


def test_app_statistics_match_pandas():
    spark = pd.read_parquet(LAKE / "app_stats").set_index("app").sort_index()
    ref = pd.read_csv(REPO / "results" / "app_classification.csv").set_index("app").sort_index()
    assert list(spark.index) == list(ref.index)                       # same 565 apps
    for col in ("mean_per_min", "acf144", "p99_over_mean"):
        np.testing.assert_allclose(spark[col], ref[col], rtol=1e-9, atol=1e-12)
    assert (spark.cls == ref.cls).all()


def test_selected_series_match_pandas():
    s = np.load(REPO / "data" / "series.npz", allow_pickle=True)
    ten = pd.read_parquet(LAKE / "series_10min")
    for app, name, y in zip(s["apps"], s["names"], s["series"]):
        got = ten[ten.app == app].sort_values("interval")["count"].to_numpy()
        np.testing.assert_array_equal(got, y)
    sel = pd.read_csv(LAKE / "selected_apps.csv")
    assert list(sel.name) == list(s["names"]) and list(sel.app) == list(s["apps"])


def test_minute_data_aggregates_to_series():
    s = np.load(REPO / "data" / "series.npz", allow_pickle=True)
    m = pd.read_parquet(LAKE / "minute_selected")
    assert len(m) == 10 * 14 * 1440
    for app, y in zip(s["apps"], s["series"]):
        mm = m[m.app == app].sort_values("minute")["count"].to_numpy()
        np.testing.assert_array_equal(mm.reshape(-1, 10).sum(1), y)
