"""The deployable forecaster must reproduce the offline study exactly."""
import json

import numpy as np
import pytest

from autoscaler._paths import REPO
from autoscaler.forecaster import Artifact, fit_artifact, predict
from autoscaler.policy import PolicyConfig, decide

SERIES = np.load(REPO / "data" / "series.npz", allow_pickle=True)
FC = np.load(REPO / "results" / "forecasts.npz", allow_pickle=True)
AS_OF = 1440                                   # start of test day 10


@pytest.mark.parametrize("k", range(10))
def test_streaming_predictions_match_offline_part1(k):
    y = SERIES["series"][k].astype(float)
    art = fit_artifact(y, AS_OF, app=str(SERIES["apps"][k]), name=str(SERIES["names"][k]))
    art = Artifact.from_dict(json.loads(json.dumps(art.to_dict())))   # JSON round trip
    for i, t in enumerate(range(AS_OF, AS_OF + 576, 37)):
        hist_start = t - 288
        mu, sg = predict(art, y[hist_start:t - 1], hist_start, t)   # only up to y[t-2]
        j = t - AS_OF
        assert mu == pytest.approx(FC["mu"][k][j], rel=1e-8)
        assert sg == pytest.approx(FC["sigma"][k][j], rel=1e-6)


def test_predict_refuses_to_peek_or_lack_history():
    y = SERIES["series"][0].astype(float)
    art = fit_artifact(y, AS_OF, app="a")
    with pytest.raises(IndexError):
        predict(art, y[1200:1439], 1200, 1440)      # missing y[t-288]
    with pytest.raises(IndexError):
        predict(art, y[1154:1440], 1154, 1442)      # missing y[t-2]


def test_validation_gate_marks_bad_forecasts_reactive_only():
    modes = {str(n): fit_artifact(s.astype(float), AS_OF, app=str(n)).mode
             for n, s in zip(SERIES["names"], SERIES["series"])}
    assert all(modes[f"P{i}"] == "hybrid" for i in range(1, 6))
    assert modes["B1"] == "reactive_only"


def test_decide_hybrid_takes_max():
    cfg = PolicyConfig(z=1.0, u_safety=0.9)
    d = decide("hybrid", mu=95.0, sigma=5.0, y_lag2=270.0, c=10.0, cfg=cfg)
    assert d == dict(n_predictive=10, n_reactive=30, n=30)
    assert decide("reactive_only", 1e9, 1e9, 9.0, 10.0, cfg)["n"] == 1
