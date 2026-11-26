import numpy as np

from simulate import metrics, overload_at_cost, predictive_policy, reactive_policy


def test_metrics_hand_computed():
    y = np.array([10.0, 30.0, 20.0])
    m = metrics(y, np.array([1, 2, 3]), c=10.0)
    assert m["cost"] == 6 and abs(m["overload"] - 10 / 60) < 1e-12


def test_policies_use_lead_time_and_floor():
    y = np.array([0, 0, 100, 0, 0, 0], float)
    n = reactive_policy(y, np.arange(2, 6), c=10.0, u=0.5)
    assert list(n) == [1, 1, 20, 1]          # spike at t=2 is seen only at t=4
    assert list(predictive_policy(np.array([5.0, 95.0]), np.array([1.0, 5.0]), 10.0, 1.0)) == [1, 10]


def test_overload_at_cost_interpolates_envelope():
    cost = np.array([1.0, 2.0, 3.0, 2.5])
    over = np.array([0.5, 0.2, 0.0, 0.3])    # (2.5, 0.3) is dominated
    assert abs(overload_at_cost(cost, over, 2.5) - 0.1) < 1e-12
    assert np.isnan(overload_at_cost(cost, over, 4.0))
