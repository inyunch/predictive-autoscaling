"""Window / decision logic of the stream scaler, without a Kafka broker."""
import numpy as np
import pytest

from autoscaler._paths import REPO
from autoscaler.forecaster import HISTORY, fit_artifact, predict
from autoscaler.policy import PolicyConfig, decide
from autoscaler.registry import Registry

pytest.importorskip("confluent_kafka")
from streaming.scaler import Scaler  # noqa: E402

S = np.load(REPO / "data" / "series.npz", allow_pickle=True)
AS_OF = 1440


class FakeProducer:
    def __init__(self):
        self.records = []

    def produce(self, topic, key, value):
        import json
        self.records.append(json.loads(value))

    def flush(self, *_):
        return 0


@pytest.fixture
def scaler(tmp_path):
    reg = Registry(tmp_path / "reg")
    for k in (0, 5):                                   # P1 (hybrid) and B1 (reactive_only)
        reg.write("v1", fit_artifact(S["series"][k].astype(float), AS_OF,
                                     app=str(S["names"][k]), name=str(S["names"][k])))
    reg.publish("v1")
    sc = Scaler.__new__(Scaler)                        # build without a broker connection
    sc.registry, sc.cfg, sc.version = reg, PolicyConfig(), None
    sc.ckpt_path = tmp_path / "ckpt.json"
    sc.reload_models()
    sc.offsets, sc.state, sc.resumed = {}, {}, False
    sc.producer = FakeProducer()
    sc.stats = dict(messages=0, decisions=0, late=0, reloads=0)
    from streaming.scaler import AppState
    for app, art in sc.models.items():
        sc.state[app] = AppState(art.trained_through - HISTORY, list(art.history_tail),
                                 art.trained_through)
    return sc


def expected(app_idx, t, art):
    y = S["series"][app_idx].astype(float)
    mu, sg = predict(art, y[t - HISTORY:t - 1], t - HISTORY, t)
    return decide(art.mode, mu, sg, y[t - 2], art.capacity, PolicyConfig())["n"]


def test_minute_windows_produce_offline_decisions(scaler):
    for k, name in ((0, "P1"), (5, "B1")):
        y = S["series"][k]
        for t in range(AS_OF, AS_OF + 40):             # split each 10-min count over 3 minutes
            c = int(y[t])
            for m, part in zip((0, 4, 9), (c // 3, c // 3, c - 2 * (c // 3))):
                scaler.on_message({"app": name, "minute": t * 10 + m, "count": part})
        scaler.on_message({"app": name, "minute": (AS_OF + 40) * 10, "count": 0, "eos": True})
    got = {(r["app"], r["interval"]): r["n"] for r in scaler.producer.records}
    for k, name in ((0, "P1"), (5, "B1")):
        art = scaler.models[name]
        for t in range(AS_OF + 2, AS_OF + 42):
            assert got[(name, t)] == expected(k, t, art)
    assert scaler.models["B1"].mode == "reactive_only"


def test_gaps_close_as_zero_and_late_data_is_dropped(scaler):
    scaler.on_message({"app": "P1", "minute": AS_OF * 10, "count": 5})
    scaler.on_message({"app": "P1", "minute": (AS_OF + 3) * 10, "count": 7})   # 3 windows close
    st = scaler.state["P1"]
    assert st.history[-3:] == [5, 0, 0] and st.open_interval == AS_OF + 3
    scaler.on_message({"app": "P1", "minute": (AS_OF + 1) * 10, "count": 99})  # late
    assert scaler.stats["late"] == 1 and st.history[-3:] == [5, 0, 0]


def test_checkpoint_round_trip(scaler):
    from streaming.scaler import Checkpoint
    from dataclasses import asdict
    scaler.on_message({"app": "P1", "minute": AS_OF * 10 + 3, "count": 11})
    scaler.offsets = {0: 42}
    scaler.checkpoint()
    ck = Checkpoint.load(scaler.ckpt_path)
    assert ck.offsets == {"0": 42}
    assert ck.apps["P1"] == asdict(scaler.state["P1"])
