"""Stream processor: invocations -> 10-minute windows -> forecast -> scaling decision.

For every app the scaler keeps an event-time tumbling window of 10 trace-minutes.
When interval k closes (first message of a later interval arrives, or end-of-stream)
it appends y_k to the app's history and decides the capacity for interval k+2,
i.e. 20 minutes ahead (the instance start-up lead time):

    mu, sigma  = forecaster(history up to y_k)                  (Poisson GLM, Newton-fitted)
    n          = max(ceil((mu + z sigma)/c), ceil(y_k/(c u)))  (hybrid; reactive_only if the
                                                                model failed its WAPE gate)

Decisions go to the `scaling-decisions` topic keyed by app (consumers upsert by
(app, interval), so re-delivered decisions are harmless).

Delivery semantics
------------------
Window state and the input offsets that produced it are checkpointed together
(atomic file replace) after the batch's decisions have been flushed. On (re)start
the consumer seeks to the checkpointed offsets, so a crash anywhere replays exactly
the messages not yet reflected in the state: window counts are never double-counted,
and decisions are at-least-once and deterministic (identical on replay).
Kafka group offsets are committed too, for lag monitoring.

Models are hot-reloaded when the registry's LATEST pointer changes (Airflow retrain).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confluent_kafka import OFFSET_BEGINNING, Consumer, Producer, TopicPartition  # noqa: E402

from autoscaler.forecaster import HISTORY, predict  # noqa: E402
from autoscaler.policy import PolicyConfig, decide  # noqa: E402
from autoscaler.registry import Registry  # noqa: E402
from streaming import config  # noqa: E402

BIN = 10
LEAD = 2
KEEP = HISTORY + 16            # history kept per app (needs y[t-288] .. y[t-2])


@dataclass
class AppState:
    hist_start: int            # absolute interval of history[0]
    history: list
    open_interval: int         # window currently being filled
    open_count: int = 0
    done: bool = False

    def close(self) -> int:
        """Close the open window; return its interval index."""
        k = self.open_interval
        self.history.append(self.open_count)
        if len(self.history) > KEEP + 64:              # trim, keep indexes consistent
            drop = len(self.history) - KEEP
            self.history = self.history[drop:]
            self.hist_start += drop
        self.open_interval, self.open_count = k + 1, 0
        return k


@dataclass
class Checkpoint:
    offsets: dict = field(default_factory=dict)      # "partition" -> next offset
    apps: dict = field(default_factory=dict)         # app -> AppState as dict

    @staticmethod
    def load(path: Path) -> "Checkpoint":
        if not path.exists():
            return Checkpoint()
        d = json.loads(path.read_text())
        return Checkpoint(d["offsets"], d["apps"])

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"offsets": self.offsets, "apps": self.apps}))
        os.replace(tmp, path)                          # atomic


class Scaler:
    def __init__(self, registry: str, checkpoint: str, cfg: PolicyConfig):
        self.registry = Registry(registry)
        self.ckpt_path = Path(checkpoint)
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.version = None
        self.models = {}
        self.reload_models()
        ck = Checkpoint.load(self.ckpt_path)
        self.offsets = {int(p): o for p, o in ck.offsets.items()}
        self.state = {a: AppState(**s) for a, s in ck.apps.items()}
        self.producer = Producer(config.PRODUCER)
        self.stats = dict(messages=0, decisions=0, late=0, reloads=0)
        self.resumed = bool(self.state)
        if not self.resumed:
            for app, art in self.models.items():      # bootstrap from the training tail
                self.state[app] = AppState(art.trained_through - HISTORY,
                                           list(art.history_tail), art.trained_through)
                for t in (art.trained_through, art.trained_through + 1):
                    self.emit(app, t, decided_after=t - LEAD)

    # ------------------------------------------------------------------ models
    def reload_models(self) -> None:
        v = self.registry.latest_version()
        if v is None:
            raise SystemExit("registry has no published version; run the training first")
        if v != self.version:
            self.models, self.version = self.registry.load(v), v
            print(f"[scaler] loaded model version {v} ({len(self.models)} apps)", flush=True)

    # --------------------------------------------------------------- decisions
    def emit(self, app: str, t: int, decided_after: int) -> None:
        st, art = self.state[app], self.models[app]
        y_lag2 = st.history[t - LEAD - st.hist_start]
        mu, sigma = predict(art, st.history, st.hist_start, t)
        d = decide(art.mode, mu, sigma, y_lag2, art.capacity, self.cfg)
        rec = dict(app=app, name=art.name, interval=t, decided_after=decided_after, **d,
                   mu=round(mu, 3), sigma=round(sigma, 3), mode=art.mode,
                   capacity=art.capacity, model_version=self.version)
        self.producer.produce(config.TOPIC_DECISIONS, key=app, value=json.dumps(rec))
        self.stats["decisions"] += 1

    def on_message(self, msg: dict) -> None:
        app = msg["app"]
        st = self.state.get(app)
        if st is None or st.done:
            return
        k = msg["minute"] // BIN
        if k < st.open_interval:                       # data for an already closed window
            self.stats["late"] += 1
            return
        while st.open_interval < k:                    # close windows (gaps count as 0)
            closed = st.close()
            self.emit(app, closed + LEAD, decided_after=closed)
        if msg.get("eos"):
            st.done = True
            return
        st.open_count += msg["count"]

    # -------------------------------------------------------------------- loop
    def checkpoint(self) -> None:
        self.producer.flush(30)                        # decisions durable before state
        Checkpoint({str(p): o for p, o in self.offsets.items()},
                   {a: asdict(s) for a, s in self.state.items()}).save(self.ckpt_path)

    def run(self, idle_timeout: float, crash_after: int | None) -> dict:
        consumer = Consumer({"bootstrap.servers": config.BOOTSTRAP, "group.id": config.GROUP_ID,
                             "enable.auto.commit": False, "auto.offset.reset": "earliest",
                             "session.timeout.ms": 10000})   # detect a crashed member quickly
        assigned = []

        def on_assign(c, parts):                       # resume from OUR checkpointed offsets
            for p in parts:
                p.offset = self.offsets.get(p.partition, OFFSET_BEGINNING)
            c.assign(parts)
            assigned.append(time.time())
            print(f"[scaler] assigned {[(p.partition, p.offset) for p in parts]} "
                  f"({'resumed from checkpoint' if self.resumed else 'fresh start'})", flush=True)

        consumer.subscribe([config.TOPIC_INVOCATIONS], on_assign=on_assign)
        t0 = last_msg = last_log = time.time()
        try:
            while True:
                batch = consumer.consume(num_messages=2000, timeout=1.0)
                if not batch:
                    if self.registry.latest_version() != self.version:   # retrain while idle
                        self.reload_models()
                        self.stats["reloads"] += 1
                    if all(s.done for s in self.state.values()):
                        break
                    if assigned and time.time() - max(last_msg, assigned[-1]) > idle_timeout:
                        break                          # idle only counts once we own partitions
                    if not assigned and time.time() - t0 > 120:
                        raise TimeoutError("no partitions assigned within 120 s")
                    continue
                last_msg = time.time()
                for m in batch:
                    if m.error():
                        raise RuntimeError(m.error())
                    self.on_message(json.loads(m.value()))
                    self.offsets[m.partition()] = m.offset() + 1
                    self.stats["messages"] += 1
                    if crash_after and self.stats["messages"] >= crash_after:
                        print(f"[scaler] simulated crash after {crash_after} messages "
                              "(state since last checkpoint is lost)", flush=True)
                        os._exit(1)
                self.checkpoint()
                consumer.commit(offsets=[TopicPartition(config.TOPIC_INVOCATIONS, p, o)
                                         for p, o in self.offsets.items()], asynchronous=False)
                v = self.registry.latest_version()
                if v != self.version:
                    self.reload_models()
                    self.stats["reloads"] += 1
                if time.time() - last_log > 5:
                    last_log = time.time()
                    print(f"[scaler] {self.stats} {self.stats['messages'] / (time.time() - t0):,.0f} msg/s",
                          flush=True)
        finally:
            self.checkpoint()
            consumer.close()
        self.stats["seconds"] = round(time.time() - t0, 1)
        print(f"[scaler] finished {self.stats}", flush=True)
        return self.stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default="platform/data/registry")
    ap.add_argument("--checkpoint", default="platform/data/scaler_checkpoint.json")
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--u-safety", type=float, default=0.9)
    ap.add_argument("--idle-timeout", type=float, default=15.0)
    ap.add_argument("--crash-after", type=int, default=None,
                    help="testing: hard-exit after N messages without checkpointing")
    a = ap.parse_args()
    Scaler(a.registry, a.checkpoint, PolicyConfig(z=a.z, u_safety=a.u_safety)).run(
        a.idle_timeout, a.crash_after)
