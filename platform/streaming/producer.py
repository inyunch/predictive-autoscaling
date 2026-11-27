"""Replay per-minute function traffic into Kafka, as if it were live telemetry.

Each message is one app-minute:  key = app,  value = {"app", "minute", "count"}.
Keying by app keeps every app's minutes in order within one partition, which is
what the scaler's per-app event-time windows rely on.

    python platform/streaming/producer.py --from-day 10 --speed 0       # as fast as possible
    python platform/streaming/producer.py --from-day 10 --speed 600     # 600 trace-minutes / s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confluent_kafka import Producer  # noqa: E402

from autoscaler.data import minute_stream  # noqa: E402
from streaming import config  # noqa: E402

MIN_PER_DAY = 1440


def replay(from_day: int, to_day: int, speed: float, eos: bool = True) -> int:
    df = minute_stream()
    df = df[(df.minute >= from_day * MIN_PER_DAY) & (df.minute < to_day * MIN_PER_DAY)]
    df = df.sort_values(["minute", "app"])          # interleave apps like real time
    producer = Producer(config.PRODUCER)
    errors = []
    on_delivery = lambda err, msg: errors.append(err) if err else None

    t0, sent, last_minute = time.time(), 0, None
    for app, minute, count in df.itertuples(index=False):
        if speed > 0 and minute != last_minute:     # pace by trace time
            target = t0 + (minute - df.minute.iloc[0]) / speed
            time.sleep(max(0.0, target - time.time()))
            last_minute = minute
        producer.produce(config.TOPIC_INVOCATIONS, key=app,
                         value=json.dumps({"app": app, "minute": int(minute), "count": int(count)}),
                         on_delivery=on_delivery)
        sent += 1
        if sent % 10000 == 0:
            producer.poll(0)
    # end-of-stream marker per app: lets the scaler close the last open window
    for app in (df.app.unique() if eos else []):
        producer.produce(config.TOPIC_INVOCATIONS, key=app,
                         value=json.dumps({"app": app, "minute": to_day * MIN_PER_DAY,
                                           "count": 0, "eos": True}),
                         on_delivery=on_delivery)
    producer.flush(60)
    if errors:
        raise RuntimeError(f"{len(errors)} delivery errors, first: {errors[0]}")
    dt = time.time() - t0
    print(f"produced {sent} messages ({df.app.nunique()} apps, days {from_day}-{to_day - 1}) "
          f"in {dt:.1f}s = {sent / dt:,.0f} msg/s")
    return sent


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-day", type=int, default=10)
    ap.add_argument("--to-day", type=int, default=14)
    ap.add_argument("--no-eos", action="store_true",
                    help="do not send end-of-stream markers (more days will follow)")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="trace minutes per wall-clock second (0 = unthrottled)")
    a = ap.parse_args()
    replay(a.from_day, a.to_day, a.speed, eos=not a.no_eos)
