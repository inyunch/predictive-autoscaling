"""Check the streamed decisions against an offline recomputation, then score them.

1. Read every record on `scaling-decisions`, upsert by (app, interval).
2. Recompute each decision offline from the full series with the same artifact.
   Any mismatch (or missing interval) means the streaming path is wrong.
3. Replay the decisions against actual traffic: cost and overload, next to the
   reactive target-tracking baseline (u = 0.7) from the course study.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from confluent_kafka import Consumer, TopicPartition  # noqa: E402

from autoscaler._paths import REPO  # noqa: E402,F401
from autoscaler.data import load_selected  # noqa: E402
from autoscaler.forecaster import HISTORY, predict  # noqa: E402
from autoscaler.policy import PolicyConfig, decide  # noqa: E402
from autoscaler.registry import Registry  # noqa: E402
from simulate import metrics, reactive_policy  # noqa: E402  (course code)
from streaming import config  # noqa: E402

END = 14 * 144


def read_decisions(timeout: float = 5.0) -> tuple[dict, int]:
    c = Consumer({"bootstrap.servers": config.BOOTSTRAP, "group.id": "verify",
                  "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    md = c.list_topics(config.TOPIC_DECISIONS, timeout=10).topics[config.TOPIC_DECISIONS]
    c.assign([TopicPartition(config.TOPIC_DECISIONS, p, 0) for p in md.partitions])
    latest, total = {}, 0
    while True:
        batch = c.consume(2000, timeout=timeout)
        if not batch:
            break
        for m in batch:
            r = json.loads(m.value())
            latest[(r["app"], r["interval"])] = r      # upsert: last write wins
            total += 1
    c.close()
    return latest, total


def main(registry: str, z: float, u_safety: float, out: str | None) -> pd.DataFrame:
    cfg = PolicyConfig(z=z, u_safety=u_safety)
    decisions, total = read_decisions()
    reg = Registry(registry)
    versions = {}                                   # each decision is recomputed with the
    load = lambda v: versions.setdefault(v, reg.load(v))   # model version that made it
    start = min(r["interval"] for r in decisions.values())
    rows, mismatches, missing, by_version = [], 0, 0, {}
    for _, r in load_selected().iterrows():
        y = r.series
        n_stream = np.zeros(END - start)
        for t in range(start, END):
            got = decisions.get((r.app, t))
            if got is None:
                missing += 1
                continue
            art = load(got["model_version"])[r.app]
            by_version[got["model_version"]] = by_version.get(got["model_version"], 0) + 1
            mu, sg = predict(art, y[t - HISTORY:t - 1], t - HISTORY, t)
            expect = decide(art.mode, mu, sg, y[t - 2], art.capacity, cfg)["n"]
            mismatches += got["n"] != expect
            n_stream[t - start] = got["n"]
        yt = y[start:END]
        m = metrics(yt, n_stream, art.capacity)
        base = metrics(yt, reactive_policy(y, np.arange(start, END), art.capacity, 0.7), art.capacity)
        rows.append(dict(app=r["name"], mode=art.mode, cost=m["cost"],
                         overload_pct=100 * m["overload"],
                         reactive070_cost=base["cost"], reactive070_overload_pct=100 * base["overload"],
                         cost_change_pct=100 * (m["cost"] / base["cost"] - 1)))
    df = pd.DataFrame(rows)
    print(f"decision records read: {total}, unique (app, interval): {len(decisions)}, "
          f"duplicates (re-delivered): {total - len(decisions)}")
    print(f"decisions per model version: {dict(sorted(by_version.items()))}")
    print(f"missing decisions: {missing}, mismatches vs offline recomputation: {mismatches}")
    print(df.round(3).to_string(index=False))
    if out:
        df.to_csv(out, index=False, float_format="%.4f")
    if missing or mismatches:
        raise SystemExit("VERIFY FAILED")
    print("VERIFY OK: streaming decisions are identical to the offline computation")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default="platform/data/registry")
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--u-safety", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.registry, a.z, a.u_safety, a.out)
