"""End-to-end demo: Kafka streaming + daily Airflow retraining + hot model reload.

  day 10 model (Airflow) -> start scaler
  for each test day d = 10..13:
      replay day d into Kafka, wait until the scaler has consumed it
      Airflow run for day d+1 retrains on the latest 8 days and publishes;
      the running scaler hot-reloads the new version
  verify every decision against an offline recomputation with the model version that made it

Prerequisites: broker running (scripts/kafka_local.sh start) and topics created.
    python platform/scripts/demo_retrain_loop.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
REPO = PLATFORM.parent
sys.path.insert(0, str(PLATFORM))
PY = sys.executable
AIRFLOW = str(Path(PY).with_name("airflow"))
TRACE_START = date(2019, 7, 15)

from confluent_kafka import Consumer, TopicPartition  # noqa: E402

from streaming import config  # noqa: E402


def airflow_env() -> dict:
    env = dict(os.environ, AIRFLOW_HOME=str(PLATFORM / "data" / "airflow-home"),
               AIRFLOW__CORE__DAGS_FOLDER=str(PLATFORM / "airflow" / "dags"),
               AIRFLOW__CORE__LOAD_EXAMPLES="False")
    return env


def run_dag(day: int) -> None:
    ds = (TRACE_START + timedelta(days=day)).isoformat()
    t0 = time.time()
    res = subprocess.run([AIRFLOW, "dags", "test", "autoscaling_pipeline", ds],
                         env=airflow_env(), capture_output=True, text=True)
    latest = (PLATFORM / "data" / "registry" / "LATEST").read_text().strip()
    ok = res.returncode == 0 and latest == f"day{day:02d}"
    print(f"[demo] airflow run {ds} (trace day {day}) -> registry LATEST={latest} "
          f"{'OK' if ok else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
    if not ok:
        print(res.stdout[-3000:], res.stderr[-3000:])
        raise SystemExit("DAG run failed")


def wait_caught_up(timeout: float = 300) -> None:
    c = Consumer({"bootstrap.servers": config.BOOTSTRAP, "group.id": config.GROUP_ID,
                  "enable.auto.commit": False})
    parts = [TopicPartition(config.TOPIC_INVOCATIONS, p) for p in range(config.PARTITIONS)]
    t0 = time.time()
    while time.time() - t0 < timeout:
        committed = c.committed(parts, timeout=10)
        lag = sum(c.get_watermark_offsets(p, timeout=10)[1] - max(cp.offset, 0)
                  for p, cp in zip(parts, committed))
        if lag == 0:
            c.close()
            return
        time.sleep(1)
    c.close()
    raise TimeoutError("scaler did not catch up")


def main() -> None:
    subprocess.run([str(PLATFORM / "scripts" / "kafka_local.sh"), "reset"], check=True,
                   capture_output=True)
    ckpt = PLATFORM / "data" / "scaler_checkpoint.json"
    ckpt.unlink(missing_ok=True)
    run_dag(10)
    log = open(PLATFORM / "data" / "scaler.log", "w")
    scaler = subprocess.Popen([PY, str(PLATFORM / "streaming" / "scaler.py"), "--idle-timeout", "600"],
                              stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    try:
        for d in range(10, 14):
            last = d == 13
            subprocess.run([PY, str(PLATFORM / "streaming" / "producer.py"), "--from-day", str(d),
                            "--to-day", str(d + 1), *([] if last else ["--no-eos"])], check=True)
            wait_caught_up()
            print(f"[demo] scaler consumed day {d}", flush=True)
            if not last:
                run_dag(d + 1)
                time.sleep(3)                       # let the idle scaler pick up the new version
        scaler.wait(timeout=120)
    finally:
        if scaler.poll() is None:
            scaler.terminate()
        log.close()
    print("".join(l for l in open(PLATFORM / "data" / "scaler.log") if "loaded model" in l
                  or "finished" in l), end="")
    out = PLATFORM / "results" / "streaming_verify.csv"
    out.parent.mkdir(exist_ok=True)
    subprocess.run([PY, str(PLATFORM / "streaming" / "verify.py"), "--out", str(out)], check=True)


if __name__ == "__main__":
    main()
