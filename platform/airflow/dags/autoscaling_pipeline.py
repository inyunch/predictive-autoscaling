"""Daily retraining pipeline for the predictive autoscaler (Airflow 3, TaskFlow API).

    spark_etl ─► list_apps ─► train[app] (dynamic task mapping, one per app) ─► publish ─► report
                 as_of_day ─┘

* as_of_day     maps the run's logical date onto a trace day (trace day 0 = TRACE_START), so a
                backfill over 2019-07-25..07-28 replays days 10-13 exactly as production would.
* spark_etl     rebuilds the Parquet lake with PySpark (skipped unless the lake is missing or
                `rebuild_lake` is set; on a cluster this would be a SparkSubmitOperator / EMR step).
* train[app]    fits the Poisson model on the 8 days before as_of with a holdout WAPE gate and
                writes the artifact into registry version dayNN. Retries independently per app.
* publish       all-or-nothing: verifies every app produced an artifact, then atomically switches
                the registry LATEST pointer. The Kafka scaler hot-reloads on its next batch.
* report        writes a JSON run report (modes, holdout WAPE, Newton iterations).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pendulum
from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, get_current_context, task

PLATFORM = Path(__file__).resolve().parents[2]          # <repo>/platform
REPO = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

TRACE_START = date(2019, 7, 15)           # nominal calendar date of trace day 0
FIRST_DAY, LAST_DAY = 10, 14              # need 2 days of lags + 8 training days
REGISTRY = str(PLATFORM / "data" / "registry")
LAKE = PLATFORM / "data" / "lake"


@dag(
    dag_id="autoscaling_pipeline",
    schedule="@daily",
    start_date=pendulum.datetime(2019, 7, 25, tz="UTC"),
    end_date=pendulum.datetime(2019, 7, 29, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    params={"rebuild_lake": False, "wape_gate": 0.25, "k_instances": 10.0},
    tags=["autoscaling", "ml", "spark"],
    doc_md=__doc__,
)
def autoscaling_pipeline():

    @task
    def as_of_day() -> int:
        ctx = get_current_context()
        day = (ctx["logical_date"].date() - TRACE_START).days
        if not FIRST_DAY <= day <= LAST_DAY:
            raise AirflowSkipException(f"trace day {day} outside [{FIRST_DAY}, {LAST_DAY}]")
        return day

    @task
    def spark_etl() -> str:
        params = get_current_context()["params"]
        if (LAKE / "series_10min").exists() and not params["rebuild_lake"]:
            return f"lake up to date at {LAKE}"
        raw = REPO / "data" / "raw"
        if not any(raw.glob("invocations_per_function_md.anon.d*.csv")):
            raise FileNotFoundError(f"raw trace missing in {raw}; run `make data` first")
        cmd = [sys.executable, str(PLATFORM / "spark" / "aggregate_trace.py"),
               "--raw", str(raw), "--out", str(LAKE)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout[-3000:])
        if res.returncode:
            print(res.stderr[-5000:])
            raise RuntimeError(f"Spark ETL failed with exit code {res.returncode}")
        return f"lake rebuilt at {LAKE}"

    @task
    def list_apps(_etl: str) -> list[str]:
        from autoscaler.data import load_selected
        return list(load_selected().app)

    @task(max_active_tis_per_dagrun=4)
    def train(app: str, day: int) -> dict:
        from autoscaler.train import train_one
        params = get_current_context()["params"]
        return train_one(app, day, REGISTRY, k_instances=params["k_instances"],
                         wape_gate=params["wape_gate"])

    @task
    def publish(results: list[dict], day: int) -> dict:
        from autoscaler.registry import Registry
        from autoscaler.train import version_for
        results = list(results)
        version = version_for(day)
        Registry(REGISTRY).publish(version, expected_apps=[r["app"] for r in results])
        return dict(version=version, day=day, results=results)

    @task
    def report(summary: dict) -> str:
        out = PLATFORM / "data" / "reports" / f"{summary['version']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        modes = [r["mode"] for r in summary["results"]]
        summary["hybrid_apps"] = modes.count("hybrid")
        summary["reactive_only_apps"] = modes.count("reactive_only")
        out.write_text(json.dumps(summary, indent=2))
        for r in summary["results"]:
            print(f"{r['name']:>3}  {r['mode']:<14} holdout WAPE {r['wape_holdout']:.3f}  "
                  f"Newton iters {r['newton_iters']}")
        return str(out)

    day = as_of_day()
    apps = list_apps(spark_etl())
    results = train.partial(day=day).expand(app=apps)
    report(publish(results, day))


autoscaling_pipeline()
