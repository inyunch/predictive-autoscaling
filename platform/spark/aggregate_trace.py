"""PySpark ETL: Azure Functions trace -> per-app series, statistics and app selection.

Distributed re-implementation of src/prepare_data.py (the pandas version used in the
course study). Same rules, same output, but written so it scales out to a cluster:

  14 daily CSVs (46k functions x 1,444 cols each, ~2 GB)
    -> sum per (app, day)                       wide aggregation
    -> busy on day 1 (>= 50/min), present all 14 days
    -> posexplode to (app, minute, count)       unpivot, 11M rows
    -> 10-minute bins                           group by
    -> mean, exact p99 per minute, lag-144 autocorrelation (window function)
    -> classify periodic / bursty, pick top 5 of each

Outputs (Parquet, under --out):
  app_stats/        one row per active app with acf, p99/mean, class, rank
  series_10min/     (app, interval, count) for all active apps
  minute_selected/  (app, name, minute, count) per-minute traffic of the 10 selected apps
                    -> replayed into Kafka by streaming/producer.py
  selected_apps.csv

Run:  python platform/spark/aggregate_trace.py --raw data/raw --out platform/data/lake
(or spark-submit with the same arguments on a cluster; paths can be s3a:// URIs)
"""
from __future__ import annotations

import argparse
import time

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType

MIN_PER_DAY, N_DAYS, BIN, LAG = 1440, 14, 10, 144
MINUTE_COLS = [str(i) for i in range(1, MIN_PER_DAY + 1)]
SCHEMA = StructType(
    [StructField(c, StringType()) for c in ("HashOwner", "HashApp", "HashFunction", "Trigger")]
    + [StructField(c, LongType()) for c in MINUTE_COLS])


def spark_session(app_name: str = "autoscaling-etl") -> SparkSession:
    return (SparkSession.builder.appName(app_name)
            .master("local[*]")                       # ignored by spark-submit on a cluster
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.sql.codegen.wholeStage", "false")   # 1,440-column aggregate
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def quiet(spark: SparkSession) -> SparkSession:
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def app_day_totals(spark: SparkSession, raw: str) -> DataFrame:
    """(app, day, minutes: array<long>[1440]) = sum over the app's functions."""
    wide = (spark.read.csv(f"{raw}/invocations_per_function_md.anon.d*.csv",
                           header=True, schema=SCHEMA)
            .withColumn("day", F.regexp_extract(F.input_file_name(), r"\.d(\d+)\.csv", 1)
                        .cast("int")))
    summed = wide.groupBy("HashApp", "day").agg(*[F.sum(c).alias(c) for c in MINUTE_COLS])
    return summed.select(F.col("HashApp").alias("app"), "day",
                         F.array(*[F.col(c) for c in MINUTE_COLS]).alias("minutes"))


def busy_active_apps(app_day: DataFrame, busy_rate: float) -> tuple[DataFrame, DataFrame]:
    """(busy on day 1, busy AND present in all 14 daily files)."""
    totals = app_day.select("app", "day",
                            F.aggregate("minutes", F.lit(0).cast("long"), lambda a, x: a + x)
                            .alias("total"))
    busy = totals.where((F.col("day") == 1) & (F.col("total") / MIN_PER_DAY >= busy_rate))
    days = totals.groupBy("app").agg(F.countDistinct("day").alias("n_days"))
    busy = busy.select("app")
    return busy, busy.join(days, "app").where(F.col("n_days") == N_DAYS).select("app")


def per_minute(app_day: DataFrame, apps: DataFrame) -> DataFrame:
    """Unpivot to (app, minute, count) with minute = absolute minute since trace start."""
    return (app_day.join(F.broadcast(apps), "app")
            .select("app", "day", F.posexplode("minutes").alias("m", "count"))
            .select("app", ((F.col("day") - 1) * MIN_PER_DAY + F.col("m")).alias("minute"),
                    "count"))


def ten_minute(minute: DataFrame) -> DataFrame:
    return (minute.groupBy("app", (F.col("minute") / BIN).cast("int").alias("interval"))
            .agg(F.sum("count").alias("count")))


def app_statistics(minute: DataFrame, ten: DataFrame) -> DataFrame:
    """mean/min, exact p99/min (numpy 'linear' definition), lag-144 autocorrelation."""
    mstats = minute.groupBy("app").agg(
        F.mean("count").alias("mean_per_min"),
        F.percentile(F.col("count").cast("double"), F.lit(0.99)).alias("p99"))
    w = Window.partitionBy("app").orderBy("interval")
    centred = (ten.withColumn("x", F.col("count") - F.mean("count").over(Window.partitionBy("app")))
               .withColumn("x_lag", F.lag("x", LAG).over(w)))
    acf = centred.groupBy("app").agg(
        F.coalesce(F.try_divide(F.sum(F.col("x") * F.col("x_lag")), F.sum(F.col("x") * F.col("x"))),
                   F.lit(0.0)).alias("acf144"))           # constant series -> 0, as in pandas
    return (mstats.join(acf, "app")
            .withColumn("p99_over_mean", F.col("p99") / F.col("mean_per_min"))
            .drop("p99"))


def classify_and_select(stats: DataFrame, n_per_class: int) -> DataFrame:
    big = F.col("mean_per_min") > 100
    cls = (F.when(big & (F.col("acf144") > 0.8) & (F.col("p99_over_mean") < 3), "periodic")
           .when(big & (F.abs("acf144") < 0.15) & (F.col("p99_over_mean") > 8), "bursty")
           .otherwise("other"))
    stats = stats.withColumn("cls", cls)
    score = F.when(F.col("cls") == "periodic", F.col("acf144")).otherwise(F.col("p99_over_mean"))
    rank = F.row_number().over(Window.partitionBy("cls").orderBy(score.desc()))
    stats = stats.withColumn("rank", F.when(F.col("cls") != "other", rank))
    name = F.when((F.col("rank") <= n_per_class) & (F.col("cls") != "other"),
                  F.concat(F.upper(F.substring("cls", 1, 1)), F.col("rank").cast("string")))
    return stats.withColumn("name", name)


def main(raw: str, out: str, busy_rate: float = 50.0, n_per_class: int = 5) -> dict:
    t0 = time.time()
    spark = quiet(spark_session())
    app_day = app_day_totals(spark, raw).cache()
    busy, apps = busy_active_apps(app_day, busy_rate)
    apps = apps.cache()
    minute = per_minute(app_day, apps).cache()
    ten = ten_minute(minute).cache()
    stats = classify_and_select(app_statistics(minute, ten), n_per_class).cache()

    stats.write.mode("overwrite").parquet(f"{out}/app_stats")
    ten.write.mode("overwrite").parquet(f"{out}/series_10min")
    chosen = stats.where(F.col("name").isNotNull()).select("app", "name", "cls")
    (minute.join(F.broadcast(chosen), "app").select("app", "name", "minute", "count")
     .repartition("app").write.mode("overwrite").parquet(f"{out}/minute_selected"))
    sel = (stats.where(F.col("name").isNotNull())
           .select("name", "app", "cls", "mean_per_min", "acf144", "p99_over_mean")
           .toPandas().sort_values("name", key=lambda s: s.map(lambda v: (v[0] != "P", v))))
    sel.to_csv(f"{out}/selected_apps.csv", index=False, float_format="%.6f")

    summary = dict(
        busy_day1=busy.count(),
        active_all_days=apps.count(),
        classes={r["cls"]: r["count"] for r in stats.groupBy("cls").count().collect()},
        minute_rows=minute.count(), seconds=round(time.time() - t0, 1))
    print(summary)
    print(sel.to_string(index=False))
    spark.stop()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="platform/data/lake")
    ap.add_argument("--busy-rate", type=float, default=50.0)
    ap.add_argument("--n-per-class", type=int, default=5)
    a = ap.parse_args()
    main(a.raw, a.out, a.busy_rate, a.n_per_class)
