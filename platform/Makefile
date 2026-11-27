# Platform layer targets. Run from the repository root:  make -f platform/Makefile <target>
PY ?= .venv-platform/bin/python
KAFKA_HOME ?= /opt/kafka
export PYTHONPATH := platform
export KAFKA_HOME

.PHONY: spark train kafka-up kafka-down topics produce scale verify demo airflow-test test docker-up

spark:                      ## PySpark ETL: raw trace -> Parquet lake
	$(PY) platform/spark/aggregate_trace.py --raw data/raw --out platform/data/lake

train:                      ## train + publish models as of day 10
	$(PY) -m autoscaler.train --as-of-day 10

kafka-up:                   ## local KRaft broker + topics
	platform/scripts/kafka_local.sh start && platform/scripts/kafka_local.sh topics

kafka-down:
	platform/scripts/kafka_local.sh stop

produce:
	$(PY) platform/streaming/producer.py --from-day 10

scale:
	$(PY) platform/streaming/scaler.py

verify:
	$(PY) platform/streaming/verify.py --out platform/results/streaming_verify.csv

demo:                       ## Airflow + Kafka daily retrain loop, then verify
	$(PY) platform/scripts/demo_retrain_loop.py

airflow-test:               ## one DAG run for trace day 10
	. platform/scripts/airflow_env.sh && $(dir $(PY))airflow db migrate >/dev/null && \
	  $(dir $(PY))airflow dags test autoscaling_pipeline 2019-07-25

test:
	$(PY) -m pytest -q platform/tests

docker-up:
	docker compose -f platform/docker-compose.yml up -d --build
