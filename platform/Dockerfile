# One image for the Airflow scheduler/API server, the Spark job (local mode) and the
# Kafka producer/scaler. Java is needed by PySpark.
FROM python:3.11-slim-bookworm

ARG AIRFLOW_VERSION=3.0.6
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Optional: a corporate/proxy CA certificate passed as a BuildKit secret
#   docker build --secret id=ca,src=/path/to/ca.crt ...
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then cp /run/secrets/ca /usr/local/share/ca-certificates/extra.crt \
      && cat /run/secrets/ca >> /etc/ssl/certs/ca-certificates.crt; fi \
 && sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless ca-certificates curl procps \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

COPY platform/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
      -c "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

WORKDIR /opt/app
ENV PYTHONPATH=/opt/app/platform \
    AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/app/platform/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False
# the repository is bind-mounted at /opt/app by docker-compose (code + data + registry)
